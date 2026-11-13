"""Stage 1: one-shot in-context classification with the unedited model (W0).

Writes per-instance predictions (with the exact prompts) that every later stage reuses.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)
from tqdm.auto import tqdm

from .data import DatasetBundle
from .prompting import (build_label_space, class_probabilities, generate_label, one_shot_messages,
                        parse_output)
from .utils import WRONG_TOKEN, RunContext

REPORT_COLS = ["accuracy", "precision", "recall", "f1", "mcc", "roc_auc"]


def exemplars_for(data: DatasetBundle, labels, seed, iteration):
    """One exemplar per label, drawn from TRAIN and never overlapping the evaluation texts."""
    eval_texts = set(data.samples["text"].astype(str))
    avail = data.train_pool[~data.train_pool["text"].astype(str).isin(eval_texts)]
    rng = np.random.default_rng(seed + 1000 * iteration)
    chosen = []
    for lab in labels:
        cand = avail[avail["label"] == lab]
        if len(cand) == 0:
            continue
        row = cand.iloc[int(rng.integers(0, len(cand)))]
        chosen.append((row["text"], lab))
    return chosen


def compute_metrics(labels, y_true, y_pred, wrong_fraction, proba=None):
    m = {"accuracy": accuracy_score(y_true, y_pred),
         "precision": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
         "recall": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
         "f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
         "f1_micro": f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0),
         "mcc": matthews_corrcoef(y_true, y_pred),
         "wrong_label_fraction": wrong_fraction,
         "roc_auc": float("nan")}
    if proba is not None:
        present = [i for i, l in enumerate(labels) if l in set(y_true)]
        try:
            yidx = np.array([labels.index(t) for t in y_true])
            m["roc_auc"] = roc_auc_score(yidx, proba[:, present], labels=present,
                                         multi_class="ovr", average="macro")
        except Exception as e:
            print(f"[ROC-AUC n/a] {e}")
    return m


def run_baseline(ctx: RunContext, data: DatasetBundle = None):
    cfg = ctx.cfg
    data = data or DatasetBundle(cfg)
    tokenizer, model = ctx.load_model()
    gen_kwargs = ctx.gen_kwargs()

    labels, verb, rev = build_label_space(data.all)
    exemplars = exemplars_for(data, labels, cfg.seed, cfg.iteration)

    records, y_true, y_pred, proba, wrong = [], [], [], [], 0
    desc = f"{cfg.dataset}|one|{cfg.prompt}|it{cfg.iteration}"
    for _, r in tqdm(data.samples.iterrows(), total=len(data.samples), desc=desc):
        messages = one_shot_messages(cfg.prompt, verb, r["text"], exemplars)
        raw_output = generate_label(model, tokenizer, messages, ctx.device, gen_kwargs)
        pred = parse_output(raw_output, rev)
        invalid = pred is None
        if invalid:
            wrong += 1
            pred = WRONG_TOKEN
        true_label = str(r["label"])
        y_true.append(true_label)
        y_pred.append(pred)
        if cfg.compute_roc_auc:
            proba.append(class_probabilities(model, tokenizer, messages, labels, verb, ctx.device))
        records.append({
            "dataset": cfg.dataset,
            "sample_id": str(r["id"]),
            "text": str(r["text"]),
            "true_label": true_label,
            "predicted_label": pred,
            "is_correct": pred == true_label,
            "is_misclassified": pred != true_label,
            "invalid_model_output": invalid,
            "shot": "one",
            "prompt": cfg.prompt,
            "iteration": int(cfg.iteration),
            "raw_model_output": str(raw_output),
            "exact_messages_json": json.dumps(messages, ensure_ascii=False),
            "exemplars_json": json.dumps(exemplars, ensure_ascii=False),
        })

    metrics = compute_metrics(labels, y_true, y_pred, wrong / len(data.samples),
                              np.vstack(proba) if cfg.compute_roc_auc else None)
    metrics.update(dataset=cfg.dataset, shot="one", prompt=cfg.prompt, iter=cfg.iteration)
    metrics_df = pd.DataFrame([metrics])
    pred_df = pd.DataFrame(records)

    pred_df.to_csv(ctx.path("one_shot_domain_predictions.csv"), index=False)
    pred_df[pred_df["is_misclassified"]].to_csv(ctx.path("one_shot_domain_misclassifications.csv"),
                                                 index=False)
    metrics_df.to_csv(ctx.path("one_shot_metrics.csv"), index=False)

    print(metrics_df[REPORT_COLS].round(4).to_string(index=False))
    print(f"Saved predictions for {len(pred_df)} samples to {ctx.path('one_shot_domain_predictions.csv')}")
    return pred_df, metrics_df
