"""Stage 5: evaluation of the edited model WT against the original W0.

Produces the classification metrics (Accuracy, macro Precision/Recall/F1, MCC, macro OvR ROC-AUC),
the knowledge-editing metrics (S, ES, RG, NS, FR, PC, GE, DR, PR) and the linguistic subgroup table.
"""
import json
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

from .editing import EditingEngine
from .linguistic import linguistic_subgroups
from .ripple_protect import load_scoped_predictions
from .utils import WRONG_TOKEN, RunContext, norm_label, to_bool


# -- 1. W0 -> WT generated predictions on every evaluation sample ------------------------
def final_predictions(engine: EditingEngine) -> pd.DataFrame:
    ctx = engine.ctx
    before = load_scoped_predictions(ctx).drop_duplicates("sample_id").reset_index(drop=True)
    prepared = {}
    for g in engine.groups:
        for ex in g["members"]:
            prepared[str(ex["sample_id"])] = ex
    for ex in engine.global_protect:
        prepared.setdefault(str(ex["sample_id"]), ex)

    rows = []
    for i, r in before.iterrows():
        sid = str(r["sample_id"])
        ex = prepared.get(sid) or engine.build_example(sid, str(r["true_label"]))
        p = engine.generated_prediction(ex)
        rows.append({"sample_id": sid, "predicted_label_WT": p["predicted_label"],
                     "raw_output_WT": p["raw_output"], "correct_WT": bool(p["correct"])})
        if (i + 1) % 50 == 0 or i + 1 == len(before):
            print(f"WT predictions: {i + 1}/{len(before)}")

    df = before.merge(pd.DataFrame(rows), on="sample_id", how="left", validate="one_to_one")
    df["correct_W0"] = to_bool(df["is_correct"])
    df["transition"] = np.select(
        [~df["correct_W0"] & df["correct_WT"], ~df["correct_W0"] & ~df["correct_WT"],
         df["correct_W0"] & df["correct_WT"]],
        ["FIXED", "STILL_WRONG", "PRESERVED"], default="DAMAGED")
    df.to_csv(ctx.edit_path("FINAL_W0_WT_generated.csv"), index=False)

    counts = df["transition"].value_counts()
    n_w0, n_wrong = int(df["correct_W0"].sum()), int((~df["correct_W0"]).sum())
    summary = {
        "total": len(df), "correct_W0": n_w0, "wrong_W0": n_wrong, "correct_WT": int(df["correct_WT"].sum()),
        **{k.lower(): int(counts.get(k, 0)) for k in ["FIXED", "STILL_WRONG", "PRESERVED", "DAMAGED"]},
        "accuracy_W0": n_w0 / len(df), "accuracy_WT": float(df["correct_WT"].mean()),
        "edit_success": counts.get("FIXED", 0) / n_wrong if n_wrong else np.nan,
        "preservation": counts.get("PRESERVED", 0) / n_w0 if n_w0 else np.nan,
        "damage_rate": counts.get("DAMAGED", 0) / n_w0 if n_w0 else np.nan,
    }
    pd.DataFrame([summary]).to_csv(ctx.edit_path("FINAL_W0_WT_metrics.csv"), index=False)
    print(pd.Series(summary).to_string())
    return df


# -- 2. Classification metrics and ROC-AUC ----------------------------------------------
def cls_metrics(y_true, y_pred):
    return {"Accuracy": accuracy_score(y_true, y_pred),
            "Precision_Macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "Recall_Macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "Macro_F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "MCC": matthews_corrcoef(y_true, y_pred)}


def _prompt_ids(tokenizer, messages_json):
    messages = json.loads(messages_json)
    content = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) \
        if isinstance(messages, list) else str(messages)
    return tokenizer(content, add_special_tokens=False, return_tensors="pt").input_ids


@torch.inference_mode()
def label_score_matrix(model, tokenizer, prompts, class_names, tag):
    """Softmax over classes of the mean teacher-forced log-likelihood of each label given the prompt."""
    model.eval()
    device = next(model.parameters()).device
    label_ids = {c: tokenizer(c, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
                 for c in class_names}
    rows = []
    for i, p in enumerate(prompts, 1):
        p = p.to(device)
        row = []
        for c in class_names:
            lids = label_ids[c]
            logits = model(input_ids=torch.cat([p, lids], 1)).logits[0, p.shape[1] - 1:p.shape[1] - 1 + lids.shape[1]]
            lp = torch.log_softmax(logits.float(), -1).gather(1, lids[0, :, None]).squeeze(1)
            row.append(float(lp.mean()))
        rows.append(row)
        if i % 50 == 0 or i == len(prompts):
            print(f"{tag} label scores: {i}/{len(prompts)}")
    s = np.asarray(rows, dtype=np.float64)
    e = np.exp(s - s.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def macro_ovr_auc(truth, probs, class_names):
    per_class = {}
    for j, c in enumerate(class_names):
        target = (np.asarray(truth) == c).astype(int)
        per_class[c] = float(roc_auc_score(target, probs[:, j])) if 0 < target.sum() < len(target) else np.nan
    return float(np.nanmean(list(per_class.values()))), per_class


def classification_report(engine: EditingEngine, df: pd.DataFrame):
    ctx, cfg = engine.ctx, engine.cfg
    y_true = df["true_label"].map(norm_label)
    y_w0, y_wt = df["predicted_label"].map(norm_label), df["predicted_label_WT"].map(norm_label)
    s0, st = cls_metrics(y_true, y_w0), cls_metrics(y_true, y_wt)

    classes = sorted(y_true.unique().tolist())
    prompts = [_prompt_ids(engine.tokenizer, m) for m in df["exact_messages_json"]]
    probs_wt = label_score_matrix(engine.model, engine.tokenizer, prompts, classes, "WT")
    auc_wt, auc_wt_cls = macro_ovr_auc(y_true, probs_wt, classes)

    # W0 scores come from a fresh copy of the original model; the edited model is left untouched.
    from transformers import AutoModelForCausalLM
    base = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=ctx.dtype).to(ctx.device)
    try:
        probs_w0 = label_score_matrix(base, engine.tokenizer, prompts, classes, "W0")
    finally:
        del base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    auc_w0, auc_w0_cls = macro_ovr_auc(y_true, probs_w0, classes)

    support = y_true.value_counts()
    pd.DataFrame([{"class": c, "support": int(support[c]), "AUC_W0": auc_w0_cls[c], "AUC_WT": auc_wt_cls[c]}
                  for c in classes]).to_csv(ctx.edit_path("ROC_AUC_per_class.csv"), index=False)

    s0["ROC_AUC"], st["ROC_AUC"] = auc_w0, auc_wt
    overall = pd.DataFrame([{"Metric": k, "Before_W0": s0[k], "After_WT": st[k], "Delta": st[k] - s0[k]}
                            for k in s0])
    overall.to_csv(ctx.edit_path("classification_before_after.csv"), index=False)
    print(overall.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return overall


# -- 3. Generation entropy (fluency) -----------------------------------------------------
def ngram_entropy(text, n):
    tokens = str(text).strip().split()
    if len(tokens) < n:
        return np.nan
    counts = Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    p = np.array(list(counts.values()), dtype=float)
    p /= p.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))


def generation_entropy(text):
    e2, e3 = ngram_entropy(text, 2), ngram_entropy(text, 3)
    if not (np.isfinite(e2) and np.isfinite(e3)):
        return np.nan
    return (2.0 / 3.0) * e2 + (4.0 / 3.0) * e3


@torch.inference_mode()
def fluency_ge(engine: EditingEngine, df: pd.DataFrame):
    cfg, tok, model = engine.cfg, engine.tokenizer, engine.model
    torch.manual_seed(cfg.seed)
    rows = []
    for _, r in df[["sample_id", "text"]].iterrows():
        prompt = ("Read the following disaster-related social media post "
                  "and briefly explain its content in natural language.\n\n"
                  f"Post: {r['text']}\n\nExplanation:")
        enc = {k: v.to(engine.device) for k, v in tok(prompt, return_tensors="pt").items()}
        out = model.generate(**enc, max_new_tokens=cfg.fluency_max_new_tokens, do_sample=True,
                             temperature=cfg.fluency_temperature, top_p=cfg.fluency_top_p,
                             pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        rows.append({"sample_id": r["sample_id"], "generated_text_WT": text, "GE": generation_entropy(text)})
    gen = pd.DataFrame(rows)
    gen.to_csv(engine.ctx.edit_path("fluency_generations.csv"), index=False)
    return float(gen["GE"].dropna().mean())


# -- 4. Paraphrase consistency and portability -------------------------------------------
class ProbeClassifier:
    """Free-form classification prompts used for the paraphrase and portability probes."""

    VARIANTS = [
        "Classify the following disaster-related social media post.\nAllowed labels:\n{labels}\n"
        "Return exactly ONE label from the allowed labels.\nDo not provide an explanation.\n"
        "Post:\n{text}\nLabel:",
        "Which category best describes this disaster-related post?\nCategories:\n{labels}\n"
        "Respond only with the exact category name.\nText:\n{text}\nCategory:",
        "Assign one disaster-response class to the text below.\nPossible classes:\n{labels}\n"
        "Output only one class name.\nInput:\n{text}\nAnswer:",
    ]

    def __init__(self, engine, labels):
        self.engine, self.labels = engine, labels
        self.label_text = ", ".join(labels)

    @torch.inference_mode()
    def generate(self, prompt, max_new_tokens):
        tok, model = self.engine.tokenizer, self.engine.model
        enc = {k: v.to(self.engine.device) for k, v in tok(prompt, return_tensors="pt").items()}
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def paraphrases(self, text, n):
        prompt = (f"Paraphrase the disaster-related social media post below in\n{n} different ways.\n"
                  "Rules:\n- Preserve exactly the same factual meaning.\n- Do not add new information.\n"
                  "- Do not remove important information.\n- Do not mention classification labels.\n"
                  f"- Return exactly {n} lines.\n- Put one paraphrase on each line.\n"
                  f"- Do not number the lines.\nOriginal post:\n{text}\nParaphrases:")
        out = []
        for line in self.generate(prompt, 180).splitlines():
            line = re.sub(r"^\s*[\-\*\d\.\)\:]+\s*", "", line.strip()).strip()
            if line and line not in out:
                out.append(line)
        return out[:n]

    def classify(self, text, variant=0):
        raw = self.generate(self.VARIANTS[variant].format(labels=self.label_text, text=text), 24)
        pred = norm_label(raw.splitlines()[0] if raw else WRONG_TOKEN)
        if pred not in self.labels:
            matches = [l for l in self.labels if l in norm_label(raw)]
            pred = matches[0] if len(matches) == 1 else WRONG_TOKEN
        return pred, raw


def paraphrase_and_portability(engine: EditingEngine, df: pd.DataFrame):
    cfg, ctx = engine.cfg, engine.ctx
    labels = sorted(df["true_label"].map(norm_label).unique().tolist())
    clf = ProbeClassifier(engine, labels)
    targets = df[df["true_label"].map(norm_label) != df["predicted_label"].map(norm_label)]

    para_rows, port_rows = [], []
    for i, (_, r) in enumerate(targets.iterrows(), 1):
        true_label = norm_label(r["true_label"])
        for k, para in enumerate(clf.paraphrases(str(r["text"]), cfg.n_paraphrases), 1):
            pred, raw = clf.classify(para, 0)
            para_rows.append({"sample_id": r["sample_id"], "true_label": true_label, "paraphrase_id": k,
                              "paraphrase": para, "prediction_WT": pred, "correct": pred == true_label,
                              "raw_prediction": raw})
        for v in range(len(ProbeClassifier.VARIANTS)):
            pred, raw = clf.classify(str(r["text"]), v)
            port_rows.append({"sample_id": r["sample_id"], "prompt_variant": v, "true_label": true_label,
                              "prediction": pred, "correct": pred == true_label, "raw_prediction": raw})
        if i % 20 == 0 or i == len(targets):
            print(f"paraphrase/portability probes: {i}/{len(targets)}")

    para, port = pd.DataFrame(para_rows), pd.DataFrame(port_rows)
    para.to_csv(ctx.edit_path("paraphrase_details.csv"), index=False)
    port.to_csv(ctx.edit_path("portability_details.csv"), index=False)
    consistency = [g["prediction_WT"].value_counts().iloc[0] / len(g) for _, g in para.groupby("sample_id")] \
        if len(para) else []
    return {"Paraphrase_Robustness": float(para["correct"].mean()) if len(para) else np.nan,
            "Paraphrase_Consistency": float(np.mean(consistency)) if consistency else np.nan,
            "Portability": float(port["correct"].mean()) if len(port) else np.nan}


# -- 5. Knowledge-editing metrics ---------------------------------------------------------
def editing_metrics(ctx: RunContext, df: pd.DataFrame, extra: dict):
    true = df["true_label"].map(norm_label).to_numpy()
    correct_w0 = df["predicted_label"].map(norm_label).to_numpy() == true
    correct_wt = df["predicted_label_WT"].map(norm_label).to_numpy() == true
    sids = df["sample_id"].astype(str)

    ripple = pd.read_csv(ctx.path("ripple_sets.csv"), dtype=str)
    protect = pd.read_csv(ctx.path("protect_sets.csv"), dtype=str)
    edit_mask = sids.isin(set(ripple["sample_id"])).to_numpy()
    protect_mask = sids.isin(set(protect["sample_id"])).to_numpy()

    # ES: initially misclassified (edit-target) instances that are correct at WT.
    ES = float(correct_wt[edit_mask].mean())

    # RG: accuracy of each Ripple group right after its own edit (W_t), over all groups.
    inst = pd.read_csv(ctx.edit_path("seq_edit_generated_per_instance.csv"))
    rip = inst[inst["set_type"] == "ripple"].drop_duplicates(["t", "group_id", "sample_id"], keep="last")
    RG = float(to_bool(rip["correct_after"]).mean())
    RG_before = float(to_bool(rip["correct_before"]).mean())

    # NS / PR / DR on the Protect memory.
    NS = float(correct_wt[protect_mask].mean())
    PR = float(correct_wt[protect_mask & correct_w0].mean())
    DR = 1.0 - PR

    # FR: 1 - retention of all earlier groups after the final edit.
    diag = pd.read_csv(ctx.edit_path("seq_edit_generated_diagnostic.csv")).sort_values("t")
    ret = pd.to_numeric(diag["prev_retention_teacher_top1"], errors="coerce").dropna()
    retention_final = float(ret.iloc[-1]) if len(ret) else np.nan
    FR = 1.0 - retention_final

    S = 3.0 / (1 / ES + 1 / RG + 1 / NS) if min(ES, RG, NS) > 0 else np.nan
    row = {"Dataset": ctx.cfg.dataset, "Variant": ctx.cfg.variant, "S": S, "ES": ES, "RG": RG, "NS": NS,
           "FR": FR, "PC": extra.get("Paraphrase_Consistency", np.nan), "GE": extra.get("GE", np.nan),
           "DR": DR, "PR": PR, "RG_before": RG_before, "Retention_final": retention_final,
           "Retention_mean": float(ret.mean()) if len(ret) else np.nan,
           "Protect_KL": float(pd.to_numeric(diag["protect_kl"], errors="coerce").mean()),
           "Paraphrase_Robustness": extra.get("Paraphrase_Robustness", np.nan),
           "Portability": extra.get("Portability", np.nan),
           "N_total": len(df), "N_edit_targets": int(edit_mask.sum()), "N_protect": int(protect_mask.sum())}
    out = pd.DataFrame([row])
    out.to_csv(ctx.edit_path("editing_metrics.csv"), index=False)
    print(out[["Dataset", "Variant", "S", "ES", "RG", "NS", "FR", "PC", "GE", "DR", "PR"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return out


def run_evaluation(ctx: RunContext, engine: EditingEngine = None):
    """Evaluate the model currently in memory (expected to be at WT after the 'edit' stage)."""
    engine = engine or EditingEngine(ctx)
    df = final_predictions(engine)
    classification_report(engine, df)
    extra = {}
    if ctx.cfg.run_fluency_eval:
        extra["GE"] = fluency_ge(engine, df)
    if ctx.cfg.run_paraphrase_eval:
        extra.update(paraphrase_and_portability(engine, df))
    metrics = editing_metrics(ctx, df, extra)
    linguistic_subgroups(df).to_csv(ctx.edit_path("linguistic_subgroups.csv"), index=False)
    return metrics
