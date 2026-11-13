"""Stage 2: Ripple groups (one per true label of the misclassified samples) and the global Protect memory."""
import pandas as pd

from .utils import WRONG_TOKEN, RunContext, require, to_bool

RIPPLE_COLS = ["ripple_group_id", "sample_id", "text", "true_label", "predicted_label", "iteration", "prompt"]
PROTECT_COLS = ["sample_id", "text", "true_label", "predicted_label", "iteration", "prompt"]


def load_scoped_predictions(ctx: RunContext) -> pd.DataFrame:
    """Predictions of the baseline restricted to one-shot / prompt / iteration."""
    path = require(ctx.path("one_shot_domain_predictions.csv"), "Run the 'baseline' stage first.")
    d = pd.read_csv(path, dtype=str, keep_default_na=False)
    cfg = ctx.cfg
    return d[(d["shot"] == "one") & (d["prompt"] == cfg.prompt)
             & (d["iteration"].str.strip() == str(cfg.iteration))].copy()


def build_ripple_protect(ctx: RunContext):
    cfg = ctx.cfg
    scoped = load_scoped_predictions(ctx)

    # Drop invalid generations; they are neither edit targets nor protected.
    valid_mask = scoped["predicted_label"] != WRONG_TOKEN
    if "invalid_model_output" in scoped.columns:
        valid_mask &= ~to_bool(scoped["invalid_model_output"])
    valid = scoped[valid_mask].copy().reset_index(drop=True)
    valid["is_correct"] = valid["predicted_label"] == valid["true_label"]
    assert not valid["sample_id"].duplicated().any(), "sample_id not unique in scope"

    mis = valid[~valid["is_correct"]]
    correct = valid[valid["is_correct"]]
    print(f"scoped={len(scoped)} valid={len(valid)} | misclassified (Ripple)={len(mis)} "
          f"| correct (Protect)={len(correct)}")

    # One Ripple group per true label, holding every misclassified sample of that class.
    ripple_rows = []
    for gid, label in enumerate(sorted(mis["true_label"].unique())):
        members = mis[mis["true_label"] == label].sort_values("sample_id")
        for _, r in members.iterrows():
            ripple_rows.append({"ripple_group_id": gid, "sample_id": r["sample_id"], "text": r["text"],
                                "true_label": label, "predicted_label": r["predicted_label"],
                                "iteration": cfg.iteration, "prompt": cfg.prompt})
    ripple = pd.DataFrame(ripple_rows, columns=RIPPLE_COLS)

    # One global Protect memory = all correctly classified samples, shared by every group.
    protect = pd.DataFrame([{"sample_id": r["sample_id"], "text": r["text"], "true_label": r["true_label"],
                             "predicted_label": r["predicted_label"], "iteration": cfg.iteration,
                             "prompt": cfg.prompt} for _, r in correct.iterrows()], columns=PROTECT_COLS)

    # Invariants: groups partition the misclassified set by label; Ripple and Protect are disjoint.
    ripple_ids, protect_ids = set(ripple["sample_id"]), set(protect["sample_id"])
    assert ripple_ids == set(mis["sample_id"]) and len(ripple) == len(mis)
    assert protect_ids == set(correct["sample_id"])
    assert not (ripple_ids & protect_ids)
    if len(ripple):
        assert (ripple.groupby("ripple_group_id")["true_label"].nunique() == 1).all()
        assert (ripple.groupby("true_label")["ripple_group_id"].nunique() == 1).all()

    summary = (ripple.groupby(["ripple_group_id", "true_label"]).size().rename("ripple_size")
               .reset_index())
    summary["protect_pool_size"] = len(protect)

    ripple.to_csv(ctx.path("ripple_sets.csv"), index=False)
    protect.to_csv(ctx.path("protect_sets.csv"), index=False)
    summary.to_csv(ctx.path("ripple_protect_summary.csv"), index=False)
    print(summary.to_string(index=False) if len(summary) else "No misclassified samples: no Ripple groups.")
    return ripple, protect, summary
