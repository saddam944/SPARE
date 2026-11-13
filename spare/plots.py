"""Figures: before/after classification bars, confusion matrices and the linguistic heatmap."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .linguistic import SUBGROUP_ORDER  # noqa: E402
from .utils import norm_label  # noqa: E402

METRICS = [("Accuracy", "Accuracy"), ("Precision_Macro", "Precision"), ("Recall_Macro", "Recall"),
           ("Macro_F1", "Macro-F1"), ("MCC", "MCC"), ("ROC_AUC", "ROC-AUC")]
COLORS = ["#005AD8", "#FF6B00", "#00A070"]


def _variant_file(run_dir: Path, dataset: str, name: str, variant="full"):
    return Path(run_dir) / variant / f"{dataset}_{name}"


def classification_bars(runs: dict, out_dir: Path, variant="full"):
    """One bar chart for W0 and one for WT, grouped by metric with a bar per dataset."""
    tables = {ds: pd.read_csv(_variant_file(d, ds, "classification_before_after.csv", variant)).set_index("Metric")
              for ds, d in runs.items()}
    for col, title, fname in [("Before_W0", "Classification Performance Before Model Editing", "before_edit.png"),
                              ("After_WT", "Classification Performance After Model Editing", "after_edit.png")]:
        fig, ax = plt.subplots(figsize=(13, 6.5))
        x = np.arange(len(METRICS))
        width = 0.8 / max(1, len(tables))
        for i, (ds, t) in enumerate(tables.items()):
            vals = [t.loc[m, col] if m in t.index else np.nan for m, _ in METRICS]
            ax.bar(x + (i - (len(tables) - 1) / 2) * width, vals, width, label=ds,
                   color=COLORS[i % len(COLORS)], edgecolor="white")
        ax.set_xticks(x, [label for _, label in METRICS], fontsize=13)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score", fontsize=13)
        ax.set_title(title, fontsize=18, fontweight="bold", pad=50)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=len(tables), frameon=False, fontsize=13)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(Path(out_dir) / fname, dpi=300)
        plt.close(fig)


def confusion_matrices(final_csv: Path, out_png: Path, abbreviate=True):
    """Row-normalized confusion matrices before (W0) and after (WT) editing for one dataset."""
    df = pd.read_csv(final_csv)
    classes = sorted(df["true_label"].map(norm_label).unique())
    cols = classes + ["wrong"]

    def matrix(pred_col):
        y = df["true_label"].map(norm_label)
        p = df[pred_col].map(norm_label).where(lambda s: s.isin(classes), "wrong")
        m = pd.crosstab(y, p).reindex(index=classes, columns=cols, fill_value=0).to_numpy(float)
        return m / np.maximum(m.sum(axis=1, keepdims=True), 1)

    names = [_abbrev(c) if abbreviate else c.replace("_", " ") for c in classes]
    fig, axes = plt.subplots(1, 2, figsize=(2 + 2.1 * len(cols), 0.9 * len(classes) + 2.5))
    for ax, col, title in [(axes[0], "predicted_label", "(a) Before Model Editing"),
                           (axes[1], "predicted_label_WT", "(b) After Model Editing")]:
        m = matrix(col)
        im = ax.imshow(m, cmap="Greens", vmin=0, vmax=1, aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=9, fontweight="bold",
                        color="white" if m[i, j] > 0.6 else "black")
        ax.set_xticks(range(len(cols)), names + ["wrong"], rotation=45, ha="right")
        ax.set_yticks(range(len(classes)), names)
        ax.set_xlabel("Predicted Label", fontweight="bold")
        ax.set_ylabel("True Label", fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=axes, label="Row-normalized proportion", fraction=0.025)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def linguistic_heatmap(runs: dict, out_png: Path, variant="full"):
    """Accuracy (%) per linguistic subgroup, before and after editing, for each dataset."""
    columns, data = [], []
    for ds, d in runs.items():
        t = pd.read_csv(_variant_file(d, ds, "linguistic_subgroups.csv", variant)).set_index("subgroup")
        t = t.reindex(SUBGROUP_ORDER)
        for col, tag in [("accuracy_before", "Before"), ("accuracy_after", "After")]:
            columns.append(f"{ds}\n{tag}")
            data.append(t[col].to_numpy(float))
    m = np.column_stack(data)
    fig, ax = plt.subplots(figsize=(2 + 1.8 * len(columns), 7))
    im = ax.imshow(np.ma.masked_invalid(m), cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            txt = "N/A" if np.isnan(m[i, j]) else f"{m[i, j]:.1f}"
            ax.text(j, i, txt, ha="center", va="center", fontweight="bold",
                    color="white" if (not np.isnan(m[i, j]) and m[i, j] > 70) else "black")
    ax.set_xticks(range(len(columns)), columns)
    ax.set_yticks(range(len(SUBGROUP_ORDER)), SUBGROUP_ORDER)
    fig.colorbar(im, ax=ax, label="Accuracy (%)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def _abbrev(label: str) -> str:
    words = [w for w in label.split("_") if w not in {"and", "or", "of"}]
    return "".join(w[0].upper() for w in words) or label


def make_all(runs: dict, out_dir: Path, variant="full"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    classification_bars(runs, out_dir, variant)
    linguistic_heatmap(runs, out_dir / "linguistic_subgroups.png", variant)
    for ds, d in runs.items():
        confusion_matrices(_variant_file(d, ds, "FINAL_W0_WT_generated.csv", variant),
                           out_dir / f"confusion_{ds}.png")
    print(f"Figures written to {out_dir}")
