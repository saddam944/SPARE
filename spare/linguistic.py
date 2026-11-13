"""Accuracy before and after editing for 12 (overlapping) linguistic subgroups of tweets.

Length-based groups split at the median of the evaluation sample; "uppercase heavy" means at
least 30% of the letters are upper case. These thresholds are the defaults of this script; adjust
them with the keyword arguments if needed.
"""
import re

import numpy as np
import pandas as pd

SUBGROUP_ORDER = ["Long Tweets", "Short Tweets", "Many Words", "Few Words", "Has Hashtags", "Has Mentions",
                  "Has URLs", "Has Numbers", "Has Special Chars", "Has Exclamation", "Has Question Mark",
                  "Uppercase Heavy"]

_URL = re.compile(r"https?://\S+|www\.\S+", re.I)


def _uppercase_ratio(text):
    letters = [c for c in text if c.isalpha()]
    return sum(c.isupper() for c in letters) / len(letters) if letters else 0.0


def subgroup_masks(texts: pd.Series, uppercase_threshold=0.3) -> dict:
    texts = texts.astype(str)
    n_chars = texts.str.len()
    n_words = texts.str.split().map(len)
    no_url = texts.map(lambda t: _URL.sub("", t))
    return {
        "Long Tweets": n_chars > n_chars.median(),
        "Short Tweets": n_chars <= n_chars.median(),
        "Many Words": n_words > n_words.median(),
        "Few Words": n_words <= n_words.median(),
        "Has Hashtags": texts.str.contains(r"#\w", regex=True),
        "Has Mentions": texts.str.contains(r"@\w", regex=True),
        "Has URLs": texts.str.contains(_URL),
        "Has Numbers": no_url.str.contains(r"\d", regex=True),
        "Has Special Chars": no_url.str.contains(r"[^\w\s#@.,!?'\"]", regex=True),
        "Has Exclamation": texts.str.contains("!", regex=False),
        "Has Question Mark": no_url.str.contains("?", regex=False),
        "Uppercase Heavy": texts.map(_uppercase_ratio) >= uppercase_threshold,
    }


def linguistic_subgroups(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Accuracy (%) per subgroup at W0 and WT; ``df`` is the FINAL_W0_WT_generated table."""
    rows = []
    for name, mask in subgroup_masks(df["text"], **kwargs).items():
        mask = mask.to_numpy()
        n = int(mask.sum())
        rows.append({"subgroup": name, "n": n,
                     "accuracy_before": 100 * float(df["correct_W0"].to_numpy()[mask].mean()) if n else np.nan,
                     "accuracy_after": 100 * float(df["correct_WT"].to_numpy()[mask].mean()) if n else np.nan})
    return pd.DataFrame(rows)
