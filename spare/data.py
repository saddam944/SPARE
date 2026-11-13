"""Dataset loading for HumAID, CrisisNLP and CrisisLexT26.

The TEST split is used for evaluation and the TRAIN split only for one-shot exemplars.
"""
import glob
import hashlib
import io
import os
import re
import zipfile

import numpy as np
import pandas as pd

from .config import SpareConfig

TEXT_CANDS = ["tweet_text", "text", "tweet", "message", "content"]
LABEL_CANDS = ["class_label", "label", "category", "class"]
ID_CANDS = ["tweet_id", "id", "tweetid"]


def _find_col(cols, cands):
    low = {c.lower().strip(): c for c in cols}
    for c in cands:
        if c in low:
            return low[c]
    for c in cands:
        for lc, orig in low.items():
            if c in lc:
                return orig
    return None


# --- HumAID --------------------------------------------------------------------------
HUMAID_REPO = "QCRI/HUMAID-all"


def _humaid_split_to_frame(split_ds):
    raw = split_ds.to_pandas().astype(str)
    tcol = _find_col(raw.columns, TEXT_CANDS)
    lcol = _find_col(raw.columns, LABEL_CANDS)
    icol = _find_col(raw.columns, ID_CANDS)
    return pd.DataFrame({"id": raw[icol] if icol else raw.index.astype(str),
                         "text": raw[tcol], "label": raw[lcol]})


def load_humaid(cfg: SpareConfig):
    from datasets import load_dataset
    ds = load_dataset(HUMAID_REPO, verification_mode="no_checks")
    print(f"[HumAID] splits: {list(ds.keys())}")
    return _humaid_split_to_frame(ds["train"]), _humaid_split_to_frame(ds["test"])


# --- CrisisNLP (raw LREC 2016 CrowdFlower release) -----------------------------------
CRISISNLP_ZIP_URLS = {
    "cf": "https://crisisnlp.qcri.org/data/lrec2016/labeled_cf/CrisisNLP_labeled_data_crowdflower_v2.zip",
    "volunteers": "https://crisisnlp.qcri.org/data/lrec2016/labeled_aidr/CrisisNLP_volunteers_labeled_data.zip",
}
CRISISNLP_INCLUDE = ("cf",)

CRISISNLP_9 = [
    "injured_or_dead_people",
    "missing_trapped_or_found_people",
    "displaced_people_and_evacuations",
    "infrastructure_and_utilities_damage",
    "donation_needs_or_offers_or_volunteering_services",
    "caution_and_advice",
    "sympathy_and_emotional_support",
    "other_useful_information",
    "not_related_or_irrelevant",
]


def _crisisnlp_norm(s):
    s = str(s).strip().lower()
    s = re.sub(r"[\s\-/]+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    return re.sub(r"_+", "_", s).strip("_")


CRISISNLP_LABEL_MAP = {}


def _reg(canon, *variants):
    for v in (canon,) + variants:
        CRISISNLP_LABEL_MAP[_crisisnlp_norm(v)] = canon


_reg("injured_or_dead_people", "injured_and_dead_people", "casualties")
_reg("missing_trapped_or_found_people", "missing_or_found_people", "missing_trapped_or_found",
     "missing_people")
_reg("displaced_people_and_evacuations", "displaced_people", "evacuations", "displaced_and_evacuations")
_reg("infrastructure_and_utilities_damage", "infrastructure_and_utility_damage",
     "infrastructure_and_utilities", "infrastructure_and_utilities_damage_or_service_interruption")
_reg("donation_needs_or_offers_or_volunteering_services", "donation_needs_or_offers",
     "donations_and_volunteering", "donation_and_volunteering", "donations_of_money_goods_or_services")
_reg("caution_and_advice", "caution_advice", "warnings")
_reg("sympathy_and_emotional_support", "sympathy_and_support", "sympathy_support")
_reg("other_useful_information", "other_relevant_information", "other_useful_info", "relevant_information")
_reg("not_related_or_irrelevant", "not_related", "not_relevant", "irrelevant", "not_applicable", "off_topic")


def _has_tables(folder):
    return os.path.isdir(folder) and bool(
        glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)
        + glob.glob(os.path.join(folder, "**", "*.tsv"), recursive=True))


def _crisisnlp_fetch(url, subdir, cache_dir):
    """Use an extracted copy or a local zip in ``cache_dir`` if present, otherwise download."""
    dest = os.path.join(cache_dir, subdir)
    if _has_tables(dest):
        print(f"[CrisisNLP] using extracted data: {dest}")
        return dest
    zip_path = os.path.join(cache_dir, os.path.basename(url))
    if os.path.isfile(zip_path):
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dest)
        print(f"[CrisisNLP] extracted local zip: {zip_path}")
        return dest
    import requests
    print(f"[CrisisNLP] downloading {url} ...")
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(dest)
    print(f"[CrisisNLP] extracted to {dest}")
    return dest


def _read_any_table(fp):
    sep = "\t" if fp.lower().endswith((".tsv", ".txt")) else ","
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(fp, sep=sep, encoding=enc, engine="python", on_bad_lines="skip")
        except Exception as e:
            err = e
    print(f"[CrisisNLP][skip] {fp}: {err}")
    return None


def load_crisisnlp(cfg: SpareConfig):
    cache_dir = str(cfg.data_dir / "crisisnlp_raw")
    frames = []
    for split in CRISISNLP_INCLUDE:
        root = _crisisnlp_fetch(CRISISNLP_ZIP_URLS[split], split, cache_dir)
        files = (glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)
                 + glob.glob(os.path.join(root, "**", "*.tsv"), recursive=True))
        print(f"[CrisisNLP] {split}: found {len(files)} table files")
        for fp in files:
            df = _read_any_table(fp)
            if df is None or df.empty:
                continue
            tcol = _find_col(df.columns, TEXT_CANDS + ["tweet"])
            lcol = _find_col(df.columns, LABEL_CANDS + ["choose_one_category", "label_text", "annotation"])
            icol = _find_col(df.columns, ID_CANDS)
            if tcol is None or lcol is None:
                print(f"[CrisisNLP][skip] {fp}: no text/label column in {list(df.columns)}")
                continue
            event = os.path.splitext(os.path.basename(fp))[0]
            frames.append(pd.DataFrame({
                "id": df[icol].astype(str) if icol else event + "_" + df.index.astype(str),
                "text": df[tcol].astype(str),
                "label": df[lcol].astype(str),
                "event": event,
                "source": f"crisisnlp-{split}",
            }))
    if not frames:
        raise FileNotFoundError(f"No CrisisNLP tables parsed. Inspect '{cache_dir}/'.")
    cn = pd.concat(frames, ignore_index=True)

    # Map raw labels onto the 9-class scheme; unmapped labels are dropped.
    mapped = cn["label"].map(_crisisnlp_norm).map(CRISISNLP_LABEL_MAP)
    keep = mapped.notna()
    cn9 = cn.loc[keep].copy()
    cn9["label"] = mapped[keep].values
    print(f"[CrisisNLP] 9-class: kept {len(cn9)} / {len(cn)} | dropped {len(cn) - len(cn9)}")
    assert set(cn9["label"].unique()) <= set(CRISISNLP_9)

    # 70/30 stratified split.
    from sklearn.model_selection import train_test_split
    strat = cn9["label"] if cn9["label"].value_counts().min() >= 2 else None
    train, test = train_test_split(cn9, test_size=0.30, random_state=cfg.seed, shuffle=True,
                                   stratify=strat)
    return train.reset_index(drop=True), test.reset_index(drop=True)


# --- CrisisLexT26 (CrisisBench humanitarian subset) ----------------------------------
CRISISBENCH_REPO = "QCRI/CrisisBench-all-lang"
CRISISBENCH_CONFIG = "humanitarian"
CRISISLEXT26_SOURCES = ("crisislext26",)
LANG_FILTER = "en"


def _crisisbench_split_to_frame(split_ds, sources):
    raw = split_ds.to_pandas()
    sel = raw[raw["source"].isin(sources)].copy()
    if LANG_FILTER is not None and "lang" in sel.columns:
        sel = sel[sel["lang"].astype(str) == LANG_FILTER]
    return pd.DataFrame({
        "id": sel["id"].astype(str),
        "text": sel["text"].astype(str),
        "label": sel["class_label"].astype(str),
        "event": sel["event"].astype(str),
        "source": sel["source"].astype(str),
    })


def load_crisislext26(cfg: SpareConfig):
    from datasets import load_dataset
    ds = load_dataset(CRISISBENCH_REPO, CRISISBENCH_CONFIG)
    print(f"[CrisisLexT26] splits: {list(ds.keys())}")
    train = _crisisbench_split_to_frame(ds["train"], CRISISLEXT26_SOURCES)
    test = _crisisbench_split_to_frame(ds["test" if "test" in ds else "dev"], CRISISLEXT26_SOURCES)
    return train.reset_index(drop=True), test.reset_index(drop=True)


LOADERS = {"HumAID": load_humaid, "CrisisNLP": load_crisisnlp, "CrisisLexT26": load_crisislext26}


# --- Evaluation sample and exemplar pool --------------------------------------------
def clean_pool(df):
    d = df.copy()
    d["text"] = d["text"].astype(str)
    d["label"] = d["label"].astype(str)
    d = d[(d["text"].str.strip() != "") & (d["label"].str.strip() != "")]
    d = d.drop_duplicates(subset=["text"])
    return d.sort_values(["id", "text"], kind="mergesort").reset_index(drop=True)


def sample_eval(df, seed, n):
    pool = clean_pool(df)
    if n is None or len(pool) <= n:
        return pool.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(pool), size=n, replace=False))
    return pool.iloc[idx].reset_index(drop=True)


class DatasetBundle:
    """Train/test splits plus the evaluation sample and the exemplar pool."""

    def __init__(self, cfg: SpareConfig):
        train, test = LOADERS[cfg.dataset](cfg)
        self.train, self.test = train, test
        self.all = pd.concat([train, test], ignore_index=True)   # label-space discovery only
        self.samples = sample_eval(test, cfg.seed, cfg.n_samples)
        self.train_pool = clean_pool(train)
        again = sample_eval(test, cfg.seed, cfg.n_samples)
        assert self.samples.equals(again), "evaluation sampling is not reproducible"
        self.fingerprint = hashlib.sha256("|".join(self.samples["id"].astype(str)).encode()).hexdigest()[:16]
        print(f"[{cfg.dataset}] TRAIN={len(train)} | TEST={len(test)} | "
              f"classes={self.all['label'].nunique()} | eval={len(self.samples)} | "
              f"exemplar pool={len(self.train_pool)} | eval-id sha256[:16]={self.fingerprint}")
