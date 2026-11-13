"""Shared helpers: seeding, run paths, model loading and label normalization."""
import json
import os
import platform
import random
import re
import sys
from datetime import datetime, timezone
from importlib.metadata import version as _pkg_version
from pathlib import Path

import numpy as np

from .config import SpareConfig

WRONG_TOKEN = "__wrong__"


def set_all_seeds(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def norm_label(x) -> str:
    """Lower-case, underscore-separated label; empty or missing values become WRONG_TOKEN."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return WRONG_TOKEN
    x = str(x).strip().lower()
    x = re.sub(r"[\s\-]+", "_", x)
    x = re.sub(r"[^a-z0-9_]", "", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x if x else WRONG_TOKEN


def to_bool(series):
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def gpu_mem(tag=""):
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 1024**3
        r = torch.cuda.memory_reserved() / 1024**3
        mx = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[gpu-mem] {tag:34s} alloc={a:6.2f}GiB reserved={r:6.2f}GiB max_alloc={mx:6.2f}GiB")


def environment_record(seed: int) -> dict:
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "seed": seed,
    }
    for pkg in ["torch", "transformers", "accelerate", "datasets", "numpy", "pandas", "scikit-learn"]:
        try:
            record[f"pkg::{pkg}"] = _pkg_version(pkg)
        except Exception:
            record[f"pkg::{pkg}"] = "not installed"
    try:
        import torch
        record["gpu_available"] = torch.cuda.is_available()
        record["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None (CPU only)"
        record["hip_version"] = getattr(torch.version, "hip", None)
        record["cuda_version"] = getattr(torch.version, "cuda", None)
    except ImportError:
        record["gpu_available"] = False
    return record


class RunContext:
    """Holds the configuration, output paths and the (lazily loaded) model for one run.

    Stages 1-3 (baseline, Ripple/Protect groups, FiNE localization) write into ``run_dir``.
    Stages 4-5 (editing, evaluation) write into ``run_dir / cfg.variant`` so the ablation
    variants can reuse the same baseline, groups and localization.
    """

    def __init__(self, cfg: SpareConfig):
        self.cfg = cfg
        self.run_dir = (cfg.out_dir / cfg.run_name).resolve()
        self.edit_dir = self.run_dir / cfg.variant
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.edit_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = None
        self.model = None

    # -- paths -------------------------------------------------------------------
    def path(self, name: str) -> str:
        return str(self.run_dir / f"{self.cfg.dataset}_{name}")

    def edit_path(self, name: str) -> str:
        return str(self.edit_dir / f"{self.cfg.dataset}_{name}")

    def save_config(self):
        with open(self.edit_dir / "config.json", "w") as f:
            json.dump(self.cfg.to_dict(), f, indent=2)
        with open(self.run_dir / "environment.json", "w") as f:
            json.dump(environment_record(self.cfg.seed), f, indent=2)

    # -- model -------------------------------------------------------------------
    @property
    def device(self) -> str:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def dtype(self):
        import torch
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32

    def load_model(self, source: str = None):
        """Load tokenizer and model once; later stages reuse the same objects."""
        if self.model is not None and source is None:
            return self.tokenizer, self.model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        source = source or self.cfg.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=self.dtype)
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        print(f"Loaded {source} on {self.device} (dtype={self.dtype}).")
        return self.tokenizer, self.model

    def gen_kwargs(self) -> dict:
        return dict(do_sample=False, num_beams=1, max_new_tokens=self.cfg.max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id)


def require(path, hint: str):
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. {hint}")
    return path
