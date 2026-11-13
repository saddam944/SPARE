"""Experiment configuration.

Defaults follow Table 4 of the paper and the values used in the notebooks.
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

DATASETS = ("HumAID", "CrisisNLP", "CrisisLexT26")


@dataclass
class SpareConfig:
    dataset: str = "HumAID"
    out_dir: Path = Path("runs")
    data_dir: Path = Path("data")

    # Base model and generation
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    max_new_tokens: int = 12          # greedy decoding for classification
    seed: int = 42

    # Baseline (one-shot in-context classification)
    n_samples: Optional[int] = 500    # evaluation samples drawn from the TEST split; None = all
    prompt: str = "domain"
    iteration: int = 0
    compute_roc_auc: bool = True

    # FiNE localization
    top_k: int = 5                    # neurons per source-object token
    exclude_last_n_layers: int = 3    # layer freezing (last FFN layers are never edited)

    # Group-level neuron mask
    fine_mask_mode: str = "member_fraction"   # or "union"
    min_member_fraction: float = 0.5          # tau in Table 4

    # Low-level editor H_theta (MEND-style, as in HiEdit)
    editor_rank: int = 128            # bottleneck dimension
    editor_blocks: int = 2            # residual blocks
    edit_init_scale: float = 1e-3
    edit_sign: float = -1.0           # initial update is a descent step on the edit loss

    # Sequential objective
    lambda_protect: float = 1.0       # KL weight; 0 gives the "w/o KL" ablation
    eta: float = 1e-2                 # update-norm weight
    mu: float = 0.9                   # memory decay for earlier groups
    memory_k: int = 2                 # re-evaluate groups [t-k, t]
    gamma: float = 1.0                # group discount
    lr: float = 1e-3
    train_steps: int = 60             # training passes Q
    grad_clip: float = 1.0
    edit_cap: Optional[float] = 0.0   # cap on ||dW|| / ||W0||; 0 or None disables it
    max_groups: Optional[int] = None

    # A-GEM
    use_agem: bool = True             # False gives the "w/o A-GEM" ablation
    agem_ref_batch: int = 64
    agem_ref_chunk: int = 8

    # Execution-only micro-batching (does not change the objective)
    nll_batch: int = 16
    kl_batch: int = 16
    before_batch: int = 16
    agem_microbatch: int = 16
    resume: bool = True

    # Evaluation
    n_paraphrases: int = 3
    fluency_max_new_tokens: int = 64
    fluency_temperature: float = 0.7
    fluency_top_p: float = 0.9
    run_paraphrase_eval: bool = True
    run_fluency_eval: bool = True
    save_edited_model: bool = False

    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.out_dir = Path(self.out_dir)
        self.data_dir = Path(self.data_dir)
        if self.dataset not in DATASETS:
            raise ValueError(f"dataset must be one of {DATASETS}, got {self.dataset!r}")

    @property
    def run_name(self) -> str:
        n = "all" if self.n_samples is None else self.n_samples
        return f"{self.dataset}_{n}_run"

    @property
    def variant(self) -> str:
        """Name of the editing variant; stages 4-5 write into a subfolder with this name."""
        no_kl = self.lambda_protect == 0
        no_agem = not self.use_agem
        if no_kl and no_agem:
            return "no_kl_no_agem"
        if no_kl:
            return "no_kl"
        if no_agem:
            return "no_agem"
        return "full"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["out_dir"] = str(self.out_dir)
        d["data_dir"] = str(self.data_dir)
        return d
