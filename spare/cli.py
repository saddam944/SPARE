"""Command-line entry point.

    python -m spare run  --dataset HumAID --out-dir runs
    python -m spare run  --dataset HumAID --out-dir runs --stages edit,evaluate --no-kl
    python -m spare plot --runs HumAID=runs/HumAID_500_run CrisisNLP=runs/CrisisNLP_500_run --out figs_out
"""
import argparse
from pathlib import Path

from .config import DATASETS, SpareConfig

STAGES = ["baseline", "groups", "localize", "edit", "evaluate"]


def _stages(arg):
    stages = STAGES if arg == "all" else [s.strip() for s in arg.split(",")]
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown stage(s) {sorted(unknown)}; choose from {STAGES}")
    return [s for s in STAGES if s in stages]


def _n_samples(arg):
    return None if arg.lower() == "all" else int(arg)


def build_parser():
    p = argparse.ArgumentParser(prog="spare", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run pipeline stages for one dataset")
    r.add_argument("--dataset", choices=DATASETS, required=True)
    r.add_argument("--stages", type=_stages, default=STAGES,
                   help=f"'all' or a comma-separated subset of {','.join(STAGES)}")
    r.add_argument("--out-dir", type=Path, default=Path("runs"))
    r.add_argument("--data-dir", type=Path, default=Path("data"),
                   help="cache for the raw CrisisNLP download (a local zip placed here is used offline)")
    r.add_argument("--model", dest="model_name", default=SpareConfig.model_name)
    r.add_argument("--n-samples", type=_n_samples, default=SpareConfig.n_samples,
                   help="evaluation samples drawn from the test split, or 'all'")
    r.add_argument("--seed", type=int, default=SpareConfig.seed)
    r.add_argument("--train-steps", type=int, default=SpareConfig.train_steps)
    r.add_argument("--max-groups", type=int, default=None)
    r.add_argument("--no-kl", action="store_true", help="ablation: lambda = 0")
    r.add_argument("--no-agem", action="store_true", help="ablation: no A-GEM projection")
    r.add_argument("--no-resume", action="store_true", help="ignore an existing training checkpoint")
    r.add_argument("--no-roc-auc", action="store_true", help="skip label scoring in the baseline")
    r.add_argument("--skip-paraphrase", action="store_true", help="skip paraphrase/portability probes")
    r.add_argument("--skip-fluency", action="store_true", help="skip the generation-entropy probe")
    r.add_argument("--save-edited-model", action="store_true",
                   help="save the edited model (about 6 GB) so 'evaluate' can run in a later job")

    f = sub.add_parser("plot", help="draw figures from finished runs")
    f.add_argument("--runs", nargs="+", required=True, metavar="DATASET=RUN_DIR")
    f.add_argument("--out", type=Path, default=Path("figs_out"))
    f.add_argument("--variant", default="full")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "plot":
        from .plots import make_all
        runs = dict(item.split("=", 1) for item in args.runs)
        make_all(runs, args.out, args.variant)
        return

    cfg = SpareConfig(dataset=args.dataset, out_dir=args.out_dir, data_dir=args.data_dir,
                      model_name=args.model_name, n_samples=args.n_samples, seed=args.seed,
                      train_steps=args.train_steps, max_groups=args.max_groups,
                      lambda_protect=0.0 if args.no_kl else SpareConfig.lambda_protect,
                      use_agem=not args.no_agem, resume=not args.no_resume,
                      compute_roc_auc=not args.no_roc_auc,
                      run_paraphrase_eval=not args.skip_paraphrase, run_fluency_eval=not args.skip_fluency,
                      save_edited_model=args.save_edited_model)
    run(cfg, args.stages)


def run(cfg: SpareConfig, stages=STAGES):
    from .utils import RunContext, set_all_seeds
    set_all_seeds(cfg.seed)
    ctx = RunContext(cfg)
    ctx.save_config()
    print(f"run dir: {ctx.run_dir} | variant: {cfg.variant} | stages: {stages}")

    if "baseline" in stages:
        from .baseline import run_baseline
        run_baseline(ctx)
    if "groups" in stages:
        from .ripple_protect import build_ripple_protect
        build_ripple_protect(ctx)
    if "localize" in stages:
        from .localization import run_localization
        run_localization(ctx)

    engine = None
    if "edit" in stages:
        from .editing import run_editing
        set_all_seeds(cfg.seed)
        engine = run_editing(ctx)
    if "evaluate" in stages:
        from .evaluation import run_evaluation
        if engine is None:
            # The edited weights only exist in memory unless they were saved with --save-edited-model.
            wt_dir = ctx.edit_dir / "FINAL_WT_MODEL"
            if not wt_dir.is_dir():
                raise SystemExit("'evaluate' needs the edited model: run it together with 'edit', "
                                 "or run 'edit' with --save-edited-model first.")
            ctx.load_model(source=str(wt_dir))
        run_evaluation(ctx, engine)
