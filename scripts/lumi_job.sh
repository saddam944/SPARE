#!/bin/bash
# SLURM job for one dataset on a LUMI-G node (one AMD MI250X GCD).
#
# Usage (from the repository root):
#   sbatch --account=project_<id> scripts/lumi_job.sh HumAID
#   sbatch --account=project_<id> scripts/lumi_job.sh CrisisNLP --no-kl      # ablation
#
# Extra arguments after the dataset name are passed to `python -m spare run`.
#SBATCH --job-name=spare
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
DATASET=${1:?usage: sbatch scripts/lumi_job.sh <HumAID|CrisisNLP|CrisisLexT26> [extra args]}
shift

module purge
module use /appl/local/csc/modulefiles
module load pytorch

# Create the virtual environment on first use, on top of CSC's ROCm PyTorch.
if [ ! -d venv-spare ]; then
    python -m venv --system-site-packages venv-spare
    source venv-spare/bin/activate
    pip install -r requirements.txt
else
    source venv-spare/bin/activate
fi

# Keep model and dataset downloads out of the home quota.
export HF_HOME=${HF_HOME:-/scratch/${SLURM_JOB_ACCOUNT}/${USER}/hf_cache}

python -m spare run --dataset "$DATASET" --out-dir runs "$@"
