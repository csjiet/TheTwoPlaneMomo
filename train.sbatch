#!/bin/bash
#SBATCH --job-name=tp_run
#SBATCH --partition=l40s_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/%u/srun_logs/%x_%j.out
#SBATCH --error=/scratch/%u/srun_logs/%x_%j.err
#SBATCH --account=torch_pr_99_tandon_advanced

set -euo pipefail

# Usage:
#   sbatch train.sbatch <make_target> [MAKE_VAR=VALUE ...]
TARGET="${1:?Usage: sbatch train.sbatch <make_target> [MAKE_VAR=VALUE ...]}"
shift || true
MAKE_ARGS=("$@")

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "/scratch/$USER/conda/envs/AMM"
cd "/scratch/$USER/projects/TheTwoPlaneMomo"

# HF caches on scratch
export HF_HOME="/scratch/$USER/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

# W&B per-job local dirs to avoid collisions when multiple jobs launch together
export WANDB_DIR="/scratch/$USER/wandb_runs/${SLURM_JOB_ID}"
export WANDB_DATA_DIR="/scratch/$USER/wandb_data/${SLURM_JOB_ID}"
mkdir -p "$WANDB_DIR" "$WANDB_DATA_DIR"

# Optional but useful for live log output in Slurm files
export PYTHONUNBUFFERED=1

DATASETS_DIR="${DATASETS_DIR:-/scratch/$USER/datasets}"
WANDB_ENTITY="${WANDB_ENTITY:-tbal}"
WANDB_PROJECT="${WANDB_PROJECT:-AdEMAMIX_adaptive}"

JOB_SUFFIX="job${SLURM_JOB_ID}"

# Priority:
# 1) If RUN_TAG is exported into the sbatch environment, use it.
# 2) Else, pull the Makefile default RUN_TAG (e.g., expr2).
if [[ -n "${RUN_TAG:-}" ]]; then
  BASE_RUN_TAG="$RUN_TAG"
else
  BASE_RUN_TAG="$(make -s print-RUN_TAG)"
  if [[ -z "$BASE_RUN_TAG" ]]; then
    echo "ERROR: Could not read RUN_TAG from Makefile (print-RUN_TAG returned empty)."
    echo "Did you add the Makefile helper target 'print-%: ; @echo \$\$(\$\$*)' ?"
    exit 1
  fi
fi

FINAL_RUN_TAG="${BASE_RUN_TAG}_${JOB_SUFFIX}"

echo "TARGET=$TARGET"
echo "MAKE_ARGS=${MAKE_ARGS[*]-}"
echo "DATASETS_DIR=$DATASETS_DIR"
echo "WANDB_ENTITY=$WANDB_ENTITY"
echo "WANDB_PROJECT=$WANDB_PROJECT"
echo "WANDB_DIR=$WANDB_DIR"
echo "WANDB_DATA_DIR=$WANDB_DATA_DIR"
echo "RUN_TAG(base)=$BASE_RUN_TAG"
echo "RUN_TAG(final)=$FINAL_RUN_TAG"
echo "NODE=$(hostname)"
echo "START=$(date -Is)"

make "$TARGET" \
  DATASETS_DIR="$DATASETS_DIR" \
  WANDB_ENTITY="$WANDB_ENTITY" \
  WANDB_PROJECT="$WANDB_PROJECT" \
  RUN_TAG="$FINAL_RUN_TAG" \
  "${MAKE_ARGS[@]}"
