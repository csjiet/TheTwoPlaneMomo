#!/bin/bash
#SBATCH --job-name=tokenize_fineweb
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=26:00:00
#SBATCH --output=/scratch/%u/srun_logs/%x_%j.out
#SBATCH --error=/scratch/%u/srun_logs/%x_%j.err
#SBATCH --account=torch_pr_99_general

set -euo pipefail

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "/scratch/$USER/conda/envs/AMM"

cd "/scratch/$USER/projects/NewTwoPlaneMomo"

export HF_HOME="/scratch/$USER/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

# optional, but good for authenticated download
# export HF_TOKEN="hf_..."

# avoid oversubscription
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

DATASETS_DIR="${DATASETS_DIR:-/scratch/$USER/datasets}"
NUM_PROC="${NUM_PROC:-64}"

mkdir -p "$DATASETS_DIR"

echo "DATASETS_DIR=$DATASETS_DIR"
echo "NUM_PROC=$NUM_PROC"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "PWD=$(pwd)"
echo "START=$(date -Is)"
echo

python -u -c "from src.data.fineweb import get_fineweb_data; print(get_fineweb_data('$DATASETS_DIR', num_proc=int('$NUM_PROC')))"
