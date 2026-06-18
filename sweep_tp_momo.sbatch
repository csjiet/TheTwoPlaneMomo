#!/bin/bash
#SBATCH --job-name=gtpmomo
#SBATCH --partition=l40s_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/%u/srun_logs/%x_%j.out
#SBATCH --error=/scratch/%u/srun_logs/%x_%j.err
#SBATCH --account=torch_pr_99_tandon_advanced

set -euo pipefail

# Usage:
#   sbatch run_gtpmomo.sbatch <lr> <beta_short> <beta_long> <preconditioner> <precond_beta2> <weight_decay_factor>
#
# Example:
#   sbatch run_gtpmomo.sbatch 1e-3 0.9 0.999 adam 0.999 0.1

LR="${1:?missing lr}"
BETA_SHORT="${2:?missing g_tp_momo_beta_short}"
BETA_LONG="${3:?missing g_tp_momo_beta_long}"
PRECONDITIONER="${4:?missing g_tp_momo_preconditioner}"
PRECOND_BETA2="${5:?missing g_tp_momo_precond_beta2}"
WEIGHT_DECAY_FACTOR="${6:?missing g_tp_momo_weight_decay_factor}"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "/scratch/$USER/conda/envs/AMM"
cd "/scratch/$USER/projects/NewTwoPlaneMomo"

export HF_HOME="/scratch/$USER/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

export WANDB_DIR="/scratch/$USER/wandb_runs/${SLURM_JOB_ID}"
export WANDB_DATA_DIR="/scratch/$USER/wandb_data/${SLURM_JOB_ID}"
mkdir -p "$WANDB_DIR" "$WANDB_DATA_DIR"

export PYTHONUNBUFFERED=1

DATASETS_DIR="${DATASETS_DIR:-/scratch/$USER/datasets}"
WANDB_ENTITY="${WANDB_ENTITY:-tbal}"
WANDB_PROJECT="${WANDB_PROJECT:-AdEMAMIX_adaptive}"
RESULTS_BASE="${RESULTS_BASE:-/scratch/$USER/exps/two_plane_momo}"

mkdir -p "$RESULTS_BASE"

echo "LR=$LR"
echo "BETA_SHORT=$BETA_SHORT"
echo "BETA_LONG=$BETA_LONG"
echo "PRECONDITIONER=$PRECONDITIONER"
echo "PRECOND_BETA2=$PRECOND_BETA2"
echo "WEIGHT_DECAY_FACTOR=$WEIGHT_DECAY_FACTOR"
echo "NODE=$(hostname)"
echo "START=$(date -Is)"

torchrun --standalone --nproc_per_node=1 ./src/main.py \
  --config_format base \
  --model llama \
  --distributed_backend nccl \
  --device cuda \
  --run_prefix "sweep_tp_momo_job${SLURM_JOB_ID}" \
  --datasets_dir "$DATASETS_DIR" \
  --n_embd 768 --n_head 12 --n_layer 12 \
  --batch_size 64 --sequence_length 512 --acc_steps 4 \
  --dataset fineweb --iterations 128000 \
  --dropout 0.0 --warmup_steps 2000 --grad_clip 0.5 --seed 0 \
  --opt gen_two_plane_momo \
  --lr "$LR" \
  --scheduler cos \
  --g_tp_momo_beta_short "$BETA_SHORT" \
  --g_tp_momo_beta_long "$BETA_LONG" \
  --g_tp_momo_eps 1e-12 \
  --g_tp_momo_preconditioner "$PRECONDITIONER" \
  --g_tp_momo_precond_beta2 "$PRECOND_BETA2" \
  --g_tp_momo_weight_decay_factor "$WEIGHT_DECAY_FACTOR" \
  --g_tp_momo_use_loss_ema \
  --g_tp_momo_alpha_scope network \
  --results_base_folder "$RESULTS_BASE" \
  --wandb \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_run_prefix "ori_llama" \
  --wandb_entity "$WANDB_ENTITY" \
  --eval_interval 115 \
  --latest_ckpt_interval 1000
