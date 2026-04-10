#!/bin/bash
set -euo pipefail

export DATASETS_DIR="/scratch/$USER/datasets"
export WANDB_ENTITY="tbal"
export WANDB_PROJECT="AdEMAMIX_adaptive"

targets=(
  g_tp_momo2
  ademamix_no_beta3_and_alpha_warmup
	g_tp_momo_no_loss_ema
	g_tp_momo_alpha_per_param
	adamw
)

for t in "${targets[@]}"; do
  echo "Submitting: make $t"
  sbatch --export=DATASETS_DIR,WANDB_ENTITY,WANDB_PROJECT train.sbatch "$t"
done
