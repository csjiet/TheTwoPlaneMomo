#!/bin/bash
set -euo pipefail

export DATASETS_DIR="/scratch/$USER/datasets"
export WANDB_ENTITY="tbal"
export WANDB_PROJECT="AdEMAMIX_adaptive"

targets=(
  # g_tp_momo_alpha_network_warmup
	# g_tp_momo_alpha_per_param_warmup
	decoupled_wd_g_tp_momo_alpha_per_param_warmup
	decoupled_wd_g_tp_momo_alpha_network_warmup
	# ademamix_only_beta3_warmup
	# g_tp_momo_alpha_per_param_beta_long_no_warmup
	# g_tp_momo_alpha_per_param_warmup_and_alpha_denom_correction1
	# g_tp_momo_alpha_per_param_warmup_and_alpha_denom_correction2
	# g_tp_momo_alpha_per_param_warmup_large_lr
	# g_tp_momo_alpha_network_warmup_large_lr
  # ademamix_beta3_and_alpha_no_warmup
	# ademamix
	# g_tp_momo_no_loss_ema
	# adamw
)

for t in "${targets[@]}"; do
  echo "Submitting: make $t"
	sbatch --export=ALL,DATASETS_DIR,WANDB_ENTITY,WANDB_PROJECT train.sbatch "$t"
done
