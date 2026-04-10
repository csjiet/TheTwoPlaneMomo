#!/bin/bash
set -euo pipefail

export DATASETS_DIR="/scratch/$USER/datasets"
export WANDB_ENTITY="tbal"
export WANDB_PROJECT="AdEMAMIX_adaptive"

lrs=( 1e-3 )

beta_shorts=( 0.9)

beta_longs=( 0.999)

preconditioners=( adam)

precond_beta2s=( 0.95 0.999)
# 0.999
weight_decay_factors=(0.0 0.01 0.05 0.1)

for lr in "${lrs[@]}"; do
  for beta_short in "${beta_shorts[@]}"; do
    for beta_long in "${beta_longs[@]}"; do
      for preconditioner in "${preconditioners[@]}"; do
        for precond_beta2 in "${precond_beta2s[@]}"; do
          for weight_decay_factor in "${weight_decay_factors[@]}"; do
            echo "Submitting: lr=$lr beta_short=$beta_short beta_long=$beta_long preconditioner=$preconditioner precond_beta2=$precond_beta2 weight_decay_factor=$weight_decay_factor"

            sbatch sweep_tp_momo.sbatch \
              "$lr" \
              "$beta_short" \
              "$beta_long" \
              "$preconditioner" \
              "$precond_beta2" \
              "$weight_decay_factor"
          done
        done
      done
    done
  done
done
