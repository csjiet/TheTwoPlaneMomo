.PHONY: \
  soap adamw mars ademamix ademamix_no_beta3_and_alpha_warmup tp_momo tp_momo2 g_tp_momo g_tp_momo2 \
  med_soap med_adamw med_mars med_ademamix med_tp_momo med_tp_momo2 \
  small_soap small_sgd small_adamw small_ademamix

WANDB_PROJECT     ?= AdEMAMIX_adaptive
WANDB_RUN_PREFIX  ?= ori_llama
WANDB_ENTITY			?= tbal
RUN_TAG      ?= expr1_16k_iter
# tp_momo_per_param_vs_network_wd_0-1
# EXPERIMENT_NAME ?= test
# test_lb_clip_new_tp_momo 

RESULTS_BASE ?= /scratch/st5494/exps/two_plane_momo/

DATASETS_DIR ?= ./src/data/datasets

LAUNCH ?= torchrun --standalone --nproc_per_node=1
DEVICE ?= cuda
DIST_BACKEND_FLAG ?= --distributed_backend nccl
# -----------------------------------------------------

# common args for the "small" CPU runs
SMALL_COMMON_ARGS = \
  --config_format base \
  --dataset shakespeare-char \
  --model base \
  --n_layer 4 --n_head 4 --n_embd 256 \
  --sequence_length 256 --batch_size 8 \
  --iterations 2000 \
  --warmup_steps 200 \
  --eval_interval 200 \
  --log_interval 10 \
  --results_base_folder $(RESULTS_BASE) \
  --wandb \
  --wandb_project "$(WANDB_PROJECT)" \
  --wandb_run_prefix "$(WANDB_RUN_PREFIX)" \
  --wandb_entity "$(WANDB_ENTITY)" \
	--experiment_name "$(EXPERIMENT_NAME)" 


# common args for the main llama runs
COMMON_ARGS = \
  --results_base_folder $(RESULTS_BASE) \
  --wandb \
  --wandb_project "$(WANDB_PROJECT)" \
  --wandb_run_prefix "$(WANDB_RUN_PREFIX)" \
  --wandb_entity "$(WANDB_ENTITY)"
	# --experiment_name "$(EXPERIMENT_NAME)" 


DATASET      ?= fineweb
N_EMBD       ?= 768
BATCH_SIZE   ?= 64
ITERATIONS   ?= 16000
# 16000
WEIGHT_DECAY ?= 0.1

print-%:
	@echo $($*)
# Main llama runs
soap:
	mkdir -p $(RESULTS_BASE)
	$(LAUNCH) ./src/main.py --config_format base --model llama $(DIST_BACKEND_FLAG) --device $(DEVICE) \
		--run_prefix "$(RUN_TAG)" \
		--datasets_dir "$(DATASETS_DIR)" \
		--n_embd $(N_EMBD) --n_head 12 --n_layer 12 \
		--batch_size $(BATCH_SIZE) --sequence_length 512 --acc_steps 4 \
		--dataset $(DATASET) --iterations $(ITERATIONS) \
		--dropout 0.0 --warmup_steps 2000 --grad_clip 0.5 --seed 0 \
		--opt soap --lr 1e-3 --weight_decay $(WEIGHT_DECAY) --scheduler cos \
		--beta1 0.9 --beta2 0.999 \
		$(COMMON_ARGS) \
		--eval_interval 115 --latest_ckpt_interval 1000

adamw:
	mkdir -p $(RESULTS_BASE)
	$(LAUNCH) ./src/main.py --config_format base --model llama $(DIST_BACKEND_FLAG) --device $(DEVICE) \
		--run_prefix "$(RUN_TAG)" \
		--datasets_dir "$(DATASETS_DIR)" \
		--n_embd $(N_EMBD) --n_head 12 --n_layer 12 \
		--batch_size $(BATCH_SIZE) --sequence_length 512 --acc_steps 4 \
		--dataset $(DATASET) --iterations $(ITERATIONS) \
		--dropout 0.0 --warmup_steps 2000 --grad_clip 0.5 --seed 0 \
		--opt adamw --lr 1e-3 --weight_decay $(WEIGHT_DECAY) --scheduler cos \
		--beta1 0.8 --beta2 0.999 \
		$(COMMON_ARGS) \
		--eval_interval 115 --latest_ckpt_interval 1000

mars:
	mkdir -p $(RESULTS_BASE)
	$(LAUNCH) ./src/main.py --config_format base --model llama $(DIST_BACKEND_FLAG) --device $(DEVICE) \
		--run_prefix "$(RUN_TAG)" \
		--datasets_dir "$(DATASETS_DIR)" \
		--n_embd $(N_EMBD) --n_head 12 --n_layer 12 \
		--batch_size $(BATCH_SIZE) --sequence_length 512 --acc_steps 4 \
		--dataset $(DATASET) --iterations $(ITERATIONS) \
		--dropout 0.0 --warmup_steps 2000 --grad_clip 0.5 --seed 0 \
		--opt mars --lr 1e-3 --mars_lr 3e-3 --weight_decay $(WEIGHT_DECAY) --scheduler cos \
		--beta1 0.8 --mars_beta1 0.95 --beta2 0.999 --mars_beta2 0.99 \
		$(COMMON_ARGS) \
		--eval_interval 115 --latest_ckpt_interval 1000

ademamix:
	mkdir -p $(RESULTS_BASE)
	$(LAUNCH) ./src/main.py --config_format base --model llama $(DIST_BACKEND_FLAG) --device $(DEVICE) \
		--run_prefix "$(RUN_TAG)" \
		--datasets_dir "$(DATASETS_DIR)" \
		--n_embd $(N_EMBD) --n_head 12 --n_layer 12 \
		--batch_size $(BATCH_SIZE) --sequence_length 512 --acc_steps 4 \
		--dataset $(DATASET) --iterations $(ITERATIONS) \
		--dropout 0.0 --warmup_steps 2000 --grad_clip 0.5 --seed 0 \
		--opt ademamix --lr 1e-3 --weight_decay $(WEIGHT_DECAY) --scheduler cos \
		--beta1 0.9 --beta2 0.999 \
		--adema_beta3 0.999 --adema_alpha 8.0 \
		--adema_beta3_warmup 128000 --adema_alpha_warmup 128000 \
		$(COMMON_ARGS) \
		--eval_interval 115 --latest_ckpt_interval 1000

ademamix_no_beta3_and_alpha_warmup:
	mkdir -p $(RESULTS_BASE)
	$(LAUNCH) ./src/main.py --config_format base --model llama $(DIST_BACKEND_FLAG) --device $(DEVICE) \
		--run_prefix "$(RUN_TAG)" \
		--datasets_dir "$(DATASETS_DIR)" \
		--n_embd $(N_EMBD) --n_head 12 --n_layer 12 \
		--batch_size $(BATCH_SIZE) --sequence_length 512 --acc_steps 4 \
		--dataset $(DATASET) --iterations $(ITERATIONS) \
		--dropout 0.0 --warmup_steps 2000 --grad_clip 0.5 --seed 0 \
		--opt ademamix --lr 1e-3 --weight_decay $(WEIGHT_DECAY) --scheduler cos \
		--beta1 0.9 --beta2 0.999 \
		--adema_beta3 0.999 --adema_alpha 8.0 \
		--adema_beta3_warmup 1 --adema_alpha_warmup 1 \
		$(COMMON_ARGS) \
		--eval_interval 115 --latest_ckpt_interval 1000

tp_momo:
	mkdir -p $(RESULTS_BASE)
	$(LAUNCH) ./src/main.py --config_format base --model llama $(DIST_BACKEND_FLAG) --device $(DEVICE) \
		--run_prefix "$(RUN_TAG)" \
		--datasets_dir "$(DATASETS_DIR)" \
		--n_embd $(N_EMBD) --n_head 12 --n_layer 12 \
		--batch_size $(BATCH_SIZE) --sequence_length 512 --acc_steps 4 \
		--dataset $(DATASET) --iterations $(ITERATIONS) \
		--dropout 0.0 --warmup_steps 2000 --grad_clip 0.5 --seed 0 \
		--opt two_plane_momo --lr 1e-3 --scheduler cos \
		--two_plane_momo_beta_short 0.9 --two_plane_momo_beta_long 0.999 \
		--two_plane_momo_eps 1e-12 \
		$(COMMON_ARGS) \
		--eval_interval 115 --latest_ckpt_interval 1000

tp_momo2:
	mkdir -p $(RESULTS_BASE)
	$(LAUNCH) ./src/main.py --config_format base --model llama $(DIST_BACKEND_FLAG) --device $(DEVICE) \
		--run_prefix "$(RUN_TAG)" \
		--datasets_dir "$(DATASETS_DIR)" \
		--n_embd $(N_EMBD) --n_head 12 --n_layer 12 \
		--batch_size $(BATCH_SIZE) --sequence_length 512 --acc_steps 4 \
		--dataset $(DATASET) --iterations $(ITERATIONS) \
		--dropout 0.0 --warmup_steps 2000 --grad_clip 0.5 --seed 0 \
		--opt two_plane_momo --lr 1e-3 --scheduler cos \
		--two_plane_momo_beta_short 0.9 --two_plane_momo_beta_long 0.999 \
		--two_plane_momo_eps 1e-12 \
		$(COMMON_ARGS) \
		--eval_interval 115 --latest_ckpt_interval 1000

g_tp_momo:
	mkdir -p $(RESULTS_BASE)
	$(LAUNCH) ./src/main.py --config_format base --model llama $(DIST_BACKEND_FLAG) --device $(DEVICE) \
		--run_prefix "$(RUN_TAG)" \
		--datasets_dir "$(DATASETS_DIR)" \
		--n_embd $(N_EMBD) --n_head 12 --n_layer 12 \
		--batch_size $(BATCH_SIZE) --sequence_length 512 --acc_steps 4 \
		--dataset $(DATASET) --iterations $(ITERATIONS) \
		--dropout 0.0 --warmup_steps 2000 --grad_clip 0.5 --seed 0 \
		--opt gen_two_plane_momo --lr 1e-3 --scheduler cos \
		--gen_two_plane_momo_beta_short 0.9 --gen_two_plane_momo_beta_long 0.999 \
		--gen_two_plane_momo_eps 1e-12 \
		--gen_two_plane_momo_preconditioner adam \
		--gen_two_plane_momo_precond_beta2 0.999 \
		--gen_two_plane_momo_weight_decay_factor $(WEIGHT_DECAY) \
		--gen_two_plane_momo_decoupled_weight_decay \
		$(COMMON_ARGS) \
		--eval_interval 115 --latest_ckpt_interval 1000

g_tp_momo2:
	mkdir -p $(RESULTS_BASE)
	$(LAUNCH) ./src/main.py --config_format base --model llama $(DIST_BACKEND_FLAG) --device $(DEVICE) \
		--run_prefix "$(RUN_TAG)" \
		--datasets_dir "$(DATASETS_DIR)" \
		--n_embd $(N_EMBD) --n_head 12 --n_layer 12 \
		--batch_size $(BATCH_SIZE) --sequence_length 512 --acc_steps 4 \
		--dataset $(DATASET) --iterations $(ITERATIONS) \
		--dropout 0.0 --warmup_steps 2000 --grad_clip 0.5 --seed 0 \
		--opt gen_two_plane_momo --lr 1e-3 --scheduler cos \
		--gen_two_plane_momo_beta_short 0.9 --gen_two_plane_momo_beta_long 0.999 \
		--gen_two_plane_momo_eps 1e-12 \
		--gen_two_plane_momo_preconditioner adam \
		--gen_two_plane_momo_precond_beta2 0.999 \
		--gen_two_plane_momo_weight_decay_factor $(WEIGHT_DECAY) \
		$(COMMON_ARGS) \
		--gen_two_plane_momo_use_loss_ema \
		--gen_two_plane_momo_alpha_scope network \
		--eval_interval 115 --latest_ckpt_interval 1000

g_tp_momo_no_loss_ema:
	mkdir -p $(RESULTS_BASE)
	$(LAUNCH) ./src/main.py --config_format base --model llama $(DIST_BACKEND_FLAG) --device $(DEVICE) \
		--run_prefix "$(RUN_TAG)" \
		--datasets_dir "$(DATASETS_DIR)" \
		--n_embd $(N_EMBD) --n_head 12 --n_layer 12 \
		--batch_size $(BATCH_SIZE) --sequence_length 512 --acc_steps 4 \
		--dataset $(DATASET) --iterations $(ITERATIONS) \
		--dropout 0.0 --warmup_steps 2000 --grad_clip 0.5 --seed 0 \
		--opt gen_two_plane_momo --lr 1e-3 --scheduler cos \
		--gen_two_plane_momo_beta_short 0.9 --gen_two_plane_momo_beta_long 0.999 \
		--gen_two_plane_momo_eps 1e-12 \
		--gen_two_plane_momo_preconditioner adam \
		--gen_two_plane_momo_precond_beta2 0.999 \
		--gen_two_plane_momo_weight_decay_factor $(WEIGHT_DECAY) \
		$(COMMON_ARGS) \
		--no-gen_two_plane_momo_use_loss_ema \
		--gen_two_plane_momo_alpha_scope network \
		--eval_interval 115 --latest_ckpt_interval 1000

g_tp_momo_alpha_per_param:
	mkdir -p $(RESULTS_BASE)
	$(LAUNCH) ./src/main.py --config_format base --model llama $(DIST_BACKEND_FLAG) --device $(DEVICE) \
		--run_prefix "$(RUN_TAG)" \
		--datasets_dir "$(DATASETS_DIR)" \
		--n_embd $(N_EMBD) --n_head 12 --n_layer 12 \
		--batch_size $(BATCH_SIZE) --sequence_length 512 --acc_steps 4 \
		--dataset $(DATASET) --iterations $(ITERATIONS) \
		--dropout 0.0 --warmup_steps 2000 --grad_clip 0.5 --seed 0 \
		--opt gen_two_plane_momo --lr 1e-3 --scheduler cos \
		--gen_two_plane_momo_beta_short 0.9 --gen_two_plane_momo_beta_long 0.999 \
		--gen_two_plane_momo_eps 1e-12 \
		--gen_two_plane_momo_preconditioner adam \
		--gen_two_plane_momo_precond_beta2 0.999 \
		--gen_two_plane_momo_weight_decay_factor $(WEIGHT_DECAY) \
		$(COMMON_ARGS) \
		--gen_two_plane_momo_use_loss_ema \
		--gen_two_plane_momo_alpha_scope parameter \
		--eval_interval 115 --latest_ckpt_interval 1000
# Medium experiment variants 
med_soap:
	mkdir -p $(RESULTS_BASE)
	$(LAUNCH) ./src/main.py --config_format base --model llama $(DIST_BACKEND_FLAG) --device $(DEVICE) \
		--run_prefix "$(RUN_TAG)" \
		--datasets_dir "$(DATASETS_DIR)" \
		--n_embd $(N_EMBD) --n_head 12 --n_layer 12 \
		--batch_size $(BATCH_SIZE) --sequence_length 512 --acc_steps 4 \
		--dataset $(DATASET) --iterations $(ITERATIONS) \
		--dropout 0.0 --warmup_steps 2000 --grad_clip 0.5 --seed 0 \
		--opt soap --lr 1e-3 --weight_decay $(WEIGHT_DECAY) --scheduler cos \
		--beta1 0.9 --beta2 0.999 \
		$(COMMON_ARGS) \
		--eval_interval 115 --latest_ckpt_interval 1000

med_adamw:
	mkdir -p $(RESULTS_BASE)
	$(LAUNCH) ./src/main.py --config_format base --model llama $(DIST_BACKEND_FLAG) --device $(DEVICE) \
		--run_prefix "$(RUN_TAG)" \
		--datasets_dir "$(DATASETS_DIR)" \
		--n_embd $(N_EMBD) --n_head 12 --n_layer 12 \
		--batch_size $(BATCH_SIZE) --sequence_length 512 --acc_steps 4 \
		--dataset $(DATASET) --iterations $(ITERATIONS) \
		--dropout 0.0 --warmup_steps 2000 --grad_clip 0.5 --seed 0 \
		--opt adamw --lr 1e-3 --weight_decay $(WEIGHT_DECAY) --scheduler cos \
		--beta1 0.9 --beta2 0.999 \
		$(COMMON_ARGS) \
		--eval_interval 115 --latest_ckpt_interval 1000

med_mars:
	mkdir -p $(RESULTS_BASE)
	$(LAUNCH) ./src/main.py --config_format base --model llama $(DIST_BACKEND_FLAG) --device $(DEVICE) \
		--run_prefix "$(RUN_TAG)" \
		--datasets_dir "$(DATASETS_DIR)" \
		--n_embd $(N_EMBD) --n_head 12 --n_layer 12 \
		--batch_size $(BATCH_SIZE) --sequence_length 512 --acc_steps 4 \
		--dataset $(DATASET) --iterations $(ITERATIONS) \
		--dropout 0.0 --warmup_steps 2000 --grad_clip 0.5 --seed 0 \
		--opt mars --lr 1e-3 --mars_lr 3e-3 --weight_decay $(WEIGHT_DECAY) --scheduler cos \
		--beta1 0.8 --mars_beta1 0.95 --beta2 0.999 --mars_beta2 0.99 \
		$(COMMON_ARGS) \
		--eval_interval 115 --latest_ckpt_interval 1000

med_ademamix:
	mkdir -p $(RESULTS_BASE)
	$(LAUNCH) ./src/main.py --config_format base --model llama $(DIST_BACKEND_FLAG) --device $(DEVICE) \
		--run_prefix "$(RUN_TAG)" \
		--datasets_dir "$(DATASETS_DIR)" \
		--n_embd $(N_EMBD) --n_head 12 --n_layer 12 \
		--batch_size $(BATCH_SIZE) --sequence_length 512 --acc_steps 4 \
		--dataset $(DATASET) --iterations $(ITERATIONS) \
		--dropout 0.0 --warmup_steps 2000 --grad_clip 0.5 --seed 0 \
		--opt ademamix --lr 1e-3 --weight_decay $(WEIGHT_DECAY) --scheduler cos \
		--beta1 0.9 --beta2 0.999 \
		--adema_beta3 0.999 --adema_alpha 8.0 \
		--adema_beta3_warmup 128000 --adema_alpha_warmup 128000 \
		$(COMMON_ARGS) \
		--eval_interval 115 --latest_ckpt_interval 1000

med_tp_momo:
	mkdir -p $(RESULTS_BASE)
	$(LAUNCH) ./src/main.py --config_format base --model llama $(DIST_BACKEND_FLAG) --device $(DEVICE) \
		--run_prefix "$(RUN_TAG)" \
		--datasets_dir "$(DATASETS_DIR)" \
		--n_embd $(N_EMBD) --n_head 12 --n_layer 12 \
		--batch_size $(BATCH_SIZE) --sequence_length 512 --acc_steps 4 \
		--dataset $(DATASET) --iterations $(ITERATIONS) \
		--dropout 0.0 --warmup_steps 2000 --grad_clip 0.5 --seed 0 \
		--opt two_plane_momo --lr 1e-6 --scheduler cos \
		--two_plane_momo_beta_short 0.9 --two_plane_momo_beta_long 0.999 \
		--two_plane_momo_eps 1e-12 \
		$(COMMON_ARGS) \
		--eval_interval 115 --latest_ckpt_interval 1000

med_tp_momo2:
	mkdir -p $(RESULTS_BASE)
	$(LAUNCH) ./src/main.py --config_format base --model llama $(DIST_BACKEND_FLAG) --device $(DEVICE) \
		--run_prefix "$(RUN_TAG)" \
		--datasets_dir "$(DATASETS_DIR)" \
		--n_embd $(N_EMBD) --n_head 12 --n_layer 12 \
		--batch_size $(BATCH_SIZE) --sequence_length 512 --acc_steps 4 \
		--dataset $(DATASET) --iterations $(ITERATIONS) \
		--dropout 0.0 --warmup_steps 2000 --grad_clip 0.5 --seed 0 \
		--opt two_plane_momo --lr 1e-3 --scheduler cos \
		--two_plane_momo_beta_short 0.9 --two_plane_momo_beta_long 0.999 \
		--two_plane_momo_eps 1e-12 \
		$(COMMON_ARGS) \
		--eval_interval 115 --latest_ckpt_interval 1000

# Small CPU experiments w/ no torchrun
small_soap:
	mkdir -p $(RESULTS_BASE)
	python ./src/main.py $(SMALL_COMMON_ARGS) --opt soap

small_sgd:
	mkdir -p $(RESULTS_BASE)
	python ./src/main.py $(SMALL_COMMON_ARGS) --opt sgd

small_adamw:
	mkdir -p $(RESULTS_BASE)
	python ./src/main.py $(SMALL_COMMON_ARGS) --opt adamw

small_ademamix:
	mkdir -p $(RESULTS_BASE)
	python ./src/main.py $(SMALL_COMMON_ARGS) --opt ademamix

