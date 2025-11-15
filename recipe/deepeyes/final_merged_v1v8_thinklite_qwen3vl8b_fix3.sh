#!/bin/bash

set -x

export LLM_AS_A_JUDGE_BASE="https://new.xi-ai.cn/v1"
export OPENAI_API_KEY="sk-eVHQ37hHIcCCpxzallhlwzphdJtQfpCxv0WGpg79woIzqzFL"
export WANDB_API_KEY="347bf026c87f5e531a207e55607805d060a53a85"
export LLM_JUDGE_MODEL="gpt-4o-mini"

PROJECT_NAME="verl_qwen3_vl_8b"
EXPERIMENT_NAME="8b_batch64_ppo32-1_n8_0.25_fix3"

BASEDIR=/pfs/wangbo/project/a_dpo/wangxuan/datasets/DeepEyes-Datasets-47k-0.02
SAVE_CHECKPOINT_DIR=/pfs/wangbo/project/a_dpo/DeepEyes-Data/verl_checkpoints
VISUAL_DATASET_TRAIN_0_1_2=${BASEDIR}/data_0.1.2_visual_toolbox_v2.parquet
VISUAL_DATASET_TRAIN_0_8=${BASEDIR}/data_v0.8_visual_toolbox_v2.parquet
EUREKA_DATASET_TRAIN=${BASEDIR}/data_thinklite_reasoning_acc.parquet
DATASET_VAL=${BASEDIR}/data_thinklite_reasoning_acc.parquet

REF_MODEL_PATH=/pfs/wangbo/project/a_dpo/DeepEyes-Data/backbone/qwen3/Qwen3-VL-8B-Instruct

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    --config-path=/pfs/wangbo/project/a_dpo/verl/recipe/deepeyes/configs \
    --config-name='deepeyes_multiturn_grpo_fix3' \
    data.train_files=[${VISUAL_DATASET_TRAIN_0_1_2},${VISUAL_DATASET_TRAIN_0_8},${EUREKA_DATASET_TRAIN}] \
    data.val_files=[${EUREKA_DATASET_TRAIN}] \
    data.train_batch_size=8 \
    data.max_prompt_length=8192 \
    data.max_response_length=20480 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=5 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=recipe/deepeyes/configs/image_zoom_in_tool_config.yaml \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb','tensorboard'] \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=24 \
    trainer.test_freq=10000 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    +trainer.tensorboard_dir=${SAVE_CHECKPOINT_DIR}/logs/tensorboard \
    +trainer.rl_logging_board_dir=${SAVE_CHECKPOINT_DIR}/logs/rl_logging_board \
    trainer.total_epochs=8 2>&1 | tee ./logs/${EXPERIMENT_NAME}.log
