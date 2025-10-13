#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DATA_DIR='${PROJECT_ROOT}/scripts/runs/reseek/data'

WAND_PROJECT='Search-R1'


export BASE_MODEL='${MODEL_DIR}/Qwen2.5-3B-Instruct'
export EXPERIMENT_NAME=searchR1_nq-search-r1-grpo-qwen2.5-3b-em-s3_v0.7
set -x

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh    
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export EMBEDDING_SERVER_URL="http://9.130.192.140:8000"

#export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues

# max_prompt_length = (config['training']['max_start_length'] + config['training']['max_response_length'] * (config['training']['max_turns'] - 1) + config['training']['max_obs_length'] * config['training']['max_turns'])
TRAIN_DATA_DIR=${DATA_DIR}/searchR1_processed_direct_v0.2
TEST_DATA_DIR=${DATA_DIR}/searchR1_processed_direct_v0.2
TIME_STAMP=$(date +%Y%m%d_%H%M%S)

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$TRAIN_DATA_DIR/train.parquet \
    data.val_files=$TEST_DATA_DIR/test.parquet \
    data.train_batch_size=512 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.shuffle=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.n_agent=5 \
    actor_rollout_ref.rollout.temperature=1 \
    algorithm.no_think_rl=false \
    trainer.logger=['console','tensorboard'] \
    trainer.balance_batch=false \
    trainer.val_only=false \
    trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=16 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=1 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    trainer.device=npu \
    reward_model.reward_manager=batch_naive \
    max_turns=4 \
    retriever.url="http://9.130.192.143:8100/retrieve" \
    retriever.topk=3 \
    2>&1 | tee logs/$EXPERIMENT_NAME_$TIME_STAMP.log
# trainer.total_training_steps=1005 \