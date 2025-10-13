# ReSeek: A Self-Correcting Framework for Search Agents with Instructive Rewards

## Environment Setup

### Basic Installation

```bash
# Clone the repository

cd ReSeek

# Install dependencies
pip install -r requirements.txt

# Run installation script
bash install.sh
```

### Optional Dependencies

**SGLang Support**:
```bash
pip install -r requirements_sglang.txt
```

**NPU (Ascend) Support**:
```bash
pip install -r requirements-npu.txt
```

**vLLM, SGLang, Megatron-Core**:
```bash
bash scripts/install_vllm_sglang_mcore.sh
```

## Environment Variables

Before running training scripts, set the following environment variables:

```bash
# Set project root directory
export PROJECT_ROOT=/path/to/ReSeek

# Set model directory
export MODEL_DIR=/path/to/models

# Set data directory (optional, can use the provided data in scripts/runs/reseek/data)
export DATA_DIR=/path/to/datasets
```

## Data Preparation

### ReSeek Dataset

```bash
python scripts/preprocess_search_r1_dataset.py \
  --hf_repo_id xxx/ReSeek_train_test \
  --local_dir ${DATA_DIR}/processed_dateset
```

## Training

**GRPO Training (NPU)**:
```bash
cd scripts/runs/reseek

# 3B model
bash train_grpo.sh

# 7B model
bash train_grpo_7b.sh
```

**PPO Training (NPU)**:
```bash
cd scripts/runs/reseek

# 3B model
bash train_ppo.sh

# 7B model
bash train_ppo_7b.sh
```

**PPO Training (CUDA)**:
```bash
cd scripts/runs/reseek
bash train_ppo_cuda.sh
```

**Multi-turn Training**:
```bash
cd scripts/runs/reseek
bash run_qwen2.5-3b_instruct_search_multiturn.sh
```


## Model Conversion

### HuggingFace to Megatron-Core

```bash
python scripts/converter_hf_to_mcore.py \
  --input_dir ${MODEL_DIR}/hf_model \
  --output_dir ${MODEL_DIR}/mcore_model \
  --tensor_parallel_size 2
```

## Device Configuration

**CUDA Devices**:
```bash
# CUDA is used by default, no extra configuration needed
python -m verl.trainer.main_ppo [arguments]
```

**NPU Devices**:
```bash
# Set up environment first
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

# Then specify device in configuration
python -m verl.trainer.main_ppo \
  trainer.device=npu \
  [other arguments]
```

## Retrieval Service

### Build Index

**Using Transformers**:
```bash
cd scripts/runs/reseek/reseek_search/search
bash build_index.sh
```

**Using vLLM**:
```bash
cd scripts/runs/reseek/reseek_search/search
bash build_index_vllm.sh
```

### Launch Retrieval Service

```bash
cd scripts/runs/reseek
bash retrieval_launch.sh
```

## License

Apache License 2.0
