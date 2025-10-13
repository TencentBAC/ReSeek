#!/bin/bash

# 使用 vLLM 构建索引的示例脚本
# 相比原版 transformers 实现，vLLM 可以显著提升推理速度

# 设置模型和数据路径
MODEL_PATH="BAAI/bge-base-en-v1.5"  # 替换为你的模型路径
CORPUS_PATH="data/corpus.jsonl"      # 替换为你的语料库路径
SAVE_DIR="indexes/vllm_index"        # 索引保存目录
EMBEDDING_PATH="${DATA_DIR}/hot_benchmark/wiki/hot-wiki-18-qwen/emb_qwen.memmap" # 替换为你的 embedding 路径

# 基本参数
RETRIEVAL_METHOD="bge"
MAX_LENGTH=512
BATCH_SIZE=256  # vLLM 可以处理更大的批次大小

# vLLM 特定参数
TENSOR_PARALLEL_SIZE=1  # 根据你的 GPU 数量调整（建议设置为 GPU 数量）
MAX_MODEL_LEN=512       # 最大模型长度

# 版本要求和性能优化建议
echo "=== vLLM 版本要求 ==="
echo "需要 vllm>=0.8.5 以支持 embedding 任务"
echo "安装命令: pip install 'vllm>=0.8.5'"
echo ""
echo "=== 性能优化建议 ==="
echo "1. 如果有多个 GPU，建议设置 TENSOR_PARALLEL_SIZE 为 GPU 数量"
echo "2. 批次大小可以设置更大，vLLM 优化了内存使用"
echo "3. 使用 fp16 可以进一步提升速度和节省显存"
echo "4. 确保 CUDA 版本与 vLLM 兼容"
echo ""

echo "开始使用 vLLM 构建索引..."
echo "模型路径: $MODEL_PATH"
echo "语料库路径: $CORPUS_PATH"
echo "保存目录: $SAVE_DIR"
echo "张量并行大小: $TENSOR_PARALLEL_SIZE"
echo "批次大小: $BATCH_SIZE"

python index_builder.py \
    --retrieval_method $RETRIEVAL_METHOD \
    --model_path $MODEL_PATH \
    --corpus_path $CORPUS_PATH \
    --save_dir $SAVE_DIR \
    --max_length $MAX_LENGTH \
    --batch_size $BATCH_SIZE \
    --use_fp16 \
    --use_vllm \
    --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
    --max_model_len $MAX_MODEL_LEN \
    --save_embedding \
    --embedding_path $EMBEDDING_PATH \
    --faiss_type "IVF1024,Flat"

echo "索引构建完成！"
echo "索引文件保存在: $SAVE_DIR"

# 对比版本：不使用 vLLM（原始 transformers 实现）
echo ""
echo "如果要使用原始 transformers 实现（较慢），可以运行："
echo "python index_builder.py \\"
echo "    --retrieval_method $RETRIEVAL_METHOD \\"
echo "    --model_path $MODEL_PATH \\"
echo "    --corpus_path $CORPUS_PATH \\"
echo "    --save_dir ${SAVE_DIR}_transformers \\"
echo "    --max_length $MAX_LENGTH \\"
echo "    --batch_size 64 \\"  # transformers 通常需要较小的批次大小
echo "    --use_fp16 \\"
echo "    --save_embedding \\"
echo "    --faiss_type \"IVF1024,Flat\""
