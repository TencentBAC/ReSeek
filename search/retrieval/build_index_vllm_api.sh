#!/bin/bash

# vLLM HTTP API 版本的索引构建脚本
# 使用方法：
# 1. 先启动 vLLM 服务器
# 2. 再运行此脚本构建索引

# 配置参数
CORPUS_PATH="/group/40077/shyuli/datasets/RL/hot_benchmark/wiki/hot-wiki-18.jsonl"         # 替换为你的语料库路径
SAVE_DIR="/group/40077/shyuli/datasets/RL/hot_benchmark/wiki"
RETRIEVAL_METHOD=qwen                 # 或者 bge, contriever 等
BATCH_SIZE=64                          # API 调用的批次大小
VLLM_API_URL="http://localhost:8000"   # vLLM 服务器地址
#EMBEDDING_PATH="/group/40077/shyuli/datasets/RL/hot_benchmark/wiki/hot-wiki-18-e5/emb_e5.memmap" # 预计算的 embedding 文件路径
corpus_file=/group/40077/shyuli/datasets/RL/hot_benchmark/wiki/hot-wiki-18.jsonl # jsonl
# change faiss_type to HNSW32/64/128 for ANN indexing
# change retriever_name to bm25 for BM25 indexing

#"vllm serve /group/40077/shyuli/models/embedding/e5-base-v2 --task embed --host 0.0.0.0 --port 8000 --data-parallel-size 2"

echo
echo "开始构建索引..."

# 运行索引构建
python index_builder_api.py \
    --retrieval_method $RETRIEVAL_METHOD \
    --corpus_path $CORPUS_PATH \
    --save_dir $SAVE_DIR \
    --batch_size $BATCH_SIZE \
    --vllm_api_url $VLLM_API_URL \
    --max_length 256 \
    --save_embedding \
    --faiss_type "Flat"

# --embedding_path $EMBEDDING_PATH \
echo "索引构建完成！"