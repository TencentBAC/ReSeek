corpus_file=/group/40077/shyuli/datasets/RL/hot_benchmark/wiki/hot-wiki-18.jsonl # jsonl
save_dir=/group/40077/shyuli/datasets/RL/wiki
retriever_name=conan
retriever_model=/group/40077/shyuli/models/ours/embedding/conan-0827/ckpts
# /group/40077/shyuli/models/embedding/bge-large-en-v1.5
# /group/40077/shyuli/models/ours/embedding/conan-0827/ckpts
# /group/40077/shyuli/models/embedding/e5-base-v2
# /group/40077/shyuli/models/embedding/qwen/Qwen3-Embedding-0.6B
#
#
# 
# 

# change faiss_type to HNSW32/64/128 for ANN indexing
# change retriever_name to bm25 for BM25 indexing
# 使用多 GPU 进行 sentence_transformers 编码
CUDA_VISIBLE_DEVICES=0,1,2,3 python index_builder.py \
    --retrieval_method $retriever_name \
    --model_path $retriever_model \
    --corpus_path $corpus_file \
    --save_dir $save_dir \
    --use_fp16 \
    --max_length 256 \
    --batch_size 64 \
    --pooling_method mean \
    --faiss_type Flat \
    --save_embedding \
    --embedding_path /group/40077/shyuli/datasets/RL/wiki/emb_conan_slices