corpus_file=${DATA_DIR}/hot_benchmark/wiki/hot-wiki-18.jsonl # jsonl
save_dir=${DATA_DIR}/hot_benchmark/wiki
retriever_name=qwen # this is for indexing naming
retriever_model=${MODEL_DIR}/embedding/qwen/Qwen3-Embedding-0.6B
EMBEDDING_PATH="${DATA_DIR}/hot_benchmark/wiki/hot-wiki-18-qwen/emb_qwen.memmap"
# change faiss_type to HNSW32/64/128 for ANN indexing
# change retriever_name to bm25 for BM25 indexing
CUDA_VISIBLE_DEVICES=0,1,2,3, python index_builder.py \
    --retrieval_method $retriever_name \
    --model_path $retriever_model \
    --corpus_path $corpus_file \
    --save_dir $save_dir \
    --use_fp16 \
    --max_length 256 \
    --batch_size 512 \
    --pooling_method mean \
    --faiss_type Flat \
    --use_vllm \
    --save_embedding \
    --embedding_path $EMBEDDING_PATH
