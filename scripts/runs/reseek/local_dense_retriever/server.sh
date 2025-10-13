save_path=${DATA_DIR}/wiki-18
index_file=$save_path/e5_Flat.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path=${MODEL_DIR}/embedding/e5-base-v2

python ${PROJECT_ROOT}/scripts/runs/reseek/local_dense_retriever/retrieval_server.py --index_path $index_file --corpus_path $corpus_file --topk 3 --retriever_name $retriever_name --retriever_model $retriever_path #--faiss_gpu