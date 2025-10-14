import os
import faiss
import json
import warnings
import numpy as np
from typing import List, Dict
import shutil
import subprocess
import argparse
from tqdm import tqdm
import datasets
import requests
import time


class VLLMEmbeddingClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.embeddings_url = f"{base_url.rstrip('/')}/v1/embeddings"
        self.task_description = "Given a web search query, retrieve relevant passages that answer the query"

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings"""
        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.embeddings_url,
                    json={"input": texts, "encoding_format": "float", "truncate_prompt_tokens": 256},
                    headers={"Content-Type": "application/json"},
                    timeout=300,  # 5 minute timeout
                )
                response.raise_for_status()
                return [item["embedding"] for item in response.json()["data"]]
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    print(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise e

        # This line will never execute, but for type checking
        return []

    def check_server_status(self):
        """Check vLLM server status"""
        try:
            health_url = self.embeddings_url.replace("/v1/embeddings", "/health")
            response = requests.get(health_url, timeout=10)
            return response.status_code == 200
        except:
            return False


def load_vllm_client(vllm_api_url: str = "http://localhost:8000"):
    """Load vLLM HTTP API client"""
    client = VLLMEmbeddingClient(base_url=vllm_api_url)

    if not client.check_server_status():
        print(f"Warning: Unable to connect to vLLM server {vllm_api_url}")
        print("Please ensure vLLM server is running, for example:")
        print(f"vllm serve your-model-path --task embed --host 0.0.0.0 --port 8000")
    else:
        print(f"✓ Successfully connected to vLLM server {vllm_api_url}")

    return client


def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset("json", data_files=corpus_path, split="train", num_proc=4)
    # Ensure return is Dataset type instead of IterableDataset
    if hasattr(corpus, "__len__"):
        return corpus
    else:
        # If IterableDataset, convert to list then back to Dataset
        corpus_list = list(corpus)
        return datasets.Dataset.from_list(corpus_list)


class Index_Builder:
    r"""A tool class used to build an index used in retrieval."""

    def __init__(
        self,
        retrieval_method,
        corpus_path,
        save_dir,
        batch_size,
        faiss_type=None,
        embedding_path=None,
        save_embedding=False,
        faiss_gpu=False,
        vllm_api_url="http://localhost:8000",
        max_length=256,
        chunk_save_interval=10000,  # Force disk flush after processing this many batches
    ):

        self.retrieval_method = retrieval_method.lower()
        self.corpus_path = corpus_path
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.faiss_type = faiss_type if faiss_type is not None else "Flat"
        self.embedding_path = embedding_path
        self.save_embedding = save_embedding
        self.faiss_gpu = faiss_gpu
        self.vllm_api_url = vllm_api_url
        self.max_length = max_length
        self.chunk_save_interval = chunk_save_interval
        # prepare save dir
        print(self.save_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        else:
            if not self._check_dir(self.save_dir):
                warnings.warn("Some files already exists in save dir and may be overwritten.", UserWarning)

        self.index_save_path = os.path.join(self.save_dir, f"{self.retrieval_method}_{self.faiss_type}.index")

        self.embedding_save_path = os.path.join(self.save_dir, f"emb_{self.retrieval_method}.memmap")

        self.corpus = load_corpus(self.corpus_path)

        print("Finish loading...")

    @staticmethod
    def _check_dir(dir_path):
        r"""Check if the dir path exists and if there is content."""

        if os.path.isdir(dir_path):
            if len(os.listdir(dir_path)) > 0:
                return False
        else:
            os.makedirs(dir_path, exist_ok=True)
        return True

    def build_index(self):
        r"""Constructing different indexes based on selective retrieval method."""
        if self.retrieval_method == "bm25":
            self.build_bm25_index()
        else:
            self.build_dense_index()

    def build_bm25_index(self):
        """Building BM25 index based on Pyserini library.

        Reference: https://github.com/castorini/pyserini/blob/master/docs/usage-index.md#building-a-bm25-index-direct-java-implementation
        """

        # to use pyserini pipeline, we first need to place jsonl file in the folder
        self.save_dir = os.path.join(self.save_dir, "bm25")
        os.makedirs(self.save_dir, exist_ok=True)
        temp_dir = self.save_dir + "/temp"
        temp_file_path = temp_dir + "/temp.jsonl"
        os.makedirs(temp_dir)

        shutil.copyfile(self.corpus_path, temp_file_path)

        print("Start building bm25 index...")
        pyserini_args = [
            "--collection",
            "JsonCollection",
            "--input",
            temp_dir,
            "--index",
            self.save_dir,
            "--generator",
            "DefaultLuceneDocumentGenerator",
            "--threads",
            "1",
        ]

        subprocess.run(["python", "-m", "pyserini.index.lucene"] + pyserini_args)

        shutil.rmtree(temp_dir)

        print("Finish!")

    def _load_embedding(self, embedding_path, corpus_size, hidden_size):
        all_embeddings = np.memmap(embedding_path, mode="r", dtype=np.float32).reshape(corpus_size, hidden_size)
        return all_embeddings

    def _save_embedding(self, all_embeddings):
        memmap = np.memmap(self.embedding_save_path, shape=all_embeddings.shape, mode="w+", dtype=all_embeddings.dtype)
        length = all_embeddings.shape[0]
        # add in batch
        save_batch_size = 10000
        if length > save_batch_size:
            for i in tqdm(range(0, length, save_batch_size), leave=False, desc="Saving Embeddings"):
                j = min(i + save_batch_size, length)
                memmap[i:j] = all_embeddings[i:j]
        else:
            memmap[:] = all_embeddings

    def encode_all_vllm_api(self):
        """Batch encoding using vLLM HTTP API (chunked save version)

        Call vLLM service via HTTP requests, save to disk while inferring to avoid memory explosion
        """
        print(f"Starting document encoding using vLLM API ({self.vllm_api_url})...")
        print("Using chunked save mechanism to avoid memory explosion...")

        # Calculate total batches and embedding dimensions
        corpus_size = len(self.corpus) if hasattr(self.corpus, "__len__") else sum(1 for _ in self.corpus)
        total_batches = (corpus_size + self.batch_size - 1) // self.batch_size

        # Get first batch embedding to determine dimensions
        if hasattr(self.corpus, "__getitem__"):
            first_batch_data = self.corpus[0 : self.batch_size]["contents"]
        else:
            # If IterableDataset, special handling needed
            first_batch_items = []
            for i, item in enumerate(self.corpus):
                if i >= self.batch_size:
                    break
                first_batch_items.append(item)
            first_batch_data = [item["contents"] for item in first_batch_items]
        if self.retrieval_method == "e5":
            first_batch_data = [f"passage: {doc}" for doc in first_batch_data]

        first_batch_embeddings = self.encoder.get_embeddings(first_batch_data)
        first_batch_embeddings = np.array(first_batch_embeddings, dtype=np.float32)

        # Normalize first batch
        if "dpr" not in self.retrieval_method:
            norms = np.linalg.norm(first_batch_embeddings, axis=1, keepdims=True)
            first_batch_embeddings = first_batch_embeddings / (norms + 1e-12)

        hidden_size = first_batch_embeddings.shape[1]

        # Create memmap file for chunked saving
        memmap = np.memmap(self.embedding_save_path, shape=(corpus_size, hidden_size), mode="w+", dtype=np.float32)

        # Save first batch
        actual_first_batch_size = first_batch_embeddings.shape[0]
        memmap[0:actual_first_batch_size] = first_batch_embeddings

        print(f"Embedding dimensions: {hidden_size}")
        print(f"Total batches: {total_batches}")
        print(f"Starting chunked processing...")

        try:
            # Process remaining batches
            for batch_idx in tqdm(range(1, total_batches), desc="vLLM API Embedding:"):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, corpus_size)

                if hasattr(self.corpus, "__getitem__"):
                    batch_data = self.corpus[start_idx:end_idx]["contents"]
                else:
                    # If IterableDataset, special handling needed
                    batch_items = []
                    for i, item in enumerate(self.corpus):
                        if i >= start_idx and i < end_idx:
                            batch_items.append(item)
                        elif i >= end_idx:
                            break
                    batch_data = [item["contents"] for item in batch_items]

                # Add prefix based on retrieval method
                if self.retrieval_method == "e5":
                    batch_data = [f"passage: {doc}" for doc in batch_data]

                # Get embeddings using vLLM HTTP API
                batch_embeddings_list = self.encoder.get_embeddings(batch_data)

                # Convert to numpy array
                batch_embeddings = np.array(batch_embeddings_list, dtype=np.float32)

                # Normalize (except for DPR models)
                if "dpr" not in self.retrieval_method:
                    # Calculate L2 norm and normalize
                    norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                    batch_embeddings = batch_embeddings / (norms + 1e-12)  # Avoid division by zero

                # Save directly to memmap file
                memmap[start_idx:end_idx] = batch_embeddings

                # Periodically force flush to disk
                if batch_idx % self.chunk_save_interval == 0:
                    memmap.flush()
                    print(f"Processed {batch_idx}/{total_batches} batches, flushed to disk")

                # Clean up memory
                del batch_embeddings, batch_embeddings_list

        except Exception as e:
            print(f"vLLM API call failed: {e}")
            print("Please ensure vLLM server is running and supports embedding tasks")
            raise e

        # Final flush to disk
        memmap.flush()
        print("All batches processed, data saved to disk")

        # Load all embeddings from memmap file
        all_embeddings = np.memmap(
            self.embedding_save_path, shape=(corpus_size, hidden_size), mode="r", dtype=np.float32
        )

        return all_embeddings

    def encode_all(self):
        """Encode using vLLM API"""
        return self.encode_all_vllm_api()

    def build_dense_index(self):
        """Build dense index using vLLM API"""

        if os.path.exists(self.index_save_path):
            print("The index file already exists and will be overwritten.")

        if self.embedding_path is not None:
            # Load from pre-computed embedding file
            # Note: Need to know embedding dimensions, assuming 768 here, adjust for actual model
            hidden_size = 768  # Adjust based on actual model
            corpus_size = len(self.corpus) if hasattr(self.corpus, "__len__") else sum(1 for _ in self.corpus)
            print(corpus_size)
            all_embeddings = self._load_embedding(self.embedding_path, corpus_size, hidden_size)
        else:
            # Calculate embeddings using API
            all_embeddings = self.encode_all()
            if self.save_embedding:
                self._save_embedding(all_embeddings)
            del self.corpus
        del self.corpus
        all_embeddings = np.array(all_embeddings, dtype=np.float16)
        print(all_embeddings.shape)
        # build index
        import pdb

        pdb.set_trace()
        print("Creating index")
        dim = all_embeddings.shape[-1]
        faiss_index = faiss.index_factory(dim, self.faiss_type, faiss.METRIC_INNER_PRODUCT)

        if self.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True  # Consistent with float16 data
            co.shard = True
            faiss_index = faiss.index_cpu_to_all_gpus(faiss_index, co)
            if not faiss_index.is_trained:
                faiss_index.train(all_embeddings)
            faiss_index.add(all_embeddings)
            faiss_index = faiss.index_gpu_to_cpu(faiss_index)
        else:
            if not faiss_index.is_trained:
                faiss_index.train(all_embeddings)
            faiss_index.add(all_embeddings)

        faiss.write_index(faiss_index, self.index_save_path)
        print("Finish!")


def main():
    parser = argparse.ArgumentParser(description="Creating index using vLLM HTTP API.")

    # Basic parameters
    parser.add_argument(
        "--retrieval_method", type=str, required=True, help="Retrieval method (e5, bge, contriever, etc.)"
    )
    parser.add_argument("--corpus_path", type=str, required=True, help="Corpus JSON file path")
    parser.add_argument("--save_dir", default="indexes/", type=str, help="Index save directory")

    # Parameters for building dense index
    parser.add_argument("--batch_size", type=int, default=32, help="API call batch size")
    parser.add_argument("--faiss_type", default="Flat", type=str, help="FAISS index type")
    parser.add_argument("--embedding_path", default=None, type=str, help="Pre-computed embedding file path")
    parser.add_argument("--save_embedding", action="store_true", default=False, help="Whether to save embedding")
    parser.add_argument("--faiss_gpu", default=False, action="store_true", help="Whether to use GPU for index building")

    # vLLM API parameters
    parser.add_argument("--vllm_api_url", type=str, default="http://localhost:8000", help="vLLM API service address")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum text length")
    parser.add_argument(
        "--chunk_save_interval", type=int, default=10000, help="Force disk flush after processing this many batches"
    )

    args = parser.parse_args()

    print("=== vLLM API Index Builder ===")
    print(f"Retrieval method: {args.retrieval_method}")
    print(f"Corpus path: {args.corpus_path}")
    print(f"Save directory: {args.save_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"vLLM API: {args.vllm_api_url}")
    print(f"FAISS type: {args.faiss_type}")
    print(f"Max length: {args.max_length}")
    print(f"Chunk save interval: {args.chunk_save_interval}")
    print()

    index_builder = Index_Builder(
        retrieval_method=args.retrieval_method,
        corpus_path=args.corpus_path,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        faiss_type=args.faiss_type,
        embedding_path=args.embedding_path,
        save_embedding=args.save_embedding,
        faiss_gpu=args.faiss_gpu,
        vllm_api_url=args.vllm_api_url,
        max_length=args.max_length,
        chunk_save_interval=args.chunk_save_interval,
    )
    index_builder.build_index()


if __name__ == "__main__":
    main()
