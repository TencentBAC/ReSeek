import os
import faiss
import json
import warnings
import numpy as np
from typing import cast, List, Dict
import shutil
import subprocess
import argparse
import torch
from tqdm import tqdm

import datasets
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sentence_transformers import SentenceTransformer
import logging


def load_model(model_path: str, use_fp16: bool = False):
    """Load model using sentence_transformers"""
    model = SentenceTransformer(model_path)
    model.eval()
    if use_fp16:
        model = model.half()
    # sentence_transformers does not need separate tokenizer
    return model, None


def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"):
    if pooling_method == "mean" and attention_mask is not None:
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler" and pooler_output is not None:
        return pooler_output
    elif pooling_method == "mean":
        # If no attention_mask, use simple average
        return last_hidden_state.mean(dim=1)
    else:
        raise NotImplementedError("Pooling method not implemented!")


def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset("json", data_files=corpus_path, split="train", num_proc=4)
    return corpus


class Index_Builder:
    r"""A tool class used to build an index used in retrieval."""

    def __init__(
        self,
        retrieval_method,
        model_path,
        corpus_path,
        save_dir,
        max_length,
        batch_size,
        use_fp16,
        pooling_method,
        faiss_type=None,
        embedding_path=None,
        save_embedding=False,
        faiss_gpu=False,
    ):

        self.retrieval_method = retrieval_method.lower()
        self.model_path = model_path
        self.corpus_path = corpus_path
        self.save_dir = save_dir
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.pooling_method = pooling_method
        self.faiss_type = faiss_type if faiss_type is not None else "Flat"
        self.embedding_path = embedding_path
        self.save_embedding = save_embedding
        self.faiss_gpu = faiss_gpu

        self.gpu_num = torch.cuda.device_count()
        # prepare save dir
        print(self.save_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        else:
            if not self._check_dir(self.save_dir):
                warnings.warn("Some files already exists in save dir and may be overwritten.", UserWarning)

        self.index_save_path = os.path.join(self.save_dir, f"{self.retrieval_method}_{self.faiss_type}.index")

        # Use npz format to save embeddings, supports chunking
        self.embedding_save_dir = os.path.join(self.save_dir, f"emb_{self.retrieval_method}_slices")
        if not os.path.exists(self.embedding_save_dir):
            os.makedirs(self.embedding_save_dir)
        self.metadata_path = os.path.join(self.embedding_save_dir, "metadata.json")

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

    def _init_streaming_save(self, corpus_size, embedding_dim, dtype=np.float16):
        """Initialize streaming save npz mechanism"""
        self.slice_size = 10000  # Each npz file saves 10000 embeddings
        self.current_slice_idx = 0
        self.current_slice_embeddings = []
        self.total_saved = 0
        self.corpus_size = corpus_size
        self.embedding_dim = embedding_dim
        self.dtype = dtype

        # Save metadata
        metadata = {
            "corpus_size": corpus_size,
            "embedding_dim": embedding_dim,
            "dtype": str(dtype),
            "slice_size": self.slice_size,
            "total_slices": (corpus_size + self.slice_size - 1) // self.slice_size,
        }
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _streaming_save_batch(self, batch_embeddings):
        """Stream save single batch embeddings to npz file"""
        batch_embeddings = batch_embeddings.astype(self.dtype)

        for embedding in batch_embeddings:
            self.current_slice_embeddings.append(embedding)

            # When current slice is full, save slice
            if len(self.current_slice_embeddings) >= self.slice_size:
                self._save_current_slice()

        self.total_saved += len(batch_embeddings)

        # Print progress every 50000 embeddings saved
        if self.total_saved % 50000 == 0:
            print(
                f"Saved {self.total_saved}/{self.corpus_size} embeddings ({self.total_saved/self.corpus_size*100:.1f}%)"
            )

    def _save_current_slice(self):
        """Save current slice to npz file"""
        if not self.current_slice_embeddings:
            return

        slice_array = np.array(self.current_slice_embeddings, dtype=self.dtype)
        slice_path = os.path.join(self.embedding_save_dir, f"slice_{self.current_slice_idx:06d}.npz")

        np.savez_compressed(slice_path, embeddings=slice_array)

        # Clean current slice data, release memory
        self.current_slice_embeddings = []
        self.current_slice_idx += 1

    def _finalize_streaming_save(self):
        """Complete streaming save, save last slice"""
        # Save last incomplete slice
        if self.current_slice_embeddings:
            self._save_current_slice()

        print(f"All embeddings saved to {self.current_slice_idx} npz files")
        print(f"Save directory: {self.embedding_save_dir}")

    def _load_embedding_from_slices(self):
        """Load embeddings from npz slice files"""
        # Read metadata
        with open(self.metadata_path, "r") as f:
            metadata = json.load(f)

        corpus_size = metadata["corpus_size"]
        embedding_dim = metadata["embedding_dim"]

        print(f"Loading {corpus_size} embeddings from slice files...")

        # Create result array
        all_embeddings = np.zeros((corpus_size, embedding_dim), dtype=np.float16)

        current_idx = 0
        slice_idx = 0

        with tqdm(total=corpus_size, desc="Loading embeddings") as pbar:
            while current_idx < corpus_size:
                slice_path = os.path.join(self.embedding_save_dir, f"slice_{slice_idx:06d}.npz")
                if not os.path.exists(slice_path):
                    break

                # Load slice
                slice_data = np.load(slice_path)
                slice_embeddings = slice_data["embeddings"]

                # Copy to result array
                end_idx = min(current_idx + len(slice_embeddings), corpus_size)
                all_embeddings[current_idx:end_idx] = slice_embeddings[: end_idx - current_idx]

                pbar.update(end_idx - current_idx)
                current_idx = end_idx
                slice_idx += 1

        return all_embeddings

    def encode_all_streaming(self):
        """Stream encoding using sentence_transformers, save while generating to avoid memory explosion"""
        # Safely get corpus size
        try:
            corpus_size = len(self.corpus)
        except TypeError:
            # If IterableDataset, need to iterate to calculate size
            corpus_size = sum(1 for _ in self.corpus)
            # Reload corpus because IterableDataset can only be iterated once
            self.corpus = load_corpus(self.corpus_path)

        print(f"Starting stream encoding {corpus_size} documents...")

        # Get first batch to determine embedding dimensions
        first_batch_size = min(32, corpus_size)
        try:
            # Try using index access
            first_batch_data = self.corpus[0:first_batch_size]["contents"]
        except (TypeError, KeyError):
            # If index not supported, use iterator
            first_batch_data = []
            for i, item in enumerate(self.corpus):
                if i >= first_batch_size:
                    break
                first_batch_data.append(item["contents"])
            # Reload corpus
            self.corpus = load_corpus(self.corpus_path)

        first_batch_data = self._add_prefix(first_batch_data)
        first_embeddings = self.encoder.encode(
            first_batch_data, batch_size=32, convert_to_numpy=True, normalize_embeddings=True
        )[:, :1024]
        embedding_dim = first_embeddings.shape[1]
        # import pdb; pdb.set_trace()
        # Initialize streaming save
        self._init_streaming_save(corpus_size, embedding_dim, dtype=np.float16)

        # Save first batch
        self._streaming_save_batch(first_embeddings.astype(np.float16))

        # Process remaining batches
        start_idx = len(first_batch_data)

        # Check if using multiple GPUs
        if self.gpu_num > 1:
            print(f"Using {self.gpu_num} GPUs for sentence_transformers encoding")
            self._encode_multi_gpu_streaming(start_idx, corpus_size)
        else:
            self._encode_single_gpu_streaming(start_idx, corpus_size)

        # Complete save
        self._finalize_streaming_save()

        print("Stream encoding completed!")

    def _add_prefix(self, batch_data):
        """Add prefix based on retrieval method"""
        if self.retrieval_method == "e5":
            return [f"passage: {doc}" for doc in batch_data]
        elif "qwen" in self.retrieval_method.lower():
            return [f"{doc}" for doc in batch_data]
        elif "bge" in self.retrieval_method.lower():
            return [f"Represent this sentence for searching relevant passages: {doc}" for doc in batch_data]
        else:
            return batch_data

    def _encode_single_gpu_streaming(self, start_idx, corpus_size):
        """Single GPU streaming encoding"""
        for batch_start in tqdm(range(start_idx, corpus_size, self.batch_size), desc="Single GPU Streaming Encoding"):
            batch_end = min(batch_start + self.batch_size, corpus_size)

            try:
                # Try using index access
                batch_data = self.corpus[batch_start:batch_end]["contents"]
            except (TypeError, KeyError):
                # If index not supported, use iterator (lower efficiency but better compatibility)
                batch_data = []
                for i, item in enumerate(self.corpus):
                    if i >= batch_end:
                        break
                    if i >= batch_start:
                        batch_data.append(item["contents"])
                # Reload corpus for next use
                self.corpus = load_corpus(self.corpus_path)

            batch_data = self._add_prefix(batch_data)

            # Encode
            embeddings = self.encoder.encode(
                batch_data, batch_size=32, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True
            )

            # Save immediately, release memory
            self._streaming_save_batch(embeddings.astype(np.float16))
            del embeddings  # Explicitly release memory

    def _encode_multi_gpu_streaming(self, start_idx, corpus_size):
        """Multi GPU streaming encoding"""
        pool = None
        try:
            # Start multi-process pool
            pool = self.encoder.start_multi_process_pool()

            # Process in batches, don't load all data at once
            mega_batch_size = self.batch_size * 16  # Process 4 batches at a time

            for mega_start in tqdm(range(start_idx, corpus_size, mega_batch_size), desc="Multi GPU Streaming Encoding"):
                mega_end = min(mega_start + mega_batch_size, corpus_size)

                try:
                    # Try using index access
                    mega_batch_data = self.corpus[mega_start:mega_end]["contents"]
                except (TypeError, KeyError):
                    # If index not supported, use iterator
                    mega_batch_data = []
                    for i, item in enumerate(self.corpus):
                        if i >= mega_end:
                            break
                        if i >= mega_start:
                            mega_batch_data.append(item["contents"])
                    # Reload corpus
                    self.corpus = load_corpus(self.corpus_path)

                mega_batch_data = self._add_prefix(mega_batch_data)

                # Use multi-process pool encoding
                embeddings = self.encoder.encode_multi_process(
                    mega_batch_data, pool, batch_size=32, show_progress_bar=False, normalize_embeddings=True
                )[:, :1024]

                # Save immediately, release memory
                self._streaming_save_batch(embeddings.astype(np.float16))
                del embeddings  # Explicitly release memory

        except Exception as e:
            print(f"Multi GPU encoding failed: {e}")
            print("Falling back to single GPU processing...")
            if pool is not None:
                try:
                    self.encoder.stop_multi_process_pool(pool)
                except:
                    pass
            self._encode_single_gpu_streaming(start_idx, corpus_size)
        finally:
            # Ensure multi-process pool is stopped
            if pool is not None:
                try:
                    self.encoder.stop_multi_process_pool(pool)
                except:
                    pass

    @torch.no_grad()
    def build_dense_index(self):
        """Obtain the representation of documents based on the embedding model(BERT-based) and
        construct a faiss index.
        """

        if os.path.exists(self.index_save_path):
            print("The index file already exists and will be overwritten.")

        self.encoder, self.tokenizer = load_model(
            model_path=self.model_path,
            use_fp16=self.use_fp16,
        )

        if self.embedding_path is not None:
            # Load embeddings from specified path
            if os.path.exists(self.metadata_path):
                # Load from npz slice files
                all_embeddings = self._load_embedding_from_slices()
            else:
                # Compatible with old memmap format
                from transformers import AutoConfig

                config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
                hidden_size = config.hidden_size
                try:
                    corpus_size = len(self.corpus)
                except TypeError:
                    corpus_size = sum(1 for _ in self.corpus)
                    self.corpus = load_corpus(self.corpus_path)
                all_embeddings = np.memmap(self.embedding_path, mode="r", dtype=np.float16).reshape(
                    corpus_size, hidden_size
                )
        else:
            # Use streaming encoding to generate embeddings
            if self.save_embedding:
                self.encode_all_streaming()
                all_embeddings = self._load_embedding_from_slices()
            else:
                # If not saving, still use streaming encoding but don't persist
                print(
                    "Warning: Not saving embeddings may cause memory shortage, recommend using --save_embedding parameter"
                )
                self.encode_all_streaming()
                all_embeddings = self._load_embedding_from_slices()

            del self.corpus  # Release corpus memory

        # build index
        print("Creating index")
        dim = all_embeddings.shape[-1]
        faiss_index = faiss.index_factory(dim, self.faiss_type, faiss.METRIC_INNER_PRODUCT)

        if self.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
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

        # save index
        faiss.write_index(faiss_index, self.index_save_path)
        print(f"Index saved to: {self.index_save_path}")


MODEL2POOLING = {"e5": "mean", "bge": "cls", "contriever": "mean", "jina": "mean", "qwen": "mean"}


def main():
    parser = argparse.ArgumentParser(description="Creating index.")

    # Basic parameters
    parser.add_argument("--retrieval_method", type=str)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--corpus_path", type=str)
    parser.add_argument("--save_dir", default="indexes/", type=str)

    # Parameters for building dense index
    parser.add_argument("--max_length", type=int, default=180)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--use_fp16", default=False, action="store_true")
    parser.add_argument("--pooling_method", type=str, default=None)
    parser.add_argument("--faiss_type", default=None, type=str)
    parser.add_argument("--embedding_path", default=None, type=str)
    parser.add_argument("--save_embedding", action="store_true", default=False)
    parser.add_argument("--faiss_gpu", default=False, action="store_true")

    args = parser.parse_args()

    if args.pooling_method is None:
        pooling_method = "mean"
        for k, v in MODEL2POOLING.items():
            if k in args.retrieval_method.lower():
                pooling_method = v
                break
    else:
        if args.pooling_method not in ["mean", "cls", "pooler"]:
            raise NotImplementedError
        else:
            pooling_method = args.pooling_method

    index_builder = Index_Builder(
        retrieval_method=args.retrieval_method,
        model_path=args.model_path,
        corpus_path=args.corpus_path,
        save_dir=args.save_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        use_fp16=args.use_fp16,
        pooling_method=pooling_method,
        faiss_type=args.faiss_type,
        embedding_path=args.embedding_path,
        save_embedding=args.save_embedding,
        faiss_gpu=args.faiss_gpu,
    )
    index_builder.build_index()


if __name__ == "__main__":
    main()
