#!/usr/bin/env python3
"""
Simple NPZ file reader

Used to read a single NPZ file and view its contents
"""

import numpy as np
import argparse


def read_npz_file(npz_path):
    """
    Read NPZ file and display information

    Args:
        npz_path: NPZ file path
    """
    print(f"Reading file: {npz_path}")

    data = np.load(npz_path)

    print(f"Keys in file: {list(data.keys())}")

    if "embeddings" in data:
        embeddings = data["embeddings"]

        print(f"\nEmbeddings info:")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Data type: {embeddings.dtype}")
        print(f"  Memory usage: {embeddings.nbytes / (1024*1024):.1f} MB")

        if len(embeddings) > 0:
            norms = np.linalg.norm(embeddings, axis=1)
            means = np.mean(embeddings, axis=1)

            print(f"  L2 norm - mean: {np.mean(norms):.6f}, std: {np.std(norms):.6f}")
            print(f"  L2 norm - min: {np.min(norms):.6f}, max: {np.max(norms):.6f}")
            print(f"  Vector mean - mean: {np.mean(means):.6f}, std: {np.std(means):.6f}")

            n_show = min(3, len(embeddings))
            print(f"\nFirst {n_show} vectors:")
            for i in range(n_show):
                print(f"  Vector {i}: norm={np.linalg.norm(embeddings[i]):.6f}, first 5 values={embeddings[i][:5]}")
    else:
        print("'embeddings' key not found in file")

        for key in data.keys():
            arr = data[key]
            print(f"\nKey '{key}':")
            print(f"  Shape: {arr.shape}")
            print(f"  Data type: {arr.dtype}")
            if arr.size <= 10:
                print(f"  Values: {arr}")
            else:
                print(f"  First 10 values: {arr.flat[:10]}")


def main():
    parser = argparse.ArgumentParser(description="Read NPZ file")
    parser.add_argument("npz_file", help="NPZ file path")

    args = parser.parse_args()

    try:
        read_npz_file(args.npz_file)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
