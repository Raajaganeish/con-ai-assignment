"""
step2_index.py — Step 2.2: Build Dense (FAISS) and Sparse (BM25) indexes from chunks.

What this does
--------------
1) Loads chunk JSONL files created in step2_chunk.py (chunks_100.jsonl and/or chunks_400.jsonl).
2) Embeds chunk texts using sentence-transformers (default: all-MiniLM-L6-v2).
3) Builds:
   - Dense FAISS index (Inner Product) with normalized vectors.
   - Sparse BM25 index using rank-bm25 on tokenized text.
4) Saves artifacts under ./artifacts:
   - faiss_index_sz{size}.index
   - faiss_ids_sz{size}.jsonl       (maps row -> chunk id + metadata)
   - bm25_sz{size}.pkl              (BM25 object)
   - bm25_ids_sz{size}.jsonl        (same id mapping as above for sparse)
   - embed_config_sz{size}.json     (model + params for provenance)

Run
---
# Build indexes for both granularities
python app/step2_index.py --artifacts_dir ./artifacts --sizes 100 400

# Or just one size
python app/step2_index.py --artifacts_dir ./artifacts --sizes 400

Notes
-----
- Everything is open-source; no external APIs.
- We normalize embeddings so FAISS IndexFlatIP is equivalent to cosine similarity.
"""

import argparse
import json
import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Iterable

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------- Config ----------
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
BATCH_SIZE = int(os.environ.get("EMBED_BATCH", "128"))

TOKEN_SPLIT = re.compile(r"[A-Za-z0-9]+")  # simple alnum tokenizer for BM25


# ---------- IO helpers ----------
def read_jsonl(path: Path) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def tokenize_for_bm25(text: str) -> List[str]:
    # Lowercase, simple alnum tokens
    return TOKEN_SPLIT.findall(text.lower())


# ---------- Dense index ----------
def build_faiss(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS IndexFlatIP. Assumes embeddings are L2-normalized.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index


def embed_texts(texts: List[str], model_name: str, batch_size: int) -> np.ndarray:
    model = SentenceTransformer(model_name)
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine-ready
    )
    return np.asarray(vecs, dtype=np.float32)


# ---------- Sparse index ----------
def build_bm25(docs_tokenized: List[List[str]]) -> BM25Okapi:
    return BM25Okapi(docs_tokenized)


# ---------- Main pipeline ----------
def process_size(size: int, artifacts_dir: Path):
    chunks_path = artifacts_dir / f"chunks_{size}.jsonl"
    if not chunks_path.exists():
        print(f"[WARN] {chunks_path.name} not found. Skipping size={size}.")
        return

    print(f"\n[INDEX] size={size} → reading chunks ...")
    ids = []
    texts = []
    metas = []

    for row in read_jsonl(chunks_path):
        ids.append(row["id"])
        texts.append(row["text"])
        metas.append(row["meta"])

    n = len(texts)
    if n == 0:
        print(f"[WARN] No chunks in {chunks_path.name}.")
        return

    # Dense embeddings
    print(f"[DENSE] Embedding {n} chunks with '{EMBED_MODEL}' (batch={BATCH_SIZE}) ...")
    X = embed_texts(texts, EMBED_MODEL, BATCH_SIZE)  # shape: (n, d), L2-normalized
    print(f"[DENSE] Vectors: {X.shape}")

    print("[DENSE] Building FAISS IndexFlatIP ...")
    faiss_index = build_faiss(X)

    # Save FAISS + ID mapping
    faiss_path = artifacts_dir / f"faiss_index_sz{size}.index"
    faiss.write_index(faiss_index, str(faiss_path))
    write_jsonl(artifacts_dir / f"faiss_ids_sz{size}.jsonl",
                ({"row": i, "id": ids[i], "meta": metas[i]} for i in range(n)))
    print(f"[OK] Saved {faiss_path.name} and faiss_ids_sz{size}.jsonl")

    # Sparse BM25
    print(f"[SPARSE] Tokenizing {n} chunks for BM25 ...")
    docs_tok = [tokenize_for_bm25(t) for t in tqdm(texts, total=n)]
    bm25 = build_bm25(docs_tok)
    with open(artifacts_dir / f"bm25_sz{size}.pkl", "wb") as f:
        pickle.dump(bm25, f)
    write_jsonl(artifacts_dir / f"bm25_ids_sz{size}.jsonl",
                ({"row": i, "id": ids[i], "meta": metas[i]} for i in range(n)))
    print(f"[OK] Saved bm25_sz{size}.pkl and bm25_ids_sz{size}.jsonl")

    # Config/provenance
    cfg = {
        "embed_model": EMBED_MODEL,
        "batch_size": BATCH_SIZE,
        "num_chunks": n,
        "size": size,
    }
    with open(artifacts_dir / f"embed_config_sz{size}.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved embed_config_sz{size}.json")


def main():
    ap = argparse.ArgumentParser(description="Step 2.2 — Build Dense (FAISS) and Sparse (BM25) indexes.")
    ap.add_argument("--artifacts_dir", type=str, default="./artifacts")
    ap.add_argument("--sizes", type=int, nargs="+", default=[100, 400],
                    help="Chunk sizes to index (default: 100 400)")
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for size in args.sizes:
        process_size(size, artifacts_dir)


if __name__ == "__main__":
    main()
