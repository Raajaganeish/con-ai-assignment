"""
retrieve.py — Hybrid Retrieval + Multi-Stage Re-ranking (tunable + filters)

Usage example
-------------
python app/retrieve.py --artifacts_dir ./artifacts --size 100 \
  --query "What was PayPal's revenue in 2023?" \
  --topk_dense 120 --topk_sparse 300 --w_dense 0.35 --w_sparse 0.65 \
  --prefer_tables --table_boost 1.6 --final_topk 8 \
  --must_terms revenues,2023,$ \
  --include_section_regex "consolidated\\s+statements?.*operations|results of operations|financial|sound financial and operational performance|item\\s*8" \
  --doc_year 2023 \
  --hard_regex "(net\\s*revenues?|total\\s*net\\s*revenues?|revenues?).*(2023)|2023.*(net\\s*revenues?|revenues?)" \
  --hard_drop_if_no_match
"""

import argparse
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

ALNUM = re.compile(r"[A-Za-z0-9]+", re.I)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ---------------- Data structures ----------------
@dataclass
class Hit:
    id: str
    text: str
    score: float
    meta: Dict


# ---------------- I/O helpers ----------------
def read_jsonl(path: Path) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_faiss_and_ids(artifacts: Path, size: int):
    index = faiss.read_index(str(artifacts / f"faiss_index_sz{size}.index"))
    idmap = list(read_jsonl(artifacts / f"faiss_ids_sz{size}.jsonl"))
    return index, idmap


def load_bm25_and_ids(artifacts: Path, size: int):
    with open(artifacts / f"bm25_sz{size}.pkl", "rb") as f:
        bm25: BM25Okapi = pickle.load(f)
    idmap = list(read_jsonl(artifacts / f"bm25_ids_sz{size}.jsonl"))
    return bm25, idmap


def load_chunks_texts(artifacts: Path, size: int) -> List[str]:
    path = artifacts / f"chunks_{size}.jsonl"
    return [row["text"] for row in read_jsonl(path)]


# ---------------- Preprocess ----------------
def preprocess_query(q: str):
    q_clean = q.strip().lower()
    tokens = ALNUM.findall(q_clean)

    # Lightweight query expansion to help BM25 on financials
    expanded = list(tokens)
    if "revenue" in q_clean or "revenues" in q_clean:
        expanded += ["revenues", "net", "total", "statement", "operations", "consolidated"]
    years = re.findall(r"\b(20\d{2})\b", q_clean)
    for y in years:
        expanded += [y, y]  # duplicate for weighting

    return q_clean, expanded


# ---------------- Retrieval primitives ----------------
def dense_search(q: str, model: SentenceTransformer, index, topk: int):
    q_vec = model.encode([q], normalize_embeddings=True)[0].astype(np.float32)
    D, I = index.search(np.expand_dims(q_vec, 0), topk)
    return [(int(I[0][i]), float(D[0][i])) for i in range(len(I[0])) if int(I[0][i]) != -1]


def sparse_search(q_tokens: List[str], bm25: BM25Okapi, topk: int):
    scores = bm25.get_scores(q_tokens)
    idx = np.argsort(scores)[::-1][:topk]
    return [(int(i), float(scores[i])) for i in idx if scores[i] > 0.0]


def weighted_fusion(dense_hits, sparse_hits, w_dense=0.6, w_sparse=0.4) -> Dict[int, float]:
    scores = {}

    def add(hits, w):
        if not hits:
            return
        vals = np.array([s for _, s in hits], dtype=np.float32)
        if len(vals) > 1:
            vmin, vmax = float(vals.min()), float(vals.max())
            normed = (vals - vmin) / (vmax - vmin) if vmax > vmin else np.ones_like(vals)
        else:
            normed = np.ones_like(vals)
        for (i, _), sv in zip(hits, normed):
            scores[i] = scores.get(i, 0.0) + w * float(sv)

    add(dense_hits, w_dense)
    add(sparse_hits, w_sparse)
    return scores  # row_index -> fused_score


def is_numeric_intent(q: str) -> bool:
    if re.search(r"\b(20\d{2})\b", q):
        return True
    if re.search(r"\b(revenue|revenues|income|sales|assets|liabilities|cash\s+flow|eps|margin|operating|net|total|growth|percent|%)\b", q):
        return True
    if re.search(r"\d", q) or "$" in q:
        return True
    return False


def any_term_present(text: str, terms: List[str]) -> bool:
    t = text.lower()
    return all(term.lower() in t for term in terms)


# ---------------- Orchestrator ----------------
class HybridReranker:
    def __init__(self, artifacts_dir: Path, size: int):
        self.size = size
        self.artifacts = artifacts_dir
        self.faiss_index, self.faiss_ids = load_faiss_and_ids(artifacts_dir, size)
        self.bm25, self.bm25_ids = load_bm25_and_ids(artifacts_dir, size)
        self.texts = load_chunks_texts(artifacts_dir, size)
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

    def query(
        self,
        q: str,
        topk_dense: int,
        topk_sparse: int,
        w_dense: float,
        w_sparse: float,
        final_topk: int,
        prefer_tables: bool,
        table_boost: float,
        must_terms: List[str],
        include_section_regex: str,
        doc_year: Optional[int],
        hard_regex: str,
        hard_drop_if_no_match: bool,
    ):
        q_clean, q_tokens = preprocess_query(q)

        # Stage A — broad retrieval (hybrid)
        dense_hits = dense_search(q_clean, self.embedder, self.faiss_index, topk_dense)
        sparse_hits = sparse_search(q_tokens, self.bm25, topk_sparse)
        fused = weighted_fusion(dense_hits, sparse_hits, w_dense, w_sparse)
        if not fused:
            return []

        # Pre-filter & bias BEFORE cross-encoder
        numeric_intent = is_numeric_intent(q_clean)
        sec_rx = re.compile(include_section_regex, re.I) if include_section_regex else None
        hard_rx = re.compile(hard_regex, re.I) if hard_regex else None

        cands = []
        for row_idx, fused_score in fused.items():
            text = self.texts[row_idx]
            idrec = self.faiss_ids[row_idx]
            meta = idrec["meta"]
            score = fused_score

            # must-have terms (drop if missing)
            if must_terms and not any_term_present(text, must_terms):
                continue

            # hard regex filtering/boost
            match = True
            if hard_rx:
                match = bool(hard_rx.search(text))
                if hard_drop_if_no_match and not match:
                    continue
                if match:
                    score *= 1.25  # light boost

            # section regex boost
            if sec_rx and meta.get("section") and sec_rx.search(str(meta["section"])):
                score *= 1.10

            # doc-year preference (doc_id contains year)
            if doc_year and meta.get("doc_id") and str(doc_year) in str(meta["doc_id"]):
                score *= 1.05

            # table bias for numeric intent
            if prefer_tables and numeric_intent and "[TABLE]" in text:
                score *= float(table_boost)

            cands.append((row_idx, score, idrec["id"], meta, text))

        if not cands:
            return []

        # Sort by boosted fused score; then Stage B — cross-encoder re-rank
        cands.sort(key=lambda t: t[1], reverse=True)
        pairs = [(q_clean, t[4]) for t in cands]
        ce_scores = self.cross_encoder.predict(pairs).tolist() if pairs else []

        # normalize CE scores; late fuse 50/50
        ce_arr = np.array(ce_scores, dtype=np.float32)
        if len(ce_arr) > 1:
            m, M = float(ce_arr.min()), float(ce_arr.max())
            ce_norm = ((ce_arr - m) / (M - m)) if M > m else np.ones_like(ce_arr)
        else:
            ce_norm = np.ones_like(ce_arr)

        hits: List[Hit] = []
        for (row_idx, fused_score, cid, meta, text), ce_s in zip(cands, ce_norm):
            final = 0.5 * fused_score + 0.5 * float(ce_s)
            hits.append(Hit(id=cid, text=text, score=final, meta=meta))

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:final_topk]


# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Hybrid + Multi-Stage Retrieval (tunable + filters)")
    ap.add_argument("--artifacts_dir", type=str, default="./artifacts")
    ap.add_argument("--size", type=int, default=400, help="Chunk size to query (100 or 400)")
    ap.add_argument("--query", type=str, required=True)
    ap.add_argument("--final_topk", type=int, default=6)

    # Stage-A knobs
    ap.add_argument("--topk_dense", type=int, default=20)
    ap.add_argument("--topk_sparse", type=int, default=20)
    ap.add_argument("--w_dense", type=float, default=0.6)
    ap.add_argument("--w_sparse", type=float, default=0.4)

    # Preferences
    ap.add_argument("--prefer_tables", action="store_true", default=True)
    ap.add_argument("--table_boost", type=float, default=1.25)

    # Filters
    ap.add_argument("--must_terms", type=str, default="", help="Comma-separated terms required in chunk text")
    ap.add_argument("--include_section_regex", type=str, default="", help="Regex to prefer certain sections")
    ap.add_argument("--doc_year", type=int, default=None)

    # Hard regex filter/boost
    ap.add_argument("--hard_regex", type=str, default="",
                    help="If set, boost candidates that match; optionally drop non-matches.")
    ap.add_argument("--hard_drop_if_no_match", action="store_true", default=False,
                    help="Drop candidates that do not match --hard_regex.")

    args = ap.parse_args()
    must_terms = [t.strip() for t in args.must_terms.split(",") if t.strip()]

    hr = HybridReranker(Path(args.artifacts_dir), size=args.size)
    hits = hr.query(
        q=args.query,
        topk_dense=args.topk_dense,
        topk_sparse=args.topk_sparse,
        w_dense=args.w_dense,
        w_sparse=args.w_sparse,
        final_topk=args.final_topk,
        prefer_tables=args.prefer_tables,
        table_boost=args.table_boost,
        must_terms=must_terms,
        include_section_regex=args.include_section_regex,
        doc_year=args.doc_year,
        hard_regex=args.hard_regex,
        hard_drop_if_no_match=args.hard_drop_if_no_match,
    )

    print("\n=== RESULTS ===")
    for i, h in enumerate(hits, 1):
        meta = h.meta
        print(f"[{i}] score={h.score:.3f}  id={h.id}")
        print(f"    doc={meta.get('doc_id')}  section={meta.get('section')}  pages={meta.get('pages')}  sz={meta.get('chunk_size')}  idx={meta.get('idx_in_section')}")
        snippet = (h.text[:300] + '…') if len(h.text) > 300 else h.text
        print("    text:", snippet.replace("\n", " "))
        print()

if __name__ == "__main__":
    main()
