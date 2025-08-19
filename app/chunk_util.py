"""
step2_chunk.py — Step 2.1: Split cleaned sections into retrieval chunks.

What this does
--------------
1) Loads artifacts/sections_*.json produced by segment.py.
2) Tokenizes each section (HF tokenizer) and creates two granularities of chunks:
     - size=100 tokens (with 20-token overlap)
     - size=400 tokens (with 40-token overlap)
3) For every chunk, saves:
     - id: globally unique id
     - text: chunk text
     - meta: {doc_id, section, pages, h2_subheads, chunk_size, idx_in_section, start_token, end_token}
4) Emits JSONL files:
     - artifacts/chunks_100.jsonl
     - artifacts/chunks_400.jsonl
   and a small summary manifest: artifacts/chunks_manifest.json

Notes
-----
- We use the tokenizer from 'sentence-transformers/all-MiniLM-L6-v2' to approximate retrieval-token lengths.
- Overlap ensures queries at boundaries still get matched.
- TABLES are preserved as "[TABLE] ... [/TABLE]" blocks inside the text.

Run
---
python app/step2_chunk.py --artifacts_dir ./artifacts
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from transformers import AutoTokenizer
from tqdm import tqdm

# ---------- Tunables ----------
CHUNK_SIZES = [100, 400]    # two granularities
OVERLAP = {100: 20, 400: 40}

MODEL_FOR_TOKENIZER = "sentence-transformers/all-MiniLM-L6-v2"


def load_sections(artifacts_dir: Path) -> List[Dict]:
    sections = []
    for jf in sorted(artifacts_dir.glob("sections_*.json")):
        with open(jf, "r", encoding="utf-8") as f:
            sections.extend(json.load(f))
    return sections


def sliding_windows(tokens: List[int], size: int, overlap: int) -> List[Tuple[int, int]]:
    """Yield (start, end) token indices for a sliding window with overlap."""
    step = max(1, size - overlap)
    i = 0
    N = len(tokens)
    out = []
    while i < N:
        j = min(N, i + size)
        out.append((i, j))
        if j == N:
            break
        i += step
    return out


def tokens_to_text(tokenizer, input_ids: List[int], a: int, b: int) -> str:
    """Decode a span of token ids to text (strip special tokens)."""
    piece = tokenizer.decode(input_ids[a:b], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return piece.strip()


def chunk_sections(sections: List[Dict], artifacts_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_FOR_TOKENIZER)

    manifests = []
    writers: Dict[int, any] = {}  # size -> file handle
    for size in CHUNK_SIZES:
        writers[size] = open(artifacts_dir / f"chunks_{size}.jsonl", "w", encoding="utf-8")

    gid = 0  # global id counter

    for sec in tqdm(sections, desc="Chunking sections"):
        doc_id = sec.get("doc_id")
        section_name = sec.get("name")
        start_page = sec.get("start_page", 0) + 1  # store as 1-based for humans
        end_page = sec.get("end_page", 0) + 1
        h2s = sec.get("h2_subheads", []) or []

        text = sec.get("merged_text", "") or ""
        if not text.strip():
            continue

        # Tokenize once; reuse for both granularities
        enc = tokenizer(text, add_special_tokens=False, return_attention_mask=False, return_offsets_mapping=False)
        ids = enc["input_ids"]
        if not ids:
            continue

        for size in CHUNK_SIZES:
            overlap = OVERLAP[size]
            for idx_in_section, (a, b) in enumerate(sliding_windows(ids, size, overlap)):
                gid += 1
                chunk_text = tokens_to_text(tokenizer, ids, a, b)
                if len(chunk_text) < 20:
                    continue

                chunk = {
                    "id": f"{doc_id}::{section_name}::sz{size}::#{idx_in_section}",
                    "text": chunk_text,
                    "meta": {
                        "doc_id": doc_id,
                        "section": section_name,
                        "pages": [start_page, end_page],
                        "h2_subheads": h2s[:10],   # keep a few
                        "chunk_size": size,
                        "idx_in_section": idx_in_section,
                        "start_token": a,
                        "end_token": b,
                    }
                }
                writers[size].write(json.dumps(chunk, ensure_ascii=False) + "\n")

        # simple manifest row per section
        manifests.append({
            "doc_id": doc_id,
            "section": section_name,
            "pages": [start_page, end_page],
            "chars": len(text),
            "tokens": len(ids),
        })

    for fh in writers.values():
        fh.close()

    with open(artifacts_dir / "chunks_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifests, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Step 2.1 — Create retrieval chunks from sections.")
    ap.add_argument("--artifacts_dir", type=str, default="./artifacts")
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    sections = load_sections(artifacts_dir)
    if not sections:
        print("No sections found. Run segment.py first.")
        return

    chunk_sections(sections, artifacts_dir)
    print("[OK] wrote chunks_100.jsonl, chunks_400.jsonl, and chunks_manifest.json")


if __name__ == "__main__":
    main()
