"""
segment.py — Heading-level segmentation (H1/H2) with font-size hierarchy.

What it does
------------
1) Scans a PDF with PyMuPDF and collects text spans with their font sizes.
2) Clusters font sizes to infer heading levels:
     - Largest cluster  => H1 (top sections)
     - Second largest   => H2 (subsections, kept as metadata)
3) Builds H1-anchored sections [start_page, end_page], attaches page content from step1a.
4) Cleans and merges text; preserves tables as [TABLE] ... [/TABLE].

Inputs
------
artifacts/raw_{docid}.jsonl   (from extract.py / step1a)

Outputs
-------
artifacts/sections_{docid}.json
artifacts/sections_{docid}.txt

Run
---
python app/segment.py --pdf_dir ./pdf --artifacts_dir ./artifacts
"""

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import fitz  # PyMuPDF
from sklearn.cluster import KMeans

# ---------------- Tunables ----------------
MIN_HEADING_LEN = 3              # ignore very tiny “titles”
MAX_HEADING_LEN = 120
H1_TOP_PERCENTILE = 0.90         # fallback if clustering fails (take top 10% font sizes as H1)
H2_TOP_PERCENTILE = 0.75         # fallback H2 threshold
ALLOW_DUP_H1_PER_PAGE = False    # usually 1-2 strong headings per page is enough

# -------------- Cleaning ------------------
HDRFOOT = re.compile(
    r"(Table of Contents|PayPal Holdings,? Inc\.?|©.*?PayPal.*|Forward-Looking Statements)",
    re.I,
)
ONLY_PAGENUM = re.compile(r"^\s*\d+\s*$")
WS = re.compile(r"\s+")

def clean_text(s: str) -> str:
    s = HDRFOOT.sub(" ", s or "")
    lines = [ln for ln in s.splitlines() if not ONLY_PAGENUM.match(ln.strip())]
    s = "\n".join(lines)
    s = s.replace("\x00", " ")
    s = WS.sub(" ", s).strip()
    return s

# -------------- Data classes --------------
@dataclass
class ContentItem:
    type: str   # text | table | ocr
    page: int
    text: str

@dataclass
class Section:
    doc_id: str
    name: str
    start_page: int
    end_page: int
    items: List[ContentItem]
    h2_subheads: List[str]  # optional H2s we saw inside this section

    def merged_text(self) -> str:
        parts = []
        for it in self.items:
            if it.type == "table":
                parts.append("\n\n[TABLE]\n" + it.text + "\n[/TABLE]\n")
            else:
                parts.append(it.text)
        merged = "\n".join([p for p in parts if p.strip()])
        return clean_text(merged)

# -------------- Heading detection ----------
def collect_spans(doc: fitz.Document):
    """Return per-page list of (text_line, max_font_size) and keep raw spans to rebuild lines."""
    per_page = []
    all_sizes = []
    for pno in range(len(doc)):
        d = doc[pno].get_text("dict")
        page_lines = []
        for b in d.get("blocks", []):
            for l in b.get("lines", []):
                texts = [s.get("text", "") for s in l.get("spans", [])]
                if not texts:
                    continue
                line_txt = "".join(texts).strip()
                if not line_txt:
                    continue
                max_size = max((s.get("size", 0) for s in l.get("spans", [])), default=0.0)
                page_lines.append((line_txt, max_size))
                all_sizes.append(max_size)
        per_page.append(page_lines)
    return per_page, np.array(all_sizes, dtype=np.float32) if all_sizes else np.array([])

def choose_h1_h2_thresholds(font_sizes: np.ndarray) -> Tuple[float, float]:
    """
    Try KMeans (k=3) to separate big/medium/small fonts.
    Return (h1_threshold, h2_threshold) as absolute font sizes.
    Fallback to percentiles if clustering fails.
    """
    if font_sizes.size == 0:
        return (float("inf"), float("inf"))
    try:
        X = font_sizes.reshape(-1, 1)
        km = KMeans(n_clusters=3, n_init="auto", random_state=42).fit(X)
        centers = sorted([c[0] for c in km.cluster_centers_])  # small < mid < large
        small, mid, large = centers
        # slightly below centers to be inclusive
        h1_thr = (mid + large) / 2.0     # anything >= this => H1
        h2_thr = (small + mid) / 2.0     # anything >= this and < h1_thr => H2
        return (h1_thr, h2_thr)
    except Exception:
        # Fallback percentiles
        h1_thr = np.quantile(font_sizes, H1_TOP_PERCENTILE)
        h2_thr = np.quantile(font_sizes, H2_TOP_PERCENTILE)
        return (h1_thr, h2_thr)

def detect_headings(per_page_lines, h1_thr: float, h2_thr: float):
    """
    Label lines as H1/H2/text based on font size thresholds.
    Returns:
      h1_marks: list of (page_index, heading_text)
      per_page_h2: dict page_index -> [subheads]
    """
    h1_marks = []
    per_page_h2 = {}
    for pidx, lines in enumerate(per_page_lines):
        page_h2 = []
        h1_count = 0
        for (txt, fsz) in lines:
            # ignore tiny or overly long lines
            if len(txt) < MIN_HEADING_LEN or len(txt) > MAX_HEADING_LEN:
                continue
            # Must contain letters (avoid lines that are numbers/totals)
            if not re.search(r"[A-Za-z]", txt):
                continue
            t = clean_text(txt)
            if fsz >= h1_thr:
                if not ALLOW_DUP_H1_PER_PAGE and h1_count >= 1:
                    # If we already picked one H1 on this page, treat others as H2
                    page_h2.append(t)
                else:
                    h1_marks.append((pidx, t))
                    h1_count += 1
            elif fsz >= h2_thr:
                page_h2.append(t)
        if page_h2:
            per_page_h2[pidx] = page_h2
    return h1_marks, per_page_h2

def make_h1_ranges(h1_marks: List[Tuple[int, str]], num_pages: int) -> List[Tuple[str, int, int]]:
    """
    Convert H1 marks into [(title, start_page, end_page)].
    Pages are 0-based here.
    """
    if not h1_marks:
        return [("Document", 0, num_pages-1)]
    # Deduplicate consecutive repeats of same title
    dedup = []
    last = None
    for pidx, title in h1_marks:
        if title != last:
            dedup.append((pidx, title))
            last = title
    ranges = []
    for i, (p, title) in enumerate(dedup):
        start = p
        end = (dedup[i+1][0] - 1) if i+1 < len(dedup) else (num_pages - 1)
        ranges.append((title.strip(), start, end))
    return ranges

# -------------- Core segmentation ----------
def segment_pdf(pdf_path: Path, raw_path: Path, out_dir: Path):
    print(f"[SEGMENT] {pdf_path.name}")
    doc = fitz.open(pdf_path)

    # 1) Collect lines + font sizes across the whole doc
    per_page_lines, all_sizes = collect_spans(doc)
    h1_thr, h2_thr = choose_h1_h2_thresholds(all_sizes)
    # 2) Detect headings per page
    h1_marks, per_page_h2 = detect_headings(per_page_lines, h1_thr, h2_thr)
    # 3) Build H1 ranges
    ranges = make_h1_ranges(h1_marks, len(doc))

    # Prepare empty sections
    sections: List[Section] = [
        Section(doc_id=pdf_path.stem, name=title, start_page=sp, end_page=ep, items=[], h2_subheads=[])
        for (title, sp, ep) in ranges
    ]

    # Load raw page content from step1a
    page_items: Dict[int, Dict[str, Any]] = {}
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            page_items[rec["page"]] = rec  # 1-based pages

    # Attach content to sections and collect H2s
    for pno in range(1, len(doc)+1):
        rec = page_items.get(pno, {})
        items = []
        if rec.get("text"):
            items.append(ContentItem(type="text", page=pno, text=clean_text(rec["text"])))
        for md in rec.get("tables_markdown", []):
            items.append(ContentItem(type="table", page=pno, text=md))
        if rec.get("ocr_text"):
            items.append(ContentItem(type="ocr", page=pno, text=clean_text(rec["ocr_text"])))

        for sec in sections:
            if sec.start_page <= (pno - 1) <= sec.end_page:
                sec.items.extend(items)
                # store any H2 on this page under the section
                if (pno - 1) in per_page_h2:
                    for h2 in per_page_h2[pno - 1]:
                        if h2 not in sec.h2_subheads:
                            sec.h2_subheads.append(h2)
                break

    # Emit files
    out_json = []
    out_txt_lines = []
    for sec in sections:
        merged = sec.merged_text()
        payload = {**asdict(sec), "merged_text": merged}
        out_json.append(payload)
        out_txt_lines.append("# " + sec.name)
        if sec.h2_subheads:
            out_txt_lines.append("## Subheads: " + " | ".join(sec.h2_subheads[:8]))
        out_txt_lines.append(merged)
        out_txt_lines.append("")

    out_jsonf = out_dir / f"sections_{pdf_path.stem}.json"
    out_txtf = out_dir / f"sections_{pdf_path.stem}.txt"
    with open(out_jsonf, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)
    out_txtf.write_text("\n".join(out_txt_lines), encoding="utf-8")
    print(f"[OK] wrote {out_jsonf.name} + {out_txtf.name}  (sections: {len(sections)})")

# -------------- CLI ------------------------
def main():
    ap = argparse.ArgumentParser(description="Step 1B (heading-based): Segment PDFs into sections.")
    ap.add_argument("--pdf_dir", type=str, default="./pdf")
    ap.add_argument("--artifacts_dir", type=str, default="./artifacts")
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    art = Path(args.artifacts_dir)

    for pdf in sorted(pdf_dir.glob("*.pdf")):
        raw = art / f"raw_{pdf.stem}.jsonl"
        if not raw.exists():
            print(f"Missing {raw.name} (Run extract.py first). Skipping {pdf.name}")
            continue
        segment_pdf(pdf, raw, art)

if __name__ == "__main__":
    main()
