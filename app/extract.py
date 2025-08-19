"""
step1a_extract.py

Purpose:
  Extract per-page content from PDFs:
    - Text (PyMuPDF)
    - Tables (pdfplumber) -> save as Markdown + CSV files
    - OCR fallback for image-heavy pages

Outputs:
  artifacts/raw_{docid}.jsonl  # one JSON object per page
  artifacts/tables/{docid}_p{page}_{i}.csv  # each detected table as CSV
  artifacts/tables/{docid}_p{page}_{i}.md   # same table as Markdown
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any

import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
from PIL import Image
import pytesseract

# --------- light cleaners ----------
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


def extract_text_blocks(page: fitz.Page) -> str:
    """Text in reading order from PyMuPDF."""
    blocks = page.get_text("blocks", flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE)
    blocks = sorted(blocks, key=lambda b: (int(b[1]), int(b[0])))  # y, x
    out = []
    for (x0, y0, x1, y1, txt, *_rest) in blocks:
        if txt and not txt.isspace():
            out.append(txt)
    return "\n".join(out)


def extract_tables(pl_page) -> List[pd.DataFrame]:
    """Return a list of DataFrames for tables detected by pdfplumber."""
    dfs = []
    try:
        tables = pl_page.find_tables()
        for t in tables:
            df = pd.DataFrame(t.extract())
            if df.empty:
                continue
            # Try to promote first row to header if it looks like headerish
            if all(isinstance(x, str) for x in df.iloc[0].tolist()):
                df.columns = df.iloc[0]
                df = df[1:]
            dfs.append(df.reset_index(drop=True))
    except Exception:
        pass
    return dfs


def ocr_page(page: fitz.Page, dpi=300) -> str:
    """OCR image of the page if text extraction is poor."""
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72), alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    try:
        text = pytesseract.image_to_string(img)
    except Exception:
        text = ""
    return text


def process_pdf(pdf_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    raw_out = out_dir / f"raw_{pdf_path.stem}.jsonl"
    print(f"[EXTRACT] {pdf_path.name} → {raw_out.name}")

    with pdfplumber.open(pdf_path) as pl, open(raw_out, "w", encoding="utf-8") as fout:
        for pno in range(len(doc)):
            page = doc[pno]
            txt = clean_text(extract_text_blocks(page))

            # tables
            md_tables = []
            csv_paths = []
            try:
                pl_page = pl.pages[pno]
                dfs = extract_tables(pl_page)
                for i, df in enumerate(dfs):
                    csv_path = tables_dir / f"{pdf_path.stem}_p{pno+1}_{i+1}.csv"
                    md_path = tables_dir / f"{pdf_path.stem}_p{pno+1}_{i+1}.md"
                    df.to_csv(csv_path, index=False)
                    md_tables.append(df.to_markdown(index=False))
                    md_path.write_text(df.to_markdown(index=False), encoding="utf-8")
                    csv_paths.append(str(csv_path))
            except Exception:
                pass

            # OCR fallback
            do_ocr = (len(txt) < 40) and (len(page.get_images(full=True)) > 0)
            ocr_txt = clean_text(ocr_page(page)) if do_ocr else ""

            record: Dict[str, Any] = {
                "page": pno + 1,
                "text": txt,
                "ocr_text": ocr_txt,
                "tables_markdown": md_tables,
                "tables_csv": csv_paths,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[OK] {raw_out}  (pages: {len(doc)})")


def main():
    ap = argparse.ArgumentParser(description="Step 1A: Extract text/tables/OCR per page.")
    ap.add_argument("--pdf_dir", type=str, default="./pdf")
    ap.add_argument("--out_dir", type=str, default="./artifacts")
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs in {pdf_dir.resolve()}")
        return

    out_dir = Path(args.out_dir)
    for p in pdfs:
        process_pdf(p, out_dir)


if __name__ == "__main__":
    main()
