"""
build_qa.py  — Step 1C: Construct >=50 question–answer pairs from sectioned content.

Overview
--------
1) Load section JSONs produced by segment.py. Each section already merges:
     - page text (cleaned)
     - table markdown blocks
     - OCR (if any)
2) Load tables from artifacts/tables/*.csv for reliable numeric extraction.
3) Identify finance metrics by keyword patterns (Revenue, Net income, Assets, etc.).
4) Extract values + optional year context and form templated Q/A pairs with provenance.
5) Deduplicate and save to JSONL + CSV.

Run
---
python app/build_qa.py --artifacts_dir ./artifacts

Outputs
-------
artifacts/qa_pairs.jsonl
artifacts/qa_pairs.csv
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

# ---------- Metric patterns (extendable) ----------
METRIC_PATTERNS = {
    "Revenue": r"(total\s+)?(net\s+)?revenue[s]?",
    "Operating income": r"\boperating\s+income\b|\bincome\s+from\s+operations\b",
    "Operating margin": r"\boperating\s+margin\b",
    "Gross profit": r"\bgross\s+profit\b",
    "Gross margin": r"\bgross\s+margin\b",
    "Net income": r"\bnet\s+(income|earnings|loss)\b",
    "Diluted EPS": r"\bdiluted\s+(earnings\s+per\s+share|eps)\b",
    "Basic EPS": r"\bbasic\s+(earnings\s+per\s+share|eps)\b",
    "Total assets": r"\btotal\s+assets\b",
    "Total liabilities": r"\btotal\s+liabilities\b",
    "Total equity": r"\btotal\s+(stockholders'?|shareholders'?)\s+equity\b",
    "Cash and cash equivalents": r"\bcash\s+and\s+cash\s+equivalents\b",
    "Operating cash flow": r"\bnet\s+cash\s+provided\s+by\s+operating\s+activities\b|\bcash\s+flows?\s+from\s+operating\b",
    "Investing cash flow": r"\bnet\s+cash\s+used\s+in\s+investing\s+activities\b|\bcash\s+flows?\s+from\s+investing\b",
    "Financing cash flow": r"\bnet\s+cash\s+provided\s+by\s+\(used\s+in\)\s+financing\s+activities\b|\bcash\s+flows?\s+from\s+financing\b",
    "Free cash flow": r"\bfree\s+cash\s+flow\b",
    "R&D expense": r"\bresearch\s+and\s+development\b",
    "Sales & marketing expense": r"\bsales\s+and\s+marketing\b",
    "G&A expense": r"\bgeneral\s+and\s+administrative\b",
    "Transaction expense": r"\btransaction\s+expense\b",
    "Transaction revenue": r"\btransaction\s+revenue\b",
}

YEAR_PAT = re.compile(r"\b(20\d{2})\b")
# pick up $ 4,123, 4.1 billion, 2.3 bn, etc.
AMOUNT_PAT = re.compile(
    r"(\$?\s?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*(?:billion|million|thousand|bn|m|k)?|\$?\s?\d+(?:\.\d+)?)",
    re.I,
)

def normalize_amount(raw: str) -> str:
    """Normalize whitespace; keep original units/format to stay faithful to source."""
    return re.sub(r"\s+", " ", str(raw)).strip()

def load_sections(artifacts_dir: Path) -> List[Dict]:
    sections = []
    for jf in sorted(artifacts_dir.glob("sections_*.json")):
        with open(jf, "r", encoding="utf-8") as f:
            sections.extend(json.load(f))
    return sections

def mine_pairs_from_text(sections: List[Dict]) -> List[Dict]:
    pairs = []
    for sec in sections:
        merged = sec.get("merged_text", "") or ""
        # Split into sentences/lines for simpler matching
        for ln in [x.strip() for x in merged.split("\n") if x.strip()]:
            for metric, pat in METRIC_PATTERNS.items():
                if re.search(pat, ln, re.I):
                    # pull multiple numbers if present (common in multi-year lines)
                    amounts = AMOUNT_PAT.findall(ln)
                    if not amounts:
                        continue
                    years = YEAR_PAT.findall(ln)
                    for amt in amounts:
                        ans = normalize_amount(amt if isinstance(amt, str) else amt[0])
                        y = years[0] if years else None
                        q = f"What was PayPal's {metric}" + (f" in {y}?" if y else "?")
                        a = f"In the section '{sec['name']}', {metric}" + (f" in {y}" if y else "") + f" was {ans}."
                        pairs.append({
                            "question": q,
                            "answer": a,
                            "metric": metric,
                            "year": y,
                            "doc_id": sec.get("doc_id"),
                            "section": sec.get("name"),
                        })
                    break
    return pairs

def mine_pairs_from_tables(tables_dir: Path) -> List[Dict]:
    pairs = []
    for csv in sorted(tables_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        if df.empty:
            continue
        # Try to detect year columns
        year_cols = []
        for c in df.columns:
            m = YEAR_PAT.search(str(c))
            if m:
                year_cols.append((c, m.group(1)))
        # Heuristic: first column is label
        label_col = df.columns[0]
        for _, row in df.iterrows():
            label = str(row.get(label_col, "")).strip()
            for metric, pat in METRIC_PATTERNS.items():
                if re.search(pat, label, re.I):
                    if year_cols:
                        for col, year in year_cols:
                            raw = str(row.get(col, "")).strip()
                            if not raw or raw.lower() == "nan":
                                continue
                            if not AMOUNT_PAT.search(raw) and not re.search(r"\d", raw):
                                continue
                            q = f"What was PayPal's {metric} in {year}?"
                            a = f"From a financial table, {metric} in {year} was {normalize_amount(raw)}."
                            pairs.append({
                                "question": q,
                                "answer": a,
                                "metric": metric,
                                "year": year,
                                "table_file": csv.name,
                            })
                    else:
                        # No year columns — still capture a generic metric value
                        raw_values = [v for v in row.tolist()[1:] if pd.notna(v)]
                        for val in raw_values[:2]:  # limit
                            if not re.search(r"\d", str(val)):
                                continue
                            q = f"What was PayPal's {metric} (value reported)?"
                            a = f"From a financial table, {metric} was {normalize_amount(val)}."
                            pairs.append({
                                "question": q,
                                "answer": a,
                                "metric": metric,
                                "table_file": csv.name,
                            })
                    break
    return pairs

def dedupe_pairs(pairs: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for p in pairs:
        key = (p.get("question","").lower(), p.get("answer","").lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts_dir", type=str, default="./artifacts")
    args = ap.parse_args()

    art = Path(args.artifacts_dir)
    sections = load_sections(art)
    text_pairs = mine_pairs_from_text(sections)
    table_pairs = mine_pairs_from_tables(art / "tables")

    pairs = dedupe_pairs(table_pairs + text_pairs)

    print(f"[INFO] generated {len(pairs)} raw pairs.")
    # Persist
    out_jsonl = art / "qa_pairs.jsonl"
    out_csv = art / "qa_pairs.csv"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    pd.DataFrame(pairs).to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_jsonl.name} and {out_csv.name}")

    # Helpful hint if we fell short
    if len(pairs) < 50:
        print("[WARN] <50 Q/A pairs found. We can easily expand by adding more metric patterns or scanning more lines per section.")

if __name__ == "__main__":
    main()
