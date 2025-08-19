# app/debug_find_text.py
import re, json
from pathlib import Path

ART = Path("./artifacts")
RX = re.compile(r"(net\s+revenues?|total\s+net\s+revenues?|revenues?)", re.I)
YEAR = "2023"

def main():
    hits = 0
    p = ART / "chunks_100.jsonl"
    if not p.exists():
        print("Missing chunks_100.jsonl; run step2_chunk first.")
        return
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            txt = row["text"]
            if (YEAR in txt) and RX.search(txt):
                hits += 1
                meta = row["meta"]
                print(f"[{hits}] id={row['id']}")
                print(f"    doc={meta['doc_id']}  section={meta['section']}  pages={meta['pages']}  sz={meta['chunk_size']}  idx={meta['idx_in_section']}")
                print("    text:", (txt[:300] + "…").replace("\n", " "))
                print()
                if hits >= 10: break
    print(f"Total first-pass matches found: {hits}")

if __name__ == "__main__":
    main()
