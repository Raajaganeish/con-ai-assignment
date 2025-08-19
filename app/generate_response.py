# app/generate_response.py
"""
Step 2.5 + 2.6 — RAG Response Generation (CPU-safe, no table-KV) + Guardrails

- Multi-stage retrieval (HybridReranker) with CLI-tunable knobs.
- Extractive-first for numeric intents: find a money value near 'revenue(s)'
  and return it VERBATIM (no unit conversion).
- Generative fallback: google/flan-t5-small (CPU-friendly).
- Guardrails:
    (1) Input-side: validate queries; block clearly harmful/irrelevant inputs.
    (2) Output-side: grounding check that flags (does NOT modify) potential hallucinations.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from retrieve import HybridReranker, is_numeric_intent

GEN_MODEL = os.environ.get("GEN_MODEL", "google/flan-t5-small")
MAX_MODEL_CTX = 512

SYS_INSTRUCTIONS = (
    "Answer the question using ONLY the provided context. "
    "Be concise and factual. If the answer is not present, say: Data not found."
)

# =========================
# Guardrails (Step 2.6)
# =========================

# Input-side: simple domain & safety filter
def validate_query(q: str) -> tuple[bool, str, str]:
    """
    Return (allowed, category, reason).
    Blocks empty, excessively long queries, and clearly harmful/irrelevant content.
    Keeps things lightweight and deterministic (no network).
    """
    query = (q or "").strip()
    if not query:
        return (False, "invalid", "Empty query")
    if len(query) > 2000:
        return (False, "invalid", "Query too long")

    ql = query.lower()

    # Non-financial / obviously irrelevant small-talk or trivia (example)
    if any(x in ql for x in ["capital of france", "weather", "song lyrics", "joke"]):
        return (False, "out_of_scope", "Query is not about company financials")

    # Harmful/sensitive categories (very coarse)
    blocked = [
        "build a bomb", "how to make a bomb", "make explosives",
        "self-harm", "kill myself", "suicide", "harm others",
        "credit card numbers", "password dump", "ssn list",
        "hack into", "exploit", "malware", "ransomware",
    ]
    if any(b in ql for b in blocked):
        return (False, "safety_block", "Harmful or sensitive content")

    # Otherwise allowed
    return (True, "ok", "Allowed")

# Output-side: grounding flag (does not modify answer)
MONEY_RX_ANS = re.compile(r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?\s?(?:billion|bn|million|m|thousand|k)", re.I)
YEAR_RX_ANS  = re.compile(r"\b(20\d{2})\b")

def grounding_check(answer: str, contexts: List[str], numeric_intent: bool) -> dict:
    """
    Flag when numbers/years in the answer do not appear in the retrieved context.
    Returns a report dict. Does NOT change the answer.
    """
    ctx = "\n".join(contexts).lower()

    nums = [m.group(0) for m in MONEY_RX_ANS.finditer(answer)]
    yrs  = [m.group(1) for m in YEAR_RX_ANS.finditer(answer)]

    missing_nums = []
    for n in nums:
        if n.lower() not in ctx:
            # also try a loosened version (strip spaces)
            n2 = re.sub(r"\s+", "", n.lower())
            if n2 not in re.sub(r"\s+", "", ctx):
                missing_nums.append(n)

    missing_yrs = []
    for y in yrs:
        if y not in ctx:
            missing_yrs.append(y)

    status = "ok"
    reasons = []
    if numeric_intent and missing_nums:
        status = "low_grounding"
        reasons.append(f"numeric tokens not found in context: {missing_nums}")
    if missing_yrs:
        status = "low_grounding" if status == "ok" else status
        reasons.append(f"years not found in context: {missing_yrs}")

    return {
        "status": status,              # "ok" | "low_grounding"
        "missing_numbers": missing_nums,
        "missing_years": missing_yrs,
        "numeric_intent": numeric_intent,
    }

# =========================
# Retrieval prompt helpers
# =========================

def build_prompt(contexts: List[str], question: str) -> str:
    context = "\n\n---\n\n".join(contexts)
    return f"{SYS_INSTRUCTIONS}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"

def fit_context_to_window(tokenizer, question: str, contexts: List[str], reserve_tokens: int = 64) -> str:
    """Greedily add contexts until we hit the token budget (verbatim, no reformat)."""
    budget = MAX_MODEL_CTX - reserve_tokens
    kept: List[str] = []
    fixed_ids = tokenizer(
        f"{SYS_INSTRUCTIONS}\n\nContext:\n\nQuestion: {question}\nAnswer:",
        return_tensors="pt", add_special_tokens=False
    ).input_ids[0]
    budget -= fixed_ids.shape[-1]
    if budget <= 0:
        return build_prompt([], question)
    for c in contexts:
        ids = tokenizer(c + "\n\n---\n\n", return_tensors="pt", add_special_tokens=False).input_ids[0]
        tlen = ids.shape[-1]
        if tlen <= budget:
            kept.append(c)
            budget -= tlen
        else:
            break
    return build_prompt(kept, question)

def retrieval_confidence(scores: List[float]) -> float:
    if not scores:
        return 0.0
    top = max(scores)
    avg3 = sum(sorted(scores, reverse=True)[:3]) / min(3, len(scores))
    return round(0.5 * top + 0.5 * avg3, 3)

# =========================
# Extractive-first (verbatim)
# =========================

MONEY_RX = re.compile(
    r"""(
        \$\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?           # $ with thousands groups
        |
        \d{1,3}(?:,\d{3})+(?:\.\d+)?                # naked thousands with commas
        |
        \$?\s?\d+(?:\.\d+)?\s?(?:billion|bn|million|m|thousand|k)  # explicit units
    )""",
    re.IGNORECASE | re.VERBOSE
)

REV_RX   = re.compile(r"\b(total\s+net\s+revenues?|net\s+revenues?|revenues?)\b", re.I)
YEAR_RX  = re.compile(r"\b(20\d{2})\b", re.I)
DELTA_RX = re.compile(r"\b(increase(?:d)?|decrease(?:d)?|decline(?:d)?|grew|growth|down|up)\b|\bby\b|\bcompared\b", re.I)

PAT_DIRECT_1 = re.compile(
    r"(revenue(?:s)?[^.]{0,80}?(?:was|were|totaled|amounted\s+to|of)\s+)(?P<money>"+MONEY_RX.pattern+")",
    re.I | re.VERBOSE,
)
PAT_DIRECT_2 = re.compile(
    r"(?P<money>"+MONEY_RX.pattern+r")\s+(?:in|of)\s+revenue(?:s)?\b",
    re.I | re.VERBOSE,
)

def extract_numeric_answer(question: str, contexts: List[str]) -> str | None:
    """Find a money value near 'revenue' and return it exactly as it appears in the text (no unit conversion)."""
    ym = YEAR_RX.search(question)
    target_year = ym.group(1) if ym else None

    best = None  # (score, money_string, year_text)
    for ctx in contexts:
        lines = [ln.strip() for ln in ctx.splitlines() if ln.strip()]

        for i, line in enumerate(lines):
            if not REV_RX.search(line):
                continue

            window = " ".join(lines[max(0, i-1):min(len(lines), i+2)])

            # Prefer direct patterns
            for pat in (PAT_DIRECT_1, PAT_DIRECT_2):
                for m in pat.finditer(window):
                    money_tok = m.group("money").strip()
                    if re.fullmatch(r"\$?\s?20\d{2}", money_tok.replace(",", "")):
                        continue
                    if "%" in money_tok:
                        continue
                    if DELTA_RX.search(window):
                        continue
                    if target_year and target_year not in window:
                        continue
                    score = 5.0
                    if "total net revenues" in window.lower():
                        score += 1.0
                    cand = (score, money_tok, target_year)
                    if (best is None) or (cand[0] > best[0]):
                        best = cand

            # Fallback: any money token in a neutral (non-delta) window
            if best is None:
                if DELTA_RX.search(window):
                    continue
                for m in MONEY_RX.finditer(window):
                    money_tok = m.group(1).strip()
                    if re.fullmatch(r"\$?\s?20\d{2}", money_tok.replace(",", "")):
                        continue
                    if "%" in money_tok:
                        continue
                    if target_year and target_year not in window:
                        continue
                    score = 2.0
                    if "billion" in money_tok.lower():
                        score += 0.5
                    if "total net revenues" in window.lower():
                        score += 0.5
                    cand = (score, money_tok, target_year)
                    if (best is None) or (cand[0] > best[0]):
                        best = cand

    if best:
        _, money_tok, yr = best
        if yr:
            return f"PayPal’s revenue in {yr} was {money_tok}."
        else:
            return f"PayPal’s revenue was {money_tok}."
    return None

# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser(description="RAG Response Generation + Guardrails (FLAN‑T5‑Small, extractive-first, verbatim)")
    ap.add_argument("--artifacts_dir", type=str, default="./artifacts")
    ap.add_argument("--size", type=int, default=400, help="Chunk size to query (100 or 400)")
    ap.add_argument("--query", type=str, required=True)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--guardrail_report", type=str, default="./artifacts/guardrail_last.json")

    # Retrieval knobs (CLI-tunable)
    ap.add_argument("--topk_dense", type=int, default=40)
    ap.add_argument("--topk_sparse", type=int, default=80)
    ap.add_argument("--w_dense", type=float, default=0.4)
    ap.add_argument("--w_sparse", type=float, default=0.6)
    ap.add_argument("--prefer_tables", action="store_true", default=True)
    ap.add_argument("--table_boost", type=float, default=1.25)
    ap.add_argument("--must_terms", type=str, default="")
    ap.add_argument("--include_section_regex", type=str, default="")
    ap.add_argument("--doc_year", type=int, default=None)
    ap.add_argument("--hard_regex", type=str, default="")
    ap.add_argument("--hard_drop_if_no_match", action="store_true", default=False)

    args = ap.parse_args()
    must_terms = [t.strip() for t in args.must_terms.split(",") if t.strip()]

    # ---- Input guardrail ----
    allowed, category, reason = validate_query(args.query)
    if not allowed:
        print("Answer: Blocked by input guardrail")
        print(f"Guardrail: {{\"type\":\"input\",\"category\":\"{category}\",\"reason\":\"{reason}\"}}")
        print("Confidence: 0.0")
        print("Provenance: (blocked)")
        Path(args.guardrail_report).write_text(json.dumps(
            {"type": "input", "status": "blocked", "category": category, "reason": reason}, indent=2
        ), encoding="utf-8")
        return

    # ---- Retrieve (multi-stage) ----
    hr = HybridReranker(Path(args.artifacts_dir), size=args.size)
    hits = hr.query(
        q=args.query,
        topk_dense=args.topk_dense, topk_sparse=args.topk_sparse,
        w_dense=args.w_dense, w_sparse=args.w_sparse,
        final_topk=args.topk,
        prefer_tables=args.prefer_tables, table_boost=args.table_boost,
        must_terms=must_terms, include_section_regex=args.include_section_regex,
        doc_year=args.doc_year,
        hard_regex=args.hard_regex, hard_drop_if_no_match=args.hard_drop_if_no_match,
    )

    if not hits:
        print("Answer: Data not found")
        print("Confidence: 0.0")
        print("Provenance: (no relevant chunks)")
        Path(args.guardrail_report).write_text(json.dumps(
            {"type": "output", "status": "low_grounding", "reason": "no retrieval hits"}, indent=2
        ), encoding="utf-8")
        return

    contexts = [h.text for h in hits]
    conf = retrieval_confidence([h.score for h in hits])
    numeric_flag = is_numeric_intent(args.query)

    # ---- Extractive-first (verbatim) ----
    extracted = extract_numeric_answer(args.query, contexts) if numeric_flag else None
    if extracted:
        print(f"Answer: {extracted}\n")
        print(f"Confidence: {conf}")
        print("Provenance:")
        for h in hits:
            m = h.meta
            print(f"- doc={m.get('doc_id')} section={m.get('section')} pages={m.get('pages')} score={h.score:.3f}")
        # Output-side guardrail report (does not modify answer)
        g = grounding_check(extracted, contexts, numeric_flag)
        print(f"Guardrail: {json.dumps({'type':'output','status': g['status']})}")
        Path(args.guardrail_report).write_text(json.dumps({"type":"output", **g}, indent=2), encoding="utf-8")
        return

    # ---- Generative fallback ----
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    prompt = fit_context_to_window(tokenizer, args.query, contexts, reserve_tokens=max(32, args.max_new_tokens + 16))
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_MODEL_CTX).to(device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            do_sample=False,
            length_penalty=1.0,
            early_stopping=True,
        )
    raw = tokenizer.decode(output[0], skip_special_tokens=True)
    ans = raw.split("Answer:", 1)[-1] if "Answer:" in raw else raw

    print(f"Answer: {ans}\n")
    print(f"Confidence: {conf}")
    print("Provenance:")
    for h in hits:
        m = h.meta
        print(f"- doc={m.get('doc_id')} section={m.get('section')} pages={m.get('pages')} score={h.score:.3f}")

    # Output-side guardrail report (does not modify answer)
    g = grounding_check(ans, contexts, numeric_flag)
    print(f"Guardrail: {json.dumps({'type':'output','status': g['status']})}")
    Path(args.guardrail_report).write_text(json.dumps({"type":"output", **g}, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
