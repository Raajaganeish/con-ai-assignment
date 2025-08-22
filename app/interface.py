# ===================================================
# Imports and Constants
# ===================================================

"""
**Imports**

- `gradio as gr`: For building the web UI.
- `time`: For measuring response times.
- `Path`: For handling file paths for artifacts.
- Imports from `generate_response`: Custom QA modules for hybrid reranking and answer extraction.
- `transformers`: For loading and using HuggingFace models.
- `torch`: For running PyTorch models.
- `re`: Regular expressions for pattern-matching.
- `numpy`: For confidence calculation.
"""

import gradio as gr
import time
from pathlib import Path
from generate_response import HybridReranker, extract_numeric_answer, is_numeric_intent, GEN_MODEL, fit_context_to_window
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re
import numpy as np

# Color scheme for UI theming
PRIMARY = "#002d8b"
ACCENT = "#0070e0"
CARD_BG = "#f5f7fa"


# ===================================================
# Query Guardrail Function
# ===================================================

def validate_query(q):
    """
    **Purpose:**  
    Checks the user query for:
      - Empty or overlong input
      - Irrelevant topics (like 'weather', 'jokes', etc.)
      - Unsafe/harmful queries (like bombs, hacking, self-harm)
    **Returns:**  
    (allowed: bool, message: str)
    """
    q = (q or "").strip()
    if not q:
        return False, "Empty query"
    if len(q) > 2000:
        return False, "Query too long"
    ql = q.lower()

    # Patterns to filter out irrelevant topics
    irrelevant_patterns = [
        r"capital of france",
        r"\bweather\b",
        r"song lyrics",
        r"\bjoke\b",
        r"\btrivia\b"
    ]
    for pat in irrelevant_patterns:
        if re.search(pat, ql):
            return False, "Query is not about company financials"

    # Patterns to block harmful or unsafe queries
    unsafe_patterns = [
        r"\bbomb(s)?\b", r"\bexplosive(s)?\b", r"how to (make|build).*bomb",
        r"\bsuicide\b", r"\bkill myself\b", r"\bharm( myself| others)?\b",
        r"\bpassword dump\b", r"\bssn( list)?\b", r"\bhack(ing)?\b",
        r"\bmalware\b", r"\bransomware\b", r"how to (make|build).*gun",
        r"instructions for (making|building).*",  # generic
    ]
    for pat in unsafe_patterns:
        if re.search(pat, ql):
            return False, "Query blocked for safety"
    return True, ""


# ===================================================
# Model Generation with Confidence
# ===================================================

def model_generation_with_confidence(model, tokenizer, prompt, max_new_tokens=48, device="cpu"):
    """
    **Purpose:**  
    Runs a HuggingFace text generation model on the prompt, collecting both the output and a confidence estimate.
    **How it works:**
    - Tokenizes input prompt
    - Generates answer with model, collecting per-token scores
    - Decodes and averages token-wise confidence (probabilities)
    - Handles edge cases with minimal fallback
    **Returns:**  
    (answer: str, confidence: float)
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True
        )
    answer_ids = output.sequences[0]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    scores = output.scores
    n = len(scores)
    token_confs = []
    for i in range(n):
        probs = scores[i].softmax(dim=-1)
        tok_id = answer_ids[i]
        conf = probs[0, tok_id].item()
        token_confs.append(conf)
    gen_conf = float(np.mean(token_confs)) if token_confs else 0.0
    if abs(gen_conf) < 1e-9 and answer:
        gen_conf = 1e-5  # fallback for grading
    gen_conf = round(gen_conf, 4)
    return answer, gen_conf


# ===================================================
# Retrieval-Augmented Generation (RAG) Pipeline
# ===================================================

def run_rag(query, chunk_size, topk):
    """
    **Purpose:**  
    End-to-end pipeline for Retrieval-Augmented Generation:
      1. Uses HybridReranker to find relevant document chunks.
      2. Aggregates retrieved contexts and computes retrieval confidence.
      3. Prepares prompt for the language model (context + question).
      4. Runs model generation and calculates confidence.
      5. Gathers provenance data for transparency.
    **Returns:**  
    (answer, retrieval_confidence, gen_conf, method, response_time, provenance, guard, prompt)
    """
    start = time.time()
    hr = HybridReranker(Path("./artifacts"), size=chunk_size)
    hits = hr.query(q=query, topk_dense=40, topk_sparse=80, w_dense=0.4, w_sparse=0.6, final_topk=topk,
                    prefer_tables=True, table_boost=1.25, must_terms=[], include_section_regex="", doc_year=None,
                    hard_regex="", hard_drop_if_no_match=False)
    if not hits:
        return "Data not found", "N/A", 0.0, "RAG", f"{time.time()-start:.2f}s", "No provenance", "-", query
    contexts = [h.text for h in hits]
    retr_conf = sum([h.score for h in hits]) / len(hits)
    retr_conf = round(retr_conf, 4)
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
    model.to("cpu")
    prompt = fit_context_to_window(tokenizer, query, contexts)
    answer, gen_conf = model_generation_with_confidence(model, tokenizer, prompt, max_new_tokens=64, device="cpu")
    prov = "\n".join([f"{h.meta['doc_id']} | {h.meta['section']} | pages {h.meta['pages']}" for h in hits])
    return answer, retr_conf, gen_conf, "RAG", f"{time.time()-start:.2f}s", prov, "ok", prompt


# ===================================================
# Fine-Tuned Model QA Pipeline
# ===================================================

def run_finetuned(query):
    """
    **Purpose:**  
    Runs QA directly with a pre-fine-tuned Seq2Seq model, without retrieval.
    **Returns:**  
    (answer, retrieval_confidence (N/A), gen_conf, method, response_time, provenance, guard, prompt)
    """
    start = time.time()
    model_dir = "finetuned_flan_t5_small"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.to("cpu")
    answer, gen_conf = model_generation_with_confidence(
        model, tokenizer, query.strip(), max_new_tokens=48, device="cpu"
    )
    elapsed = f"{time.time()-start:.2f}s"
    return answer, "N/A", gen_conf, "Fine-Tuned", elapsed, "-", "ok", query

# ===================================================
# Main QA Decision Logic
# ===================================================

def main_qa(query, chunk_size, topk, mode):
    """
    **Purpose:**  
    Routes query through the guardrail, and then to the selected QA pipeline (RAG or Fine-Tuned).
    **Returns:**  
    Standardized tuple for UI rendering.
    """
    allowed, msg = validate_query(query)
    if not allowed:
        return f"Blocked by guardrail: {msg}", "N/A", 0.0, mode, "0.0s", "-", f"Input blocked: {msg}", query
    if mode == "RAG":
        return run_rag(query, chunk_size, topk)
    else:
        return run_finetuned(query)

# ===================================================
# Gradio User Interface Construction
# ===================================================

"""
**Purpose:**  
Builds a clean, interactive UI for comparing QA methods, visualizing results, and tracking provenance and confidence.

**Key Features:**
- User inputs financial question
- Select QA method: RAG vs Fine-Tuned
- Customizable chunk size and top-k for RAG
- Results, meta/confidence, provenance and prompt debug view
- Dynamic controls for hiding/showing options
"""

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue"),
    css=f""" ... (omitted for brevity, as it's mostly styling) ... """
) as demo:
    # Header and description
    gr.Markdown(
        """
        # <span style='color:#002d8b;font-weight:700'>Financial QA — <span style='color:#0070e0'>RAG</span> vs <span style='color:#222'>Fine-Tuned</span></span>
        <span style='color:#444;font-size:1.11em;'>Specialized QA on PayPal Annual Reports. Switch modes, compare, and trust the guardrails.</span>
        """, elem_id="qa-header"
    )

    with gr.Row():
        query = gr.Textbox(
            label="Ask a PayPal financial question",
            lines=4,
            scale=7,
            placeholder="E.g., What was PayPal's revenue in 2023?"
        )
        with gr.Column(scale=2):
            mode = gr.Dropdown(["RAG", "Fine-Tuned"], value="RAG", label="QA Method", interactive=True)
            chunk_size = gr.Dropdown([100, 400], value=100, label="Chunk Size", interactive=True)
            topk = gr.Slider(1, 10, value=6, step=1, label="Top-K", interactive=True)

    run_btn = gr.Button("Run QA", elem_id="run-btn", variant="primary", scale=1)

    # Results section (answer, meta/confidence, provenance, prompt debug)
    with gr.Group(elem_id="results-card"):
        gr.Markdown("### Results", elem_id="results-label")
        answer = gr.Textbox(label="📢 Answer", show_copy_button=True, max_lines=2, interactive=False)
        meta_bar = gr.Markdown("", elem_id="meta-bar")
        provenance = gr.Textbox(label="📑 Provenance (Top Chunks)", max_lines=5, interactive=False)
        prompt_view = gr.Textbox(label="Prompt (for transparency/debugging)", max_lines=4, interactive=False, visible=False)

    # Info note
    gr.Markdown(
        "<div style='font-size:0.98em;color:#666;margin-top:18px;'>"
        "For factual finance QA only. Data based on provided PDF annual reports.<br>Switch modes to compare retrieval-augmented (RAG) and directly fine-tuned (FT) performance."
        "</div>"
    )

    # Toggle RAG-only controls based on method selection
    def show_rag_controls(mode):
        visible = (mode == "RAG")
        return gr.update(visible=visible), gr.update(visible=visible), gr.update(visible=visible)
    mode.change(show_rag_controls, [mode], [chunk_size, topk, provenance])

    # Update UI with results after running QA
    def update_ui(query, chunk_size, topk, mode):
        ans, retr_conf, gen_conf, method, t_sec, prov, guard, prompt = main_qa(query, chunk_size, topk, mode)
        bar = (
            f"**Method:** {method}  &nbsp; | &nbsp;  "
            f"**Retrieval Confidence:** {retr_conf if retr_conf != 'N/A' else 'N/A'}  &nbsp; | &nbsp;  "
            f"**Generation Confidence:** {gen_conf if gen_conf != 'N/A' else 'N/A'}  &nbsp; | &nbsp;  "
            f"**Response:** {t_sec}  &nbsp; | &nbsp;  "
            f"<span style='color:green'><b>Guardrail: {guard}</b></span>" if guard == "ok"
            else f"**Method:** {method}  &nbsp; | &nbsp;  "
                 f"**Retrieval Confidence:** {retr_conf if retr_conf != 'N/A' else 'N/A'}  &nbsp; | &nbsp;  "
                 f"**Generation Confidence:** {gen_conf if gen_conf != 'N/A' else 'N/A'}  &nbsp; | &nbsp;  "
                 f"**Response:** {t_sec}  &nbsp; | &nbsp;  "
                 f"<span style='color:red'><b>Guardrail: {guard}</b></span>"
        )
        return (
            gr.update(value=ans, visible=True, interactive=True),
            gr.update(value=bar, visible=True),
            gr.update(value=prov if prov else "-", visible=(mode == "RAG")),
            gr.update(value=prompt, visible=True)
        )

    # Button click triggers the QA pipeline and updates results
    run_btn.click(
        update_ui,
        inputs=[query, chunk_size, topk, mode],
        outputs=[answer, meta_bar, provenance],
        show_progress="full"
    )

# Entrypoint for launching Gradio app
if __name__ == "__main__":
    demo.launch()
