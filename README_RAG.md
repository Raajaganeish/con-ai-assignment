# Comparative Financial QA — RAG System (PayPal 2023–2024)

This repository implements the **RAG (Retrieval‑Augmented Generation)** half of the assignment using PayPal’s last two annual reports. It covers the pipeline end‑to‑end—from PDF ingestion and sectioning, to hybrid dense+sparse retrieval with a **multi‑stage re‑ranker**, to answer generation with a small open‑source model, plus input/output **guardrails** and a simple **Gradio** UI.

> ✅ **Scope**: This README focuses on the RAG system only. The Fine‑Tuned model path can be added later and plugged into the same UI.

---

## What it does (Functionality)

1. **Data Collection & Preprocessing**
   - Converts PDFs → text using high‑quality PDF parsers (with OCR fallback if needed).
   - Cleans noise (headers/footers/page numbers).
   - **Segments** documents using structural cues (PDF headings/H1/H2) into logical sections.

2. **Chunking**
   - Builds retrieval chunks at two sizes (**100** and **400** tokens). Each chunk is saved with provenance (doc name, section, pages, chunk index).

3. **Embedding & Indexing**
   - Dense embeddings via `sentence-transformers/all-MiniLM-L6-v2` → **FAISS** index.
   - Sparse retrieval via **BM25** (rank‑bm25).
   - Stores index manifests + id mapping for reproducibility.

4. **Hybrid Retrieval (Stage‑A)**
   - Queries both dense (FAISS) and sparse (BM25) indices.
   - Combines results with **weighted score fusion** (configurable).

5. **Multi‑Stage Retrieval (Stage‑B Re‑ranker)**
   - Re‑ranks the union of candidates with a cross‑encoder to prioritize passages that best match the query.

6. **Response Generation**
   - **Extractive‑first** for numeric questions (pulls $-values near “revenue(s)” in the target year **verbatim**, no reformatting).
   - **Generative fallback** with `google/flan-t5-small` (lightweight, CPU‑safe).
   - Prompts include concatenated retrieved passages + the user query.
   - Token‑based prompt fitting to the model’s context window.

7. **Guardrails (2.6)**
   - **Input‑side**: blocks harmful/irrelevant queries (e.g., “build a bomb”, “capital of France?”).
   - **Output‑side**: **flags** (does not change) low‑grounding answers if numbers/years are not found in retrieved context. A JSON report is saved to `artifacts/guardrail_last.json`.

8. **Interface (2.7)**
   - Gradio UI to query the system, see answer, retrieval confidence, method used, response time, and provenance. (Currently **RAG** only; FT can be added later.)

---

## Environment

- **Python**: 3.10.13
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

> Note (macOS/Apple Silicon): We use CPU‑safe models to avoid MPS “bus error” crashes. If you have CUDA, you can switch device flags in code later.

---

## Data

Place the two PayPal annual reports into `./pdf/`:

- `pdf/2023-Annual-Report.pdf`
- `pdf/2024-Annual-Report.pdf`

(You already downloaded the official documents and placed them here.)

---

## Directory Layout (key paths)

```
app/
  extract.py               # Step 1A: PDF → raw text JSONL
  segment.py               # Step 1B: Clean + segment into sections (H1/H2-aware)
  build_qa.py              # (Optional) heuristic Q/A mining for dataset construction
  chunk_util.py            # Step 2.1: Build retrieval chunks (100 & 400 tokens)
  embedding_indexing.py    # Step 2.2: Build FAISS (dense) + BM25 (sparse) indices
  retrieve.py              # Step 2.3 + 2.4: Hybrid retrieval + multi-stage re-ranking
  generate_response.py     # Step 2.5 + 2.6: RAG generation (extractive-first) + guardrails
  interface.py             # Step 2.7: Gradio UI

artifacts/
  raw_*.jsonl                      # raw page text per PDF
  sections_*.json / *.txt          # cleaned sections
  chunks_100.jsonl / chunks_400.jsonl
  faiss_index_sz100.index / ids
  faiss_index_sz400.index / ids
  bm25_sz100.pkl / bm25_sz400.pkl
  embed_config_*.json / chunks_manifest.json
  guardrail_last.json              # last guardrail report
```

---

## End‑to‑End Execution Sequence

> Run each step from the repo root. Artifacts land in `./artifacts/`.

### 1) Extract text from PDFs
```bash
python app/extract.py --pdf_dir ./pdf --artifacts_dir ./artifacts
```
Outputs: `artifacts/raw_*.jsonl`

### 2) Clean & segment into sections
```bash
python app/segment.py --pdf_dir ./pdf --artifacts_dir ./artifacts
```
Outputs: `artifacts/sections_*.json` and `sections_*.txt`

### 3) (Optional) Build Q/A dataset (Step 1 requirement)
```bash
python app/build_qa.py --artifacts_dir ./artifacts
```
Outputs: `artifacts/qa_pairs.jsonl` and `qa_pairs.csv`

### 4) Create retrieval chunks (100 & 400)
```bash
python app/chunk_util.py --artifacts_dir ./artifacts
```
Outputs: `artifacts/chunks_100.jsonl`, `artifacts/chunks_400.jsonl`, `chunks_manifest.json`

### 5) Build dense + sparse indices
```bash
python app/embedding_indexing.py --artifacts_dir ./artifacts
```
Outputs:
- FAISS: `faiss_index_sz100.index`, `faiss_index_sz400.index` (+ ids jsonl)
- BM25: `bm25_sz100.pkl`, `bm25_sz400.pkl`
- `embed_config_sz*.json`

### 6) Debug retrieval (optional)
```bash
python app/retrieve.py --artifacts_dir ./artifacts --size 100   --query "What was PayPal's revenue in 2023?"
```
Tip: you can pass advanced knobs (`--topk_dense`, `--topk_sparse`, `--w_dense`, `--w_sparse`, `--must_terms`, etc.).

### 7) Generate answers (RAG, CLI)
```bash
python app/generate_response.py --artifacts_dir ./artifacts --size 100   --query "What was PayPal's revenue in 2023?"   --topk 8 --max_new_tokens 48
```
- Prints: **Answer**, **Confidence**, **Provenance**, **Guardrail status**
- Saves guardrail JSON: `artifacts/guardrail_last.json`

### 8) Launch the UI (Gradio)
```bash
python app/interface.py
```
Open the local URL shown (e.g., `http://127.0.0.1:7860`).

---

## Inference / How to Use

### A) From CLI
- Ask a fact question:
  ```bash
  python app/generate_response.py --artifacts_dir ./artifacts --size 100     --query "What was PayPal's revenue in 2023?"     --topk 8
  ```
- You’ll see:
  - **Answer**: either an extractive (verbatim) value near “revenue(s)” + year or a short generated answer.
  - **Confidence**: derived from retrieval scores (0–1).
  - **Provenance**: document, section, pages for the top hits.
  - **Guardrail**: “ok” or “low_grounding”. The answer is **not** modified by guardrails.

### B) From UI (Gradio)
- Enter your question, pick **Chunk Size** (100 or 400), adjust **Top‑K**.
- Click **Run RAG**.
- The app displays **Answer**, **Retrieval Confidence**, **Method (RAG)**, **Response Time**, and **Provenance**.

> Later, when you implement the Fine‑Tuned model, add a dropdown to switch modes and route to your FT inference path using the same interface.

---

## Advanced Retrieval (2.4: Multi‑Stage Retrieval)

- Stage‑A: Hybrid sparse+dense retrieval (BM25 + FAISS) with **weighted score fusion**.
- Stage‑B: Cross‑encoder **re‑ranking** of the union to promote semantically precise passages.
- Options in `retrieve.py` and `generate_response.py` allow tuning:
  - `--topk_dense`, `--topk_sparse`, `--w_dense`, `--w_sparse`
  - `--prefer_tables`, `--table_boost`
  - `--must_terms`, `--include_section_regex`, `--doc_year`
  - Hard filters (regex) if desired.

---

## Guardrails (2.6)

- **Input‑side**: Blocks clearly harmful or off‑topic queries; prints a block reason (no answer generated).
- **Output‑side**: **Flags** (does not change) answers whose numbers/years don’t appear in retrieved context; saves report to `artifacts/guardrail_last.json` for the UI/logs.

---

## Troubleshooting

- **macOS “bus error”** when loading large causal LMs: we default to `google/flan-t5-small` (encoder‑decoder, CPU‑friendly). Avoid MPS for now.
- **Chunk size effects**: `--size 100` improves granularity (good for exact figures); `--size 400` improves context breadth.
- **If extraction returns the wrong figure** (e.g., a delta like “increased by $602 million”):
  - Increase `topk` and use `--must_terms revenues,2023,$` or a section regex like `--include_section_regex "sound financial and operational performance"`.
  - The extractor prioritizes direct statements (“revenue … was $X”).

---

## Next (when you add Fine‑Tuning)

- Reuse the same Q/A pairs to fine‑tune a small model (e.g., DistilBERT, GPT‑2 small, FLAN‑T5 base).
- Add a **Mode** dropdown in `interface.py` to switch between **RAG** and **FT** paths. Show the same metadata (confidence, runtime, provenance or dataset note).

---

**Authors**: Group 21  
**Contact**: raachandrasekara@paypal.com



### Latest Update
### 1\. Environment Setup

`# (Recommended: use a virtual environment)
python3 -m venv venv
source venv/bin/activate  # or 'venv\Scripts\activate' on Windows

# Install all dependencies
pip install -r requirements.txt`

* * * * *

### 2\. Place Input PDFs

Place your financial statements (e.g. PayPal Annual Reports) in the `pdf/` folder:

`pdf/2023-Annual-Report.pdf
pdf/2024-Annual-Report.pdf`

* * * * *

### 3\. Data Extraction & Preprocessing Pipeline

`# Extract text from PDFs
python app/extract.py --pdf_dir ./pdf --artifacts_dir ./artifacts

# Clean & segment into logical sections
python app/segment.py --pdf_dir ./pdf --artifacts_dir ./artifacts

# (Optional: Auto-generate Q/A pairs for fine-tuning/eval)
python app/build_qa.py --artifacts_dir ./artifacts

# Create retrieval chunks (100 & 400 tokens)
python app/chunk_util.py --artifacts_dir ./artifacts

# Build dense and sparse retrieval indices
python app/embedding_indexing.py --artifacts_dir ./artifacts`

* * * * *

### 4\. Run Baseline Evaluation (Pre-Fine-Tuning)

`python app/baseline_benchmark.py`

* * * * *

### 5\. Fine-Tune the Model (Flan-T5 Small, Supervised Q/A)

`python app/fine_tune_flan_t5.py`

Model checkpoints save to `finetuned_flan_t5_small/`.

* * * * *

### 6\. Evaluate Fine-Tuned Model

`python app/evaluate_finetuned_model.py`

* * * * *

### 7\. (Optional) RAG CLI Inference

`python app/generate_response.py --artifacts_dir ./artifacts --size 100\
  --query "What was PayPal's revenue in 2023?" --topk 8 --max_new_tokens 48`

* * * * *

### 8\. Launch the Interactive QA UI

`python app/interface.py`

Open the link (usually <http://127.0.0.1:7860>) in your browser.

* * * * *

### 💡 Pro Tips

-   Re-run any step if you update PDFs or want to retrain.

-   All outputs, models, and results are saved in `artifacts/` and project root.

-   Large files (PDFs, models, artifacts) are excluded from git via `.gitignore`.

* * * * *