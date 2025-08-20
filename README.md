
# PayPal Financial QA — Quickstart

This project is a Retrieval-Augmented Generation (RAG) QA system for PayPal's 2023–2024 annual reports. It extracts, indexes, and answers questions from the reports using hybrid retrieval and a small open-source model. You can run everything locally on CPU.

---

## 1. Setup

**Requirements:**
- Python 3.10+
- All dependencies in `requirements.txt`

**Install:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 2. Prepare Data

Download these files and place them in the `pdf/` folder:
- `pdf/2023-Annual-Report.pdf`
- `pdf/2024-Annual-Report.pdf`

---

## 3. Run the Pipeline (from repo root)

**Step 1: Extract text from PDFs**
```bash
python app/extract.py --pdf_dir ./pdf --artifacts_dir ./artifacts
```

**Step 2: Segment into sections**
```bash
python app/segment.py --pdf_dir ./pdf --artifacts_dir ./artifacts
```

**Step 3 (Optional): Build Q/A dataset**
```bash
python app/build_qa.py --artifacts_dir ./artifacts
```

**Step 4: Create retrieval chunks**
```bash
python app/chunk_util.py --artifacts_dir ./artifacts
```

**Step 5: Build retrieval indices**
```bash
python app/embedding_indexing.py --artifacts_dir ./artifacts
```

---

## 4. Run QA (Ask Questions)

**From CLI:**
```bash
python app/generate_response.py --artifacts_dir ./artifacts --size 100 --query "What was PayPal's revenue in 2023?" --topk 8
```

**From UI:**
```bash
python app/interface.py
```
Then open the link shown in your terminal (usually http://127.0.0.1:7860).

---

## 5. (Optional) Fine-Tune & Evaluate

**Run baseline evaluation:**
```bash
python app/baseline_benchmark.py
```

**Fine-tune the model:**
```bash
python app/fine_tune_flan_t5.py
```

**Evaluate fine-tuned model:**
```bash
python app/evaluate_finetuned_model.py
```

---

## Notes
- All outputs and models are saved in `artifacts/` and `finetuned_flan_t5_small/`.
- If you update PDFs or want to retrain, just re-run the relevant steps.
- Large files are excluded from git.

**Contact:** raachandrasekara@paypal.com