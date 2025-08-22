Fine-Tuned Model QA Pipeline: Implementation Details
====================================================

**Overview**
------------

The Fine-Tuned QA model scenario leverages **a sequence-to-sequence transformer (FLAN-T5-small)** that is specifically fine-tuned on finance-related question-answer (QA) pairs. This approach is a strong baseline for comparison against retrieval-augmented generation (RAG), especially when you have a moderate but focused set of Q/A pairs. The fine-tuned model directly maps financial questions to answers, without retrieving document chunks at inference.

* * * * *

**1\. Fine-Tuning Process**
---------------------------

### **File:** `fine_tune_flan_t5.py`

#### **Workflow**

-   **Data:** Loads a JSONL dataset of financial Q/A pairs (`paypal_QA_FineTuning.jsonl`), each with a `prompt` (question or input) and `completion` (answer).

-   **Model:** Starts from `"google/flan-t5-small"`---a compact, efficient seq2seq model from HuggingFace.

-   **Tokenization:**

    -   Inputs (`prompt`) are tokenized (max 128 tokens).

    -   Targets (`completion`) are tokenized (max 64 tokens).

-   **Training:**

    -   Arguments (epochs, batch size, learning rate, etc.) are customizable.

    -   Uses HuggingFace `Seq2SeqTrainer` for robust, reproducible fine-tuning.

-   **Saving:** The trained model is saved to `./finetuned_flan_t5_small` for later inference.

**Code Snippet for Preprocessing:**

```
def preprocess(example):
    inputs = example["prompt"].strip()
    targets = example["completion"].strip()
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    labels = tokenizer(targets, max_length=64, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

```

**Trainer Configuration:**

-   `evaluation_strategy="no"` (no validation during training, can be changed)

-   `num_train_epochs=6`

-   `learning_rate=5e-5`

-   `per_device_train_batch_size=4`

-   Model checkpoints are saved each epoch.

**Result:**\
A fine-tuned FLAN-T5 model specialized for PayPal/finance Q&A, ready for inference.

* * * * *

**2\. Inference / QA with Fine-Tuned Model**
--------------------------------------------

### **File:** `interface.py`, `generate_response.py`

#### **How Inference Works**

-   **No retrieval.** The model is called directly with the user's query.

-   **Prompting:** The user's question is passed as-is (or with minimal formatting) to the model.

-   **Prediction:** The model generates a text answer.

-   **Confidence Estimation:** The token-level probabilities are averaged to provide a generation confidence metric (optional, for UX/debugging).

-   **UI:** The interface allows users to select the "Fine-Tuned" mode and run their query against the fine-tuned model, showing the answer and meta-info.

**Sample Function:**

```
def run_finetuned(query):
    model_dir = "finetuned_flan_t5_small"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.to("cpu")
    answer, gen_conf = model_generation_with_confidence(
        model, tokenizer, query.strip(), max_new_tokens=48, device="cpu"
    )
    return answer, "N/A", gen_conf, "Fine-Tuned", ..., "-", "ok", query

```

-   `"retrieval_confidence"` is `"N/A"` since no retrieval is performed.

* * * * *

**3\. Evaluation & Benchmarking**
---------------------------------

### **Files:** `evaluate_finetuned_model.py`, `baseline_benchmark.py`

#### **Evaluation Steps**

-   **Inputs:** Both scripts load a set of Q/A pairs (usually a test split from your main dataset).

-   **Inference:** Each question is run through either:

    -   The baseline model (e.g., original `flan-t5-small`) for baseline comparison.

    -   The fine-tuned model (`finetuned_flan_t5_small`) for evaluation.

-   **Metrics:** For each prediction:

    -   **Exact match**: Checks if the generated answer exactly matches the gold/reference answer (can be replaced by fuzzy metrics for robustness).

    -   **Inference time**: Measures time per answer for performance evaluation.

-   **Reporting:**

    -   Results are printed in tabular format.

    -   JSON result files are saved for further analysis.

**Result Fields per Question:**

-   `question`

-   `predicted_answer`

-   `reference_answer`

-   `correct` (True/False)

-   `inference_time_s`

**Example Output:**

```
Q#   Correct   Inference(s)   Question
1    True      0.121          What was PayPal's revenue in 2023?
...

```

**Purpose:**

-   Enables objective comparison between baseline, fine-tuned, and RAG systems.

* * * * *

**4\. Comparison to RAG (Retrieval-Augmented Generation)**
----------------------------------------------------------

-   **Fine-Tuned Model:**

    -   **Strengths:** Fast inference, low latency, answers directly from model memory.

    -   **Weaknesses:** May hallucinate or miss details not seen in training. No provenance or evidence trace for the answer.

-   **RAG Model:**

    -   **Strengths:** Can reference arbitrary data from recent documents, always provides provenance and supports transparency.

    -   **Weaknesses:** More complex, requires robust chunking and indexing.

**The Gradio UI allows direct head-to-head comparison of both modes.**

* * * * *

**5\. High-Level Flow Diagram**
-------------------------------

```
flowchart LR
    A[Training Data<br>(Q/A pairs)] --> B[Fine-Tune<br>FLAN-T5-small]
    B --> C[Fine-Tuned Model]
    C --> D[Inference<br>(User Query)]
    D --> E[Answer Generated]

```

* * * * *

**6\. Key Implementation Files**
--------------------------------

-   `fine_tune_flan_t5.py`: Data loading, tokenization, model training, and saving.

-   `evaluate_finetuned_model.py`: Runs evaluation over test set, records metrics.

-   `baseline_benchmark.py`: Baseline inference for comparison.

-   `interface.py`: Integrates fine-tuned model as one mode in the UI, handles guardrails, displays results.

-   `generate_response.py`: (Shares some RAG-related code; not directly used for pure fine-tuned scenario.)

* * * * *

**7\. Example Usage**
---------------------

-   **Fine-tune:**\
    `python fine_tune_flan_t5.py`

-   **Evaluate:**\
    `python evaluate_finetuned_model.py`

-   **Launch UI:**\
    `python interface.py` (select "Fine-Tuned" mode)

* * * * *

**References**
==============

-   **Fine-Tuning Script:** [`fine_tune_flan_t5.py`][26†source]

-   **Evaluation Script:** [`evaluate_finetuned_model.py`][25†source]

-   **UI/Inference Integration:** [`interface.py`][28†source], [`generate_response.py`][27†source]

-   **Baseline Evaluation:** [`baseline_benchmark.py`][24†source]

* * * * *

**Let me know if you need an even deeper breakdown of any part (e.g., loss curves, advanced metrics, hyperparameter tuning), or want comparison plots/tables for your docs!**