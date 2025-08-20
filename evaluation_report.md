| # | Question                                | Method      | Model Answer                               | Confidence | Time (s) | Correct (Y/N) |
|---|-----------------------------------------|-------------|--------------------------------------------|------------|----------|---------------|
| 1 | What was PayPal's revenue in 2023?      | RAG         | $1.2 billion    | 0.6796     | 12.22    | N             |
| 1 | What was PayPal's revenue in 2023?      | Fine-Tuned  | PayPal’s revenue in 2023 was $1.8 billion. | 0.0001     | 1.13     | Y             |
| 2 | How much did PayPal spend on R&D in 2023?| RAG         | PayPal’s revenue in 2023 was 926 m.         | 0.6092     | 8.48    | N             |
| 2 | How much did PayPal spend on R&D in 2023?| Fine-Tuned  | PayPal spent $1.25 billion on R&D in 2023. | 0.0        | 1.06     | N             |
| 3 | What is the capital of France?          | RAG         | Blocked by guardrail: Query is not about company financials | N/A | 0.0 | Y |
| 3 | What is the capital of France?          | Fine-Tuned  | Blocked by guardrail: Query is not about company financials | N/A | 0.0 | Y |

| # | Question                                | Method      | Model Answer                               | Confidence | Time (s) | Correct (Y/N) |
|---|-----------------------------------------|-------------|--------------------------------------------|------------|----------|---------------|
| 1 | What was PayPal's total payment volume in 2023?      | RAG         | $1.1 billion    | 0.6735     | 11.35    | N             |
| 1 | What was PayPal's total payment volume in 2023?      | Fine-Tuned         | PayPal's total payment volume in 2023 was $1.8 billion.    | 0.0     | 1.16    | N             |
| 2 | How much revenue did PayPal generate in 2023?       | RAG         | $1.2 billion                                               | 0.6796     | 13.03    | N             |
| 2 | How much revenue did PayPal generate in 2023?       | Fine-Tuned  | PayPal generated $1.8 billion in revenue in 2023.          | 0.0002     | 1.05     | N             |
| 3 | How many active accounts did PayPal have at the end of 2023? | RAG        | 2                                                          | 0.6471     | 12.40    | N             |
| 3 | How many active accounts did PayPal have at the end of 2023? | Fine-Tuned | PayPal had a total of.3 billion active accounts at the end of 2023. | 0.0002 | 1.10     | N             |
| 4 | What was PayPal's free cash flow in 2023?               | RAG         | PayPal’s revenue in 2023 was 926 m.                        | 0.6563     | 8.64     | N             |
| 4 | What was PayPal's free cash flow in 2023?               | Fine-Tuned  | PayPal's free cash flow in 2023 was $1.25 billion.         | 0.0        | 0.90     | N             |
| 5 | What was PayPal's non-GAAP operating margin in 2023?    | RAG         | a 5%                                                       | 0.7938     | 12.18    | N             |
| 5 | What was PayPal's non-GAAP operating margin in 2023?    | Fine-Tuned  | PayPal's non-GAAP operating margin in 2023 was.            | 0.0006     | 1.07     | N             |
| 6 | How many payment transactions did PayPal process in 2023?| RAG        | 2                                                          | 0.5729     | 11.62    | N             |
| 6 | How many payment transactions did PayPal process in 2023?| Fine-Tuned | PayPal processed a total of 1.2 billion payment transactions in 2023. | 0.0   | 1.07     | N             |
| 7 | How much did PayPal return to shareholders via share repurchases in 2023? | RAG        | $500,000                                        | 0.6338     | 11.61    | N             |
| 7 | How much did PayPal return to shareholders via share repurchases in 2023? | Fine-Tuned | PayPal returned to shareholders via share repurchases in 2023. | 0.0   | 1.09 | N |
| 8 | What were PayPal's net revenues in 2024?                    | RAG         | $1.1 billion                                               | 0.6952     | 11.62    | N             |
| 8 | What were PayPal's net revenues in 2024?                    | Fine-Tuned  | PayPal's net revenues in 2024 were $1.8 billion.           | 0.0001     | 1.12     | N             |
| 9 | What was PayPal's free cash flow in 2024?                   | RAG         | PayPal’s revenue in 2024 was 926 m.                        | 0.6707     | 8.80     | N             |
| 9 | What was PayPal's free cash flow in 2024?                   | Fine-Tuned  | PayPal's free cash flow in 2024 was $1.25 billion.         | 0.0        | 0.84     | N             |
|10 | How much did Venmo's volume grow in Q4 2024?                | RAG         | a million                                                  | 0.4874     | 11.02    | N             |
|10 | How much did Venmo's volume grow in Q4 2024?                | Fine-Tuned  | Venmo's volume rose by 6% in Q4 2024.                      | 0.0001     | 1.08     | N             |




## 4.4 Analysis: RAG vs Fine-Tuned QA

### **Accuracy Comparison**

Across the 10 selected financial questions:
- **Both RAG and Fine-Tuned models produced incorrect answers for all 10 questions** (accuracy 0/10). 
- Most model outputs either had the wrong value, an incomplete number, or a hallucinated statement not matching the true financial data.

### **Average Inference Speed**

- **RAG:**
    - Mean response: ~10–12 seconds per question
    - (Multi-stage retrieval + reranking + generation, slower)
- **Fine-Tuned:**
    - Mean response: ~1 second per question
    - (Fast direct inference, no retrieval)

### **Confidence Scores**

- **RAG:**
    - Retrieval confidence generally between 0.48–0.79, but this only reflects retrieval’s own scoring (not answer factuality).
    - Generation confidence very low, showing high model uncertainty.
- **Fine-Tuned:**
    - Generation confidence consistently low (typically 0.0000–0.0006), showing little certainty, which aligns with low answer quality.

---

### **Strengths & Weaknesses**

#### **RAG (Retrieval-Augmented Generation)**
- **Strengths:**
    - Provides source passage provenance.
    - Robust guardrails (safely blocks irrelevant or harmful queries).
    - Adaptable to new documents—can answer questions about *any* content present in the PDFs.
- **Weaknesses:**
    - Slower response times.
    - When retrieval fails (e.g., no relevant passage found), answer is likely wrong or “hallucinated.”
    - Accuracy in this test was very low due to chunking/recall limitations.

#### **Fine-Tuned Model**
- **Strengths:**
    - Very fast and efficient (no retrieval step).
    - Answers are often more fluent in natural language.
- **Weaknesses:**
    - Accuracy was low, frequently hallucinating plausible but incorrect facts or numbers.
    - No provenance—cannot trace answer to the source document.
    - Needs much larger, higher-quality fine-tuning data for good factuality.

---

### **Robustness to Irrelevant Queries**

Both systems **correctly blocked** irrelevant or unsafe questions using input-side guardrails (e.g., “What is the capital of France?”).

---

### **Practical Trade-Offs**

- **RAG** is better when provenance and traceability are critical, and when the document source is updated or questions are open-ended.
- **Fine-Tuned** is suitable for high-speed scenarios, but only with strong, diverse training data.
- In low-data, high-stakes scenarios (like this test), neither system is reliable for factual financial QA without further improvement.

---

**Summary:**  
*In this evaluation, both RAG and Fine-Tuned pipelines struggled to produce accurate answers on unseen PayPal financial questions, highlighting the challenges of both retrieval recall and small-data fine-tuning for robust factual QA. Further work—like larger datasets, improved chunking, or better retrieval—would be required for production use.*

