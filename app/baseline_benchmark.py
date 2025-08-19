import json
import time
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "google/flan-t5-small"
QA_PATH = "paypal_QA_RAG.json"  # path to your Q/A pairs
RESULTS_PATH = "baseline_results.json"

# Load Q/A pairs (first 10 for baseline)
with open(QA_PATH, "r") as f:
    qa_list = json.load(f)
test_qa = qa_list[:10]   # (or use all for more thorough baseline)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.to("cpu")  # Use "cuda" if you have a GPU

results = []

for item in test_qa:
    question = item["question"]
    gold = item["answer"]

    # Prompt style
    prompt = f"Q: {question} A:"
    inputs = tokenizer(prompt, return_tensors="pt")
    t0 = time.time()
    outputs = model.generate(**inputs, max_new_tokens=48)
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    t1 = time.time()
    elapsed = round(t1 - t0, 3)

    # Simple exact match for correctness (can be made more robust)
    correct = pred.strip().lower() == gold.strip().lower()

    results.append({
        "question": question,
        "predicted_answer": pred.strip(),
        "reference_answer": gold.strip(),
        "correct": correct,
        "inference_time_s": elapsed
    })

# Save results
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)

# Print results table
print(f"{'Q#':<4} {'Correct':<8} {'Inference(s)':<14} Question")
for i, r in enumerate(results):
    print(f"{i+1:<4} {str(r['correct']):<8} {r['inference_time_s']:<14} {r['question']}")
print(f"\nResults saved to {RESULTS_PATH}")
