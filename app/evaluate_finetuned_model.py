import json
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_DIR = "finetuned_flan_t5_small"
QA_PATH = "paypal_QA_RAG.json"
RESULTS_PATH = "finetuned_results.json"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
model.to("cpu")

with open(QA_PATH) as f:
    test_qa = json.load(f)
test_qa = test_qa[:10]  # or your test split

results = []
for item in test_qa:
    question = item["question"]
    gold = item["answer"]
    prompt = f"{question.strip()}"
    inputs = tokenizer(prompt, return_tensors="pt")
    t0 = time.time()
    outputs = model.generate(**inputs, max_new_tokens=48)
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    t1 = time.time()
    elapsed = round(t1 - t0, 3)
    correct = pred.strip().lower() == gold.strip().lower()

    results.append({
        "question": question,
        "predicted_answer": pred.strip(),
        "reference_answer": gold.strip(),
        "correct": correct,
        "inference_time_s": elapsed
    })

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"{'Q#':<4} {'Correct':<8} {'Inference(s)':<14} Question")
for i, r in enumerate(results):
    print(f"{i+1:<4} {str(r['correct']):<8} {r['inference_time_s']:<14} {r['question']}")
print(f"\nResults saved to {RESULTS_PATH}")
