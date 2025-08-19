from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Model selection
MODEL_NAME = "google/flan-t5-small"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print(f"Loaded model: {MODEL_NAME}")
