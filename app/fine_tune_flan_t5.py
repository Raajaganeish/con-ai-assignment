import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

MODEL_NAME = "google/flan-t5-small"
DATA_PATH = "paypal_QA_FineTuning.jsonl"
OUTPUT_DIR = "./finetuned_flan_t5_small"
BATCH_SIZE = 4
EPOCHS = 6
LEARNING_RATE = 5e-5

# Load data (HuggingFace Datasets reads JSONL easily)
ds = load_dataset("json", data_files=DATA_PATH, split="train")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Preprocessing
def preprocess(example):
    # Optionally prepend "Q: ... A:" (but your prompts are clear already)
    inputs = example["prompt"].strip()
    targets = example["completion"].strip()
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    labels = tokenizer(targets, max_length=64, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_ds = ds.map(preprocess, batched=False)

# Training arguments
args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="no",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    save_strategy="epoch",
    logging_steps=1,
    predict_with_generate=True,
    fp16=False,  # Set True if using CUDA and supported
    report_to="none"
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# Train!
trainer.train()

# Save the final model
trainer.save_model(OUTPUT_DIR)
print(f"Model fine-tuned and saved to {OUTPUT_DIR}")
