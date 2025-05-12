import math
import csv
import time
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset


# === CONFIG ===
model_name = "distilgpt2"
train_file = "my_corpus.txt"
output_dir = "./finetuned-distilgpt2-full"
num_epochs = 100
batch_size = 2
max_length = 128


# === Load tokenizer and model ===
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)


# === Count parameters ===
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    percent = 100 * trainable / total
    print("\nFull Fine-Tune Parameter Report")
    print("──────────────────────────────────")
    print(f"Total parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}")
    print(f"Trainable ratio      : {percent:.2f}%")
    print("──────────────────────────────────")

count_params(model)


# === Load and tokenize dataset ===
dataset = load_dataset("text", data_files={"train": train_file})

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length
    )

tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized = tokenized.map(lambda x: {"labels": x["input_ids"]}, batched=True)


# === Data collator ===
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)


# === Training arguments ===
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    num_train_epochs=1,  # manual loop
    save_strategy="no",
    logging_strategy="no",
    report_to="none",
    learning_rate=5e-4
)


# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    tokenizer=tokenizer,
    data_collator=data_collator
)


# === Manual training loop ===
log_history = []

for epoch in range(num_epochs):
    print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
    start_time = time.time()

    trainer.train()

    end_time = time.time()
    duration = round(end_time - start_time, 2)

    # Extract train loss
    train_loss = next(
        (entry.get("loss", entry.get("train_loss"))
         for entry in reversed(trainer.state.log_history)
         if "loss" in entry or "train_loss" in entry),
        None
    )

    # Compute perplexity
    perplexity = math.exp(train_loss) if train_loss is not None else None

    print(f"[Epoch {epoch + 1}] train_loss={train_loss}, ppl={perplexity:.2f}, time={duration}s")

    log_history.append({
        "epoch": epoch + 1,
        "train_loss": round(train_loss, 4) if train_loss else None,
        "perplexity": round(perplexity, 2) if perplexity else None,
        "train_time_sec": duration
    })


# === Save training log ===
with open("training_log_full.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "perplexity", "train_time_sec"])
    writer.writeheader()
    writer.writerows(log_history)


# === Save final model ===
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("\nFull fine-tuning complete. Log saved to training_log_full.csv")
