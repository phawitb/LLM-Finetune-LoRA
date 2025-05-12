import pandas as pd
import matplotlib.pyplot as plt

# Load both logs (LoRA and Full)
lora_path = "training_log.csv"
full_path = "training_log_full.csv"

# Load into pandas
lora = pd.read_csv(lora_path)
full = pd.read_csv(full_path)

# Plot Train Loss
plt.figure(figsize=(10, 5))
plt.plot(lora["epoch"], lora["train_loss"], label="LoRA", marker='o')
plt.plot(full["epoch"], full["train_loss"], label="Full Fine-Tune", marker='x')
plt.title("Train Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.legend()
plt.grid(True)
plt.savefig("compare_train_loss.png")

# Plot Perplexity
plt.figure(figsize=(10, 5))
plt.plot(lora["epoch"], lora["perplexity"], label="LoRA", marker='o')
plt.plot(full["epoch"], full["perplexity"], label="Full Fine-Tune", marker='x')
plt.title("Perplexity Comparison")
plt.xlabel("Epoch")
plt.ylabel("Perplexity")
plt.legend()
plt.grid(True)
plt.savefig("compare_perplexity.png")

