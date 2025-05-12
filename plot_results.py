import pandas as pd
import matplotlib.pyplot as plt

# Create the CSV content and save to file
csv_path = "training_log.csv"

# Load the CSV
df = pd.read_csv(csv_path)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(df["epoch"], df["train_loss"], label="Train Loss", marker='o')
plt.plot(df["epoch"], df["perplexity"], label="Perplexity", marker='x')
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Training Loss and Perplexity Over Epochs")
plt.legend()
plt.grid(True)

# Save plot
plot_path = "train_loss_perplexity.png"
plt.savefig(plot_path)


