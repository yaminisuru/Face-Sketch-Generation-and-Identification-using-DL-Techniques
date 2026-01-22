import matplotlib.pyplot as plt

# Example data: replace these with your actual training values
epochs = list(range(1, 21))  # 20 epochs
training_loss = [1.2, 1.1, 0.95, 0.88, 0.82, 0.78, 0.72, 0.68, 0.65, 0.62, 0.60, 0.58, 0.56, 0.54, 0.53, 0.52, 0.50, 0.49, 0.48, 0.47]
top1_acc = [0.12, 0.18, 0.25, 0.32, 0.38, 0.42, 0.47, 0.50, 0.53, 0.57, 0.60, 0.63, 0.66, 0.68, 0.70, 0.72, 0.74, 0.75, 0.76, 0.78]
top5_acc = [0.35, 0.42, 0.50, 0.56, 0.61, 0.65, 0.68, 0.71, 0.73, 0.76, 0.78, 0.80, 0.82, 0.83, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90]

# -----------------------------
# Plot Training Loss
plt.figure(figsize=(6,4))
plt.plot(epochs, training_loss, marker='o', color='red', label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss vs Epoch')
plt.grid(True)
plt.legend()
plt.show()

# Plot Top-1 Accuracy
plt.figure(figsize=(6,4))
plt.plot(epochs, top1_acc, marker='o', color='blue', label='Top-1 Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Top-1 Accuracy vs Epoch')
plt.grid(True)
plt.legend()
plt.show()

# Plot Top-5 Accuracy
plt.figure(figsize=(6,4))
plt.plot(epochs, top5_acc, marker='o', color='green', label='Top-5 Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Top-5 Accuracy vs Epoch')
plt.grid(True)
plt.legend()
plt.show()
