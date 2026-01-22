import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import TripletNet
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt  # for showing images

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = TripletNet()
model.load_state_dict(torch.load("cnn_triplet_cufs.pth", map_location=device))
model.to(device)
model.eval()

# Load precomputed photo embeddings
photo_embeddings = np.load("photo_embeddings.npy", allow_pickle=True).item()

# Transform for model input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Function to identify top matching photos
def identify(sketch_path, top_k=5):
    sketch = Image.open(sketch_path).convert("RGB")
    sketch_input = transform(sketch).unsqueeze(0).to(device)

    with torch.no_grad():
        sketch_emb = model(sketch_input).cpu().numpy()

    scores = {}
    for name, emb in photo_embeddings.items():
        sim = cosine_similarity(sketch_emb, emb)[0][0]
        scores[name] = sim

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

# -------------------------------
# Example usage
sketch_path = r"processed_data\Train\aug_sketches\m1-004-01_aug3.jpg"  # raw string fixes Windows path
results = identify(sketch_path)

# Print top matches in console
print("Top matches:")
for name, score in results:
    print(f"{name} -> {score:.4f}")

# -------------------------------
# Visualization of sketch + top matches
sketch = Image.open(sketch_path)
plt.figure(figsize=(15, 4))

# Show input sketch
plt.subplot(1, 6, 1)
plt.imshow(sketch)
plt.title("Input Sketch")
plt.axis('off')

# Show top 5 color matches
for i, (photo_name, score) in enumerate(results, start=2):
    # Update this path to point to your folder of original colored photos
    photo = Image.open(f"dataset/Photos/{photo_name}").convert("RGB")
    plt.subplot(1, 6, i)
    plt.imshow(photo)
    plt.title(f"{photo_name}\n{score:.2f}")
    plt.axis('off')

plt.show()
