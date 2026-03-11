import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import TripletNet
from sklearn.metrics.pairwise import cosine_similarity
import sys
import json
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# Get script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "cnn_triplet_cufs.pth")
embedding_path = os.path.join(BASE_DIR, "photo_embeddings.npy")

# Load model
model = TripletNet()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Load embeddings
photo_embeddings = np.load(embedding_path, allow_pickle=True).item()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def identify(sketch_path, top_k=5):

    sketch = Image.open(sketch_path).convert("RGB")
    sketch_input = transform(sketch).unsqueeze(0).to(device)

    with torch.no_grad():
        sketch_emb = model(sketch_input).cpu().numpy()

    results = []

    for name, emb in photo_embeddings.items():

        sim = cosine_similarity(sketch_emb, emb)[0][0]

        results.append({
            "name": name,
            "score": float(sim)
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return results[:top_k]


if __name__ == "__main__":

    sketch_path = sys.argv[1]

    matches = identify(sketch_path)

    # 🔹 Print nicely in console
    print("\nTop 5 Matches:\n")

    for i, m in enumerate(matches):
        print(f"{i+1}. {m['name']}  |  Similarity: {m['score']:.4f}")

    # 🔹 JSON output for backend/frontend
    print("\nJSON Output:\n")
    print(json.dumps(matches))