import torch
import numpy as np
from PIL import Image
from model import SiameseNetwork
import torchvision.transforms as transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SiameseNetwork().to(device)
model.load_state_dict(torch.load("siamese_model.pth"))
model.eval()

photo_embeddings = np.load("photo_embeddings.npy", allow_pickle=True).item()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def identify(sketch_path):
    sketch = Image.open(sketch_path).convert("RGB")
    sketch = transform(sketch).unsqueeze(0).to(device)

    with torch.no_grad():
        sketch_emb = model.forward_once(sketch).cpu().numpy()

    min_dist = float("inf")
    best_match = None

    for name, emb in photo_embeddings.items():
        dist = np.linalg.norm(sketch_emb - emb)
        if dist < min_dist:
            min_dist = dist
            best_match = name

    print("Matched Photo:", best_match)
    print("Similarity Distance:", min_dist)

identify("processed_data/test/sketches/f-025-01.jpg")
