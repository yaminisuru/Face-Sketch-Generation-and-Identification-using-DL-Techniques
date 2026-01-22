import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import TripletNet

device = "cuda" if torch.cuda.is_available() else "cpu"

model = TripletNet()
model.load_state_dict(torch.load("cnn_triplet_cufs.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

photo_dir = "processed_data/test/photos"
embeddings = {}

with torch.no_grad():
    for img in os.listdir(photo_dir):
        path = os.path.join(photo_dir, img)
        image = Image.open(path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        emb = model(image).cpu().numpy()
        embeddings[img] = emb

np.save("photo_embeddings.npy", embeddings)
print("Embeddings saved successfully")
