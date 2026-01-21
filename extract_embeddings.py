import os
import torch
import numpy as np
from PIL import Image
from model import SiameseNetwork
import torchvision.transforms as transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SiameseNetwork().to(device)
model.load_state_dict(torch.load("siamese_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

photo_dir = "processed_data/train/photos"
embeddings = {}

with torch.no_grad():
    for img in os.listdir(photo_dir):
        image = Image.open(os.path.join(photo_dir, img)).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        emb = model.forward_once(tensor).cpu().numpy()
        embeddings[img] = emb

np.save("photo_embeddings.npy", embeddings)
