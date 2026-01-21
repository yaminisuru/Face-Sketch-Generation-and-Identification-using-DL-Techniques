import torch
from torch.utils.data import DataLoader
from model import SiameseNetwork
from dataset import CUFSSketchPhotoDataset
from loss import ContrastiveLoss

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = CUFSSketchPhotoDataset("processed_data/train")
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = SiameseNetwork().to(device)
criterion = ContrastiveLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(25):
    total_loss = 0
    for sketch, photo, label in loader:
        sketch, photo = sketch.to(device), photo.to(device)
        label = label.float().to(device)

        out1, out2 = model(sketch, photo)
        loss = criterion(out1, out2, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "siamese_model.pth")
