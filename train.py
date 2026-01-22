# train.py
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from dataset import TripletDataset
from model import TripletNet
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset
train_dataset = TripletDataset("processed_data/train", mode="train")
val_dataset = TripletDataset("processed_data/train", mode="val")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Model
model = TripletNet().to(device)
criterion = nn.TripletMarginLoss(margin=1.0)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 40

# Validation accuracy
def compute_accuracy():
    model.eval()
    photo_dir = "processed_data/train/photos"
    sketch_dir = "processed_data/train/sketches"

    photos = sorted(os.listdir(photo_dir))
    sketches = sorted(os.listdir(sketch_dir))
    sketches = sketches[int(0.8 * len(sketches)):]  # val set

    from PIL import Image
    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    photo_embs = []
    with torch.no_grad():
        for p in photos:
            img = Image.open(os.path.join(photo_dir, p)).convert("RGB")
            emb = model(transform(img).unsqueeze(0).to(device))
            photo_embs.append(emb.squeeze(0))
    photo_embs = torch.stack(photo_embs)
    photo_embs = photo_embs / photo_embs.norm(dim=1, keepdim=True)

    top1, top5 = 0, 0
    with torch.no_grad():
        for s in sketches:
            img = Image.open(os.path.join(sketch_dir, s)).convert("RGB")
            emb = model(transform(img).unsqueeze(0).to(device))
            emb = emb / emb.norm(dim=1, keepdim=True)

            sim = torch.matmul(emb, photo_embs.T)
            topk = torch.topk(sim, k=5).indices[0].cpu().numpy()
            topk_names = [photos[i] for i in topk]

            if s == topk_names[0]:
                top1 += 1
            if s in topk_names:
                top5 += 1

    return top1/len(sketches), top5/len(sketches)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for anchor, positive, negative in train_loader:
        anchor, positive, negative = (
            anchor.to(device),
            positive.to(device),
            negative.to(device)
        )

        emb_a = model(anchor)
        emb_p = model(positive)
        emb_n = model(negative)

        loss = criterion(emb_a, emb_p, emb_n)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    top1, top5 = compute_accuracy()

    print(
        f"Epoch {epoch+1:02d} | "
        f"Loss: {avg_loss:.4f} | "
        f"Top-1: {top1*100:.2f}% | "
        f"Top-5: {top5*100:.2f}%"
    )

torch.save(model.state_dict(), "cnn_triplet_cufs.pth")
print("Training completed.")
