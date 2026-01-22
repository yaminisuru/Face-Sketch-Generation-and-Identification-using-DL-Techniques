# dataset.py
import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class TripletDataset(Dataset):
    def __init__(self, root, mode="train"):
        self.sketch_dir = os.path.join(root, "sketches")
        self.photo_dir = os.path.join(root, "photos")

        self.sketches = sorted(os.listdir(self.sketch_dir))
        self.photos = sorted(os.listdir(self.photo_dir))

        # Split into train/test
        split_idx = int(0.8 * len(self.sketches))
        if mode == "train":
            self.sketches = self.sketches[:split_idx]
        else:
            self.sketches = self.sketches[split_idx:]

        # Transforms
        self.sketch_transform = T.Compose([
            T.Resize((224, 224)),
            T.RandomRotation(10),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        self.photo_transform = T.Compose([
            T.Resize((224, 224)),
            T.ColorJitter(0.1,0.1,0.1,0.1),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def __len__(self):
        return len(self.sketches)

    def __getitem__(self, idx):
        anchor_name = self.sketches[idx]

        # Anchor sketch
        anchor = Image.open(os.path.join(self.sketch_dir, anchor_name)).convert("RGB")
        anchor = self.sketch_transform(anchor)

        # Positive photo
        positive = Image.open(os.path.join(self.photo_dir, anchor_name)).convert("RGB")
        positive = self.photo_transform(positive)

        # Negative photo (random for initial batch)
        neg_name = random.choice(self.photos)
        while neg_name == anchor_name:
            neg_name = random.choice(self.photos)
        negative = Image.open(os.path.join(self.photo_dir, neg_name)).convert("RGB")
        negative = self.photo_transform(negative)

        return anchor, positive, negative
