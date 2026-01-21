import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CUFSSketchPhotoDataset(Dataset):
    def __init__(self, base_path):
        self.photo_dir = os.path.join(base_path, "photos")
        self.sketch_dir = os.path.join(base_path, "sketches")
        self.aug_dir = os.path.join(base_path, "aug_sketches")

        self.ids = os.listdir(self.sketch_dir)
        self.augmented = os.listdir(self.aug_dir)

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.ids) * 2

    def __getitem__(self, idx):
        same = idx % 2
        base_id = self.ids[idx // 2]

        # randomly choose original or augmented sketch
        if random.random() < 0.5:
            sketch_path = os.path.join(self.sketch_dir, base_id)
        else:
            aug = random.choice(self.augmented)
            sketch_path = os.path.join(self.aug_dir, aug)

        sketch = Image.open(sketch_path).convert("RGB")

        if same:
            photo = Image.open(os.path.join(self.photo_dir, base_id)).convert("RGB")
            label = 1
        else:
            neg_id = random.choice([x for x in self.ids if x != base_id])
            photo = Image.open(os.path.join(self.photo_dir, neg_id)).convert("RGB")
            label = 0

        return (
            self.transform(sketch),
            self.transform(photo),
            label
        )
