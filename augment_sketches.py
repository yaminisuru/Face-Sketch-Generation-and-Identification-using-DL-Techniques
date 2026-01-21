import cv2
import os
import numpy as np

SKETCH_DIR = "processed_data/train/sketches"
AUG_DIR = "processed_data/train_aug/sketches"

os.makedirs(AUG_DIR, exist_ok=True)

def augment(img):
    augmented = []

    # Original
    augmented.append(img)

    # Flip
    augmented.append(cv2.flip(img, 1))

    # Rotate
    rows, cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 10, 1)
    augmented.append(cv2.warpAffine(img, M, (cols, rows)))

    # Noise
    noise = np.random.normal(0, 10, img.shape)
    noisy_img = np.clip(img + noise, 0, 255).astype(np.uint8)
    augmented.append(noisy_img)

    return augmented

for file in os.listdir(SKETCH_DIR):
    img = cv2.imread(os.path.join(SKETCH_DIR, file), 0)
    if img is None:
        continue

    augmented_images = augment(img)

    for i, aug_img in enumerate(augmented_images):
        cv2.imwrite(os.path.join(AUG_DIR, f"{file[:-4]}_aug{i}.jpg"), aug_img)

print("Sketch augmentation completed!")
