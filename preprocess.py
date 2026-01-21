import os
import cv2

PHOTO_DIR = "C:\\Users\\durga\\OneDrive\\Desktop\\face_sketch_project\\dataset\\photos"
SKETCH_DIR = "C:\\Users\\durga\\OneDrive\\Desktop\\face_sketch_project\\dataset\\sketches"

OUT_PHOTO_DIR = "C:\\Users\\durga\\OneDrive\\Desktop\\face_sketch_project\\processed_data\\photos"
OUT_SKETCH_DIR = "C:\\Users\\durga\\OneDrive\\Desktop\\face_sketch_project\\processed_data\\sketches"

os.makedirs(OUT_PHOTO_DIR, exist_ok=True)
os.makedirs(OUT_SKETCH_DIR, exist_ok=True)

IMG_SIZE = 256

def process_images(input_dir, output_dir):
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(os.path.join(output_dir, img_name), img)

process_images(PHOTO_DIR, OUT_PHOTO_DIR)
process_images(SKETCH_DIR, OUT_SKETCH_DIR)

print("Preprocessing completed successfully!")
