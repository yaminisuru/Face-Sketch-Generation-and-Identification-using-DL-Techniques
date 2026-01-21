import os
import shutil
import random

BASE_DIR = "C:\\Users\\durga\\OneDrive\\Desktop\\face_sketch_project\\processed_data"
PHOTO_DIR = os.path.join(BASE_DIR, "photos")
SKETCH_DIR = os.path.join(BASE_DIR, "sketches")

TRAIN_PHOTO_DIR = "C:\\Users\\durga\\OneDrive\\Desktop\\face_sketch_project\\processed_data\\train\\photos"
TRAIN_SKETCH_DIR = "C:\\Users\\durga\\OneDrive\\Desktop\\face_sketch_project\\processed_data\\train\\sketches"
TEST_PHOTO_DIR = "C:\\Users\\durga\\OneDrive\\Desktop\\face_sketch_project\\processed_data\\test\\photos"
TEST_SKETCH_DIR = "C:\\Users\\durga\\OneDrive\\Desktop\\face_sketch_project\\processed_data\\test\\sketches"

os.makedirs(TRAIN_PHOTO_DIR, exist_ok=True)
os.makedirs(TRAIN_SKETCH_DIR, exist_ok=True)
os.makedirs(TEST_PHOTO_DIR, exist_ok=True)
os.makedirs(TEST_SKETCH_DIR, exist_ok=True)

files = sorted(os.listdir(PHOTO_DIR))
random.shuffle(files)

split_idx = int(0.8 * len(files))
train_files = files[:split_idx]
test_files = files[split_idx:]

def copy_files(file_list, src_photo, src_sketch, dst_photo, dst_sketch):
    for f in file_list:
        shutil.copy(os.path.join(src_photo, f), os.path.join(dst_photo, f))
        shutil.copy(os.path.join(src_sketch, f), os.path.join(dst_sketch, f))

copy_files(train_files, PHOTO_DIR, SKETCH_DIR, TRAIN_PHOTO_DIR, TRAIN_SKETCH_DIR)
copy_files(test_files, PHOTO_DIR, SKETCH_DIR, TEST_PHOTO_DIR, TEST_SKETCH_DIR)

print("Dataset split completed!")
