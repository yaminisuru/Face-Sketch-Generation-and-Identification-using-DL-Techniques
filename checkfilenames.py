import os

PHOTO_DIR = "C:\\Users\\durga\\OneDrive\\Desktop\\face_sketch_project\\dataset\\photos"
SKETCH_DIR = "C:\\Users\\durga\\OneDrive\\Desktop\\face_sketch_project\\dataset\\sketches"

photo_files = set(os.listdir(PHOTO_DIR))
sketch_files = set(os.listdir(SKETCH_DIR))

# Files present in photos but missing in sketches
missing_sketches = photo_files - sketch_files

# Files present in sketches but missing in photos
missing_photos = sketch_files - photo_files

print("Total photos:", len(photo_files))
print("Total sketches:", len(sketch_files))

if not missing_sketches and not missing_photos:
    print(" All photo and sketch filenames are perfectly matched!")
else:
    print("\n Mismatched files found:\n")

    if missing_sketches:
        print("Photos without sketches:")
        for f in sorted(missing_sketches):
            print("  ", f)

    if missing_photos:
        print("\nSketches without photos:")
        for f in sorted(missing_photos):
            print("  ", f)
