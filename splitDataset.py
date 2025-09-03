import os
import cv2
from sklearn.model_selection import train_test_split

SRC_IMAGES = "JPEGImages"
SRC_MASKS  = "SegmentationClass"
DST_ROOT   = "datasets"
IMG_SIZE   = (256, 256)
VAL_SPLIT  = 0.1

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def resize_and_save(src_path, dst_path, size, is_mask=False):
    img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    resized = cv2.resize(img, size, interpolation=interp)
    cv2.imwrite(dst_path, resized)

def preprocess_dataset():
    image_files = [f for f in os.listdir(SRC_IMAGES) if f.lower().endswith((".jpg", ".png"))]
    mask_files  = [f for f in os.listdir(SRC_MASKS) if f.lower().endswith(".png")]
    paired = []
    for img in image_files:
        base = os.path.splitext(img)[0]
        mask_name = f"{base}.png"
        if mask_name in mask_files:
            paired.append((img, mask_name))
    train_pairs, val_pairs = train_test_split(paired, test_size=VAL_SPLIT, random_state=42)
    for split, pairs in [("train", train_pairs), ("val", val_pairs)]:
        img_out = os.path.join(DST_ROOT, split, "images")
        mask_out = os.path.join(DST_ROOT, split, "masks")
        ensure_dir(img_out)
        ensure_dir(mask_out)
        for img_name, mask_name in pairs:
            src_img  = os.path.join(SRC_IMAGES, img_name)
            src_mask = os.path.join(SRC_MASKS, mask_name)
            dst_img  = os.path.join(img_out, img_name)
            dst_mask = os.path.join(mask_out, mask_name)
            resize_and_save(src_img, dst_img, IMG_SIZE, is_mask=False)
            resize_and_save(src_mask, dst_mask, IMG_SIZE, is_mask=True)

if __name__ == "__main__":
    preprocess_dataset()
