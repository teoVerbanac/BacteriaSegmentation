import os
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import jaccard_score, f1_score
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects, binary_closing, disk
from scipy import ndimage as ndi

def apply_clahe_gray(image, clipLimit=2.0, tileGridSize=(8,8)):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(gray)

def get_bacteria_mask(mask_img):
    if mask_img is None:
        return None
    if len(mask_img.shape) == 3:
        gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = mask_img
    _, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return (th > 0).astype(np.uint8)

def watershed_segmentation_best(image, min_size=200):
    gray = apply_clahe_gray(image)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 35, -5)
    kernel = np.ones((3,3), np.uint8)
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)
    dist = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
    _, fg = cv2.threshold(dist, 0.4 * dist.max(), 1, 0)
    fg = fg.astype(bool)
    bg = cv2.dilate(opened, kernel, iterations=3)
    bg = bg.astype(bool)
    markers, _ = ndi.label(fg)
    edges = cv2.Canny(gray, 50, 150)
    edges = edges > 0
    labels = watershed(-dist, markers, mask=bg & ~edges)
    mask = labels > 0
    mask = binary_closing(mask, disk(3))
    mask = remove_small_objects(mask, min_size=min_size)
    mask = ndi.binary_fill_holes(mask)
    return mask.astype(np.uint8)

def compute_metrics(gt, pred):
    gt_flat = gt.flatten().astype(np.int32)
    pred_flat = pred.flatten().astype(np.int32)
    gt_sum = int(gt_flat.sum())
    pred_sum = int(pred_flat.sum())
    if gt_sum == 0 and pred_sum == 0:
        return 1.0, 1.0, gt_sum, pred_sum
    if gt_sum == 0 or pred_sum == 0:
        return 0.0, 0.0, gt_sum, pred_sum
    iou = jaccard_score(gt_flat, pred_flat)
    dice = f1_score(gt_flat, pred_flat)
    return iou, dice, gt_sum, pred_sum

def main(images_dir="JPEGImages", masks_dir="SegmentationClass",
         out_csv="results_watershed_best.csv", min_size=200):
    files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    results = []
    for filename in files:
        img_path = os.path.join(images_dir, filename)
        mask_path = os.path.join(masks_dir, filename)
        image = cv2.imread(img_path)
        mask_img = cv2.imread(mask_path)
        if image is None or mask_img is None:
            continue
        gt_mask = get_bacteria_mask(mask_img)
        if gt_mask is None:
            continue
        pred_ws = watershed_segmentation_best(image, min_size=min_size)
        iou, dice, gt_area, pred_area = compute_metrics(gt_mask, pred_ws)
        results.append([filename, "watershed_best", float(dice), float(iou), int(gt_area), int(pred_area)])
    df = pd.DataFrame(results, columns=["image", "method", "dice", "iou", "gt_area", "pred_area"])
    df.to_csv(out_csv, index=False)

if __name__ == "__main__":
    main()
