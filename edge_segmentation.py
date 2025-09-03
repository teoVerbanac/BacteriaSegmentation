import os
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import jaccard_score, f1_score

#CLAHE
def apply_clahe_bgr(image, clipLimit=2.0, tileGridSize=(8,8)):
    if image is None:
        return None
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    cl = clahe.apply(l)
    lab_cl = cv2.merge((cl, a, b))
    return cv2.cvtColor(lab_cl, cv2.COLOR_LAB2BGR)

def get_bacteria_mask(mask_img):
    if mask_img is None:
        return None
    if len(mask_img.shape) == 2 or (len(mask_img.shape) == 3 and mask_img.shape[2] == 1):
        gray = mask_img if len(mask_img.shape) == 2 else mask_img[:, :, 0]
    else:
        gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return (th > 0).astype(np.uint8)

def edge_based_segmentation_simple(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 25, 120)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours_info = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
    filled = np.zeros_like(mask)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, thickness=-1)
    return (filled > 0).astype(np.uint8)

def filter_small_objects(mask, min_size=200):
    mask_u8 = (mask > 0).astype(np.uint8)
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if nb_components <= 1:
        return mask_u8
    sizes = stats[1:, -1]
    filtered = np.zeros(mask.shape, dtype=np.uint8)
    for i, sz in enumerate(sizes):
        if sz >= min_size:
            filtered[output == i + 1] = 1
    return filtered

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

def main(images_dir="JPEGImages", masks_dir="SegmentationClass", out_csv="results_edge.csv", min_size=200):
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
        image_clahe = apply_clahe_bgr(image)
        pred_edge = edge_based_segmentation_simple(image_clahe)
        pred_edge = filter_small_objects(pred_edge, min_size=min_size)
        iou, dice, gt_area, pred_area = compute_metrics(gt_mask, pred_edge)
        results.append([filename, "edge_clahe", float(dice), float(iou), int(gt_area), int(pred_area)])
    pd.DataFrame(results, columns=["image", "method", "dice", "iou", "gt_area", "pred_area"]).to_csv(out_csv, index=False)

if __name__ == "__main__":
    main()
