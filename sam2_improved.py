import os
import cv2
import torch
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from peft import PeftModel

def dice_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-6)

def iou_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return intersection / (union + 1e-6)

def run_sam2_inference(images_dir="JPEGImages", masks_dir="SegmentationClass", output_dir="output_masks"):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = r"D:\Users\tverbanac\imagesComplete\SAM2\sam2\model_weights\sam2_hiera_l.yaml"
    ckpt_path  = r"D:\Users\tverbanac\imagesComplete\SAM2\sam2\model_weights\sam2_hiera_large.pt"
    model = build_sam2(model_name, ckpt_path, device=device)
    model = PeftModel.from_pretrained(model, "sam2_lora_adapter")
    model.to(device)
    predictor = SAM2ImagePredictor(model)

    all_results = []

    for filename in os.listdir(images_dir):
        if not (filename.endswith(".jpg") or filename.endswith(".png")):
            continue

        image_path = os.path.join(images_dir, filename)
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)

        h, w, _ = image.shape
        input_box = [0, 0, w, h]

        with torch.no_grad():
            masks, scores, _ = predictor.predict(box=input_box, multimask_output=True)
        mask = np.any(masks, axis=0).astype(np.uint8)
        mask = (mask > 0.5).astype(np.uint8)

        mask_path = os.path.join(output_dir, filename.replace(".jpg", "_mask.png"))
        cv2.imwrite(mask_path, mask * 255)

        gt_path = os.path.join(masks_dir, filename.replace(".jpg", ".png"))
        if os.path.exists(gt_path):
            gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            gt_mask = cv2.resize(gt_mask, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            gt_mask = (gt_mask > 127).astype(np.uint8)
            dice = dice_score(gt_mask, mask)
            iou = iou_score(gt_mask, mask)
            all_results.append((filename, dice, iou))

    results_path = os.path.join(output_dir, "results_summary.txt")
    with open(results_path, "w") as f:
        f.write("Filename,Dice,IoU\n")
        for filename, dice, iou in all_results:
            f.write(f"{filename},{dice:.4f},{iou:.4f}\n")

if __name__ == "__main__":
    run_sam2_inference()
