import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from sam2.build_sam import build_sam2
from peft import LoraConfig, get_peft_model, PeftModel
import torch.nn.functional as F

class ColoniesDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_size=1024):
        self.images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg','.png'))]
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        fname = self.images[idx]
        img_path = os.path.join(self.images_dir, fname)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = torch.from_numpy(img).permute(2,0,1).float() / 255.0
        base = os.path.splitext(fname)[0]
        mask_path = os.path.join(self.masks_dir, base + ".png")
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.float32)
            mask = torch.from_numpy(mask).unsqueeze(0)
        else:
            mask = torch.zeros((1, self.img_size, self.img_size), dtype=torch.float32)
        return img, mask

def dice_score(y_true, y_pred, eps=1e-6):
    intersection = (y_true * y_pred).sum()
    return (2.0 * intersection) / (y_true.sum() + y_pred.sum() + eps)

def iou_score(y_true, y_pred, eps=1e-6):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return intersection / (union + eps)

def soft_dice_loss(pred_probs, targets, eps=1e-6):
    num = 2.0 * (pred_probs * targets).sum(dim=(1,2,3))
    den = pred_probs.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) + eps
    loss_per_sample = 1.0 - (num / den)
    return loss_per_sample.mean()

device = "cuda" if torch.cuda.is_available() else "cpu"

images_dir = "datasets/train/images"
masks_dir  = "datasets/train/masks"
dataset = ColoniesDataset(images_dir, masks_dir, img_size=1024)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

val_images_dir = "datasets/val/images"
val_masks_dir  = "datasets/val/masks"
val_dataset = ColoniesDataset(val_images_dir, val_masks_dir, img_size=1024)
val_loader  = DataLoader(val_dataset, batch_size=2, shuffle=False)

config_yaml = "model_weights/sam2_hiera_l.yaml"
ckpt_path   ="SAM2/sam2/model_weights/sam2_hiera_large.pt"
sam_model = build_sam2(config_yaml, ckpt_path, device=device, mode="eval")

peft_config = LoraConfig(
    task_type="FEATURE_EXTRACTION", inference_mode=False,
    r=24, lora_alpha=96, lora_dropout=0.1,
    target_modules=["q_proj","k_proj","v_proj","out_proj","mlp.fc1","mlp.fc2"]  #ovdje bi treblo malo istrayit
)
model = get_peft_model(sam_model, peft_config)
model.to(device)
model.train()

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)

for epoch in range(50):
    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()

        backbone_out = model.model.forward_image(imgs)
        backbone_out, vision_feats, vision_pos, feat_sizes = model.model._prepare_backbone_features(backbone_out)
        fpn_feats = backbone_out["backbone_fpn"]
        image_embeddings = fpn_feats[-1]
        high_res_feats = [fpn_feats[i] for i in range(len(fpn_feats)-1)]
        B, _, H, W = imgs.shape
        box = torch.tensor([[0.0, 0.0, float(W), float(H)]] * imgs.shape[0], dtype=torch.float32).to(device)
        sparse_emb, dense_emb = model.model.sam_prompt_encoder(points=None, boxes=box, masks=None)
        outputs = model.model.sam_mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=model.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
            repeat_image=False,
            high_res_features=[feat for feat in high_res_feats]
        )

        mask_logits = None
        if isinstance(outputs, torch.Tensor):
            mask_logits = outputs
        elif isinstance(outputs, (list, tuple)):
            for item in outputs:
                if isinstance(item, torch.Tensor):
                    mask_logits = item
                    break
                if isinstance(item, (list, tuple)):
                    for sub in item:
                        if isinstance(sub, torch.Tensor):
                            mask_logits = sub
                            break
                    if mask_logits is not None:
                        break

        if mask_logits.ndim == 2:
            mask_logits = mask_logits.unsqueeze(0).unsqueeze(0)
        elif mask_logits.ndim == 3:
            if mask_logits.shape[0] == imgs.shape[0]:
                mask_logits = mask_logits.unsqueeze(1)
            else:
                mask_logits = mask_logits.unsqueeze(0)

        if mask_logits.shape[-2:] != masks.shape[-2:]:
            mask_logits = F.interpolate(mask_logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

        lambda_dice = 1.5
        bce_loss = criterion(mask_logits, masks)
        probs = torch.sigmoid(mask_logits)
        dice_loss = soft_dice_loss(probs, masks)
        loss = bce_loss + lambda_dice * dice_loss
        loss.backward()
        optimizer.step()
        del backbone_out, vision_feats, image_embeddings, high_res_feats, sparse_emb, dense_emb, outputs, mask_logits
        torch.cuda.empty_cache()

output_dir = "sam2_lora_adapter"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
