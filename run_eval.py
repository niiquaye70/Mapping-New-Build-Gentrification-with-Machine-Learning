import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# --- Local imports from src ---
from src.data.dataset import SiameseGentrificationDataset
from src.data.transforms import val_transform
from src.models.siamese_resnet50 import SiameseResNet50
from src.utils.seed import set_seed
from src.utils.logging_config import get_logger

# -------------------------
# Configuration
# -------------------------
TEST_CSV   = "data/test_set.csv"      # test split CSV
IMAGE_DIR  = "images/"                # path to image folder
BATCH_SIZE = 16
CKPT_PATH  = "outputs/checkpoints/best_siamese_model.pth"

# -------------------------
# Main script
# -------------------------
def main():
    set_seed(42)
    logger = get_logger("outputs/logs/eval.log")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test dataset
    test_dataset = SiameseGentrificationDataset(TEST_CSV, IMAGE_DIR, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    model = SiameseResNet50(pretrained=False).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    # Collect predictions
    all_labels, all_probs = [], []
    with torch.no_grad():
        for img1, img2, labels in test_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            outputs = model(img1, img2).squeeze()
            all_probs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # -------------------------
    # Find best threshold (F1)
    # -------------------------
    thresholds = np.linspace(0, 1, 101)
    f1_scores = []
    for t in thresholds:
        preds_t = (all_probs >= t).astype(int)
        f1_scores.append(f1_score(all_labels, preds_t, zero_division=0))

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    logger.info(f"Best threshold (F1): {best_threshold:.2f} | F1={f1_scores[best_idx]:.4f}")

    # -------------------------
    # Final metrics @ best threshold
    # -------------------------
    final_preds = (all_probs >= best_threshold).astype(int)

    acc  = accuracy_score(all_labels, final_preds)
    prec = precision_score(all_labels, final_preds, zero_division=0)
    rec  = recall_score(all_labels, final_preds, zero_division=0)
    f1   = f1_score(all_labels, final_preds, zero_division=0)
    auc  = roc_auc_score(all_labels, all_probs)

    logger.info(f"Test Accuracy : {acc:.4f}")
    logger.info(f"Test Precision: {prec:.4f}")
    logger.info(f"Test Recall   : {rec:.4f}")
    logger.info(f"Test F1       : {f1:.4f}")
    logger.info(f"Test AUC      : {auc:.4f}")

    print("=== Test Results ===")
    print(f"Best Threshold: {best_threshold:.2f}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")

if __name__ == "__main__":
    main()
