import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

# --- Local imports from src ---
from src.data.dataset import SiameseGentrificationDataset
from src.data.transforms import train_transform, val_transform
from src.models.siamese_resnet50 import SiameseResNet50
from src.training.train import train
from src.training.evaluate import evaluate
from src.utils.seed import set_seed
from src.utils.logging_config import get_logger

# -------------------------
# Configuration
# -------------------------
CSV_FILE   = "train_pairs.csv"       # your full dataset CSV
IMAGE_DIR  = "images/"         # path to image folder
BATCH_SIZE = 16
EPOCHS     = 15
LR         = 2.75e-4
WEIGHT_DECAY = 0.006

# -------------------------
# Main script
# -------------------------
def main():
    # Fix randomness
    set_seed(42)

    # Ensure output dirs exist
    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)

    # Logger
    logger = get_logger("outputs/logs/train.log")

    # Load CSV
    df = pd.read_csv(CSV_FILE)

    # Split into train/val
    train_df, val_df = train_test_split(
        df, test_size=0.3, stratify=df['Label'], random_state=42
    )
    os.makedirs("data/splits", exist_ok=True)
    train_df.to_csv("data/splits/train_set.csv", index=False)
    val_df.to_csv("data/splits/val_set.csv", index=False)

    # Datasets
    train_dataset = SiameseGentrificationDataset("data/splits/train_set.csv", IMAGE_DIR, transform=train_transform)
    val_dataset   = SiameseGentrificationDataset("data/splits/val_set.csv", IMAGE_DIR, transform=val_transform)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseResNet50(pretrained=True).to(device)

    # Loss + Optimizer + Scheduler
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2)

    # Training
    best_acc = train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, logger, epochs=EPOCHS)

    # Save final model
    final_ckpt = "outputs/checkpoints/final_siamese_model.pth"
    torch.save(model.state_dict(), final_ckpt)
    logger.info(f"Training finished. Best Val Acc={best_acc:.4f}. Final model saved → {final_ckpt}")

if __name__ == "__main__":
    main()
