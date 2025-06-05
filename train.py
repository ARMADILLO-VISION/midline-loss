import os
import torch
import argparse
import warnings
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

from config import *
from line_aware_loss import CombinedLoss
from pointnet_data_loader import get_dataloaders
from models.pointnet2_model import PointNet2Seg
from models.pointnet_model import PointNetSeg
from torch.utils.data import DataLoader
from early_stopping import EarlyStopping
from utils.per_epoch import process_epoch

warnings.filterwarnings("ignore", message=".*CuDNN.*")

def save_model(model, name="final"):
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    save_path = os.path.join(SAVE_DIR, f"pointnet_{name}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def compute_class_weights(train_loader, num_classes=3):
    # Inverse Square Root Class Frequency Weighting.
    train_labels = []
    for _, labels in train_loader:
        train_labels.extend(labels.cpu().numpy().flatten())
    
    class_counts = np.bincount(train_labels, minlength=num_classes).astype(np.float32)
    class_weights = 1.0 / (np.sqrt(class_counts) + 1e-6)
    class_weights /= class_weights.sum()
    return torch.tensor(class_weights, dtype=torch.float32)

def compute_manual_class_weights(weights=CLASS_WEIGHTS):
    weights = np.array(weights, dtype=np.float32)
    normalized_weights = weights / (weights.sum() + 1e-6)
    return torch.tensor(normalized_weights, dtype=torch.float32)

def train_model(alpha_ce, lam_ridge, lam_lig, learning_rate=LEARNING_RATE):
    train_loader, val_loader, test_loader = get_dataloaders(
        data_split_dir=NPZ_PATH,
        num_points=NUM_POINTS,
        batch_size=BATCH_SIZE,
        normalise=True,
        visualise=False,
        augment=True
    )

    class_weights = compute_class_weights(train_loader, num_classes=3)
    print(f"Class Weights: {class_weights}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PointNet2Seg(num_classes=3).to(device)
    
    loss_fn = CombinedLoss(
        class_weights=class_weights,  # Class weighting
        alpha_ce=alpha_ce,            # Weight for the cross-entropy loss
        lam_ridge=lam_ridge,          # Scaling factor for the ridge
        lam_lig=lam_lig,              # Scaling factor for the ligament
        lam_nll=0.0,                  # Scaling factor for the NLL loss term
        lam_midline=0.5,              # Controls the trade-off between thickness loss and midline alignment loss.
        ridge_class_idx=1,            # Index of the ridge class
        lig_class_idx=2,              # Index of the ligament class
        lambda_thickness=5.0,         # Lambda parameter controlling the sharpness of the soft snapping in the thickness loss.
        visualise=False,
    ).to(device)
    

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    train_losses = []
    val_losses = []

    early_stopping = EarlyStopping(patience=100)
    
    fixed_train_sample = None
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0

        for b, (points, labels) in enumerate(train_loader):
            points, labels = points.to(device), labels.to(device)

            if fixed_train_sample is None:
                fixed_train_sample = (points.clone(), labels.clone())

            optimizer.zero_grad()
            seg_logits, _ = model(points)
            loss = loss_fn(seg_logits, labels.long(), points)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        print(f"[Epoch {epoch+1}/{EPOCHS}]\n  Train Loss: {avg_train_loss:.4f}")

        with torch.no_grad():
            pts_fixed, _     = fixed_train_sample
            pts_fixed_device = pts_fixed.to(device)
            seg_logits_fs, _ = model(pts_fixed_device)
            preds_fs         = torch.argmax(seg_logits_fs, dim=-1).cpu().numpy()[0]
            pts_fs_np        = pts_fixed_device.cpu().numpy()[0]

            process_epoch(
                epoch=epoch,
                points=pts_fs_np,
                preds=preds_fs,
                base_dir="visualisations/train",
                visualise=False
            )

        Validation.
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for points, labels in val_loader:
                points, labels = points.to(device), labels.to(device)
                seg_logits, _ = model(points)
                val_loss = loss_fn(seg_logits, labels, points)
                # val_loss = loss_fn(seg_logits, labels)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        print(f"    Val Loss: {avg_val_loss:.4f}")
        
        if early_stopping.check(avg_val_loss):
            break

    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss", color='blue')
    plt.plot(val_losses, label="Val Loss", color='red')
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_plot.png")
    plt.show()

    print("\n===== Final Results =====")
    print(f"FINAL_TRAIN_LOSS: {avg_train_loss:.4f}")
    print(f"FINAL_VAL_LOSS: {avg_val_loss:.4f}")

    save_model(model, name="final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha_ce", type=float, default=0.0, help="Weight for cross-entropy loss")
    parser.add_argument("--lam_ridge", type=float, default=0.0, help="Weight for ridge Tversky loss")
    parser.add_argument("--lam_lig", type=float, default=0.0, help="Weight for ligament Tversky loss")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate from config")
    args = parser.parse_args()

    learning_rate = args.lr if args.lr is not None else LEARNING_RATE
    train_model(alpha_ce=args.alpha_ce, lam_ridge=args.lam_ridge, lam_lig=args.lam_lig, learning_rate=learning_rate)
