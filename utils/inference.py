import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import point_cloud_utils as pcu

import matplotlib.pyplot as plt

from config import (
    NPZ_PATH, NUM_POINTS, BATCH_SIZE, CLASS_NAMES, CLASS_COLORS, 
    SAVE_DIR, NUM_CLASSES
)
from pointnet_data_loader import get_dataloaders
from mpl_toolkits.mplot3d import Axes3D
from models.pointnet2_model import PointNet2Seg
from models.pointnet_model import PointNetSeg

def visualise_predictions(model, data_loader):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = data_loader.dataset

    points, labels = dataset[0]
    if points.ndim == 2:
        points = points.unsqueeze(0)
        labels = labels.unsqueeze(0)
    points, labels = points.to(device), labels.to(device)
    
    with torch.no_grad():
        seg_logits, _ = model(points)
        if hasattr(model, "logits_adapter"):
            seg_logits = model.logits_adapter(seg_logits)
        if hasattr(model, "geometry_adapter"):
            seg_logits = model.geometry_adapter(seg_logits, points)
        refined_probs = F.softmax(seg_logits, dim=-1)
        _, predictions = torch.max(refined_probs, dim=2)
    
    points_np = points.cpu().numpy()[0]
    # rotate points for a better visual perspective
    points_np = rotate_points(points_np, angle_degrees=0, axis='y')
    labels_np = labels.cpu().numpy()[0]
    preds_np = predictions.cpu().numpy()[0]
    
    # Define the default view parameters
    elev_default = -8
    azim_default = 60

    # # ===== Ground Truth Figure =====
    # fig_gt = plt.figure(figsize=(8, 8))
    # fig_gt.canvas.manager.set_window_title("Ground Truth")
    # ax_gt = fig_gt.add_subplot(111, projection="3d")
    # ax_gt.axis("off")
    # ax_gt.scatter(*points_np[labels_np == 0].T, color=CLASS_COLORS[0], s=1)
    # ax_gt.scatter(*points_np[labels_np == 1].T, color=CLASS_COLORS[1], s=10)
    # ax_gt.scatter(*points_np[labels_np == 2].T, color=CLASS_COLORS[2], s=10)
    # ax_gt.view_init(elev=elev_default, azim=azim_default)

    # # Dynamic update for ground truth window title
    # def update_gt_view_info(event):
    #     if event.inaxes == ax_gt:
    #         elev = ax_gt.elev
    #         azim = ax_gt.azim
    #         title = f"Ground Truth | Elev: {elev:.1f}, Azim: {azim:.1f}"
    #         fig_gt.canvas.manager.set_window_title(title)

    # fig_gt.canvas.mpl_connect('motion_notify_event', update_gt_view_info)
    
    # ===== Prediction Figure =====
    fig_pred = plt.figure(figsize=(8, 8))
    fig_pred.patch.set_alpha(0.0)
    fig_pred.canvas.manager.set_window_title("Prediction")
    ax_pred = fig_pred.add_subplot(111, projection="3d")
    ax_pred.set_facecolor('#121212') 
    ax_pred.axis("off")
    ax_pred.scatter(*points_np[preds_np == 0].T, color="white", s=1)
    ax_pred.scatter(*points_np[preds_np == 1].T, color="white", s=1)
    ax_pred.scatter(*points_np[preds_np == 2].T, color="white", s=1)
    ax_pred.view_init(elev=elev_default, azim=azim_default)

    # update for prediction window title
    def update_pred_view_info(event):
        if event.inaxes == ax_pred:
            elev = ax_pred.elev
            azim = ax_pred.azim
            title = f"Prediction | Elev: {elev:.1f}, Azim: {azim:.1f}"
            fig_pred.canvas.manager.set_window_title(title)

    fig_pred.canvas.mpl_connect('motion_notify_event', update_pred_view_info)

    plt.show()

def rotate_points(points, angle_degrees, axis='z'):
    theta = np.radians(angle_degrees)
    if axis == 'z':
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
        ])
    elif axis == 'x':
        R = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta),  np.cos(theta)]
        ])
    elif axis == 'y':
        R = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")
    return points @ R.T

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference + Visualisation (Single Sample, Dual Windows)")
    parser.add_argument("--checkpoint", default="pointnet_final.pth",
                        help="Path to the saved model checkpoint")
    args = parser.parse_args()

    _, _, test_loader = get_dataloaders(
        data_split_dir=NPZ_PATH,
        num_points=NUM_POINTS,
        batch_size=1,
        normalise=True,
        visualise=False,
        augment=False,
        fine_tune=args.fine_tuned
    )

    model = PointNet2Seg(num_classes=NUM_CLASSES)
    
    checkpoint_path = os.path.join(SAVE_DIR, args.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    print(f"Loading checkpoint from {checkpoint_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)

    visualise_predictions(model, test_loader)
