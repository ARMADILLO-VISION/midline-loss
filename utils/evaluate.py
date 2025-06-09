#!/usr/bin/env python3
import os
import argparse
import csv
import numpy as np
import torch
import torch.nn.functional as F
import point_cloud_utils as pcu
import open3d as o3d
import sys

sys.path.append(os.path.abspath('./'))

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull

from utils.inference import (
    find_vertical_extremes,
    find_two_farthest_points,
    build_shortest_path_line,
    map_bfs_path_to_global_indices
)

from config import (
    NPZ_PATH, NUM_POINTS, BATCH_SIZE, CLASS_NAMES, CLASS_COLORS,
    SAVE_DIR, NUM_CLASSES
)
from pointnet_data_loader import get_dataloaders
from models.pointnet2_model import PointNet2Seg

################################################################################
# Hausdorff Distance
################################################################################

def Hausdorff_dist(vol_a, vol_b):
    """
    A naive implementation of Hausdorff distance:
    For each point in vol_a, find the minimum distance to any point in vol_b;
    then take the maximum of these minimum distances.
    """
    if len(vol_a) == 0 or len(vol_b) == 0:
        return 0.0
    dist_lst = []
    for idx in range(len(vol_a)):
        dist_min = float('inf')
        for idx2 in range(len(vol_b)):
            dist = np.linalg.norm(vol_a[idx] - vol_b[idx2])
            if dist < dist_min:
                dist_min = dist
        dist_lst.append(dist_min)
    return np.max(dist_lst)

################################################################################
# Utility Functions
################################################################################

def _denormalize_points(points_norm, global_min, global_max, global_mean):
    """
    Convert normalised points back to mm using dataset stats.
    """
    scale = (global_max - global_min)
    points_mm = (points_norm * scale) + global_mean
    return points_mm

def chamfer_distance_pcu(gt_points_mm, pred_points_mm):
    """
    Returns the two-sided squared Chamfer Distance using PCU.
    Returns 0.0 if either set is empty.
    """
    if gt_points_mm.shape[0] == 0 or pred_points_mm.shape[0] == 0:
        return 0.0
    cd_val = pcu.chamfer_distance(gt_points_mm, pred_points_mm)
    return cd_val

def compute_avg_nn_distance(points_3d):
    """
    Uses Open3D to compute the mean nearest-neighbour distance for a set of points.
    Returns 0.0 if `points_3d` is empty.
    """
    if points_3d.shape[0] == 0:
        return 0.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    distances = pcd.compute_nearest_neighbor_distance()
    return np.mean(distances)

################################################################################
# Evaluation Function
################################################################################

def evaluate_all_metrics(model, test_loader, global_min, global_max, global_mean, use_bfs=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    results_list = []
    sample_idx = 0

    with torch.no_grad():
        for points, labels in test_loader:
            # We assume batch size = 1 for evaluation
            points, labels = points.to(device), labels.to(device)
            points_norm = points.cpu().numpy()[0]  # [N, 3]
            gt_labels_np = labels.cpu().numpy()[0]  # [N]

            # Forward pass
            points_eval = torch.from_numpy(points_norm).unsqueeze(0).to(device).float()
            seg_logits, _ = model(points_eval)
            if hasattr(model, "logits_adapter"):
                seg_logits = model.logits_adapter(seg_logits)
            if hasattr(model, "geometry_adapter"):
                seg_logits = model.geometry_adapter(seg_logits, points_eval)
            probs = torch.softmax(seg_logits, dim=-1)
            _, predictions = torch.max(probs, dim=2)  # shape: [B, N]
            predictions_np = predictions.cpu().numpy()[0]

            # Count GT & prediction for each class
            gt_count_r   = np.sum(gt_labels_np == 1)
            pred_count_r = np.sum(predictions_np == 1)
            gt_count_l   = np.sum(gt_labels_np == 2)
            pred_count_l = np.sum(predictions_np == 2)

            no_pred_r = (pred_count_r == 0)
            no_pred_l = (pred_count_l == 0)

            # BFS refinement if requested
            if use_bfs:
                refined_preds = np.zeros_like(predictions_np)
                # --- BFS for Ridge (class 1)
                ridge_indices = np.where(predictions_np == 1)[0]
                BFS_ridge = []
                if ridge_indices.shape[0] == 1:
                    BFS_ridge = ridge_indices
                elif ridge_indices.shape[0] >= 2:
                    ridge_xyz = points_norm[ridge_indices]
                    try:
                        hull = ConvexHull(ridge_xyz)
                        hull_indices = np.unique(hull.vertices)
                        hull_points = ridge_xyz[hull_indices]
                        pA, pB, iA_temp, iB_temp = find_two_farthest_points(hull_points)
                        iA = hull_indices[iA_temp]
                        iB = hull_indices[iB_temp]
                    except:
                        pA, pB, iA, iB = find_two_farthest_points(ridge_xyz)
                    if pA is not None:
                        lambda_param = 0.5  # penalty weight
                        path_3d = build_shortest_path_line(
                            ridge_xyz, iA, iB, k=20, vertical_penalty_weight=lambda_param
                        )
                        if path_3d.shape[0] > 1:
                            BFS_ridge = map_bfs_path_to_global_indices(path_3d, ridge_xyz, ridge_indices)
                    if len(BFS_ridge) == 0:
                        BFS_ridge = ridge_indices
                refined_preds[BFS_ridge] = 1

                # --- BFS for Ligament (class 2)
                lig_indices = np.where(predictions_np == 2)[0]
                BFS_lig = []
                if lig_indices.shape[0] == 1:
                    BFS_lig = lig_indices
                elif lig_indices.shape[0] >= 2:
                    lig_xyz = points_norm[lig_indices]
                    top_pt, bottom_pt, iT, iB = find_vertical_extremes(lig_xyz)
                    if top_pt is not None:
                        path_3d = build_shortest_path_line(lig_xyz, iB, iT, k=20)
                        if path_3d.shape[0] > 1:
                            BFS_lig = map_bfs_path_to_global_indices(path_3d, lig_xyz, lig_indices)
                    if len(BFS_lig) == 0:
                        BFS_lig = lig_indices
                refined_preds[BFS_lig] = 2

                predictions_np = refined_preds

                # Re-check predictions
                pred_count_r = np.sum(predictions_np == 1)
                pred_count_l = np.sum(predictions_np == 2)
                no_pred_r = (pred_count_r == 0)
                no_pred_l = (pred_count_l == 0)

            points_mm = _denormalize_points(points_norm, global_min, global_max, global_mean)

            # Class 1 (ridge)
            gt_points_r = points_mm[gt_labels_np == 1]
            pd_points_r = points_mm[predictions_np == 1]

            ch_r = 0.0
            hd_r = 0.0
            nn_r = 0.0
            if gt_points_r.shape[0] > 0 and pd_points_r.shape[0] > 0:
                ch_r = chamfer_distance_pcu(gt_points_r, pd_points_r)
                hd_r = Hausdorff_dist(gt_points_r, pd_points_r)

                avg_dist_gt_r = compute_avg_nn_distance(gt_points_r)
                avg_dist_pd_r = compute_avg_nn_distance(pd_points_r)
                nn_r = abs(avg_dist_pd_r - avg_dist_gt_r)

            # Class 2 (ligament)
            gt_points_l = points_mm[gt_labels_np == 2]
            pd_points_l = points_mm[predictions_np == 2]

            ch_l = 0.0
            hd_l = 0.0
            nn_l = 0.0
            if gt_points_l.shape[0] > 0 and pd_points_l.shape[0] > 0:
                ch_l = chamfer_distance_pcu(gt_points_l, pd_points_l)
                hd_l = Hausdorff_dist(gt_points_l, pd_points_l)

                avg_dist_gt_l = compute_avg_nn_distance(gt_points_l)
                avg_dist_pd_l = compute_avg_nn_distance(pd_points_l)
                nn_l = abs(avg_dist_pd_l - avg_dist_gt_l)

            row = {
                "patientname":  f"case_{sample_idx}",
                "ch_r": ch_r,
                "ch_l": ch_l,
                "hd_r": hd_r,
                "hd_l": hd_l,
                "nn_r": nn_r,
                "nn_l": nn_l,
                "no_pred_r": no_pred_r,
                "no_pred_l": no_pred_l,
                "gt_count_r": int(gt_count_r),
                "pred_count_r": int(pred_count_r),
                "gt_count_l": int(gt_count_l),
                "pred_count_l": int(pred_count_l)
            }
            results_list.append(row)
            sample_idx += 1

    return results_list

################################################################################

def print_and_save_results(results, output_csv):
    """
    Print and save evaluation results to a CSV file.
    """
    headers = ["patientname", "ch_r", "ch_l", "hd_r", "hd_l", "nn_r", "nn_l"]

    print("=== Evaluation Results ===")
    print(f"{'PatientName':>12} | {'ch_r':>14} | {'ch_l':>14} | "
          f"{'hd_r':>14} | {'hd_l':>14} | {'nn_r':>14} | {'nn_l':>14}")
    print("-" * 110)

    def format_metric(val, pred_count, gt_count):
        if val == 0.0:
            if pred_count == 0:
                return f"{val:.4f} (NP)"
            elif gt_count == 0:
                return f"{val:.4f} (GT0)"
        return f"{val:.4f}"

    # Lists for mean calculations
    ch_r_vals, ch_l_vals = [], []
    hd_r_vals, hd_l_vals = [], []
    nn_r_vals, nn_l_vals = [], []

    no_pred_r_count = 0
    no_pred_l_count = 0

    for row in results:
        patient = row["patientname"]

        # Class 1 (Ridge)
        if row["no_pred_r"] or row["gt_count_r"] == 0:
            if row["no_pred_r"]:
                display_ch_r = "NP"
                display_hd_r = "NP"
                display_nn_r = "NP"
                no_pred_r_count += 1
            else:
                display_ch_r = format_metric(row["ch_r"], row["pred_count_r"], row["gt_count_r"])
                display_hd_r = format_metric(row["hd_r"], row["pred_count_r"], row["gt_count_r"])
                display_nn_r = format_metric(row["nn_r"], row["pred_count_r"], row["gt_count_r"])
        else:
            display_ch_r = format_metric(row["ch_r"], row["pred_count_r"], row["gt_count_r"])
            display_hd_r = format_metric(row["hd_r"], row["pred_count_r"], row["gt_count_r"])
            display_nn_r = format_metric(row["nn_r"], row["pred_count_r"], row["gt_count_r"])
            ch_r_vals.append(row["ch_r"])
            hd_r_vals.append(row["hd_r"])
            nn_r_vals.append(row["nn_r"])

        # Class 2 (Ligament)
        if row["no_pred_l"] or row["gt_count_l"] == 0:
            if row["no_pred_l"]:
                display_ch_l = "NP"
                display_hd_l = "NP"
                display_nn_l = "NP"
                no_pred_l_count += 1
            else:
                display_ch_l = format_metric(row["ch_l"], row["pred_count_l"], row["gt_count_l"])
                display_hd_l = format_metric(row["hd_l"], row["pred_count_l"], row["gt_count_l"])
                display_nn_l = format_metric(row["nn_l"], row["pred_count_l"], row["gt_count_l"])
        else:
            display_ch_l = format_metric(row["ch_l"], row["pred_count_l"], row["gt_count_l"])
            display_hd_l = format_metric(row["hd_l"], row["pred_count_l"], row["gt_count_l"])
            display_nn_l = format_metric(row["nn_l"], row["pred_count_l"], row["gt_count_l"])
            ch_l_vals.append(row["ch_l"])
            hd_l_vals.append(row["hd_l"])
            nn_l_vals.append(row["nn_l"])

        print(f"{patient:>12} | {display_ch_r:>14} | {display_ch_l:>14} | "
              f"{display_hd_r:>14} | {display_hd_l:>14} | {display_nn_r:>14} | {display_nn_l:>14}")

    print("-" * 110)

    # Mean calculations (only for valid samples)
    mean_ch_r = np.mean(ch_r_vals) if ch_r_vals else 0.0
    mean_ch_l = np.mean(ch_l_vals) if ch_l_vals else 0.0
    mean_hd_r = np.mean(hd_r_vals) if hd_r_vals else 0.0
    mean_hd_l = np.mean(hd_l_vals) if hd_l_vals else 0.0
    mean_nn_r = np.mean(nn_r_vals) if nn_r_vals else 0.0
    mean_nn_l = np.mean(nn_l_vals) if nn_l_vals else 0.0

    print(f"{'MEAN':>12} | {mean_ch_r:>14.4f} | {mean_ch_l:>14.4f} | "
          f"{mean_hd_r:>14.4f} | {mean_hd_l:>14.4f} | {mean_nn_r:>14.4f} | {mean_nn_l:>14.4f}")
    print()
    print(f"No predictions for ridge (class 1) in {no_pred_r_count} out of {len(results)} samples.")
    print(f"No predictions for ligament (class 2) in {no_pred_l_count} out of {len(results)} samples.")

    with open(output_csv, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in results:
            writer.writerow({
                "patientname": row["patientname"],
                "ch_r": row["ch_r"],
                "ch_l": row["ch_l"],
                "hd_r": row["hd_r"],
                "hd_l": row["hd_l"],
                "nn_r": row["nn_r"],
                "nn_l": row["nn_l"],
            })
        writer.writerow({
            "patientname": "MEAN",
            "ch_r": mean_ch_r,
            "ch_l": mean_ch_l,
            "hd_r": mean_hd_r,
            "hd_l": mean_hd_l,
            "nn_r": mean_nn_r,
            "nn_l": mean_nn_l,
        })

    print(f"\nResults saved to {output_csv}")

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Chamfer distance, Hausdorff distance, and avg NN-distance differences for classes (1) ridge & (2) ligament. "
                    "Optional BFS refinement can be applied.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--checkpoint", default="pointnet_final.pth",
                        help="Path to the saved model checkpoint")
    parser.add_argument("--output_csv", default="evaluation.csv",
                        help="CSV output file for combined results")
    parser.add_argument("--eval_on_val", action="store_true",
                        help="Evaluate on the validation set instead of the test set")
    args = parser.parse_args()

    # Build and load data
    _, val_loader, test_loader = get_dataloaders(
        data_split_dir=NPZ_PATH,
        num_points=NUM_POINTS,
        batch_size=1,
        normalise=True,
        visualise=False,
        augment=False,
        fine_tune=args.fine_tuned
    )

    if args.eval_on_val:
        print("Evaluating on the validation set.")
        eval_loader = val_loader
    else:
        print("Evaluating on the test set.")
        eval_loader = test_loader

    # Retrieve dataset stats
    eval_dataset = eval_loader.dataset
    global_min  = eval_dataset.global_min
    global_max  = eval_dataset.global_max
    global_mean = eval_dataset.global_mean

    # Build the model
    model = PointNet2Seg(num_classes=NUM_CLASSES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.join(SAVE_DIR, args.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
    print(f"Loading checkpoint from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)

    # Evaluate on chosen set
    results = evaluate_all_metrics(
        model=model,
        test_loader=eval_loader,
        global_min=global_min,
        global_max=global_max,
        global_mean=global_mean,
        use_bfs=args.bfs
    )

    # Print and save results
    print_and_save_results(results, args.output_csv)

if __name__ == "__main__":
    main()
