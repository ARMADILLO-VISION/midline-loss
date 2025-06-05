import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from config import CLASS_COLORS, CLASS_NAMES


def differentiable_midline(points, weights, num_line_points=50, lambda_factor=10.0):
    B, N, _ = points.shape
    skeleton_list = []
    for b in range(B):
        x = points[b]           # [N, 3]
        w = weights[b]          # [N]
        w = w.to(x.device)
        w_sum = w.sum() + 1e-6
        mean = torch.sum(w.unsqueeze(1) * x, dim=0) / w_sum  # [3]
        xm = x - mean.unsqueeze(0)                           # [N, 3]
        cov = torch.matmul((w.unsqueeze(1) * xm).T, xm) / w_sum  # [3, 3]
        eigvals, eigvecs = torch.linalg.eigh(cov)
        principal = eigvecs[:, -1]  # [3]
        proj = torch.matmul(xm, principal)  # [N]
        t_min = proj.min()
        t_max = proj.max()
        t_lin = torch.linspace(t_min.item(), t_max.item(), num_line_points, device=points.device)
        candidates = mean.unsqueeze(0) + t_lin.unsqueeze(1) * principal.unsqueeze(0)  # [num_line_points, 3]
        snapped = []
        for i in range(num_line_points):
            cand = candidates[i]  # [3]
            dists = torch.norm(x - cand.unsqueeze(0), dim=1)  # [N]
            soft_weights = F.softmax(-lambda_factor * dists, dim=0)  # [N]
            soft_snapped = torch.sum(soft_weights.unsqueeze(1) * x, dim=0)  # [3]
            snapped.append(soft_snapped)
        snapped = torch.stack(snapped, dim=0)  # [num_line_points, 3]
        skeleton_list.append(snapped)
    skeleton_points = torch.stack(skeleton_list, dim=0)  # [B, num_line_points, 3]
    return skeleton_points

def differentiable_deviation_loss(points, weights, skeleton_points, alpha=1.0):
    B, N, _ = points.shape
    M = skeleton_points.shape[1]
    
    # get distances from each point to every skeleton point
    diff = points.unsqueeze(2) - skeleton_points.unsqueeze(1)  # [B, N, M, 3]
    dists = torch.norm(diff, dim=-1)  # [B, N, M]
    
    # get soft weights for each point assignment to skeleton points
    soft_weights = F.softmax(-alpha * dists, dim=-1)  # [B, N, M]
    
    # get the soft nearest skeleton point for each point
    soft_nearest = torch.sum(soft_weights.unsqueeze(-1) * skeleton_points.unsqueeze(1), dim=2)  # [B, N, 3]
    
    # get the full deviation between points and their soft nearest skeleton point
    deviation_error = torch.norm(points - soft_nearest, dim=-1)
    deviation_error = torch.clamp(deviation_error, max=2.0)  # stop explosion
    
    exp_penalty = torch.exp(alpha * deviation_error) - 1
    
    # weight and average the penalty
    loss = torch.sum(weights * exp_penalty, dim=1) / (torch.sum(weights, dim=1) + 1e-6)
    return loss.mean()

def thickness_loss(predicted_logits, points, thin_class, lambda_factor=1.0):
    probs = F.softmax(predicted_logits, dim=-1)  # [B, N, C]
    thin_probs = probs[..., thin_class]           # [B, N]
    skel = differentiable_midline(points, thin_probs, num_line_points=50, lambda_factor=lambda_factor)
    loss = differentiable_deviation_loss(points, thin_probs, skel)
    return loss


def resample_midline_torch(midline, num_points):
    midline = midline.unsqueeze(0).transpose(1, 2)
    midline_resampled = F.interpolate(midline, size=num_points, mode='linear', align_corners=True)
    midline_resampled = midline_resampled.transpose(1, 2).squeeze(0)
    return midline_resampled

def soft_chamfer_distance(pred, gt, alpha=1.0):
    diff = pred.unsqueeze(1) - gt.unsqueeze(0)  # [N, M, 3]
    dists = torch.norm(diff, dim=-1)  # [N, M]
    softmin_pred_to_gt = torch.mean(torch.sum(F.softmax(-alpha * dists, dim=1) * dists, dim=1))
    softmin_gt_to_pred = torch.mean(torch.sum(F.softmax(-alpha * dists, dim=0) * dists, dim=0))
    return (softmin_pred_to_gt + softmin_gt_to_pred) / 2

def midline_alignment_loss(pred_logits, gt_labels, coords, thin_class, lambda_factor=10.0, alpha=1.0):
    probs = F.softmax(pred_logits, dim=-1)  # [B, N, C]
    thin_probs = probs[..., thin_class]      # [B, N]
    
    # get predicted midline
    pred_midline = differentiable_midline(coords, thin_probs, num_line_points=50, lambda_factor=lambda_factor)
    
    # get ground truth points for the thin structure
    gt_mask = (gt_labels[0] == thin_class)
    gt_region = coords[0][gt_mask]  # [M, 3]
    if gt_region.shape[0] == 0:
        return torch.tensor(0.0, device=pred_logits.device)
    
    # resampling of the ground truth midline
    gt_midline = resample_midline_torch(gt_region, num_points=50)  # [50, 3]
    
    # get the soft Chamfer distance between the predicted midline and the GT midline
    loss = soft_chamfer_distance(pred_midline[0], gt_midline, alpha=alpha)
    return loss

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None, reduction='mean'):
        super().__init__()
        self.class_weights = class_weights
        self.reduction = reduction

    def forward(self, logits, targets):
        B, N, C = logits.shape
        logits_flat = logits.reshape(-1, C)
        targets_flat = targets.reshape(-1)
        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
        return F.cross_entropy(logits_flat, targets_flat, weight=weight, reduction=self.reduction)

class NLLLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.class_weights = class_weights

    def forward(self, logits, targets):
        B, N, C = logits.shape
        logits_flat = logits.reshape(-1, C)
        targets_flat = targets.reshape(-1)
        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
        return F.nll_loss(logits_flat, targets_flat, weight=weight)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, class_weights=None, ignore_index=0):
        """
        Weighted Dice Loss that ignores a specified class (e.g. background).
        :param smooth: Smoothing constant to avoid division by zero.
        :param class_weights: Optional list or tensor of weights for each class (excluding ignore_index).
        :param ignore_index: Class index to ignore (e.g. 0 for background).
        """
        super().__init__()
        self.smooth = smooth
        self.class_weights = class_weights
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        :param logits: Tensor of shape [B, N, C] (logits for each point and class).
        :param targets: Tensor of shape [B, N] (ground truth class indices for each point).
        :return: Scalar weighted Dice loss computed only on classes that are not ignore_index.
        """
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)  # [B, N, C]
        B, N, C = logits.shape

        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=C).float()  # [B, N, C]

        # Determine the valid class indices (i.e. ignore the background)
        valid_classes = [c for c in range(C) if c != self.ignore_index]

        dice_scores = []
        for c in valid_classes:
            prob_c = probs[..., c]           # [B, N]
            target_c = targets_one_hot[..., c] # [B, N]
            # Compute Dice coefficient per batch sample
            intersection = (prob_c * target_c).sum(dim=1)
            union = prob_c.sum(dim=1) + target_c.sum(dim=1)
            dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice_score)
        
        # Stack dice scores: shape [B, num_valid_classes]
        dice_scores = torch.stack(dice_scores, dim=-1)
        
        # Compute Dice loss for each sample over valid classes
        if self.class_weights is not None:
            weights = torch.tensor(self.class_weights, device=logits.device).float()
            weights = weights[[c for c in range(C) if c != self.ignore_index]]
            dice_loss = 1 - (dice_scores * weights).sum(dim=-1) / weights.sum()
        else:
            dice_loss = 1 - dice_scores.mean(dim=-1)
        
        # Return the average loss over the batch
        return dice_loss.mean()

def compute_midline(points, num_line_points=100):
    mean = np.mean(points, axis=0)
    centered = points - mean
    u, s, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    projections = np.dot(centered, direction)
    t_min, t_max = projections.min(), projections.max()
    t = np.linspace(t_min, t_max, num_line_points)
    midline = mean + np.outer(t, direction)
    return midline

def smooth_skeleton(skeleton_points, window_size=5):
    smoothed = np.copy(skeleton_points)
    for i in range(len(skeleton_points)):
        start = max(0, i - window_size // 2)
        end = min(len(skeleton_points), i + window_size // 2 + 1)
        smoothed[i] = np.mean(skeleton_points[start:end], axis=0)
    return smoothed

def compute_skeleton(region_points, num_line_points=100, epsilon=1e-3):
    midline = compute_midline(region_points, num_line_points)
    skeleton_points = []
    for mp in midline:
        snapped_point = region_points[np.argmin(np.linalg.norm(region_points - mp, axis=1))]
        if len(skeleton_points) == 0 or np.linalg.norm(snapped_point - skeleton_points[-1]) > epsilon:
            skeleton_points.append(snapped_point)
    skeleton_points = np.array(skeleton_points)
    skeleton_points = smooth_skeleton(skeleton_points, window_size=5)
    return skeleton_points

def compute_thickness_penalty(points, skeleton_points):
    penalties = np.zeros(points.shape[0])
    for j, pt in enumerate(points):
        dists = np.linalg.norm(pt - skeleton_points, axis=1)
        idx = np.argmin(dists)
        closest_point = skeleton_points[idx]
        if len(skeleton_points) < 2:
            tangent = np.zeros_like(closest_point)
        else:
            if idx == 0:
                tangent = skeleton_points[1] - skeleton_points[0]
            elif idx == len(skeleton_points) - 1:
                tangent = skeleton_points[-1] - skeleton_points[-2]
            else:
                tangent = (skeleton_points[idx + 1] - skeleton_points[idx - 1]) / 2.0
        tangent_norm = np.linalg.norm(tangent) + 1e-6
        tangent = tangent / tangent_norm
        d_vec = pt - closest_point
        d_parallel = np.dot(d_vec, tangent) * tangent
        d_perp = d_vec - d_parallel
        penalties[j] = np.linalg.norm(d_perp)
    return penalties

##############################################################################
# Combined Loss Function
##############################################################################
class CombinedLoss(nn.Module):
    def __init__(self,
                 class_weights=None,
                 alpha_ce=1.0,
                 lam_ridge=1.0,
                 lam_lig=1.0,
                 lam_nll=0.0,
                 lam_midline=1.0,
                 lam_dice=1.0,
                 ridge_class_idx=1,
                 lig_class_idx=2,
                 lambda_thickness=10.0,
                 reduction='mean',
                 visualise=True):
        super().__init__()
        self.alpha_ce = alpha_ce
        self.lam_ridge = lam_ridge
        self.lam_lig = lam_lig
        self.lam_nll = lam_nll
        self.lam_midline = lam_midline
        self.lam_dice = lam_dice
        self.ridge_class_idx = ridge_class_idx
        self.lig_class_idx = lig_class_idx
        self.lambda_thickness = lambda_thickness
        self.visualise = visualise
        self.ce_loss = WeightedCrossEntropyLoss(class_weights=class_weights, reduction=reduction)
        self.nll_loss = NLLLoss(class_weights=class_weights)
        self.dice_loss = DiceLoss()

    def forward(self, logits, targets, points, debug=False):
        ce_val = self.ce_loss(logits, targets)
        log_probs = F.log_softmax(logits, dim=-1)
        nll_val = self.nll_loss(log_probs, targets)
        
        # thickness losses
        thickness_ridge = thickness_loss(logits, points, self.ridge_class_idx, lambda_factor=self.lambda_thickness)
        thickness_lig = thickness_loss(logits, points, self.lig_class_idx, lambda_factor=self.lambda_thickness)
        
        # midline alignment losses
        midline_ridge = midline_alignment_loss(logits, targets, points, self.ridge_class_idx, lambda_factor=self.lambda_thickness)
        midline_lig = midline_alignment_loss(logits, targets, points, self.lig_class_idx, lambda_factor=self.lambda_thickness)
        
        # combine geometry losses
        ridge_geom = (1 - self.lam_midline) * thickness_ridge + self.lam_midline * midline_ridge
        lig_geom = (1 - self.lam_midline) * thickness_lig + self.lam_midline * midline_lig
        
        # scaling factor 
        avg_geom = ((ridge_geom.detach() + lig_geom.detach()) / 2.0) + 1e-6
        scale_factor = ce_val.detach() / avg_geom
        
        # apply the scale factor to geometry losses
        scaled_ridge_geom = scale_factor * ridge_geom
        scaled_lig_geom = scale_factor * lig_geom
        
        # Dice loss
        dice_loss_val = self.dice_loss(logits, targets)
        
        total_loss = (self.alpha_ce * ce_val +
                      self.lam_ridge * scaled_ridge_geom +
                      self.lam_lig * scaled_lig_geom +
                      self.lam_nll * nll_val +
                      self.lam_dice * dice_loss_val)
        
        if debug:
            print(f"CE Loss: {ce_val.item():.6f} | NLL Loss: {nll_val.item():.6f} | "
                  f"Thickness Ridge: {thickness_ridge.item():.6f} | Thickness Lig: {thickness_lig.item():.6f} | "
                  f"Midline Ridge: {midline_ridge.item():.6f} | Midline Lig: {midline_lig.item():.6f} | "
                  f"Dice Loss: {dice_loss_val.item():.6f} | Total Loss: {total_loss.item():.6f}")

        if self.visualise:
            points_np = points[0].detach().cpu().numpy()  # [N, 3]
            targets_np = targets[0].detach().cpu().numpy()  # [N]
            preds_np = torch.argmax(logits[0], dim=-1).detach().cpu().numpy()  # [N]
            
            fig = plt.figure(figsize=(18, 8))
            ax1 = fig.add_subplot(121, projection="3d")
            ax1.set_title("Ground Truth")
            gt_liver = points_np[targets_np == 0]
            gt_ridge = points_np[targets_np == self.ridge_class_idx]
            gt_ligam = points_np[targets_np == self.lig_class_idx]
            ax1.scatter(gt_liver[:, 0], gt_liver[:, 1], gt_liver[:, 2],
                        color=CLASS_COLORS[0], s=1, label=CLASS_NAMES[0])
            ax1.scatter(gt_ridge[:, 0], gt_ridge[:, 1], gt_ridge[:, 2],
                        color=CLASS_COLORS[1], s=10, label=CLASS_NAMES[1])
            ax1.scatter(gt_ligam[:, 0], gt_ligam[:, 1], gt_ligam[:, 2],
                        color=CLASS_COLORS[2], s=10, label=CLASS_NAMES[2])
            
            classes_of_interest = {self.ridge_class_idx: 'ridge', self.lig_class_idx: 'ligament'}
            for class_idx, class_name in classes_of_interest.items():
                gt_region_points = points_np[targets_np == class_idx]
                if gt_region_points.shape[0] == 0:
                    print(f"No ground truth points found for class {class_name}.")
                    continue
                gt_skeleton = compute_skeleton(gt_region_points, num_line_points=20)
                color_line = 'black' if class_idx == self.ridge_class_idx else 'orange'
                ax1.plot(gt_skeleton[:, 0], gt_skeleton[:, 1], gt_skeleton[:, 2],
                         color=color_line, linewidth=3, label=f"GT Skeleton {class_name}")
            
            ax1.legend(loc="upper right")
            
            ax2 = fig.add_subplot(122, projection="3d")
            ax2.set_title("Raw Predictions with Skeleton & Thickness Penalty")
            pred_liver = points_np[preds_np == 0]
            pred_ridge = points_np[preds_np == self.ridge_class_idx]
            pred_ligam = points_np[preds_np == self.lig_class_idx]
            ax2.scatter(pred_liver[:, 0], pred_liver[:, 1], pred_liver[:, 2],
                        color=CLASS_COLORS[0], s=1, label=CLASS_NAMES[0])
            ax2.scatter(pred_ridge[:, 0], pred_ridge[:, 1], pred_ridge[:, 2],
                        color=CLASS_COLORS[1], s=10, label=CLASS_NAMES[1])
            ax2.scatter(pred_ligam[:, 0], pred_ligam[:, 1], pred_ligam[:, 2],
                        color=CLASS_COLORS[2], s=10, label=CLASS_NAMES[2])
            
            for class_idx, class_name in classes_of_interest.items():
                region_points = points_np[preds_np == class_idx]
                if region_points.shape[0] == 0:
                    continue
                
                skeleton_points = compute_skeleton(region_points, num_line_points=200)
                thickness_penalties = compute_thickness_penalty(region_points, skeleton_points)
            
                color_line = 'black' if class_idx == self.ridge_class_idx else 'orange'
                ax2.plot(skeleton_points[:, 0], skeleton_points[:, 1], skeleton_points[:, 2],
                        color=color_line, linewidth=3, label=f"Skeleton {class_name}")
                
                line_threshold = 0.01
                
                on_line_mask = thickness_penalties < line_threshold
                off_line_mask = ~on_line_mask
                
                ax2.scatter(region_points[on_line_mask, 0],
                            region_points[on_line_mask, 1],
                            region_points[on_line_mask, 2],
                            color=CLASS_COLORS[class_idx],
                            s=20,
                            alpha=0.6,
                            label=f"{class_name} on-line points")
                
                ax2.scatter(region_points[off_line_mask, 0],
                            region_points[off_line_mask, 1],
                            region_points[off_line_mask, 2],
                            c=thickness_penalties[off_line_mask],
                            cmap='viridis',
                            s=20,
                            alpha=0.6,
                            label=f"{class_name} thickness penalty")
            
            ax2.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
            plt.subplots_adjust(right=0.8)
            plt.savefig("per_epoch.png")
            plt.close(fig)
        
        return total_loss
