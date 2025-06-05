import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import (
    PointNetSetAbstractionMsg,
    PointNetSetAbstraction,
    PointNetFeaturePropagation
)


"""
This implementation is adapted from the following repository:
https://github.com/yanx27/Pointnet_Pointnet2_pytorch-
Credits to the original authors for the design of this PointNet++ implementation and its utilities.
"""

class PointNet2Seg(nn.Module):
    def __init__(self, num_classes=3, label_dim=16):
        super(PointNet2Seg, self).__init__()
        self.num_classes = num_classes
        self.label_dim   = label_dim

        # -------------------- ENCODER --------------------
        #
        # 1) MSG: sample 512 points with 3 radii => [0.1, 0.2, 0.4]
        #    nsample => [32, 64, 128]
        #    Each scale uses an MLP:
        #      scale-1: [32, 32, 64]
        #      scale-2: [64, 64, 128]
        #      scale-3: [64, 96, 128]
        #    => final feature dimension = 64 + 128 + 128 = 320
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=512,
            radius_list=[0.1, 0.2, 0.4],
            nsample_list=[32, 64, 128],
            in_channel=0,   # only (x,y,z)
            mlp_list=[
                [32, 32, 64],
                [64, 64, 128],
                [64, 96, 128]
            ]
        )

        # 2) MSG: sample 128 points, 2 radii => [0.4, 0.8]
        #    nsample => [64, 128]
        #    MLPs => [128,128,256], [128,196,256]
        #    => final dimension = 256 + 256 = 512
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=128,
            radius_list=[0.4, 0.8],
            nsample_list=[64, 128],
            in_channel=320,   # from SA1
            mlp_list=[
                [128, 128, 256],
                [128, 196, 256]
            ]
        )

        # 3) Single-scale group-all for global feature
        #    MLP => [256, 512, 1024]
        #    => final dimension = 1024
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=512 + 3,   # 512 from above + xyz
            mlp=[256, 512, 1024],
            group_all=True
        )

        # -------------------- DECODER --------------------
        self.fp3 = PointNetFeaturePropagation(
            in_channel=1024 + 512,   # 1536
            mlp=[256, 256]
        )
        # for upsampling from SA3 to SA2

        self.fp2 = PointNetFeaturePropagation(
            in_channel=256 + 320,    # 576
            mlp=[256, 128]
        )
        # for upsampling from SA2 to SA1

        self.fp1 = PointNetFeaturePropagation(
            in_channel=128 + 3 + 0,  # xyz + ??? typically we have 128 from above plus 3 from xyz
            mlp=[128, 128]
        )

        # -------------------- HEADS --------------------
        # final 1D conv => per-point seg, plus a coordinate regression
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1   = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)  # => seg logits

        self.coord_fc = nn.Conv1d(128, 3, 1)

    def forward(self, xyz, cls_label=None):
        """
        xyz: [B, N, 3]
        cls_label: [B, label_dim], e.g. 16-dim
        Returns:
            seg_logits: [B, N, num_classes]
            pred_points: [B, N, 3]
        """
        if cls_label is None:
            # fallback if no label is provided
            B = xyz.shape[0]
            cls_label = torch.zeros(B, self.label_dim, device=xyz.device)

        B, N, _ = xyz.shape

        # -------------------- ENCODER --------------------
        # sa1 => MSG with 3 scales => out dim=320
        l1_xyz, l1_points = self.sa1(xyz.transpose(1, 2), None)
        # l1_xyz: [B, 3, 512]
        # l1_points: [B, 320, 512]

        # sa2 => MSG with 2 scales => out dim=512
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # l2_xyz: [B, 3, 128]
        # l2_points: [B, 512, 128]

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # l3_xyz: [B, 3, 1]
        # l3_points: [B, 1024, 1]

        # -------------------- DECODER --------------------
        # 1) FP3: from global back to SA2
        #    input channels: (1024 from global + 512 skip) => 1536 => MLP [256,256]
        l2_points_up = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        # => [B, 256, 128]

        # 2) FP2: from 128->512 
        #    in_channel= (256 + 320) => 576 => [256,128]
        l1_points_up = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points_up)
        # => [B, 128, 512]

        # 3) FP1: from 512->N

        # Expand the class label to [B, label_dim, N]
        cls_label_expanded = cls_label.unsqueeze(2).repeat(1, 1, N)  # => [B, label_dim, N]
        # We'll also pass the raw xyz => [B, 3, N]
        xyz_t = xyz.transpose(1, 2) # => [B, 3, N]
        points1_cat = torch.cat([cls_label_expanded, xyz_t], dim=1)  # => [B, label_dim+3, N]

        # Now we upsample
        l0_points_up = self.fp1(
            xyz_t,           # xyz1
            l1_xyz,          # xyz2
            points1_cat,     # points1
            l1_points_up     # points2
        )
        # => [B, 128, N]

        # -------------------- HEADS --------------------
        # final MLP => seg logits + coords
        x = F.relu(self.bn1(self.conv1(l0_points_up)))  # => [B, 128, N]
        x = self.drop1(x)

        # seg => [B, N, num_classes]
        seg_logits = self.conv2(x).transpose(1, 2)

        # coords => [B, N, 3]
        coords = self.coord_fc(x).transpose(1, 2)

        return seg_logits, coords
