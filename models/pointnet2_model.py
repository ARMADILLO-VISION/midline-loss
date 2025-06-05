import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation, PointNetSetAbstractionMsg

"""
This implementation is adapted from the following repository:
https://github.com/yanx27/Pointnet_Pointnet2_pytorch-
Credits to the original authors for the design of this PointNet++ implementation and its utilities.
"""

class PointNet2Seg(nn.Module):
    def __init__(self, num_classes=3):
        super(PointNet2Seg, self).__init__()
        self.num_classes = num_classes

        # ------------------- ENCODER (MSG) -------------------
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=512,
            radius_list=[0.1, 0.2],
            nsample_list=[32, 64],
            in_channel=0,
            mlp_list=[[32, 32, 64], [32, 32, 64]],
            group_all=False
        )

        self.sa2 = PointNetSetAbstractionMsg(
            npoint=128,
            radius_list=[0.4, 0.8],
            nsample_list=[32, 64],
            in_channel=128,
            mlp_list=[[64, 64, 128], [64, 64, 128]],
            group_all=False
        )

        self.sa3 = PointNetSetAbstractionMsg(
            npoint=None,
            radius_list=[None],
            nsample_list=[None],
            in_channel=256,
            mlp_list=[[128, 256, 512]],
            group_all=True
        )

        # ------------------- DECODER (Feature Propagation) -------------------
        self.fp3 = PointNetFeaturePropagation(
            in_channel=512 + 256,
            mlp=[256, 256]
        )

        self.fp2 = PointNetFeaturePropagation(
            in_channel=256 + 128,
            mlp=[128, 128]
        )

        self.fp1 = PointNetFeaturePropagation(
            in_channel=128,
            mlp=[128, 128, 128]
        )

        # ------------------- HEADS -------------------
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)
        self.coord_fc = nn.Conv1d(128, 3, 1)

    def forward(self, xyz):
        B, N, _ = xyz.shape

        # Encoder
        l1_xyz, l1_points = self.sa1(xyz.transpose(1, 2), None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Decoder
        l2_points_up = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points_up = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points_up)
        l0_points_up = self.fp1(xyz.transpose(1, 2), l1_xyz, None, l1_points_up)

        # Segmentation head
        x = F.relu(self.bn1(self.conv1(l0_points_up)))
        x = self.drop1(x)
        seg_logits = self.conv2(x).transpose(1, 2)
        coords = self.coord_fc(x).transpose(1, 2)

        return seg_logits, coords