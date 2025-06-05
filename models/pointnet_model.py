import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet_utils import PointNetEncoder

class PointNetSeg(nn.Module):
    def __init__(self,  num_classes=3):
        super(PointNetSeg, self).__init__()
        self.k =  num_classes
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=3)

        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, self.k, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, N, 3] 
               (or [B, N, <C>] if you are using more input channels).
        Returns:
            logits: [B, N, num_class] (raw logits, no softmax)
            trans_feat: feature transform matrix (for regularisation if needed)
        """

        # 1) Transpose for PointNetEncoder, which expects [B, C, N]
        x = x.transpose(1, 2).contiguous()  # => [B, 3, N] or [B, <C>, N]

        # 2) Forward pass through encoder
        x, trans, trans_feat = self.feat(x)  # -> x is now [B, 1088, N] if channel=3

        # 3) Convolutions + BatchNorm + ReLU
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 512, N]
        x = F.relu(self.bn2(self.conv2(x)))  # [B, 256, N]
        x = F.relu(self.bn3(self.conv3(x)))  # [B, 128, N]

        # 4) Final conv => raw logits
        x = self.conv4(x)                   # [B, num_class, N]

        # 5) Return in the shape [B, N, num_class]
        x = x.transpose(2, 1).contiguous()  # => [B, N, num_class]

        return x, trans_feat
