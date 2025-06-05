import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    """
    Normalises the point cloud to be zero-centroid and within unit sphere.
    """
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src: [B, N, C]
    dst: [B, M, C]
    return: dist: [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: [B, N, C]
        idx: [B, S] (or [B, S, k] with more dims)
    Return:
        new_points: [B, S, C] (or [B, S, k, C])
    """
    device = points.device
    B = points.shape[0]
    # Expand batch dim to match idx shape
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: [B, N, 3]
        npoint: number of samples
    Return:
        centroids: [B, npoint] (indices)
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius, nsample
        xyz: [B, N, 3]
        new_xyz: [B, S, 3]
    Return:
        group_idx: [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint, radius, nsample
        xyz: [B, N, 3]
        points: [B, N, D]
    Return:
        new_xyz: [B, npoint, 3]
        new_points: [B, npoint, nsample, 3 + D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)  # [B, S, 3]
    idx = query_ball_point(radius, nsample, xyz, new_xyz)  # [B, S, nsample]
    grouped_xyz = index_points(xyz, idx)                   # [B, S, nsample, 3]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)  # [B, S, nsample, D]
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, S, nsample, 3 + D]
    else:
        new_points = grouped_xyz_norm

    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points

def sample_and_group_all(xyz, points):
    """
    Group all points into one group.
    xyz: [B, N, 3]
    points: [B, N, D]
    Return:
        new_xyz: [B, 1, 3]
        new_points: [B, 1, N, 3 + D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

class PointNetSetAbstraction(nn.Module):
    """
    Single-scale grouping abstraction (SSG).
    """
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        Input:
            xyz: [B, C, N]
            points: [B, D, N]
        Return:
            new_xyz: [B, C, S]
            new_points: [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)   # [B, N, C]
        if points is not None:
            points = points.permute(0, 2, 1)  # [B, N, D]

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: [B, S, 3], new_points: [B, S, nsample, 3 + D]

        # [B, C+D, nsample, S]
        new_points = new_points.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        # Pool across nsample dimension
        new_points = torch.max(new_points, 2)[0]  # [B, mlp[-1], S]

        # Reformat
        new_xyz = new_xyz.permute(0, 2, 1)        # [B, 3, S]
        return new_xyz, new_points

class PointNetSetAbstractionMsg(nn.Module):
    """
    Multi-scale grouping (MSG) abstraction.
    """
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list, group_all=False):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.group_all = group_all

        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        # Each scale in MLP uses (in_channel + 3) because we concat local coordinates
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: [B, C, N]
            points: [B, D, N]
        Return:
            new_xyz: [B, C, S]
            new_points_concat: [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)   # [B, N, C]
        if points is not None:
            points = points.permute(0, 2, 1)  # [B, N, D]

        B, N, C = xyz.shape
        S = self.npoint

        if self.group_all or (S is None):
            # "Global" grouping
            new_xyz, new_points = sample_and_group_all(xyz, points)
            new_xyz = new_xyz.permute(0, 2, 1)  # [B, 3, 1]
            # We can still apply MLPs across "scales" if you want a single scale. 
            # But typically you'd define radius_list, nsample_list = [None], [None].
            # We'll keep it consistent below.
        else:
            # Farthest point sample
            fps_idx = farthest_point_sample(xyz, S)    # [B, S]
            new_xyz = index_points(xyz, fps_idx)       # [B, S, 3]
            new_xyz = new_xyz.permute(0, 2, 1)         # [B, 3, S]

        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            if self.group_all or (S is None):
                # If global, we already grouped everything into "new_points" 
                # and don't really do radius-based grouping again
                grouped_points = new_points.permute(0, 3, 2, 1)
            else:
                # Query ball
                group_idx = query_ball_point(radius, K, xyz, new_xyz.permute(0, 2, 1))
                grouped_xyz = index_points(xyz, group_idx)                      # [B, S, K, 3]
                grouped_xyz -= new_xyz.permute(0, 2, 1).unsqueeze(2)            # local coords
                if points is not None:
                    grouped_points = index_points(points, group_idx)            # [B, S, K, D]
                    grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
                else:
                    grouped_points = grouped_xyz
                grouped_points = grouped_points.permute(0, 3, 2, 1)             # [B, D+3, K, S]

            # Apply MLP for this scale
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            # Pool across K
            new_points = torch.max(grouped_points, 2)[0]  # [B, out_dim, S]
            new_points_list.append(new_points)

        # Concat along feature dim
        new_points_concat = torch.cat(new_points_list, dim=1)  # [B, sum_of_out_dims, S]
        return new_xyz, new_points_concat

class PointNetFeaturePropagation(nn.Module):
    """
    Decoder / feature propagation
    """
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: [B, C, N] 
            xyz2: [B, C, S]
            points1: [B, D, N]
            points2: [B, D, S]
        Return:
            new_points: [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)  # [B, N, C]
        xyz2 = xyz2.permute(0, 2, 1)  # [B, S, C]

        points2 = points2.permute(0, 2, 1)  # [B, S, D]
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        # 1) three-NN interpolation
        if S == 1:
            # If there's only one point in xyz2, just tile it
            interpolated_points = points2.repeat(1, N, 1)  # [B, N, D]
        else:
            dists = square_distance(xyz1, xyz2)       # [B, N, S]
            dists, idx = dists.sort(dim=-1)           # [B, N, S], [B, N, S]
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # knn=3
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm  # [B, N, 3]
            interpolated_points = torch.sum(index_points(points2, idx) * weight.unsqueeze(-1), dim=2)

        # 2) Concatenate skip connection
        if points1 is not None:
            points1 = points1.permute(0, 2, 1)  # [B, N, D1]
            new_points = torch.cat([points1, interpolated_points], dim=-1)  # [B, N, D1 + D2]
        else:
            new_points = interpolated_points

        # 3) MLP on upsampled
        new_points = new_points.permute(0, 2, 1)  # => [B, D1+D2, N]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points
