import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from scipy.spatial import KDTree
from collections import Counter
from config import NUM_POINTS, BATCH_SIZE, DILATION_RADIUS, CLASS_COLORS, CLASS_NAMES

class PointNetDataset(Dataset):
    def __init__(
        self,
        data_dir,
        num_points=NUM_POINTS,
        normalise=True,
        visualise=True,
        dilation_radius=DILATION_RADIUS,
        augment=False,
        global_min=None,
        global_max=None,
        global_mean=None,
        compute_stats=False
    ):
        self.data_dir = data_dir
        self.num_points = num_points
        self.normalise = normalise
        self.visualise = visualise
        self.dilation_radius = dilation_radius
        self.augment = augment

        # Gather .npz files
        self.file_list = [
            os.path.join(data_dir, f) 
            for f in os.listdir(data_dir) 
            if f.endswith('.npz')
        ]

        if compute_stats and self.file_list:
            self.global_min, self.global_max, self.global_mean = self._compute_global_stats()
        else:
            self.global_min = global_min
            self.global_max = global_max
            self.global_mean = global_mean

        if self.normalise and (self.global_min is None or 
                               self.global_max is None or 
                               self.global_mean is None):
            raise ValueError(
                "No global stats provided or computed, but normalisation=True. "
                "Please set `compute_stats=True` or pass stats from the train set."
            )

    def _compute_global_stats(self):
        all_points = []
        for npz_path in self.file_list:
            data = np.load(npz_path)
            if 'vertices' in data:
                all_points.append(data['vertices'])

        all_points = np.concatenate(all_points, axis=0)  # Merge all point clouds

        global_min = all_points.min(axis=0)
        global_max = all_points.max(axis=0)
        global_mean = all_points.mean(axis=0)

        return global_min, global_max, global_mean

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        npz_path = self.file_list[idx]
        data = np.load(npz_path)

        if 'vertices' not in data or 'labels' not in data:
            raise ValueError(
                f"Invalid data format in {npz_path}. Required keys: 'vertices', 'labels'."
            )

        vertices = np.asarray(data['vertices'], dtype=np.float32)  # (N, 3)
        labels = np.asarray(data['labels'], dtype=np.int64)        # (N,)

        # Apply label dilation
        labels = self._dilate_labels(vertices, labels, self.dilation_radius)

        # Downsampling/Upsampling tfor class balance
        vertices, labels = self._balance_classes(vertices, labels)

        # Normalisation
        if self.normalise:
            vertices = self._normalise_point_cloud(vertices)

        # Visualisation if need
        if self.visualise:
            self._visualise_pointcloud(vertices, labels)

        vertices_t = torch.tensor(vertices, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.long)
        return vertices_t, labels_t

    def _dilate_labels(self, vertices, labels, dilation_radius):
        tree = KDTree(vertices)
        new_labels = labels.copy()

        for class_id in [1, 2]:  # Ridge and Ligament
            class_mask = labels == class_id
            if np.any(class_mask):
                class_points = vertices[class_mask]
                neighbors = tree.query_ball_point(class_points, dilation_radius)
                for neighbor_idx in neighbors:
                    new_labels[neighbor_idx] = class_id

        return new_labels

    def _balance_classes(self, vertices, labels):
        lig_mask      = (labels == 2)
        lig_vertices  = vertices[lig_mask]
        lig_labels    = labels[lig_mask]
        n_lig         = lig_vertices.shape[0]

        # If ligament alone exceeds num_points => partial sampling
        if n_lig > self.num_points:
            idx_lig = self._farthest_point_sampling(lig_vertices, self.num_points)
            lig_vertices = lig_vertices[idx_lig]
            lig_labels   = lig_labels[idx_lig]

            return lig_vertices, lig_labels

        # 2) Gather other classes
        other_mask     = ~lig_mask
        other_vertices = vertices[other_mask]
        other_labels   = labels[other_mask]
        n_other        = other_vertices.shape[0]
        leftover = self.num_points - n_lig

        # 3) If leftover >= n_other => we can keep all other points
        if n_other <= leftover:
            final_vertices = np.concatenate([lig_vertices, other_vertices], axis=0)
            final_labels   = np.concatenate([lig_labels, other_labels], axis=0)

            # Now we might be below the total. If we still don't have enough, do upsampling
            n_final = final_vertices.shape[0]
            if n_final < self.num_points:
                shortfall = self.num_points - n_final

                idx_upsample = np.random.choice(n_final, shortfall, replace=True)
                up_verts = final_vertices[idx_upsample]
                up_labs  = final_labels[idx_upsample]

                final_vertices = np.concatenate([final_vertices, up_verts], axis=0)
                final_labels   = np.concatenate([final_labels,   up_labs ], axis=0)
        else:
            idx_other = self._farthest_point_sampling(other_vertices, leftover)
            samp_verts = other_vertices[idx_other]
            samp_labs  = other_labels[idx_other]

            final_vertices = np.concatenate([lig_vertices, samp_verts], axis=0)
            final_labels   = np.concatenate([lig_labels, samp_labs], axis=0)

        # final_vertices should be exactly self.num_points
        assert final_vertices.shape[0] == self.num_points, \
            f"Got shape {final_vertices.shape[0]} vs. expected {self.num_points}"

        idx_shuffle = np.arange(self.num_points)
        np.random.shuffle(idx_shuffle)
        final_vertices = final_vertices[idx_shuffle]
        final_labels   = final_labels[idx_shuffle]

        return final_vertices, final_labels


    def _normalise_point_cloud(self, points):
        scale = np.maximum(self.global_max - self.global_min, 1e-8)
        points = (points - self.global_mean) / scale

        return points

    def _farthest_point_sampling(self, points, k):
        N = points.shape[0]
        sampled_indices = np.zeros(k, dtype=np.int64)

        # random starting point
        current_index = np.random.randint(N)
        sampled_indices[0] = current_index
        distances = np.full(N, np.inf)

        last_point = points[current_index]

        for i in range(1, k):
            dist_to_last = np.sum((points - last_point) ** 2, axis=1)
            distances = np.minimum(distances, dist_to_last)
            next_index = np.argmax(distances)
            sampled_indices[i] = next_index
            last_point = points[next_index]

        return sampled_indices
    
    def _visualise_pointcloud(self, vertices, labels):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        unique_labels = np.unique(labels)
        for class_id in unique_labels:
            mask = (labels == class_id)
            ax.scatter(
                vertices[mask, 0], 
                vertices[mask, 1], 
                vertices[mask, 2],
                c=CLASS_COLORS[class_id], 
                label=CLASS_NAMES[class_id], 
                s=2
            )
        ax.set_title("3D Point Cloud Visualisation")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend(loc='upper right')
        plt.show()


def get_dataloaders(
    data_split_dir,
    num_points=NUM_POINTS,
    batch_size=BATCH_SIZE,
    num_workers=8,
    normalise=True,
    visualise=True,
    augment=False,
    fine_tune=False
):
    #### 1) Make a temporary dataset for the train folder to compute stats only
    train_stats_dataset = PointNetDataset(
        data_dir=os.path.join(data_split_dir, 'train'),
        num_points=num_points,
        normalise=False,
        visualise=False,
        augment=False,
        compute_stats=True
    )
    global_min = train_stats_dataset.global_min
    global_max = train_stats_dataset.global_max
    global_mean = train_stats_dataset.global_mean

    # print(global_min, global_max, global_mean)
    if fine_tune:
        data_split_dir = data_split_dir.replace("data_split", "tuning_data_split")

    #### 2) Build the real train dataset that uses those stats
    train_dataset = PointNetDataset(
        data_dir=os.path.join(data_split_dir, 'train'),
        num_points=num_points,
        normalise=normalise,
        visualise=visualise,
        augment=augment,
        global_min=global_min,
        global_max=global_max,
        global_mean=global_mean,
        compute_stats=False
    )

    #### 3) Build val dataset with train’s stats
    val_dataset = PointNetDataset(
        data_dir=os.path.join(data_split_dir, 'val'),
        num_points=num_points,
        normalise=normalise,
        visualise=visualise,
        augment=augment,
        global_min=global_min,
        global_max=global_max,
        global_mean=global_mean,
        compute_stats=False
    )

    #### 4) Build test dataset with train’s stats
    test_dataset = PointNetDataset(
        data_dir=os.path.join(data_split_dir, 'test'),
        num_points=num_points,
        normalise=normalise,
        visualise=visualise,
        augment=False,
        global_min=global_min,
        global_max=global_max,
        global_mean=global_mean,
        compute_stats=False
    )

    #### 5) Create the actual DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
