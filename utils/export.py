import os
import json
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import SAVE_DIR, NUM_CLASSES, NPZ_PATH
from pointnet_data_loader import get_dataloaders
from models.pointnet2_model import PointNet2Seg


def export_obj_json(output_dir, vertices, labels):
    os.makedirs(output_dir, exist_ok=True)

    obj_path = os.path.join(output_dir, "mesh.obj")
    json_path = os.path.join(output_dir, "labels.json")

    with open(obj_path, "w") as f:
        for (x, y, z) in vertices:
            f.write(f"v {x} {y} {z}\n")

    if torch.is_tensor(labels):
        labels_list = labels.cpu().numpy().tolist()
    else:
        labels_list = labels.tolist()

    data_dict = {"labels": labels_list}
    with open(json_path, "w") as jf:
        json.dump(data_dict, jf, indent=2)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, test_loader = get_dataloaders(
        data_split_dir=args.data_split_dir,
        num_points=args.num_points,
        batch_size=1,
        normalise=True,
        visualise=False,
        augment=False,
        fine_tune=False
    )

    model = PointNet2Seg(num_classes=NUM_CLASSES).to(device)
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    for i, (points, _) in enumerate(test_loader):
        # points: [B=1, N, 3]
        points = points.to(device)
        with torch.no_grad():
            seg_logits, pred_coords = model(points)
            refined_probs = F.softmax(seg_logits, dim=-1)  # shape [1, N, C]
            _, predicted_labels = torch.max(refined_probs, dim=2)  # shape [1, N]

        final_verts = points[0].cpu().numpy()
        final_labels = predicted_labels[0].cpu().numpy()

        sample_dir = os.path.join(args.output_dir, f"patient_{i}")
        export_obj_json(sample_dir, final_verts, final_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_split_dir", type=str, default=NPZ_PATH,
    )
    parser.add_argument(
        "--checkpoint", type=str,
    )
    parser.add_argument(
        "--output_dir", type=str, default="test_inference_outputs",
    )
    parser.add_argument(
        "--num_points", type=int, default=4096,
    )
    args = parser.parse_args()

    main(args)
