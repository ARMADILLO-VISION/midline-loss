import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from config import CLASS_NAMES, CLASS_COLORS

# -----------------------------------------------------------------------------
# ONE-OFF OBJ EXPORT
# -----------------------------------------------------------------------------
def export_mesh_once(base_dir: str, vertices: np.ndarray):
    mesh_path = os.path.join(base_dir, "mesh.obj")
    if os.path.exists(mesh_path):
        return
    os.makedirs(base_dir, exist_ok=True)
    with open(mesh_path, "w") as f:
        for x,y,z in vertices:
            f.write(f"v {x} {y} {z}\n")
    # print(f"[VIS] mesh.obj written → {mesh_path}")

# -----------------------------------------------------------------------------
# PER-EPOCH LABELS JSON
# -----------------------------------------------------------------------------
def save_epoch_labels_json(epoch: int,
                           labels: np.ndarray,
                           base_dir: str):
    """
    Writes base_dir/labels/epoch_XXX.json containing {"labels":[…]}.
    """
    labels_dir = os.path.join(base_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)

    out = {
        "labels": labels.tolist()
    }
    path = os.path.join(labels_dir, f"epoch_{epoch:03d}.json")
    with open(path, "w") as fp:
        json.dump(out, fp, indent=2)
    # print(f"[VIS] labels JSON → {path}")

# -----------------------------------------------------------------------------
# PER-EPOCH PNG
# -----------------------------------------------------------------------------
def save_epoch_prediction_png(epoch: int,
                              points: np.ndarray,
                              preds: np.ndarray,
                              save_dir: str):
    png_dir = os.path.join(save_dir, "pngs")
    os.makedirs(png_dir, exist_ok=True)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    ax.axis("off")

    for cls in range(len(CLASS_NAMES)):
        mask = preds == cls
        if not mask.any():
            continue
        # Set size based on class name
        size = 1 if CLASS_NAMES[cls].lower() == 'liver' else 10
        ax.scatter(
            points[mask, 0], points[mask, 1], points[mask, 2],
            s=size,
            color=CLASS_COLORS[cls],
            depthshade=False
        )

    ax.view_init(elev=-8, azim=60)
    out = os.path.join(png_dir, f"epoch_{epoch:03d}.png")
    plt.tight_layout(pad=0)
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    # print(f"[VIS] PNG → {out}")

# -----------------------------------------------------------------------------
# NEW: VISUALISE IN-WINDOW
# -----------------------------------------------------------------------------
def visualize_epoch_obj(epoch: int, base_dir: str):
    """
    Load mesh.obj + labels/epoch_XXX.json and plot 3D scatter in a window.
    """
    mesh_path = os.path.join(base_dir, "mesh.obj")
    labels_path = os.path.join(base_dir, "labels", f"epoch_{epoch:03d}.json")
    if not os.path.exists(mesh_path) or not os.path.exists(labels_path):
        raise FileNotFoundError("Need mesh.obj and labels JSON for epoch")

    # load vertices
    verts = []
    with open(mesh_path) as f:
        for line in f:
            if not line.startswith("v "): continue
            _, x,y,z = line.split()
            verts.append([float(x), float(y), float(z)])
    verts = np.array(verts)   # (N,3)

    # load labels
    with open(labels_path) as f:
        data = json.load(f)
    labels = np.array(data["labels"], dtype=int)

    # plot
    fig = plt.figure(figsize=(8,8))
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    ax.axis("off")

    for cls in range(len(CLASS_NAMES)):
        mask = labels==cls
        if not mask.any(): continue
        ax.scatter(verts[mask,0], verts[mask,1], verts[mask,2],
                   s=5, color=CLASS_COLORS[cls], depthshade=False)

    ax.view_init(elev=-8, azim=60)
    plt.show()

# -----------------------------------------------------------------------------
# SINGLE ENTRYPOINT
# -----------------------------------------------------------------------------
def process_epoch(epoch: int,
                  points: np.ndarray,
                  preds: np.ndarray,
                  base_dir: str,
                  visualise: bool = False):
    """
    Writes mesh.obj once, then per-epoch labels JSON.
    If visualise=True: show in a window.
    Else: also save the PNG.
    """
    # 1) Always export the mesh if not already done
    export_mesh_once(base_dir, points)

    # 2) Always save the labels JSON for this epoch
    save_epoch_labels_json(epoch, preds, base_dir)

    if visualise:
        save_epoch_prediction_png(epoch, points, preds, base_dir)
        # 3a) Now that mesh.obj & labels exist, open a window
        visualize_epoch_obj(epoch, base_dir)
    else:
        # 3b) Save the per-epoch PNG
        save_epoch_prediction_png(epoch, points, preds, base_dir)
