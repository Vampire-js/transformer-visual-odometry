# Visual Odometry — Transformer-style Baseline

A minimal baseline for monocular visual odometry that regresses the **relative camera pose** between two consecutive frames using a patch-embedding image encoder and an MLP head.

This repo is intentionally small: it is meant as a starting point that can later be extended into a full Vision Transformer / cross-attention architecture.

---

## Idea

Given two consecutive RGB frames $I_t$ and $I_{t+1}$ from a driving sequence, predict the rigid-body transform $T_{t \to t+1} \in SE(3)$ that maps the camera frame at time $t$ to the camera frame at time $t+1$.

The target is the flattened $3 \times 4$ pose matrix (12 numbers): the top three rows of the $4 \times 4$ homogeneous transform.

---

## Architecture

```
        ┌──────────────┐                ┌──────────────┐
  I_t ─►│  Encoder (E) │── z1 ──┐  ┌──── z2 ──┤  Encoder (E) │◄─ I_{t+1}
        └──────────────┘        │  │          └──────────────┘
                                ▼  ▼
                  [ z1 | z2 | (z2 − z1) ]   (3 · D)
                                │
                                ▼
                          mean over patches
                                │
                                ▼
                  ┌─────────────────────────┐
                  │  MLP: 3D → H → H → 12   │
                  └─────────────────────────┘
                                │
                                ▼
                  T_rel ∈ R^{12}  (flattened 3×4)
```

### Patch Encoder

The encoder turns an image into a set of patch embeddings using a single strided convolution — equivalent to non-overlapping linear patch projection used in ViT:

- `Conv2d(3, EMBED_DIM, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)`
- Output is reshaped to `(B, N_patches, EMBED_DIM)`.

The **same encoder weights** are shared between the two frames (Siamese-style).

### Pose Head

Per-patch features from both frames are combined as

$$
z = [\, z_1 \;\|\; z_2 \;\|\; z_2 - z_1 \,] \in \mathbb{R}^{N \times 3D}
$$

then mean-pooled over patches to a single $3D$ vector, and passed through a small MLP that outputs a 12-dim vector — the flattened $3 \times 4$ relative transform.

### Hyperparameters (defaults in [main.py](main.py))

| Name         | Value | Meaning                              |
| ------------ | ----- | ------------------------------------ |
| `EMBED_DIM`  | 128   | Per-patch embedding dimension        |
| `PATCH_SIZE` | 16    | Patch side length in pixels          |
| `MLP_NODES`  | 128   | Hidden width of the regression head  |
| `OUT_DIM`    | 12    | Flattened 3×4 relative pose          |

---

## Dataset

[KITTI Odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php). The loader is in [utils/KITTIOdometryDataset.py](utils/KITTIOdometryDataset.py).

Expected layout:

```
<root>/
├── sequences/
│   └── 00/
│       └── image_2/      # left color images: 000000.png, 000001.png, ...
└── poses/
    └── 00.txt            # one 3×4 row-major pose per line (cam-to-world)
```

For each index `i`, the dataset yields the consecutive pair $(I_i, I_{i+1})$ and computes the relative pose label as

$$
T_{\text{rel}} = T_i^{-1} \, T_{i+1}
$$

where $T_i, T_{i+1}$ are the absolute poses from `poses/<seq>.txt`. The top $3 \times 4$ block is flattened into a 12-vector used as the regression target.

Each sample is a dict:

```python
{
  "img1":   FloatTensor [3, H, W],
  "img2":   FloatTensor [3, H, W],
  "target": FloatTensor [12],   # flattened 3×4 relative pose
}
```

---

## Project Layout

```
.
├── main.py                       # model + training loop
├── vo1.ipynb                     # exploratory notebook
├── pyproject.toml
├── README.md
└── utils/
    └── KITTIOdometryDataset.py   # KITTI loader, relative-pose targets
```

---

## Setup

Requires Python ≥ 3.12. Using [uv](https://docs.astral.sh/uv/):

```bash
uv sync
source .venv/bin/activate
```

Or with pip:

```bash
pip install "torch>=2.12.0" "torchvision>=0.27.0" "matplotlib>=3.10.9"
```

---

## Training

Edit the dataset `root` in [main.py](main.py) to point at your local KITTI odometry directory, then:

```bash
python main.py
```

Defaults:

- Optimizer: Adam, `lr = 1e-3`
- Loss: MSE on the 12-dim flattened pose vector
- Split: 80% train / 20% test, random split of sequence `00`
- Batch size: 16, Epochs: 10

A `transform` (e.g. `torchvision.transforms.Compose([Resize(...), ToTensor()])`) must be provided to the dataset to convert PIL images to tensors of a consistent size.

---

## Known Limitations

This is a **baseline**, not a competitive VO system:

- The MLP regresses pose components directly; the predicted $3 \times 4$ block is **not constrained to lie on $SE(3)$** (rotation part is not orthonormal).
- MSE on raw matrix entries weights translation and rotation arbitrarily.
- No positional embeddings, no self-attention, no temporal context beyond two frames.
- No data normalization of pose targets — translation magnitudes between consecutive KITTI frames are small, which can make the loss landscape ill-conditioned.
- Trains and evaluates on the same sequence (`00`); no cross-sequence generalization is measured.

These are intentional starting points for follow-up experiments (ViT encoder, $SO(3)$-aware loss, sequence models, etc.).
