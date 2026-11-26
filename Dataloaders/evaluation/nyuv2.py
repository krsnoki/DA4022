"""NYU Depth V2 (labeled subset) with the Eigen split.

Files (not committed, download them yourself):
  nyu_depth_v2_labeled.mat  HDF5 (MATLAB v7.3), 1449 RGB-D frames, about 2.8 GB
  splits.mat                Eigen split: testNdxs (654), trainNdxs (795), 1-based

Roles:
  vision.nyuv2.test         the 654 Eigen test images, evaluation only
  vision.nyuv2.calibration  16 images drawn from the 795 train images (seed 0)
Calibration never touches the test split.
"""

import h5py
import numpy as np
import scipy.io

EIGEN_CROP = (45, 471, 41, 601)      # rows 45:471, cols 41:601 of the 480x640 frame
MIN_DEPTH, MAX_DEPTH = 1e-3, 10.0


def load_split_indices(split_path):
    s = scipy.io.loadmat(split_path)
    return {
        "test": (s["testNdxs"].ravel() - 1).astype(int).tolist(),     # MATLAB is 1-based
        "train": (s["trainNdxs"].ravel() - 1).astype(int).tolist(),
    }


def sample_calibration(train_indices, n=16, seed=0):
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(train_indices, size=n, replace=False).tolist())


def load_nyu(mat_path, indices):
    """List of (rgb uint8 480x640x3, depth float32 480x640 in metres)."""
    out = []
    with h5py.File(mat_path, "r") as f:
        images, depths = f["images"], f["depths"]
        for i in indices:
            rgb = np.asarray(images[i]).transpose(2, 1, 0)              # (3,640,480) -> (480,640,3)
            depth = np.asarray(depths[i]).T.astype(np.float32)          # (640,480) -> (480,640)
            out.append((np.ascontiguousarray(rgb, dtype=np.uint8), depth))
    return out


def valid_mask(depth):
    m = (depth > MIN_DEPTH) & (depth <= MAX_DEPTH)
    crop = np.zeros_like(m)
    r0, r1, c0, c1 = EIGEN_CROP
    crop[r0:r1, c0:c1] = True
    return m & crop
