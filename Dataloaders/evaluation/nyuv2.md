# Dataset card: NYU Depth V2 (labeled) with the Eigen split

- **Dataset IDs:** `vision.nyuv2.test`, `vision.nyuv2.calibration`
- **Source URL / version:** https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html; `nyu_depth_v2_labeled.mat` and the Eigen `splits.mat`. SHA-256:
  - `nyu_depth_v2_labeled.mat` (2972037809 bytes): `2d724b0c0ab358aa1ce5df855e5bd14a2279ab7202efe22f32a30d612dcb86aa`
  - `splits.mat` (2626 bytes): `6e081404491a8bfba2f066beaa9713ef6046f9007aa5f799f2a350506a580cee`
  - Computed on copies downloaded again on 2026-11-27, after the reported run. The files used in that run were not hashed.
- **License/access terms:** unknown (check the dataset page before redistributing; nothing is committed here)
- **Modality/task:** RGB-D, monocular depth evaluation
- **Role:** test (654 Eigen test frames) and calibration (16 Eigen train frames)
- **Number of examples:** 1449 frames in the file; 654 test, 795 train
- **Fields:** `images` (1449, 3, 640, 480) uint8, `depths` (1449, 640, 480) float32 metres. Both are stored transposed; the loader returns (480, 640, 3) and (480, 640).
- **Seed/sampling:** calibration = `default_rng(0).choice(train, 16, replace=False)`
- **Cache/download location:** anywhere local, passed with `--mat` and `--splits`

## Loader contract

`load_nyu(mat_path, indices)` -> list of `(rgb uint8 HxWx3, depth float32 HxW)`. `valid_mask(depth)` -> bool HxW (Eigen crop, depth in (0.001, 10]).

## Minimal smoke test

The unit tests do not need the file. `run_ora_q.py --num-images 20` is a quick real-data check.

## Leakage and privacy checks

Calibration indices come only from `trainNdxs`; evaluation only from `testNdxs`. The two sets are disjoint by construction of the Eigen split.
