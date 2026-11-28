# Dataset card: NYU Depth V2 (labeled) with the Eigen split

- **Dataset IDs:** `vision.nyuv2.test`, `vision.nyuv2.calibration`
- **Source URL / version:** https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html; `nyu_depth_v2_labeled.mat` and the Eigen `splits.mat`. SHA-256:
  - `nyu_depth_v2_labeled.mat` (2972037809 bytes): `2d724b0c0ab358aa1ce5df855e5bd14a2279ab7202efe22f32a30d612dcb86aa`
  - `splits.mat` (2626 bytes): `6e081404491a8bfba2f066beaa9713ef6046f9007aa5f799f2a350506a580cee`
  - Computed on copies downloaded again on 2026-11-27, after the reported run. The files used in that run were not hashed.
- **Download:** `splits.mat` is small; the labeled file is about 2.8 GB.
  ```bash
  wget -O splits.mat http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat
  wget -O nyu_depth_v2_labeled.mat http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
  sha256sum nyu_depth_v2_labeled.mat splits.mat
  ```
  Checked on 2026-11-28: both links answer with the byte sizes above, and the `splits.mat` download matches its checksum. The large file was not downloaded again for this check. Review the data terms before downloading.
- **License/access terms:** unknown (check the dataset page before redistributing; nothing is committed here)
- **Modality/task:** RGB-D, monocular depth evaluation
- **Role:** test (654 Eigen test frames) and calibration (16 Eigen train frames)
- **Split:** Eigen split from `splits.mat`: `testNdxs` for evaluation, `trainNdxs` for calibration
- **Preprocessing ID/version:** `dav2.hf_processor` (no separate version; fixed by the model revision). Evaluation crop and depth range as in the loader contract.
- **Tokenizer/processor revision:** no tokenizer; image processor at model revision `5426e4f0f36572d16453bbda7a8389317b1bef99`
- **Number of examples:** 1449 frames in the file; 654 test, 795 train
- **Fields:** `images` (1449, 3, 640, 480) uint8, `depths` (1449, 640, 480) float32 metres. Both are stored transposed; the loader returns (480, 640, 3) and (480, 640).
- **Seed/sampling:** calibration = `default_rng(0).choice(train, 16, replace=False)`
- **Cache/download location:** anywhere local, passed with `--mat` and `--splits`

## Loader contract

`load_nyu(mat_path, indices)` -> list of `(rgb uint8 HxWx3, depth float32 HxW)`. `valid_mask(depth)` -> bool HxW (Eigen crop, depth in (0.001, 10]).

There is no collator or padding. The runner preprocesses and evaluates one image at a time: `pixel_values` is float32 (1, 3, 518, 686). No augmentation is applied.

## Minimal smoke test

`python -m pytest -q Quantization_and_Pruning/tests/test_nyuv2_and_config.py` uses tiny synthetic files and needs no download. `run_ora_q.py --num-images 20` is a quick real-data check.

## Leakage and privacy checks

Calibration indices come only from `trainNdxs`; evaluation only from `testNdxs`. The two sets are disjoint by construction of the Eigen split.
