# CT-Derived Colonoscopy Coverage Ground-Truth Benchmark

Code for the benchmark and experiments in:

> *A Multi-Geometry CT-Derived Coverage Ground-Truth Benchmark for Colonoscopy:
> Quantitative Evaluation and Characterization of Coverage Estimation under
> Reconstruction Degradation* (under review, IJCARS).

This repository releases the **code** that builds the benchmark and runs the
evaluation: a fly-through coverage ground-truth engine, depth-only coverage
estimators, a controllable reconstruction-degradation axis, and the
evaluation / statistics / figure scripts.

## What this is

From each public **HQColon** CT colon segmentation we derive a watertight
surface and a centerline, simulate an idealized centerline fly-through, and
ray-cast per-surface seen/unseen labels to obtain an area-weighted *geometric*
coverage ground truth. On the resulting many-geometry benchmark we evaluate
depth-only coverage estimation and test reconstruction-reliability-gated
abstention under a pre-specified effect-size gate.

**Scope note.** The ground truth is *geometric* coverage on CT-derived shapes,
not clinical coverage on real colonoscopy. See the paper for the full bounds of
the claims.

## Data is not redistributed here

The coverage ground truth is **derived from HQColon**, which carries its own
license. This repository does **not** redistribute HQColon, the derived meshes,
or the coverage labels. To reproduce the benchmark:

1. Download HQColon from OSF: <https://doi.org/10.17605/OSF.IO/8TKPM>
   (read and comply with its license terms).
2. Run the engine (`engine/`) to regenerate the watertight surfaces,
   centerlines, and per-surface coverage ground truth.
3. Build the patient-level split (`configs/make_split.py`). The exact split used
   in the paper is in `configs/split_patients.json` (SHA-256
   `20991c76...f36088`; train/cal/test/holdout = 118/54/44/29 patients).

## Layout

| Path | Contents |
|---|---|
| `engine/` | coverage ground-truth engine (mesh, centerline, fly-through, ray-cast) |
| `training/` | depth-only coverage estimators (model / dataset / train / eval) |
| `gates/` | evaluation gates (depth-modality, degradation, bootstrap) |
| `recount/` | statistics / AURC recount and seed verification |
| `scripts/` | evidence-figure generation (`make_evidence_figures.py`, ...) |
| `protocol/` | study protocol and metric definition |
| `configs/` | split construction and the locked patient-level split |

## Requirements

Python 3.10+ and the packages in `requirements.txt`
(`numpy`, `scipy`, `scikit-image`, `SimpleITK`, `matplotlib`, `torch`).

## Citation

The BibTeX entry will be added once the paper is accepted.

## License

Code in this repository is released under the MIT License (see `LICENSE`).
HQColon and any data derived from it remain under their respective licenses.
