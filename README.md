# ct-colon-coverage-benchmark

Code for a CT-derived, multi-geometry benchmark of **geometric** colonoscopy
surface coverage. This repository holds the code that generates the coverage
ground truth, builds the patient-level split, and recounts the reference
coverage-MAE leaderboard. The derived data are released as a separate deposit
(see **Data** below).

> Scope boundary: coverage here is a **geometric surface-visibility fraction**
> computed from CT-derived colon meshes by a scripted virtual fly-through. It is
> **not** clinical mucosal-inspection completeness and **not** real colonoscopy
> video coverage ground truth.

## What is in this repository

```
engine/
  coverage_gt_engine.py      coverage-label generator (mesh -> fly-through -> ray-cast coverage)
  produce_full_gt.py         batch reproduction entry point
  render_selected_colons.py  optional visual rendering
  README.md                  engine pipeline and usage
configs/
  make_split.py              patient-level train/cal/test/holdout split generator
  split_patients.json        the exact split used for the reference leaderboard
recount/
  recount_mae.py             independent per-method coverage-MAE leaderboard recount
```

## Data

The coverage ground truth is **derived from HQColon CT colon segmentations** and
is released as a separate data deposit (per-case coverage labels, quality-control
metadata, the patient-level split, reconstruction-degradation records, and the
reference leaderboard evaluation rows). The deposit link is given in the
accompanying paper's Data Availability statement. The raw HQColon masks are
**not** redistributed here; obtain them from the HQColon DOI:
<https://doi.org/10.17605/OSF.IO/8TKPM> (CC BY 4.0).

## Quick start (recount, no GPU)

`recount/recount_mae.py` needs only `numpy`. Place the released leaderboard
evaluation CSV at `data/coverage_leaderboard_eval.csv`, then:

```bash
python3 recount/recount_mae.py data/coverage_leaderboard_eval.csv --group-col method
```

Expected per-method coverage MAE (patient-level macro):

| method | MAE |
|---|---|
| oracle_upper | ~0.005 |
| attentionpool_head | ~0.074 |
| const_lower | ~0.082 |
| random | ~0.107 |
| visible_area_heuristic | ~0.159 |

To regenerate coverage labels from source meshes (needs the full dependency
stack in `requirements.txt` and the raw HQColon masks), see `engine/README.md`.

## Reuse guidance

- Use the **317 watertight usable** geometries (from 245 patients, of 435
  processed) for coverage statistics; the full set includes QC-failing cases.
- Respect the patient-level split: do not mix same-patient scans across
  train/cal/test/holdout, and do not tune on the test or holdout patients.
- Do not mix lite-mesh and full-mesh coverage ground truth; the full-mesh
  outputs are authoritative.
- Do not interpret these labels as clinical coverage or real-colonoscopy
  completeness.

## License

Code in this repository is released under the MIT License (see `LICENSE`). The
derived-data deposit carries its own license (CC BY 4.0), as does HQColon.
