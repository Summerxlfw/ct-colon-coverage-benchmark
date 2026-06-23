# Study Protocol

## Study Objective

- Primary objective: evaluate whether reconstruction-reliability-gated selective coverage estimation yields (a) a risk–coverage curve (AURC) that dominates a non-selective estimator and a naive image-quality/confidence abstention baseline, and (b) cross-geometry empirical interval coverage near nominal (1−α), on a multi-geometry CT-mesh benchmark with exact geometric coverage GT.
- Secondary objective: characterize abstention behavior under controllable reconstruction degradations (blur, frame dropout, fluid/occlusion, depth noise); phantom-anchored realism check on C3VDv2.

## Data Sources

| Dataset | Role | N | Label source | Access / license | Notes |
|---|---|---:|---|---|---|
| HQColon (CT meshes) | train / val / cal / test (geometry-split) | 315 patients (435 supine/prone scans) | geometric coverage GT via fly-through + raycast | OSF 10.17605/OSF.IO/8TKPM; **dataset license TBD-verify (poss. CC BY-NC-ND)** | spot-check 5/5 watertight; usable full count pending |
| C3VDv2 | realism / abstention-failure anchor (not calibration substrate) | 2 geometries, 192 videos | shipped mesh coverage GT + artifacts | CC BY 4.0 (JHU, doi:10.7281/T1/JC64MK) | 2 geometries → realism anchor only, NOT cross-geometry calibration |
| SimCol3D / VR-Caps (optional) | rendered-RGB sim-to-real bridge | 3 / several | depth+pose | CC BY / GitHub (verify) | optional; orthogonal to coverage GT |

## Split Plan

- Split unit: **colon geometry** (HQColon: patient-level; supine+prone of one patient stay in the same split). NOT by frame, scan, or texture.
- Train: subset of HQColon geometries.
- Validation: held-out HQColon geometries (model selection only).
- Calibration: independent HQColon geometries from the train side (conformal calibration set; ≥~50 geometries so cross-geometry realized-coverage band is tight — pre-flight [B2]).
- Test: HQColon geometries fully held out from train/val/cal; + C3VDv2 as separate realism-anchor test.
- External / hidden holdout: a reserved block of HQColon geometries never touched until final; optionally raw-TCIA re-segmented geometries.
- Leakage risks: supine/prone of same patient across splits (same anatomy); C3VDv2 textures/trajectories of one geometry across splits; conformal cal containing train or test geometries.
- Leakage checks: per-scan geometry/patient-id manifest; assert split disjointness at patient level; **split hash on disk**; conformal cal ∩ (train ∪ test) = ∅ (memory: conformal exchangeability, drop train).

## Model / Method

- Method under test: reconstruction-reliability-gated selective coverage estimator = coverage head + reconstruction-reliability signal (independent of coverage GT) + split-conformal interval + abstention threshold τ (selected on calibration set, never test). [architecture details: G2]
- Frozen components: reconstruction/depth backbone (borrowed, e.g. Gaussian Pancakes / existing depth net) — TBD.
- Trainable components: coverage head + reliability head — TBD.
- Pretraining: TBD (depth/reconstruction backbone may be pretrained).
- Initialization: TBD.

## Baselines

See `baseline_registry.yaml`.

## Metrics

See `metric_definition.md`.

## Statistical Plan

- Primary endpoint: AURC (area under risk–coverage curve) + cross-geometry empirical interval coverage gap |empirical − nominal| at α=0.10.
- Confidence interval: **paired cluster bootstrap over patients (subject_id)** (binding独立单元=病人; supine/prone 同病人同 cluster; 71/245 病人双扫描, 按几何 bootstrap=伪复制; frame/scan/geometry counts 不虚增 n)。
- Significance test: non-overlapping cluster-bootstrap CI between method and naive-confidence baseline at matched operating points.
- Multiple comparison handling: report all operating points; pre-register the primary operating point.
- Seed policy: ≥3 seeds; escalate to ≥5 if borderline (GSDS lesson: across-seed CI must exclude the no-effect point for any significance claim, else report as directional).

## Fairness Rules

- Same data access: all methods see identical train/cal geometries; none see test.
- Same augmentation policy: identical degradation/augmentation pipeline across method and baselines.
- Same early stopping policy: identical val-based stopping.
- Same hyperparameter budget: identical tuning budget; abstention/conformal thresholds chosen on calibration set only.
- Same post-processing: identical mesh/raycast coverage-GT pipeline; identical interval construction protocol where applicable.

## Change Control

Protocol changes after first result must be logged here:

| Date | Change | Reason | Affected results | Approved by |
|---|---|---|---|---|
| 2026-06-12 | Protocol locked at G1 (pre-experiment) | 立项 from pre-flight verdict | none yet | summer |
