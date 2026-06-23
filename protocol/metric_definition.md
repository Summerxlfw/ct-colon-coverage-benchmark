# Metric Definition

## Primary Metric

- Name: AURC (Area Under the Risk–Coverage curve)
- Formula: AURC = ∫_0^1 risk(c) dc, where for retained fraction c (= 1 − abstention rate), risk(c) = mean |Ĉ − C_GT| over the retained geometries/segments ranked by the reliability signal; coverage error |Ĉ − C_GT| uses geometric coverage C ∈ [0,1] = examined_surface_area / total_surface_area.
- Unit: dimensionless (coverage-error × retained-fraction)
- Direction: lower is better
- Aggregation level: **patient (subject_id)** (cluster bootstrap over **patients** for CI; supine/prone 同病人同 cluster)
- Confidence interval: 95% **paired** cluster bootstrap, resampling unit = **patient** (NOT geometry; 71/245 病人有双扫描, 按几何 bootstrap=伪复制→假 YES), ≥1000 resamples

## Secondary Metrics

| Metric | Formula | Purpose | Direction |
|---|---|---|---|
| AURC | Area Under the Risk–Coverage curve (PRIMARY endpoint; full definition in the Primary Metric section above) | selective-prediction / risk–coverage ranking quality | lower |
| Cross-geometry interval coverage | empirical fraction of held-out geometries whose true coverage lies in the predicted (1−α) interval | calibration / validity (now powered via HQColon n_cal≥50) | → nominal (1−α) |
| Coverage MAE | mean |Ĉ − C_GT| at full coverage (no abstention) | point accuracy (report with CI, not pinned) | lower |
| ECE (coverage interval) | expected calibration error of predicted intervals | calibration quality | lower |
| Abstention rate @ target risk | fraction abstained to reach a target retained risk | operating-point characterization | context |
| Retained MAE @ τ | mean error on retained set at abstention rate τ | risk-coverage operating points | lower |

## Exclusion Rules

- Invalid sample: a geometry whose mask fails watertight single-component meshing is excluded at the data stage (logged), NOT silently dropped.
- Missing label: n/a (coverage GT is computed, always available for meshed geometries).
- Ambiguous label: n/a.
- Failed inference: handled by **abstention**, not exclusion — a frame/segment the method cannot estimate is abstained and counted in the risk-coverage accounting, never dropped from the denominator.

## Reporting Rules

- Decimal places: 3 for coverage/MAE/coverage-gap; 3 for AURC.
- CI format: point [lo, hi] 95% cluster-bootstrap over geometries.
- Per-dataset reporting: HQColon (primary, cross-geometry) and C3VDv2 (anchor) reported separately; never pooled.
- Macro / micro averaging: **patient-level macro** (each病人 weighted equally) is primary; geometry/segment-level as secondary.
- Patient-level vs image-level: **patient (subject_id)** is the binding unit for all CIs (NOT geometry, NOT frame). eval_test.csv 每行带 subject_id, bootstrap 按 subject_id 分组。

## Metric Anti-cheating Checks

- No test-set threshold tuning (abstention τ and conformal quantile chosen on calibration set only).
- No metric switching after seeing results (AURC + cross-geometry interval coverage pre-registered as primary).
- No cherry-picking best seed as main result (report seed distribution; ≥3 seeds).
- If a metric is exploratory, label it as exploratory (C3VDv2 sim-to-real / real-video proxies are exploratory).
- Coverage interval coverage is reported with a cluster-bootstrap CI; a single point estimate near nominal is NOT a validity claim.
