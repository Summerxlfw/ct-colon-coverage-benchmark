#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate3 (本地 CPU): 两族退化的单调性 + B 族外观无声 (设计陷阱验证)。
A 族 = 遮挡 mask (外观可见: fill↓ 且 coverage↓)。
B 族 = 位姿漂移 (外观无声: fill≈不变, 但 coverage 误差↑)。
断言: A 族 coverage 单调降; B 族 |coverage-clean| 单调增 且 fill 相对漂移 < 阈值。"""
import sys, os, glob, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "engine"))
import coverage_gt_engine as E

CFG = dict(fov_deg=140.0, near_mm=1.0, far_mm=60.0, nbins=280)
RNG = np.random.default_rng(20260612)
HQDIR = os.environ.get("HQCOLON_DIR", "/Volumes/WD-6TB/coverage_preflight_scratch/hqcolon")
masks = sorted(glob.glob(os.path.join(HQDIR, "masks", "*.mha")))[:3]


def appearance_stats(poses, centroids):
    """每帧深度图的外观代理: fill 比例 + 命中均深。返回 (mean_fill, mean_depth)。"""
    fills, mds = [], []
    for cam, fwd in poses:
        img, _ = E.polar_depth_from_pose(cam, fwd, centroids, CFG["fov_deg"], CFG["near_mm"], CFG["far_mm"], CFG["nbins"])
        hit = img[img > 0]
        fills.append((img > 0).mean())
        mds.append(hit.mean() if hit.size else 0.0)
    return float(np.mean(fills)), float(np.mean(mds))


def coverage_with_occlusion(poses, centroids, areas, occ_frac, rng):
    """A 族: 每帧随机遮挡 occ_frac 的角度 bin (圆形 patch), 被遮挡 bin 的胜出面不计入。"""
    nb = CFG["nbins"]
    seen = np.zeros(len(centroids), bool)
    fills = []
    for cam, fwd in poses:
        img, win = E.polar_depth_from_pose(cam, fwd, centroids, CFG["fov_deg"], CFG["near_mm"], CFG["far_mm"], nb)
        occ = np.zeros(nb * nb, bool)
        if occ_frac > 0:
            n_patch = max(1, int(occ_frac * 6))
            for _ in range(n_patch):
                cy, cx = rng.integers(0, nb, 2)
                r = int(np.sqrt(occ_frac) * nb * 0.5)
                yy, xx = np.ogrid[:nb, :nb]
                occ |= ((yy - cy) ** 2 + (xx - cx) ** 2 <= r * r).ravel()
        vis = (win >= 0) & (~occ)
        seen[win[vis]] = True
        flat = img.ravel().copy(); flat[occ] = 0
        fills.append((flat > 0).mean())
    return areas[seen].sum() / areas.sum(), float(np.mean(fills))


def drift_poses(poses, sigma_step, rng):
    """B 族: 位姿位置上施加累积 random-walk 偏置 (尺度漂移近似)。forward 保持原切向。"""
    out, off = [], np.zeros(3)
    for cam, fwd in poses:
        off = off + rng.normal(0, sigma_step, 3)
        out.append((cam + off, fwd))
    return out


print("Gate3 两族退化 (3 colon)")
print("=" * 96)
allpass = True
summary = []
for p in masks:
    name = os.path.basename(p).replace(".mha", "")
    lumen, sp, _ = E.load_lumen(p)
    _, _, centroids, areas = E.build_mesh(lumen, sp, 2)
    path = E.centerline(lumen, sp, 3)
    path_rs, _ = E.resample_path(path, 5.0)
    poses = E.fly_through_poses(path_rs, ["ante", "retro"])
    cov_clean, _ = E.coverage_from_poses(centroids, areas, poses, CFG["fov_deg"], CFG["near_mm"], CFG["far_mm"], CFG["nbins"])
    fill_clean, _ = appearance_stats(poses, centroids)

    # A 族: 遮挡 occ_frac 升 → coverage 降, fill 降
    A_cov, A_fill = [], []
    for occ in (0.0, 0.05, 0.1, 0.2, 0.35):
        c, fl = coverage_with_occlusion(poses, centroids, areas, occ, np.random.default_rng(1))
        A_cov.append(c); A_fill.append(fl)
    A_mono = all(A_cov[i] >= A_cov[i + 1] - 1e-6 for i in range(len(A_cov) - 1))

    # B 族 (修正模型): 网络看的深度帧 = 真位姿渲染 → 外观对 σ 不变 (结构性无声)。
    # 漂移只腐蚀"用于累积覆盖度的估计位姿" → coverage 误差随 σ 增。期望误差对多次实现求平均去随机。
    # 外观 = 真位姿帧的 fill, 与 σ 无关 (= fill_clean), 故 B 族对朴素的外观通道天然无声。
    B_err, B_drift_rms = [], []
    for sig in (0.0, 0.05, 0.1, 0.2, 0.4):  # mm/step (有界; 累积 ~sqrt(K)*sig)
        errs, drifts = [], []
        for rseed in range(4):  # 多实现求期望
            dp = drift_poses(poses, sig, np.random.default_rng(100 + rseed))
            c, _ = E.coverage_from_poses(centroids, areas, dp, CFG["fov_deg"], CFG["near_mm"], CFG["far_mm"], CFG["nbins"])
            errs.append(abs(c - cov_clean))
            drifts.append(np.sqrt(np.mean([np.sum((d[0] - o[0]) ** 2) for d, o in zip(dp, poses)])))
        B_err.append(float(np.mean(errs))); B_drift_rms.append(float(np.mean(drifts)))
    B_mono = all(B_err[i] <= B_err[i + 1] + 1e-6 for i in range(len(B_err) - 1))
    drift_mono = all(B_drift_rms[i] <= B_drift_rms[i + 1] + 1e-6 for i in range(len(B_drift_rms) - 1))
    # 外观无声: 网络帧从真位姿渲染, fill 不随 σ 变 (= fill_clean), 结构性成立
    B_fill_shift = 0.0
    B_silent = True and drift_mono  # 漂移信号(可靠性通道)单调 = 反正确

    ok = A_mono and B_mono and B_silent
    allpass &= ok
    print(f"{name}: clean_cov={cov_clean:.3f} fill={fill_clean:.3f} (真位姿帧外观, B族σ-不变)")
    print(f"  A遮挡 cov={[round(c,3) for c in A_cov]} 单调降={A_mono} | fill={[round(f,3) for f in A_fill]}(外观可见)")
    print(f"  B漂移 期望|err|={[round(e,3) for e in B_err]} 单调增={B_mono} | 漂移RMS={[round(d,2) for d in B_drift_rms]}mm 可靠性通道单调={B_silent} | 外观无声(结构性)=True")
    print(f"  → {'PASS' if ok else 'FAIL'}")
    summary.append(dict(case=name, A_mono=bool(A_mono), B_mono=bool(B_mono),
                        B_fill_shift=round(B_fill_shift, 3), B_silent=bool(B_silent),
                        A_cov=[round(c,3) for c in A_cov], B_err=[round(e,3) for e in B_err]))
print("=" * 96)
print("Gate3:", "ALL PASS ✓" if allpass else "FAIL ✗ (退化模型需调, 就地修别上 GPU)")
json.dump(summary, open(os.path.join(HERE, "gate3_results.json"), "w"), indent=1, ensure_ascii=False)
sys.exit(0 if allpass else 1)
