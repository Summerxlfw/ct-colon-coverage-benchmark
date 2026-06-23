#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate1 (本地 CPU, 上 GPU 前必过): 深度模态 + 一致性。
从引擎极坐标 z-buffer 出逐位姿深度图 → 重算覆盖度 vs 引擎 fly_through_coverage 差 ≤ ±0.02。
并报深度张量统计 (shape/fill/range/有无全空帧)。在 5 个 pilot colon 上。"""
import sys, os, glob, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "engine"))
import coverage_gt_engine as E

CFG = dict(fov_deg=140.0, pose_step_mm=5.0, near_mm=1.0, far_mm=60.0, nbins=280, n_seg=6,
           directions=["ante", "retro"])
TOL = 0.02
HQDIR = os.environ.get("HQCOLON_DIR", "/Volumes/WD-6TB/coverage_preflight_scratch/hqcolon")
masks = sorted(glob.glob(os.path.join(HQDIR, "masks", "*.mha")))

print("Gate1 深度模态 + 一致性 (tol ±%.2f)" % TOL)
print("=" * 92)
allpass = True
rows = []
for p in masks:
    name = os.path.basename(p).replace(".mha", "")
    lumen, sp, _ = E.load_lumen(p)
    verts, faces, centroids, areas = E.build_mesh(lumen, sp, downsample=2)
    path = E.centerline(lumen, sp, 3)
    path_rs, _ = E.resample_path(path, CFG["pose_step_mm"])
    # 引擎覆盖度 (seen-set 累积)
    cov_eng, seen_eng, _ = E.fly_through_coverage(centroids, areas, path_rs, CFG)
    # 深度图覆盖度 (从极坐标深度图的胜出面累积) + 深度张量统计
    poses = E.fly_through_poses(path_rs, CFG["directions"])
    seen_depth = np.zeros(len(centroids), bool)
    fills, drng_lo, drng_hi, empty_frames = [], [], [], 0
    for cam, fwd in poses:
        img, win = E.polar_depth_from_pose(cam, fwd, centroids, CFG["fov_deg"], CFG["near_mm"], CFG["far_mm"], CFG["nbins"])
        w = win[win >= 0]
        seen_depth[w] = True
        hit = img[img > 0]
        fills.append((img > 0).mean())
        if hit.size == 0:
            empty_frames += 1
        else:
            drng_lo.append(hit.min()); drng_hi.append(hit.max())
    cov_depth = areas[seen_depth].sum() / areas.sum()
    diff = abs(cov_depth - cov_eng)
    seen_match = float((seen_depth == seen_eng).mean())
    # gate 判据 = 一致性 (diff<=tol); 空帧只警告 (端部位姿看向腔外, >1% 才异常)
    ok = diff <= TOL
    empty_warn = empty_frames > max(1, int(0.01 * len(poses)))
    allpass &= ok and not empty_warn
    print(f"{name}: cov_eng={cov_eng:.4f} cov_depth={cov_depth:.4f} |diff|={diff:.4f} {'PASS' if ok else 'FAIL'}"
          f" | seen一致={seen_match:.4f} K帧={len(poses)} fill均={np.mean(fills):.3f}"
          f" depth范围[{np.min(drng_lo):.1f},{np.max(drng_hi):.1f}]mm 空帧={empty_frames}{' ⚠' if empty_warn else ''}")
    rows.append(dict(case=name, cov_eng=round(cov_eng,4), cov_depth=round(cov_depth,4),
                     diff=round(diff,4), seen_match=round(seen_match,4),
                     K=len(poses), fill_mean=round(float(np.mean(fills)),3),
                     empty_frames=int(empty_frames), pass_=bool(ok)))
print("=" * 92)
print("Gate1:", "ALL PASS ✓" if allpass else "FAIL ✗")
json.dump(rows, open(os.path.join(HERE, "gate1_results.json"), "w"), indent=1)
sys.exit(0 if allpass else 1)
