#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 HQColon 源 zip 抽取指定 colon → 跑覆盖度 engine (--save-render) → 落 *_render.npz。
三图终稿渲染素材的真数据生成器 (本地, CPU)。与全量产 produce_full_gt.py 同一 engine/同一 CFG,
故每几何的 coverage 应与 results/coverage_gt_full435.csv 一致 (可反 source 核)。

用法:
  HQCOLON_DIR=/Volumes/WD-6TB/coverage_preflight_scratch/hqcolon \
  python engine/render_selected_colons.py colon_011 colon_198 colon_080 --out results/render
"""
import argparse, os, sys, json, zipfile, tempfile, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import coverage_gt_engine as E

SCRATCH = os.environ.get("HQCOLON_DIR", "/Volumes/WD-6TB/coverage_preflight_scratch/hqcolon")
ZIP = os.path.join(SCRATCH, "gas-filled.zip")
# 与 produce_full_gt.py 完全一致的产线配置 (口径必须同, 否则 coverage 对不上 full435.csv)
CFG = dict(fov_deg=140.0, pose_step_mm=5.0, near_mm=1.0, far_mm=60.0,
           nbins=280, n_seg=6, directions=["ante", "retro"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="+", help="如 colon_011 colon_198")
    ap.add_argument("--out", default="results/render")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with zipfile.ZipFile(ZIP) as z:
        entries = {os.path.basename(n).replace(".mha", ""): n
                   for n in z.namelist() if n.endswith(".mha")}
    for case in a.cases:
        if case not in entries:
            print(f"[skip] {case}: 不在 zip"); continue
        tmpd = tempfile.mkdtemp()
        try:
            with zipfile.ZipFile(ZIP) as z:
                data = z.read(entries[case])
            local = os.path.join(tmpd, f"{case}.mha")
            with open(local, "wb") as f:
                f.write(data)
            r = E.run(local, a.out, mesh_downsample=2, cl_downsample=3,
                      cfg=CFG, save_npz=False, save_render=True)
            print(json.dumps(dict(case=r["case"], coverage=r["coverage"],
                                  n_faces=r["n_faces"], n_poses=r["n_poses"],
                                  centerline_len_cm=r["centerline_len_cm"],
                                  watertight=r["watertight"]), ensure_ascii=False))
        finally:
            shutil.rmtree(tmpd, ignore_errors=True)


if __name__ == "__main__":
    main()
