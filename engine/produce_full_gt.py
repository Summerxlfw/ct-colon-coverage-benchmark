#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全量产覆盖度 GT — 跑全 435 个 HQColon 充气 mask。
流式解压(每个解出→处理→删, 不占 60GB), 6 进程并行, 增量存 CSV, 容错续跑。
产出: 每几何 coverage + per-segment + watertight(usable 判据); 末尾汇总分布 + 病人级几何数。"""
import sys, os, zipfile, tempfile, shutil, csv, time, json
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import coverage_gt_engine as E

SCRATCH = os.environ.get("HQCOLON_DIR", "/Volumes/WD-6TB/coverage_preflight_scratch/hqcolon")
ZIP = os.path.join(SCRATCH, "gas-filled.zip")
TMPROOT = os.path.join(SCRATCH, "_prod_tmp")
META = os.path.join(SCRATCH, "meta-data.json")
RESULTS = os.path.join(HERE, "..", "results", "coverage_gt_full435.csv")
PROG = os.path.join(HERE, "..", "results", "coverage_gt_full435.progress.txt")
CFG = dict(fov_deg=140.0, pose_step_mm=5.0, near_mm=1.0, far_mm=60.0,
           nbins=280, n_seg=6, directions=["ante", "retro"])
NWORKERS = 6

FIELDS = ["case", "ok", "watertight", "centerline_ok", "lumen_largest_frac", "coverage",
          "n_faces", "n_seen", "total_area_cm2", "centerline_len_cm", "bbox_diag_cm",
          "n_poses", "seg_coverage", "sec", "error"]


def worker(entry):
    os.makedirs(TMPROOT, exist_ok=True)
    tmpd = tempfile.mkdtemp(dir=TMPROOT)
    case = os.path.basename(entry).replace(".mha", "")
    try:
        with zipfile.ZipFile(ZIP) as z:
            data = z.read(entry)
        local = os.path.join(tmpd, os.path.basename(entry))
        with open(local, "wb") as f:
            f.write(data)
        r = E.run(local, tmpd, mesh_downsample=2, cl_downsample=3, cfg=CFG, save_npz=False)
        r["ok"] = True
        r["seg_coverage"] = "|".join(str(x) for x in r["seg_coverage"])
        return r
    except Exception as ex:
        return dict(case=case, ok=False, error=f"{type(ex).__name__}: {ex}")
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)


def main():
    with zipfile.ZipFile(ZIP) as z:
        entries = sorted(n for n in z.namelist() if n.endswith(".mha"))
    t0 = time.time()
    done, rows = 0, []
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        with ProcessPoolExecutor(max_workers=NWORKERS) as ex:
            futs = {ex.submit(worker, e): e for e in entries}
            for fut in as_completed(futs):
                r = fut.result()
                rows.append(r)
                w.writerow(r); f.flush()
                done += 1
                if done % 10 == 0 or done == len(entries):
                    el = time.time() - t0
                    nok = sum(1 for x in rows if x.get("ok"))
                    msg = f"{done}/{len(entries)} done | ok={nok} | {el:.0f}s | eta {el/done*(len(entries)-done):.0f}s"
                    with open(PROG, "w") as pf:
                        pf.write(msg + "\n")
                    print(msg, flush=True)
    summarize(rows)


def summarize(rows):
    import statistics as st
    ok = [r for r in rows if r.get("ok")]
    fail = [r for r in rows if not r.get("ok")]
    wt = [r for r in ok if r.get("watertight")]
    degen = [r for r in ok if not r.get("centerline_ok")]
    usable = [r for r in ok if r.get("watertight") and float(r.get("lumen_largest_frac", 0)) >= 0.90
              and r.get("centerline_ok")]
    covs = sorted(float(r["coverage"]) for r in usable)
    # 病人级几何数 (meta join)
    npat = "n/a"
    try:
        recs = [json.loads(l) for l in open(META) if l.strip()]
        label2sub = {r["nnunet_label_file"].replace(".mha", ""): r["subject_id"] for r in recs}
        pats = {label2sub.get(r["case"]) for r in usable if label2sub.get(r["case"])}
        npat = len([p for p in pats if p])
    except Exception as e:
        npat = f"meta-join-fail: {e}"

    def q(p):
        return covs[int(p * (len(covs) - 1))] if covs else None
    print("\n" + "=" * 70)
    print("全量产覆盖度 GT 汇总")
    print(f"  总 mask: {len(rows)}  | ok: {len(ok)}  | fail(异常): {len(fail)}  | 中心线退化: {len(degen)}")
    print(f"  watertight: {len(wt)}  | usable(watertight & largest_frac>=0.9 & centerline_ok): {len(usable)}")
    print(f"  usable 几何对应独立病人数: {npat}")
    if covs:
        print(f"  coverage 分布 (usable): mean={st.mean(covs):.3f} median={q(0.5):.3f} "
              f"min={covs[0]:.3f} p10={q(0.1):.3f} p90={q(0.9):.3f} max={covs[-1]:.3f}")
    if fail:
        print(f"  失败案例 (前10): {[r['case']+':'+r.get('error','')[:40] for r in fail[:10]]}")
    print(f"  CSV: {RESULTS}")
    print("=" * 70)
    # 汇总写文件
    with open(os.path.join(os.path.dirname(RESULTS), "coverage_gt_full435_summary.json"), "w") as f:
        json.dump(dict(n_total=len(rows), n_ok=len(ok), n_watertight=len(wt),
                       n_centerline_degenerate=len(degen),
                       n_usable=len(usable), n_patients=npat,
                       cov_mean=(st.mean(covs) if covs else None),
                       cov_median=(q(0.5) if covs else None),
                       cov_min=(covs[0] if covs else None), cov_max=(covs[-1] if covs else None),
                       n_fail=len(fail),
                       fails=[{"case": r["case"], "error": r.get("error")} for r in fail]),
                  f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
