#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate4 (本地 CPU): 决胜 bootstrap harness 干跑。
合成已知答案的 eval_test.csv → 验: (1) 按病人(非几何)配对 cluster bootstrap;
(2) verdict 逻辑 YES/PARTIAL/NO/INCONCLUSIVE; (3) 演示按几何聚类的伪复制(CI 偏窄→假 YES)。
不依赖真实训练; 只验统计管线正确。"""
import numpy as np

RNG = np.random.default_rng(20260612)
MDE = 0.01  # 预注册最小可检 AURC 差


def aurc(errs, scores):
    """risk-coverage: 按 score 降序(高置信先保留), risk(c)=前 c 比例的累积平均误差; AURC=∫risk dc。低=好。"""
    scores = np.asarray(scores)
    order = np.argsort(-scores)
    e = np.asarray(errs)[order]
    cum = np.cumsum(e) / np.arange(1, len(e) + 1)
    return float(cum.mean())


def paired_cluster_bootstrap(df, cluster_key, B=2000, rng=RNG):
    """配对 cluster bootstrap: 每次重采样 cluster, 同一组上算两臂 AURC 取差。返回 (点估计, lo, hi)。"""
    clusters = {}
    for r in df:
        clusters.setdefault(r[cluster_key], []).append(r)
    keys = list(clusters)
    # 点估计 (全样本)
    def aurc_diff(rows):
        rel = aurc([x["abs_err"] for x in rows], [x["rel_score"] for x in rows])
        nai = aurc([x["abs_err"] for x in rows], [x["naive_score"] for x in rows])
        return nai - rel  # >0 = reliability 更优 (AURC 更低)
    allrows = [x for c in keys for x in clusters[c]]
    point = aurc_diff(allrows)
    ds = np.empty(B)
    for b in range(B):
        samp = rng.choice(keys, len(keys), replace=True)
        rows = [x for c in samp for x in clusters[c]]
        ds[b] = aurc_diff(rows)
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return point, float(lo), float(hi)


def verdict(point, lo, hi, cov_gap, mde=MDE):
    """复合 verdict + 功率护栏。cov_gap = |empirical interval coverage - 0.90|。"""
    aurc_sig = lo > 0
    cal_ok = cov_gap <= 0.05
    if aurc_sig and cal_ok:
        return "YES"
    if aurc_sig and not cal_ok:
        return "PARTIAL (ranking-only)"
    # 不显著: 功率护栏
    halfwidth = (hi - lo) / 2
    if point > 0 and halfwidth > mde:
        return "INCONCLUSIVE-UNDERPOWERED (扩全test)"
    return "NO"


def make_data(n_patient, traj_per_geom, rel_adv, two_geom_frac=0.3, seed=0):
    """合成: 部分病人有 2 几何; reliability score 比 naive 更贴 -abs_err (rel_adv 控优势)。"""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_patient):
        ngeom = 2 if rng.random() < two_geom_frac else 1
        for g in range(ngeom):
            for t in range(traj_per_geom):
                err = abs(rng.normal(0.08, 0.05))
                rel = -err + rng.normal(0, 0.02)                 # 强信号
                naive = -err + rng.normal(0, 0.02 + rel_adv)     # rel_adv 越大 naive 越差
                rows.append(dict(subject_id=f"p{pid}", geometry_id=f"p{pid}g{g}",
                                 abs_err=err, rel_score=rel, naive_score=naive))
    return rows


print("Gate4 决胜 bootstrap harness 干跑")
print("=" * 78)
allpass = True

# 场景1: reliability 有真优势 → 期望 YES (CI 排 0)
d1 = make_data(14, 8, rel_adv=0.06, seed=1)
p, lo, hi = paired_cluster_bootstrap(d1, "subject_id")
v1 = verdict(p, lo, hi, cov_gap=0.03)
ok1 = (lo > 0) and v1.startswith("YES")
print(f"[场景1 真优势] AURC差点估={p:.4f} 病人CI=[{lo:.4f},{hi:.4f}] verdict={v1} {'✓' if ok1 else '✗ 期望YES'}")

# 场景2: 无优势 (rel_adv=0) → 期望 NO 或 INCONCLUSIVE (CI 含 0)
d2 = make_data(14, 8, rel_adv=0.0, seed=2)
p, lo, hi = paired_cluster_bootstrap(d2, "subject_id")
v2 = verdict(p, lo, hi, cov_gap=0.03)
ok2 = (lo <= 0) and (not v2.startswith("YES"))
print(f"[场景2 无优势] AURC差点估={p:.4f} 病人CI=[{lo:.4f},{hi:.4f}] verdict={v2} {'✓' if ok2 else '✗ 期望非YES'}")

# 场景3: 真优势 + 校准差 (cov_gap 大) → 期望 PARTIAL
p, lo, hi = paired_cluster_bootstrap(d1, "subject_id")
v3 = verdict(p, lo, hi, cov_gap=0.12)
ok3 = v3.startswith("PARTIAL")
print(f"[场景3 优势但校准差] verdict={v3} {'✓' if ok3 else '✗ 期望PARTIAL'}")

# 场景4: 伪复制演示 — 同病人两几何强相关(supine/prone 同解剖), 按几何聚类低估方差
def make_correlated(n_patient, traj, rel_adv, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_patient):
        g0 = []
        for t in range(traj):
            err = abs(rng.normal(0.08, 0.05))
            rel = -err + rng.normal(0, 0.02)
            naive = -err + rng.normal(0, 0.02 + rel_adv)
            g0.append(dict(subject_id=f"p{pid}", geometry_id=f"p{pid}g0", abs_err=err, rel_score=rel, naive_score=naive))
        rows += g0
        # 第二几何 = 第一个的近复制 (同解剖 → 强相关), 占 70% 病人
        if rng.random() < 0.7:
            for x in g0:
                j = rng.normal(0, 0.003)  # 极小抖动
                rows.append(dict(subject_id=f"p{pid}", geometry_id=f"p{pid}g1",
                                 abs_err=max(0, x["abs_err"] + j), rel_score=x["rel_score"] + j,
                                 naive_score=x["naive_score"] + j))
    return rows
d4 = make_correlated(14, 8, rel_adv=0.06, seed=7)
pp, plo, phi = paired_cluster_bootstrap(d4, "subject_id")
gp, glo, ghi = paired_cluster_bootstrap(d4, "geometry_id")
patient_hw = (phi - plo) / 2; geom_hw = (ghi - glo) / 2
ok4 = geom_hw < patient_hw  # 按几何 CI 更窄 = 伪复制 (红队警告: 假 YES 风险)
print(f"[场景4 伪复制] 强相关双几何: 病人CI半宽={patient_hw:.4f} vs 几何CI半宽={geom_hw:.4f} "
      f"→ 按几何{'更窄(伪复制→假信心, 故必须按病人)✓' if ok4 else '未更窄✗'}")

allpass = ok1 and ok2 and ok3 and ok4
print("=" * 78)
print("Gate4:", "ALL PASS ✓" if allpass else "FAIL ✗")
import sys; sys.exit(0 if allpass else 1)
