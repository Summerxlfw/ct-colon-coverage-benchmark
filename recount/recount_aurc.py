#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P11 独立 recount: patient-level macro AURC + 配对病人 bootstrap + conformal + MAE。
口径对齐 metric_definition.md (patient-level macro 为 primary) 与 E001_rerun verdict.json。

用法:
  # 负结果 verdict (AURC, family B): E011 单 csv
  python recount/recount_aurc.py <eval.csv> --family B
  # E011 若把头变体放进一列 (如 rel_head=base/mlp/seq), 分别判 verdict (看强化头是否翻盘 YES):
  python recount/recount_aurc.py <eval.csv> --family B --group-col rel_head
  # E010 排行榜: per-method coverage MAE + 病人 bootstrap CI:
  python recount/recount_aurc.py <eval.csv> --mae --group-col method
对全量 wave-1 回传的 eval_*.csv 复用; conformal 覆盖率从原始残差重算, 不信列。
"""
import csv, json, argparse
import numpy as np
from collections import defaultdict


def aurc(errors, scores):
    """AURC = risk-coverage 曲线下面积; 按 score 降序保留, risk=累积MAE, 梯形积分。lower better。"""
    e = np.asarray(errors, float); s = np.asarray(scores, float); n = len(e)
    idx = np.argsort(-s, kind='stable'); se = e[idx]   # 决策A: stable tie-break, 剩余 tie 按行序(=strength序)确定化, per-patient bit-exact
    risks = np.cumsum(se) / np.arange(1, n + 1); fracs = np.arange(1, n + 1) / n
    return float(np.trapezoid(risks, fracs))


def aurc_verdict(fam_rows, nboot, seed, mde):
    """patient-level macro: 病人内合池算 AURC(rel)/AURC(naive), diff=naive-rel, 跨病人均值 + 配对病人 bootstrap。"""
    byp = defaultdict(list)
    for r in fam_rows:
        byp[r['subject_id']].append(r)
    patients = sorted(byp)
    ppd = {}
    for p, recs in byp.items():
        e = [r['abs_err'] for r in recs]; rs = [r['reliability_score'] for r in recs]; ns = [r['naive_score'] for r in recs]
        ppd[p] = aurc(e, ns) - aurc(e, rs)            # 正=reliability 更优
    point = float(np.mean([ppd[p] for p in patients]))
    rng = np.random.default_rng(seed); n = len(patients); bd = []
    for _ in range(nboot):
        smp = rng.choice(patients, size=n, replace=True)
        bd.append(np.mean([ppd[p] for p in smp]))
    bd = np.array(bd)
    lo, hi = float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5)); hw = (hi - lo) / 2
    # pooled (全局) 对照
    e = [r['abs_err'] for r in fam_rows]; rs = [r['reliability_score'] for r in fam_rows]; ns = [r['naive_score'] for r in fam_rows]
    pooled = aurc(e, ns) - aurc(e, rs)
    # seed 符号
    seed_means = defaultdict(list)
    for p, recs in byp.items():
        bys = defaultdict(list)
        for r in recs:
            bys[r['seed']].append(r)
        for s, sr in bys.items():
            ee = [r['abs_err'] for r in sr]; rr = [r['reliability_score'] for r in sr]; nn = [r['naive_score'] for r in sr]
            seed_means[s].append(aurc(ee, nn) - aurc(ee, rr))
    seed_overall = {s: float(np.mean(v)) for s, v in seed_means.items()}
    npos = sum(1 for v in seed_overall.values() if v > 0)
    # §3b effect-gate 双闸(镜像 eval.py): lo>0 必须 AND |effect|>=MDE 才算有意义 YES; 否则 sub-MDE 不升级。
    # 旧版 `if lo>0: YES` 无 effect 闸, 会把 sub-MDE 小正 edge 误报为 reliability 获胜(overclaim)。
    at_least_mde = abs(point) >= mde
    if lo > 0 and at_least_mde:
        verdict = 'YES_MEANINGFUL'
    elif lo > 0:
        verdict = 'MARGINAL_SUB_MDE'
    elif hw < mde and lo <= 0 <= hi:
        verdict = 'NO_POWERED'
    else:
        verdict = 'INCONCLUSIVE_UNDERPOWERED'
    return {'caliber': 'patient_level_macro', 'n_patients': n, 'point_est_macro': round(point, 6),
            'ci': [round(lo, 6), round(hi, 6)], 'ci_half_width': round(hw, 6), 'mde': mde, 'verdict': verdict,
            'pooled_diff_naive_minus_rel': round(pooled, 6),
            'seed_overall': {str(k): round(v, 6) for k, v in seed_overall.items()}, 'n_seeds_positive': npos,
            'per_patient_aurc_diff': {p: round(ppd[p], 6) for p in patients}}


def conformal_recount(fam_rows, all_rows):
    """从原始残差重算覆盖 = (abs_err <= conformal_q); 与列对拍; per-strength。不信 interval_coverage 列。"""
    if not any(r.get('conformal_q', '') != '' for r in all_rows):
        return {}
    def cov_raw(recs):
        v = [1.0 if r['abs_err'] <= float(r['conformal_q']) else 0.0 for r in recs if r.get('conformal_q', '') != '']
        return float(np.mean(v)) if v else None
    covB = cov_raw(fam_rows); covAll = cov_raw(all_rows)
    out = {'interval_coverage_B_raw': round(covB, 4) if covB is not None else None,
           'interval_coverage_overall_raw': round(covAll, 4) if covAll is not None else None,
           'nominal': 0.90, 'meets_nominal_B': (covB is not None and covB >= 0.88)}
    col = [(1.0 if r['abs_err'] <= float(r['conformal_q']) else 0.0, float(r['interval_coverage']))
           for r in all_rows if r.get('conformal_q', '') != '' and r.get('interval_coverage', '') not in ('', 'None')]
    if col:
        out['column_vs_raw_mismatch_rows'] = sum(1 for a, b in col if int(round(a)) != int(round(b)))
    out['per_strength_B'] = {s: round(cov_raw([r for r in fam_rows if r['strength'] == s]), 4)
                             for s in sorted(set(r['strength'] for r in fam_rows))
                             if cov_raw([r for r in fam_rows if r['strength'] == s]) is not None}
    return out


def mae_table(rows, group_col, nboot, seed):
    """E010 排行榜: per-group(method) coverage MAE, patient-level macro + 配对病人 bootstrap CI。"""
    groups = sorted(set(r.get(group_col, 'all') for r in rows))
    rng = np.random.default_rng(seed)
    res = {}
    for g in groups:
        gr = [r for r in rows if r.get(group_col, 'all') == g]
        byp = defaultdict(list)
        for r in gr:
            byp[r['subject_id']].append(r)
        patients = sorted(byp)
        per_patient = {p: float(np.mean([r['abs_err'] for r in recs])) for p, recs in byp.items()}
        point = float(np.mean([per_patient[p] for p in patients]))
        n = len(patients); bd = []
        for _ in range(nboot):
            smp = rng.choice(patients, size=n, replace=True)
            bd.append(np.mean([per_patient[p] for p in smp]))
        lo, hi = float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))
        res[g] = {'mae_macro': round(point, 6), 'ci': [round(lo, 6), round(hi, 6)], 'n_patients': n, 'n_rows': len(gr)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv'); ap.add_argument('--family', default='B')
    ap.add_argument('--group-col', default=None, help='按此列分组分别 recount (如 rel_head / method)')
    ap.add_argument('--mae', action='store_true', help='E010 排行榜模式: per-group coverage MAE')
    ap.add_argument('--nboot', type=int, default=2000); ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--mde', type=float, default=0.01)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv)))
    for r in rows:
        for k in ('abs_err', 'reliability_score', 'naive_score'):
            if k in r and r[k] != '':
                r[k] = float(r[k])

    if a.mae:
        gc = a.group_col or 'method'
        out = {'csv': a.csv, 'mode': 'coverage_MAE_leaderboard', 'group_col': gc, 'caliber': 'patient_level_macro',
               'table': mae_table(rows, gc, a.nboot, a.seed)}
        print(json.dumps(out, indent=2, ensure_ascii=False)); return

    def one(subrows, label):
        fam = [r for r in subrows if r['family'] == a.family]
        d = {'label': label, 'family': a.family}
        d.update(aurc_verdict(fam, a.nboot, a.seed, a.mde))
        d['conformal'] = conformal_recount(fam, subrows)
        return d

    if a.group_col:
        groups = sorted(set(r.get(a.group_col, '?') for r in rows))
        out = {'csv': a.csv, 'group_col': a.group_col,
               'by_group': [one([r for r in rows if r.get(a.group_col) == g], g) for g in groups]}
        # 强化头翻盘警报
        flips = [d['label'] for d in out['by_group'] if d['verdict'] == 'YES_MEANINGFUL']
        out['ALERT_heads_that_flip_to_YES'] = flips  # 非空=有头达 >=MDE 有意义改进=负结论被推翻, 停下报 user
    else:
        out = one(rows, 'all')
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
