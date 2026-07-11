#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P11 独立 recount: per-method coverage MAE, patient-level macro + 配对病人 bootstrap CI。
口径对齐 metric_definition.md (patient-level macro 为 primary)。

用法:
  # 排行榜模式: per-method coverage MAE + 病人 bootstrap CI:
  python recount/recount_mae.py <eval.csv> --group-col method
对 the full returned eval set 的 eval_*.csv 复用。
"""
import csv, json, argparse
import numpy as np
from collections import defaultdict


def mae_table(rows, group_col, nboot, seed):
    """排行榜模式: per-group(method) coverage MAE, patient-level macro + 配对病人 bootstrap CI。"""
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
    ap.add_argument('csv')
    ap.add_argument('--group-col', default='method', help='按此列分组分别 recount (如 method)')
    ap.add_argument('--mae', action='store_true', help='排行榜模式: per-group coverage MAE (默认即此模式)')
    ap.add_argument('--nboot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv)))
    for r in rows:
        if 'abs_err' in r and r['abs_err'] != '':
            r['abs_err'] = float(r['abs_err'])
    gc = a.group_col or 'method'
    out = {'csv': a.csv, 'mode': 'coverage_MAE_leaderboard', 'group_col': gc, 'caliber': 'patient_level_macro',
           'table': mae_table(rows, gc, a.nboot, a.seed)}
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
