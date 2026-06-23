#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""主会话独立 5-seed bit-exact 对拍 (不信服务器 verdict, 不复用可能 stale 的 recount_aurc.py)。
口径 = patient_level_macro (病人内合池 AURC, diff=naive-rel, 跨病人均值 + 配对病人 bootstrap seed=42)。
对每头: (1) per-patient AURC diff bit-exact vs verdict.json; (2) point/CI/seed_overall/n_pos;
(3) effect-gate verdict; (4) tie-robust 微扰交叉验(独立 code path); (5) conformal B-cov 重算。
用法: python recount/verify_e011_5seed.py
"""
import csv, json, os
import numpy as np
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, '..', 'results', 'E011_5seed')
HEADS = ['base', 'mlp', 'seq']
MDE, SEED, NBOOT = 0.01, 42, 2000


def aurc(errors, scores, kind='stable'):
    e = np.asarray(errors, float); s = np.asarray(scores, float); n = len(e)
    idx = np.argsort(-s, kind=kind); se = e[idx]
    risks = np.cumsum(se) / np.arange(1, n + 1); fracs = np.arange(1, n + 1) / n
    return float(np.trapezoid(risks, fracs))


def load(head):
    rows = list(csv.DictReader(open(os.path.join(BASE, head, 'eval_test.csv'))))
    for r in rows:
        for k in ('abs_err', 'reliability_score', 'naive_score'):
            r[k] = float(r[k])
    return rows


def per_patient(fam, kind='stable'):
    byp = defaultdict(list)
    for r in fam:
        byp[r['subject_id']].append(r)
    ppd = {}
    for p, recs in byp.items():
        e = [r['abs_err'] for r in recs]; rs = [r['reliability_score'] for r in recs]; ns = [r['naive_score'] for r in recs]
        ppd[p] = aurc(e, ns, kind) - aurc(e, rs, kind)
    return ppd, byp


def recount(head):
    rows = load(head); fam = [r for r in rows if r['family'] == 'B']
    ppd, byp = per_patient(fam)
    patients = sorted(ppd)
    point = float(np.mean([ppd[p] for p in patients]))
    rng = np.random.default_rng(SEED); n = len(patients); bd = []
    for _ in range(NBOOT):
        smp = rng.choice(patients, size=n, replace=True)
        bd.append(np.mean([ppd[p] for p in smp]))
    bd = np.array(bd); lo, hi = float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5)); hw = (hi - lo) / 2
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
    at_least = abs(point) >= MDE
    if lo > 0 and at_least:
        verdict = 'YES_MEANINGFUL'
    elif lo > 0:
        verdict = 'MARGINAL_SUB_MDE'
    elif hw < MDE and lo <= 0 <= hi:
        verdict = 'NO_POWERED'
    else:
        verdict = 'INCONCLUSIVE_UNDERPOWERED'
    return dict(point=point, lo=lo, hi=hi, hw=hw, effect_pct=round(100 * point / MDE, 2),
                seed_overall={str(k): v for k, v in seed_overall.items()}, n_pos=npos,
                verdict=verdict, ppd=ppd, n_patients=n)


def tie_robust(head, reps=200, eps=1e-6):
    """独立 code path: naive_score 加高斯微扰, 看 lo>0 是否稳健(消除粗量化 tie-break 任意性)。"""
    rows = load(head); fam = [r for r in rows if r['family'] == 'B']
    byp = defaultdict(list)
    for r in fam:
        byp[r['subject_id']].append(r)
    patients = sorted(byp)
    rng = np.random.default_rng(123)
    points, los, npos_lo = [], [], 0
    for _ in range(reps):
        ppd = {}
        for p, recs in byp.items():
            e = np.array([r['abs_err'] for r in recs]); rs = np.array([r['reliability_score'] for r in recs])
            ns = np.array([r['naive_score'] for r in recs]) + rng.normal(0, eps, len(recs))
            ppd[p] = aurc(e, ns) - aurc(e, rs)
        points.append(np.mean([ppd[p] for p in patients]))
        bcr = np.random.default_rng(SEED)
        bd = [np.mean([ppd[p] for p in bcr.choice(patients, len(patients), replace=True)]) for _ in range(500)]
        lo = np.percentile(bd, 2.5); los.append(lo)
        if lo > 0:
            npos_lo += 1
    return dict(point_mean=float(np.mean(points)), lo_mean=float(np.mean(los)), lo_gt0=f'{npos_lo}/{reps}')


def conformal_bcov(head):
    rows = load(head); fam = [r for r in rows if r['family'] == 'B']
    raw = [1.0 if r['abs_err'] <= float(r['conformal_q']) else 0.0 for r in fam if r.get('conformal_q', '') not in ('', 'None')]
    col = [(1.0 if r['abs_err'] <= float(r['conformal_q']) else 0.0, float(r['interval_coverage']))
           for r in fam if r.get('conformal_q', '') not in ('', 'None') and r.get('interval_coverage', '') not in ('', 'None')]
    mismatch = sum(1 for a, b in col if int(round(a)) != int(round(b)))
    return dict(b_cov_raw=round(float(np.mean(raw)), 4) if raw else None, col_mismatch=mismatch)


def degeneracy(head):
    """reliability_score / naive_score 跨 (patient,seed) 组的唯一值分布 → 验证 seq 是否退化同 naive。"""
    rows = load(head); fam = [r for r in rows if r['family'] == 'B']
    g = defaultdict(lambda: {'rel': set(), 'nv': set()})
    for r in fam:
        key = (r['subject_id'], r['seed'])
        g[key]['rel'].add(round(r['reliability_score'], 9)); g[key]['nv'].add(round(r['naive_score'], 9))
    rel_u = sorted(len(v['rel']) for v in g.values()); nv_u = sorted(len(v['nv']) for v in g.values())
    from collections import Counter
    return dict(rel_uniq_hist=dict(Counter(rel_u)), rel_median=int(np.median(rel_u)),
                naive_uniq_hist=dict(Counter(nv_u)), naive_median=int(np.median(nv_u)))


def main():
    server = {h: json.load(open(os.path.join(BASE, h, 'verdict.json'))) for h in HEADS}
    report = {}
    print('=' * 70)
    for h in HEADS:
        rc = recount(h); sv = server[h]
        # bit-exact per-patient
        sppd = sv['per_patient_aurc_diff']
        deltas = {p: abs(rc['ppd'][p] - sppd[p]) for p in sppd}
        maxd = max(deltas.values()); argmax = max(deltas, key=deltas.get)
        flips = sum(1 for p in sppd if (rc['ppd'][p] > 0) != (sppd[p] > 0) and abs(rc['ppd'][p]) > 1e-9 and abs(sppd[p]) > 1e-9)
        tr = tie_robust(h); cf = conformal_bcov(h); dg = degeneracy(h)
        report[h] = dict(recount=dict(point=rc['point'], ci=[rc['lo'], rc['hi']], hw=rc['hw'],
                                      effect_pct=rc['effect_pct'], n_pos=f"{rc['n_pos']}/5",
                                      verdict=rc['verdict'], seed_overall=rc['seed_overall']),
                         vs_server=dict(point_d=abs(rc['point'] - sv['point']), lo_d=abs(rc['lo'] - sv['ci'][0]),
                                        hi_d=abs(rc['hi'] - sv['ci'][1]), verdict_match=rc['verdict'] == sv['verdict'],
                                        per_patient_maxd=maxd, per_patient_maxd_at=argmax, sign_flips=flips),
                         tie_robust=tr, conformal=cf, degeneracy=dg)
        print(f"[{h}] verdict={rc['verdict']}  point={rc['point']:+.6f}  CI=[{rc['lo']:+.6f},{rc['hi']:+.6f}]  "
              f"eff={rc['effect_pct']}%  n_pos={rc['n_pos']}/5")
        print(f"     vs-server: point_d={abs(rc['point']-sv['point']):.2e}  lo_d={abs(rc['lo']-sv['ci'][0]):.2e}  "
              f"verdict_match={rc['verdict']==sv['verdict']}  per_patient_maxd={maxd:.2e}@{argmax}  flips={flips}")
        print(f"     tie-robust: point={tr['point_mean']:+.6f}  lo_mean={tr['lo_mean']:+.6f}  lo>0={tr['lo_gt0']}")
        print(f"     conformal B-cov(raw)={cf['b_cov_raw']}  col_mismatch={cf['col_mismatch']}")
        print(f"     degeneracy rel_median={dg['rel_median']} {dg['rel_uniq_hist']} | naive_median={dg['naive_median']} {dg['naive_uniq_hist']}")
        print('-' * 70)
    json.dump(report, open(os.path.join(BASE, 'MAIN_SESSION_RECOUNT_5seed.json'), 'w'), indent=2, ensure_ascii=False)
    print('written:', os.path.join(BASE, 'MAIN_SESSION_RECOUNT_5seed.json'))


if __name__ == '__main__':
    main()
