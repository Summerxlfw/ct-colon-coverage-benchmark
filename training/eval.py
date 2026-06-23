#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P11 E001 评估脚本: 配对病人-cluster bootstrap + verdict

用法:
  python training/eval.py --config configs/E001_decisive_pilot.yaml \
      --seeds 0 1 2 --hqcolon-dir ~/data/hqcolon \
      --out-dir ~/checkpoints/P11_coverage_reliability/E001_decisive_pilot
"""
import os, sys, json, argparse, csv, time
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from model import P11Model
from dataset import HQColonDataset
from train import collate_fn


def _risk_at_coverage(errors, scores, retain_frac):
    """给定保留比例, 返回 retained 子集的风险 (MAE)"""
    n = len(errors)
    n_keep = max(1, int(n * retain_frac))
    # 按 score 降序排 (高可靠度优先保留)
    idx = np.argsort(-np.array(scores))
    kept = idx[:n_keep]
    return float(np.mean(np.array(errors)[kept]))


def compute_aurc(errors, scores):
    """计算 AURC: risk-coverage 曲线下面积。
    errors: 绝对误差数组, scores: 弃权排序信号 (高=保留优先)
    AURC = 对 retain_frac 从 1.0→最小 的 risk 积分"""
    n = len(errors)
    idx = np.argsort(-np.array(scores))  # 高分优先
    sorted_err = np.array(errors)[idx]

    # 累积 risk
    cum_err = np.cumsum(sorted_err)
    fracs = np.arange(1, n + 1) / n
    risks = cum_err / np.arange(1, n + 1)

    # 梯形积分
    aurc = float(np.trapezoid(risks, fracs))
    return aurc


def paired_bootstrap_verdict(records_by_patient, n_bootstrap=1000, MDE=0.01,
                              cal_tolerance=0.05, nominal_coverage=0.90,
                              seed=42):
    """配对病人-cluster bootstrap + 预注册 verdict

    records_by_patient: {subject_id: [{'aurc_rel', 'aurc_naive', 'interval_coverage', ...}, ...]}
    返回: verdict dict
    """
    rng = np.random.default_rng(seed)
    patients = sorted(records_by_patient.keys())
    n_patients = len(patients)

    # 每病人跨 seed/trajectory 平均
    diffs_per_patient = {}
    cov_per_patient = {}
    for p in patients:
        recs = records_by_patient[p]
        d = np.mean([r['aurc_naive'] - r['aurc_rel'] for r in recs])  # 正=reliability 更优
        diffs_per_patient[p] = d
        cov_per_patient[p] = np.mean([r.get('interval_coverage', nominal_coverage) for r in recs])

    # Bootstrap
    boot_diffs = []
    for _ in range(n_bootstrap):
        # 重采样病人 (有放回)
        sample = rng.choice(patients, size=n_patients, replace=True)
        d = np.mean([diffs_per_patient[p] for p in sample])
        boot_diffs.append(d)
    boot_diffs = np.array(boot_diffs)

    # 95% CI
    lo = float(np.percentile(boot_diffs, 2.5))
    hi = float(np.percentile(boot_diffs, 97.5))
    half_width = (hi - lo) / 2
    point_est = float(np.mean(list(diffs_per_patient.values())))

    # 区间覆盖率
    mean_coverage = float(np.mean(list(cov_per_patient.values())))
    cov_deviation = abs(mean_coverage - nominal_coverage)

    # Seed 符号一致性 (robust gate)
    seed_signs = {}
    for p in patients:
        recs = records_by_patient[p]
        # 按 seed 分组
        by_seed = defaultdict(list)
        for r in recs:
            by_seed[r.get('seed', 0)].append(r['aurc_naive'] - r['aurc_rel'])
        signs = []
        for s, vals in by_seed.items():
            mean_d = np.mean(vals)
            signs.append(1 if mean_d > 0 else (-1 if mean_d < 0 else 0))
        seed_signs[p] = signs

    # 总体 seed 符号
    all_seed_means = defaultdict(list)
    for p, recs in records_by_patient.items():
        by_seed = defaultdict(list)
        for r in recs:
            by_seed[r.get('seed', 0)].append(r['aurc_naive'] - r['aurc_rel'])
        for s, vals in by_seed.items():
            all_seed_means[s].extend(vals)
    seed_overall = {s: np.mean(v) for s, v in all_seed_means.items()}
    n_positive = sum(1 for v in seed_overall.values() if v > 0)

    # Verdict 判定 (预注册)
    if lo > 0:
        # AURC CI 排 0, reliability 更优
        if cov_deviation <= cal_tolerance:
            verdict = 'YES'
        else:
            verdict = 'PARTIAL'
    else:
        # CI 含 0
        if half_width < MDE:
            verdict = 'NO'
        else:
            verdict = 'INCONCLUSIVE-UNDERPOWERED'

    return {
        'verdict': verdict,
        'point_est': point_est,
        'ci_lo': lo, 'ci_hi': hi, 'ci_half_width': half_width,
        'mean_interval_coverage': mean_coverage,
        'coverage_deviation': cov_deviation,
        'seed_overall': {str(k): round(v, 6) for k, v in seed_overall.items()},
        'n_seeds_positive': n_positive,
        'n_seeds_total': len(seed_overall),
        'n_patients': n_patients,
    }


def evaluate(args):
    """主评估函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    cfg_engine = {
        'fov_deg': config.get('input_contract', {}).get('fov_deg', 140.0),
        'near_mm': config.get('input_contract', {}).get('depth_clip_mm', [1, 60])[0],
        'far_mm': config.get('input_contract', {}).get('depth_clip_mm', [1, 60])[1],
        'nbins': 280, 'H': 256, 'W': 256, 'K_poses': 128,
        'directions': config.get('input_contract', {}).get('directions', ['ante', 'retro']),
        'pose_step_mm': 5.0, 'sigma_max': 2.0, 'scale_max': 0.05,
    }

    config_dir = os.path.dirname(os.path.abspath(args.config))
    manifest = os.path.join(config_dir, 'E001_pilot_manifest.json')

    # Test 数据 (含退化)
    test_ds = HQColonDataset(args.hqcolon_dir, manifest,
                              os.path.join(config_dir, 'split_patients.json'),
                              split='test', cfg=cfg_engine, seed=999)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    # 按 seed 评估
    all_records = []  # 每行 = 一条 trajectory 的评估结果
    records_by_patient = defaultdict(list)

    for seed in args.seeds:
        ckpt_path = os.path.join(args.out_dir, f"seed{seed}", 'checkpoint_best.pth')
        if not os.path.exists(ckpt_path):
            print(f"警告: seed{seed} checkpoint 不存在, 跳过")
            continue

        model = P11Model(embed_dim=128).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        model.eval()

        with torch.no_grad():
            for batch in test_loader:
                depth = batch['depth'].to(device)
                vmask = batch['validity_mask'].to(device)
                cov_clean = batch['coverage_clean'].to(device)
                rel_feat = batch['rel_features'].to(device)
                naive_feat = batch['naive_features'].to(device)

                out = model(depth, vmask, rel_feat, naive_feat)
                cov_pred = out['coverage_pred'].cpu().numpy()
                rel_score = out['reliability_score'].cpu().numpy()
                naive_score = out['naive_score'].cpu().numpy()
                cov_clean_np = cov_clean.cpu().numpy()

                B = len(batch['subject_id'])
                for i in range(B):
                    abs_err = abs(float(cov_pred[i]) - float(batch['coverage_gt'][i]))
                    record = {
                        'subject_id': batch['subject_id'][i],
                        'geometry_id': batch['geometry_id'][i],
                        'family': batch['family'][i],
                        'strength': batch['strength'][i],
                        'method': 'both',  # 同一模型, 两头同时输出
                        'coverage_pred': round(float(cov_pred[i]), 6),
                        'coverage_gt': round(float(batch['coverage_gt'][i]), 6),
                        'abs_err': round(abs_err, 6),
                        'x_naive_pred_var': round(float(batch['naive_features'][i][0]), 6),
                        'x_naive_frame_count': round(float(batch['naive_features'][i][1]), 6),
                        'x_naive_obs_proxy': round(float(batch['naive_features'][i][2]), 6),
                        'x_naive_flow_mag': round(float(batch['naive_features'][i][3]), 6),
                        'x_rel_depth_cons': round(float(batch['rel_features'][i][0]), 6),
                        'x_rel_pose_drift': round(float(batch['rel_features'][i][1]), 6),
                        'x_rel_obs_density': round(float(batch['rel_features'][i][2]), 6),
                        'reliability_score': round(float(rel_score[i]), 6),
                        'naive_score': round(float(naive_score[i]), 6),
                        'seed': seed,
                    }
                    all_records.append(record)

    if not all_records:
        print("错误: 无评估记录")
        return

    # 写 eval_test.csv
    csv_path = os.path.join(args.out_dir, 'eval_test.csv')
    os.makedirs(args.out_dir, exist_ok=True)
    fields = list(all_records[0].keys())
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_records)
    print(f"eval_test.csv 写入 {len(all_records)} 行 → {csv_path}")

    # 按 trajectory 计算 AURC
    # 每个 (subject, geometry, family, strength, seed) 是一条 trajectory
    from itertools import groupby
    key_fn = lambda r: (r['subject_id'], r['geometry_id'],
                         r['family'], r['strength'], r['seed'])

    # 按 trajectory 聚合 (当前每 trajectory 只有 1 行, AURC 需要多条;
    # 在单 trajectory 上 AURC = MAE, 多 trajectory 聚合后才有排序意义)
    # 实际 AURC 在 "所有 B 族 trajectory" 上做: 每条给 (error, reliability_score, naive_score)
    b_records = [r for r in all_records if r['family'] == 'B']

    if b_records:
        errors = np.array([r['abs_err'] for r in b_records])
        rel_scores = np.array([r['reliability_score'] for r in b_records])
        naive_scores = np.array([r['naive_score'] for r in b_records])

        aurc_rel = compute_aurc(errors, rel_scores)
        aurc_naive = compute_aurc(errors, naive_scores)
        print(f"B 族 AURC: reliability={aurc_rel:.6f}, naive={aurc_naive:.6f}, diff={aurc_naive-aurc_rel:.6f}")

    # 按病人聚合 bootstrap
    # AURC 差: 每次重采样病人, 在重采样子集上分别算 AURC(reliability) vs AURC(naive)
    patient_records = defaultdict(list)
    for r in all_records:
        if r['family'] == 'B':
            patient_records[r['subject_id']].append(r)

    if len(patient_records) >= 2:
        decision = config.get('decision', {})
        n_boot = decision.get('n_bootstrap', 1000)
        MDE = decision.get('MDE_aurc_diff', 0.01)
        cal_tol = decision.get('cal_tolerance', 0.05)

        patients = sorted(patient_records.keys())
        n_patients = len(patients)
        rng = np.random.default_rng(42)

        boot_diffs = []
        for _ in range(n_boot):
            sample_patients = rng.choice(patients, size=n_patients, replace=True)
            # 收集重采样子集的所有记录
            recs = []
            for p in sample_patients:
                recs.extend(patient_records[p])
            errors = np.array([r['abs_err'] for r in recs])
            rel_scores = np.array([r['reliability_score'] for r in recs])
            naive_scores = np.array([r['naive_score'] for r in recs])
            aurc_r = compute_aurc(errors, rel_scores)
            aurc_n = compute_aurc(errors, naive_scores)
            boot_diffs.append(aurc_n - aurc_r)  # 正=reliability 更优
        boot_diffs = np.array(boot_diffs)

        lo = float(np.percentile(boot_diffs, 2.5))
        hi = float(np.percentile(boot_diffs, 97.5))
        half_width = (hi - lo) / 2
        point_est = float(np.mean(boot_diffs))

        # Seed 符号
        seed_means = defaultdict(list)
        for p, recs in patient_records.items():
            by_seed = defaultdict(list)
            for r in recs:
                by_seed[r['seed']].append(r)
            for s, sr in by_seed.items():
                errs = np.array([r['abs_err'] for r in sr])
                rel_s = np.array([r['reliability_score'] for r in sr])
                naiv_s = np.array([r['naive_score'] for r in sr])
                seed_means[s].append(compute_aurc(errs, naiv_s) - compute_aurc(errs, rel_s))
        seed_overall = {s: float(np.mean(v)) for s, v in seed_means.items()}
        n_positive = sum(1 for v in seed_overall.values() if v > 0)

        # Verdict
        if lo > 0:
            verdict = 'YES'
        else:
            if half_width < MDE:
                verdict = 'NO'
            else:
                verdict = 'INCONCLUSIVE-UNDERPOWERED'

        print(f"\n{'='*60}")
        print(f"VERDICT: {verdict}")
        print(f"AURC 差 (naive-rel): {point_est:.6f}")
        print(f"95% CI: [{lo:.6f}, {hi:.6f}]  half_width={half_width:.6f}")
        print(f"全局 AURC: rel={aurc_rel:.6f} naive={aurc_naive:.6f}")
        print(f"Seed 符号: {seed_overall}")
        print(f"n_positive={n_positive}/{len(seed_overall)}")
        print(f"Test 病人数: {n_patients}")
        print(f"{'='*60}")

        # 写 readout
        readout_path = os.path.join(args.out_dir, 'readout.md')
        with open(readout_path, 'w') as f:
            f.write(f"# E001 Decisive Pilot Readout\n\n")
            f.write(f"## Verdict: **{verdict}**\n\n")
            f.write(f"| 指标 | 值 |\n|---|---|\n")
            f.write(f"| AURC diff (naive-rel) | {point_est:.6f} |\n")
            f.write(f"| 95% CI | [{lo:.6f}, {hi:.6f}] |\n")
            f.write(f"| CI half-width | {half_width:.6f} |\n")
            f.write(f"| MDE | {MDE} |\n")
            f.write(f"| 全局 AURC reliability | {aurc_rel:.6f} |\n")
            f.write(f"| 全局 AURC naive | {aurc_naive:.6f} |\n")
            f.write(f"| Seeds positive | {n_positive}/{len(seed_overall)} |\n")
            f.write(f"| Test patients | {n_patients} |\n")
            f.write(f"\n### Seed 详情\n```json\n{json.dumps({str(k):round(v,6) for k,v in seed_overall.items()}, indent=2)}\n```\n")

        # 写 verdict JSON
        verdict_path = os.path.join(args.out_dir, 'verdict.json')
        verdict_data = {
            'verdict': verdict,
            'point_est': point_est,
            'ci_lo': lo, 'ci_hi': hi, 'ci_half_width': half_width,
            'aurc_reliability': aurc_rel, 'aurc_naive': aurc_naive,
            'seed_overall': {str(k): round(v, 6) for k, v in seed_overall.items()},
            'n_seeds_positive': n_positive,
            'n_patients': n_patients,
        }
        with open(verdict_path, 'w') as f:
            json.dump(verdict_data, f, indent=2, ensure_ascii=False)

        return verdict_data
    else:
        print(f"警告: B 族 test 病人数={len(patient_records)}, 不够 bootstrap")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--hqcolon-dir', default='~/data/hqcolon')
    ap.add_argument('--out-dir', default='~/checkpoints/P11_coverage_reliability/E001_decisive_pilot')
    args = ap.parse_args()
    args.hqcolon_dir = os.path.expanduser(args.hqcolon_dir)
    args.out_dir = os.path.expanduser(args.out_dir)
    evaluate(args)


if __name__ == '__main__':
    main()
