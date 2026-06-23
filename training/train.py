#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P11 E001 训练脚本: 3-seed coverage 头 + 可靠性头 + 朴素臂

用法:
  python training/train.py --config configs/E001_decisive_pilot.yaml \
      --seed 0 --hqcolon-dir ~/data/hqcolon --out-dir ~/checkpoints/...
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from model import P11Model
from dataset import HQColonDataset


def collate_fn(batch):
    """自定义 collate: depth 帧数 K 可能不同, 需 pad 到一致"""
    max_k = max(b['depth'].shape[0] for b in batch)
    B = len(batch)
    H, W = batch[0]['depth'].shape[2], batch[0]['depth'].shape[3]

    depth = torch.zeros(B, max_k, 1, H, W)
    vmask = torch.zeros(B, max_k, dtype=torch.bool)
    for i, b in enumerate(batch):
        k = b['depth'].shape[0]
        depth[i, :k] = b['depth']
        vmask[i, :k] = b['validity_mask']

    return {
        'depth': depth,
        'validity_mask': vmask,
        'coverage_gt': torch.stack([b['coverage_gt'] for b in batch]),
        'coverage_clean': torch.stack([b['coverage_clean'] for b in batch]),
        'rel_features': torch.stack([b['rel_features'] for b in batch]),
        'naive_features': torch.stack([b['naive_features'] for b in batch]),
        'family': [b['family'] for b in batch],
        'strength': [b['strength'] for b in batch],
        'subject_id': [b['subject_id'] for b in batch],
        'geometry_id': [b['geometry_id'] for b in batch],
    }


def train_one_seed(args, seed):
    """单 seed 训练"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[seed={seed}] 设备: {device}")

    # 数据
    cfg_engine = {
        'fov_deg': args.config_data.get('input_contract', {}).get('fov_deg', 140.0),
        'near_mm': args.config_data.get('input_contract', {}).get('depth_clip_mm', [1, 60])[0],
        'far_mm': args.config_data.get('input_contract', {}).get('depth_clip_mm', [1, 60])[1],
        'nbins': 280, 'H': 256, 'W': 256, 'K_poses': 128,
        'directions': args.config_data.get('input_contract', {}).get('directions', ['ante', 'retro']),
        'pose_step_mm': 5.0,
        'sigma_max': 2.0, 'scale_max': 0.05,
    }

    train_ds = HQColonDataset(args.hqcolon_dir, args.manifest, args.split_file,
                               split='train', cfg=cfg_engine, seed=seed)
    cal_ds = HQColonDataset(args.hqcolon_dir, args.manifest, args.split_file,
                             split='cal', cfg=cfg_engine, seed=seed + 100)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn,
                              num_workers=0, pin_memory=True)
    cal_loader = DataLoader(cal_ds, batch_size=1, shuffle=False, collate_fn=collate_fn,
                            num_workers=0, pin_memory=True)

    # 模型
    model = P11Model(embed_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 损失
    cov_loss_fn = nn.MSELoss()
    rel_loss_fn = nn.MSELoss()  # 可靠性目标 = 重建保真度 (depth reprojection error 代理)
    naive_loss_fn = nn.MSELoss()

    best_cal_loss = float('inf')
    patience = 10
    no_improve = 0
    t0 = time.time()
    log_lines = []

    for epoch in range(args.epochs):
        # 时间限制
        elapsed = (time.time() - t0) / 60
        if elapsed > args.max_minutes:
            print(f"[seed={seed}] 时间到 {elapsed:.1f}min > {args.max_minutes}min, 停止")
            break

        model.train()
        epoch_loss = 0.0
        n_batch = 0
        for batch in train_loader:
            depth = batch['depth'].to(device)
            vmask = batch['validity_mask'].to(device)
            cov_gt = batch['coverage_gt'].to(device)
            cov_clean = batch['coverage_clean'].to(device)
            rel_feat = batch['rel_features'].to(device)
            naive_feat = batch['naive_features'].to(device)
            families = batch['family']

            out = model(depth, vmask, rel_feat, naive_feat)
            cov_pred = out['coverage_pred']

            # Coverage loss: 预测漂移后覆盖度 (B族) 或 clean (A族)
            # B 族 target = drifted coverage; A 族 target = clean (退化只影响输入)
            target = cov_gt.clone()
            for i, fam in enumerate(families):
                if fam == 'A' or fam == 'clean':
                    target[i] = cov_clean[i]

            loss_cov = cov_loss_fn(cov_pred, target)

            # 可靠性 loss: 目标 = |coverage_error| 的反 (更可靠 = 误差更小)
            cov_err = torch.abs(cov_pred - cov_clean)
            rel_target = 1.0 / (1.0 + cov_err * 10)  # 归一化到 [0,1], 误差小→可靠度高
            loss_rel = rel_loss_fn(out['reliability_score'], rel_target.detach())

            # 朴素 loss: 同样的目标
            loss_naive = naive_loss_fn(out['naive_score'], rel_target.detach())

            loss = loss_cov + 0.1 * loss_rel + 0.1 * loss_naive

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batch += 1

        scheduler.step()
        avg_loss = epoch_loss / max(1, n_batch)

        # 验证
        cal_loss = _evaluate(model, cal_loader, device, cov_loss_fn)

        # 日志
        log_line = {'seed': seed, 'epoch': epoch, 'train_loss': round(avg_loss, 6),
                    'cal_loss': round(cal_loss, 6), 'elapsed_min': round(elapsed, 1)}
        log_lines.append(log_line)
        print(f"[seed={seed}] ep{epoch} train={avg_loss:.6f} cal={cal_loss:.6f} t={elapsed:.1f}m")

        # Early stopping
        if cal_loss < best_cal_loss:
            best_cal_loss = cal_loss
            no_improve = 0
            # 存 best checkpoint
            ckpt_path = os.path.join(args.out_dir, f"seed{seed}", 'checkpoint_best.pth')
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'epoch': epoch, 'cal_loss': cal_loss, 'seed': seed}, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[seed={seed}] early stop at epoch {epoch}")
                break

    # 存训练日志
    log_path = os.path.join(args.out_dir, f"seed{seed}", 'train_log.jsonl')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w') as f:
        for line in log_lines:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

    # 存 config 副本
    cfg_copy = os.path.join(args.out_dir, f"seed{seed}", 'config.yaml')
    import shutil
    shutil.copy2(args.config, cfg_copy)

    print(f"[seed={seed}] 完成. best_cal={best_cal_loss:.6f}")
    return best_cal_loss


def _evaluate(model, loader, device, loss_fn):
    """验证集评估"""
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            depth = batch['depth'].to(device)
            vmask = batch['validity_mask'].to(device)
            cov_gt = batch['coverage_gt'].to(device)
            cov_clean = batch['coverage_clean'].to(device)
            rel_feat = batch['rel_features'].to(device)
            naive_feat = batch['naive_features'].to(device)

            out = model(depth, vmask, rel_feat, naive_feat)
            # 验证用 clean GT 作为 target
            loss = loss_fn(out['coverage_pred'], cov_clean)
            total_loss += loss.item()
            n += 1
    return total_loss / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='E001_decisive_pilot.yaml 路径')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--hqcolon-dir', default='~/data/hqcolon')
    ap.add_argument('--out-dir', default='~/checkpoints/P11_coverage_reliability/E001_decisive_pilot')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--max-minutes', type=float, default=90)
    args = ap.parse_args()

    # 路径展开
    args.hqcolon_dir = os.path.expanduser(args.hqcolon_dir)
    args.out_dir = os.path.expanduser(args.out_dir)

    # 加载 config
    import yaml
    with open(args.config) as f:
        args.config_data = yaml.safe_load(f)

    # 定位 manifest 和 split 文件
    config_dir = os.path.dirname(os.path.abspath(args.config))
    args.manifest = os.path.join(config_dir, 'E001_pilot_manifest.json')
    args.split_file = os.path.join(config_dir, 'split_patients.json')

    train_one_seed(args, args.seed)


if __name__ == '__main__':
    main()
