#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P11 E001 数据集: HQColon 深度张量 + 两族退化 + 可靠性/朴素特征

优化: 全部几何数据(深度+poses+centroids+areas)缓存到 NPZ, 避免重算 centerline。
B 族退化 coverage target 预计算并缓存, 避免 __getitem__ 在线跑引擎。
"""
import os, sys, json, time
import numpy as np
import torch
from torch.utils.data import Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.join(HERE, "..", "engine")
sys.path.insert(0, ENGINE_DIR)
import coverage_gt_engine as E


def _load_geometry(mask_path, cfg, cache_dir=None):
    """加载单个几何, 全部数据缓存。缓存读 ~0.2s, 首次 ~17s。"""
    name = os.path.basename(mask_path).replace(".mha", "")
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{name}_full.npz")
        meta_path = os.path.join(cache_dir, f"{name}_meta.json")
        if os.path.exists(cache_path) and os.path.exists(meta_path):
            npz = np.load(cache_path, allow_pickle=False)
            meta = json.load(open(meta_path))
            cam_arr = npz['poses_cam']
            fwd_arr = npz['poses_fwd']
            poses = [(cam_arr[i], fwd_arr[i]) for i in range(len(cam_arr))]
            return (npz['depth'], npz['validity'], poses,
                    npz['centroids'], npz['areas'],
                    npz.get('centroids_lite', npz['centroids']),
                    npz.get('areas_lite', npz['areas']),
                    meta['cov_clean'], name, meta['n_poses'])

    # 从头计算
    lumen, sp, _ = E.load_lumen(mask_path)
    _, _, centroids, areas = E.build_mesh(lumen, sp, 2)
    # 降采样 centroids: 用于 B 族退化加速 (保留面积加权近似)
    MAX_CENTROIDS = 50000
    if len(centroids) > MAX_CENTROIDS:
        idx = np.random.default_rng(42).choice(len(centroids), MAX_CENTROIDS, replace=False)
        centroids_lite = centroids[idx]
        areas_lite = areas[idx]
    else:
        centroids_lite = centroids
        areas_lite = areas
    path = E.centerline(lumen, sp, 3)
    path_rs, _ = E.resample_path(path, cfg.get('pose_step_mm', 5.0))
    poses = E.fly_through_poses(path_rs, cfg.get('directions', ['ante', 'retro']))
    K = len(poses)
    nbins = cfg.get('nbins', 280)
    H = W = cfg.get('H', 256)

    depth_all = np.zeros((K, nbins, nbins), np.float32)
    for i, (cam, fwd) in enumerate(poses):
        img, _ = E.polar_depth_from_pose(cam, fwd, centroids,
                                          cfg['fov_deg'], cfg['near_mm'], cfg['far_mm'], nbins)
        depth_all[i] = img

    from scipy.ndimage import zoom
    if nbins != H:
        depth_rs = zoom(depth_all, (1, H / nbins, H / nbins), order=1)
    else:
        depth_rs = depth_all

    near, far = cfg['near_mm'], cfg['far_mm']
    depth_rs = np.clip(depth_rs, near, far)
    validity = (depth_rs > 0).astype(np.uint8)
    depth_norm = depth_rs / far

    cov_clean, _ = E.coverage_from_poses(centroids, areas, poses,
                                          cfg['fov_deg'], cfg['near_mm'], cfg['far_mm'], nbins)

    if cache_dir:
        np.savez_compressed(cache_path,
                            depth=depth_norm.astype(np.float16), validity=validity,
                            poses_cam=np.array([p[0] for p in poses], np.float32),
                            poses_fwd=np.array([p[1] for p in poses], np.float32),
                            centroids=centroids.astype(np.float32),
                            areas=areas.astype(np.float32),
                            centroids_lite=centroids_lite.astype(np.float32),
                            areas_lite=areas_lite.astype(np.float32))
        json.dump({'cov_clean': float(cov_clean), 'n_poses': K}, open(meta_path, 'w'))

    return (depth_norm.astype(np.float16), validity, poses,
            centroids, areas, centroids_lite, areas_lite,
            float(cov_clean), name, K)


def _drift_poses_fast(poses_cam, poses_fwd, sigma_step, scale_rate, rng):
    """向量化位姿漂移: 输入 numpy 数组, 返回漂移后的 cam 数组"""
    K = len(poses_cam)
    off = np.cumsum(rng.normal(0, sigma_step, (K, 3)), axis=0)
    t = np.linspace(0, 1, K).reshape(-1, 1)
    scale = 1.0 + scale_rate * t
    return poses_cam * scale + off  # [K,3]


def _compute_features(depth, validity, n_poses, poses_cam, drifted_cam):
    """一次性算全部 7 个特征 (3 可靠性 + 4 朴素)。降采样到每 8 帧取 1 帧加速"""
    SUB = 8  # 降采样因子
    depth_s = depth[::SUB]
    validity_s = validity[::SUB]
    K_s = depth_s.shape[0]
    # 相邻帧深度差 (降采样后)
    if K_s > 1:
        diff = np.abs(depth_s[1:] - depth_s[:-1])
        vm = (validity_s[1:] > 0) & (validity_s[:-1] > 0)
        vals = diff[vm]
        frame_diff_mean = float(vals.mean()) if vals.size > 0 else 0.0
    else:
        frame_diff_mean = 0.0

    # pose_drift RMS (降采样后)
    if drifted_cam is not None:
        dc = drifted_cam[::SUB]
        pc = poses_cam[::SUB]
        pd = float(np.sqrt(np.sum((dc - pc)**2, axis=1)).mean())
    else:
        pd = 0.0

    # obs_density (降采样后)
    od = float((validity_s > 0).mean())

    # predicted_var (降采样后)
    vm2 = validity_s > 0
    vd = depth_s[vm2]
    pv = float(vd.var()) if vd.size > 0 else 0.0

    # frame_count
    n_valid = int((validity_s.sum(axis=(1, 2)) > 0).sum())
    fc = n_valid / max(1, len(depth_s))

    rel = np.array([frame_diff_mean, pd, od], np.float32)
    naive = np.array([pv, fc, od, frame_diff_mean], np.float32)
    return rel, naive


class HQColonDataset(Dataset):
    """P11 E001 pilot 数据集 — 预计算所有退化, __getitem__ 纯查表+张量"""

    def __init__(self, hqcolon_dir, manifest_path, split_path, split='train',
                 cfg=None, seed=0, n_strength=5):
        self.cfg = cfg or {}
        self.n_strength = n_strength
        self.rng = np.random.default_rng(seed)
        self.cache_dir = os.path.join(hqcolon_dir, 'cache')
        self.split = split

        with open(manifest_path) as f:
            manifest = json.load(f)

        split_map = {'train': 'pilot_train', 'cal': 'pilot_cal', 'test': 'pilot_test'}
        patients = manifest[split_map[split]]
        patient_to_cases = manifest['patient_to_cases']
        masks_dir = os.path.join(hqcolon_dir, 'masks')

        self.samples = []
        for sub in patients:
            if sub not in patient_to_cases:
                continue
            for case in patient_to_cases[sub]:
                mha_path = os.path.join(masks_dir, f"{case}.mha")
                if os.path.exists(mha_path):
                    self.samples.append((sub, case, mha_path))

        # 预加载几何
        self.geometries = {}
        self._preload_geometries()

        # 预计算 B 族退化 (最慢的部分, 一次性做完)
        self.b_cache = {}  # (case, strength_idx) → (cov_target, drifted_cam, rel_feat)
        if split != 'cal':  # cal 不需要退化
            self._precompute_B_degradation(seed)

        # 构建索引
        self.index = []
        for sub, case, _ in self.samples:
            for fam in (['A', 'B'] if split != 'test' else ['clean']):
                for s in range(n_strength):
                    self.index.append((sub, case, fam, s))
        if split == 'test':
            for sub, case, _ in self.samples:
                for fam in ['A', 'B']:
                    for s in range(n_strength):
                        self.index.append((sub, case, fam, s))

    def _preload_geometries(self):
        t0 = time.time()
        total = len(self.samples)
        for idx, (sub, case, mha_path) in enumerate(self.samples):
            if case in self.geometries:
                continue
            depth, validity, poses, centroids, areas, centroids_lite, areas_lite, cov_clean, geo_id, n_poses = \
                _load_geometry(mha_path, self.cfg, self.cache_dir)
            poses_cam = np.array([p[0] for p in poses], np.float32)
            poses_fwd = np.array([p[1] for p in poses], np.float32)
            self.geometries[case] = {
                'depth': depth, 'validity': validity,
                'poses': poses, 'poses_cam': poses_cam, 'poses_fwd': poses_fwd,
                'centroids': centroids, 'areas': areas,
                'centroids_lite': centroids_lite, 'areas_lite': areas_lite,
                'cov_clean': cov_clean, 'n_poses': n_poses, 'subject_id': sub,
            }
            if (idx + 1) % 20 == 0:
                print(f"  加载 {idx+1}/{total} ({time.time()-t0:.0f}s)")
        print(f"  预加载完成: {total} 几何, {time.time()-t0:.1f}s")

    def _precompute_B_degradation(self, seed):
        """预计算 B 族退化 coverage target, 结果缓存到磁盘"""
        t0 = time.time()
        # 缓存文件名按几何集合 hash → train/cal/test 各自独立缓存, 不互相覆盖、不 miss
        import hashlib
        geohash = hashlib.sha256("|".join(sorted(self.geometries)).encode()).hexdigest()[:12]
        b_cache_path = os.path.join(self.cache_dir, f'b_targets_{geohash}.npz')
        b_meta_path = os.path.join(self.cache_dir, f'b_targets_{geohash}.json')

        # 尝试从磁盘缓存读 (但必须覆盖当前 split 所有几何, 否则 stale 缓存会让 test 几何 miss)
        if os.path.exists(b_cache_path) and os.path.exists(b_meta_path):
            npz = np.load(b_cache_path, allow_pickle=False)
            meta = json.load(open(b_meta_path))
            for key, m in meta['entries'].items():
                case, s_idx = key.rsplit('_', 1)
                s_idx = int(s_idx)
                self.b_cache[(case, s_idx)] = {
                    'cov_target': m['cov_target'],
                    'drifted_cam': npz[key] if key in npz else None,
                }
            need = {(c, s) for c in self.geometries for s in range(self.n_strength)}
            missing = need - set(self.b_cache)
            if not missing:
                print(f"  B族缓存读入: {len(self.b_cache)} 条 (覆盖全部 {len(self.geometries)} 几何), {time.time()-t0:.1f}s")
                return
            # 缓存不覆盖当前 split (典型: train 缓存被 eval/test run 复用) → 重算, 不盲用
            print(f"  ⚠ B族缓存缺 {len(missing)} 条 (缺几何如 {sorted({m[0] for m in missing})[:3]}) → 重算覆盖全 split")
            self.b_cache = {}

        sigma_max = self.cfg.get('sigma_max', 2.0)
        scale_max = self.cfg.get('scale_max', 0.05)
        cfg_engine = {
            'fov_deg': self.cfg.get('fov_deg', 140.0),
            'near_mm': self.cfg.get('near_mm', 1.0),
            'far_mm': self.cfg.get('far_mm', 60.0),
            'nbins': self.cfg.get('nbins', 280),
        }

        n_geo = len(self.geometries)
        all_drifted = {}
        meta_entries = {}

        for gi, (case, geo) in enumerate(self.geometries.items()):
            for s_idx in range(self.n_strength):
                s_frac = s_idx / max(1, self.n_strength - 1)
                sigma = s_frac * sigma_max
                scale_rate = s_frac * scale_max
                arr_key = f"{case}_{s_idx}"

                if sigma == 0 and scale_rate == 0:
                    self.b_cache[(case, s_idx)] = {
                        'cov_target': geo['cov_clean'],
                        'drifted_cam': geo['poses_cam'].copy(),
                    }
                    all_drifted[arr_key] = geo['poses_cam'].copy()
                    meta_entries[arr_key] = {'cov_target': geo['cov_clean']}
                    continue

                rng = np.random.default_rng(seed + hash(case) % 10000 + s_idx)
                drifted_cam = _drift_poses_fast(
                    geo['poses_cam'], geo['poses_fwd'], sigma, scale_rate, rng)
                drifted_poses = [(drifted_cam[i], geo['poses_fwd'][i])
                                 for i in range(len(drifted_cam))]
                cov_target, _ = E.coverage_from_poses(
                    geo['centroids_lite'], geo['areas_lite'], drifted_poses,
                    cfg_engine['fov_deg'], cfg_engine['near_mm'],
                    cfg_engine['far_mm'], cfg_engine['nbins'])
                self.b_cache[(case, s_idx)] = {
                    'cov_target': cov_target,
                    'drifted_cam': drifted_cam,
                }
                all_drifted[arr_key] = drifted_cam
                meta_entries[arr_key] = {'cov_target': cov_target}
            if (gi + 1) % 20 == 0:
                print(f"  B族预计算 {gi+1}/{n_geo} ({time.time()-t0:.0f}s)")

        # 缓存到磁盘
        np.savez_compressed(b_cache_path, **all_drifted)
        json.dump({'entries': meta_entries}, open(b_meta_path, 'w'))
        print(f"  B族预计算+缓存完成: {len(self.b_cache)} 条, {time.time()-t0:.1f}s")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        sub, case, family, strength = self.index[idx]
        geo = self.geometries[case]
        depth = geo['depth'].astype(np.float32).copy()
        validity = geo['validity'].copy()
        cov_clean = geo['cov_clean']
        poses_cam = geo['poses_cam']
        K = geo['n_poses']
        s_frac = strength / max(1, self.n_strength - 1)
        drifted_cam = None

        if family == 'A':
            self._apply_family_A_inplace(depth, validity, s_frac)
            cov_target = cov_clean
        elif family == 'B':
            bc = self.b_cache.get((case, strength))
            if bc is None:
                # 不许静默回退 clean (会让 Family-B 惰性, 决胜失效) → fail loud
                raise KeyError(
                    f"B-cache miss for ({case}, s={strength}): B 族退化未对此几何预计算; "
                    f"静默用 clean target 会让 Family-B 在 eval 上惰性 (决胜失效)。"
                    f"确保 _precompute_B_degradation 覆盖当前 split 的所有几何 (含 test)。")
            cov_target = bc['cov_target']
            drifted_cam = bc['drifted_cam']
        else:
            cov_target = cov_clean

        rel_feat, naive_feat = _compute_features(
            depth, validity, K, poses_cam, drifted_cam)

        depth_t = torch.from_numpy(depth[:, np.newaxis])
        valid_t = torch.from_numpy(validity)

        return {
            'depth': depth_t,
            'validity_mask': (valid_t.sum(dim=(1, 2)) > 0),
            'coverage_gt': torch.tensor(cov_target, dtype=torch.float32),
            'coverage_clean': torch.tensor(cov_clean, dtype=torch.float32),
            'family': family,
            'strength': strength,
            'subject_id': sub,
            'geometry_id': case,
            'rel_features': torch.from_numpy(rel_feat),
            'naive_features': torch.from_numpy(naive_feat),
        }

    def _apply_family_A_inplace(self, depth, validity, s):
        """A族退化 (就地修改 depth/validity)"""
        if s == 0:
            return
        K, H, W = depth.shape
        # 遮挡
        n_masks = int(s * 4)
        for _ in range(n_masks):
            cy, cx = self.rng.integers(0, H), self.rng.integers(0, W)
            r = max(1, int(np.sqrt(self.rng.uniform(0.02, 0.15)) * H * 0.5))
            yy, xx = np.ogrid[:H, :W]
            mask = ((yy - cy)**2 + (xx - cx)**2 <= r*r)
            depth[:, mask] = 0.0
            validity[:, mask] = 0
        # 变暗
        depth *= (1.0 - s * 0.6)
        # 丢帧
        drop_rate = s * 0.5
        if drop_rate > 0:
            drop_mask = self.rng.random(K) < drop_rate
            depth[drop_mask] = 0.0
            validity[drop_mask] = 0
