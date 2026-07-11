#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
覆盖度 GT 引擎 v0 — P11. 把 CT 充气结肠 mask 转成
"虚拟内镜 fly-through 看到了哪些表面" 的几何覆盖度真值。

管线: gas-filled lumen mask
  → 最大连通块 → marching_cubes 水密腔面 (verts/faces/面积, mm 坐标, (z,y,x) frame)
  → 中心线 (降采样体上 EDT + 两次测地最远点定端 + inverse-DT 加权 dijkstra 取路径)
  → 沿中心线放虚拟内镜相机 (antegrade + retrograde, 可配 FOV/步长/朝向)
  → 每位姿"角度 z-buffer"可见性 (每角度 bin 取最近面 = 该方向看到的腔壁, 不需法向定向)
  → 累积 seen 面集合 → coverage = 已见面积 / 总面积 + per-segment(按弧长分段)。

可见性是 centroid 级角度 z-buffer 近似 (v0): O(F) per pose, 无三角光栅化。
故意"看不全"皱襞背面 = 真实漏诊覆盖度的几何来源, 正是要测的量。

只用 SimpleITK + numpy + scipy + skimage。结果落 CSV/NPZ。
用法: python coverage_gt_engine.py <mask.mha> --out <dir> [--mesh-downsample 2] [--fov-deg 140] ...
"""
import argparse, os, json, time
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage import measure


# ----------------------------------------------------------------- 载入 + 网格
def load_lumen(path):
    img = sitk.ReadImage(path)
    sp = np.array(img.GetSpacing(), float)              # (sx,sy,sz) mm
    a = sitk.GetArrayFromImage(img)                      # (z,y,x)
    lumen = (a == 1)
    lab, n = ndimage.label(lumen, structure=np.ones((3, 3, 3)))
    if n == 0:
        raise RuntimeError("no lumen voxels")
    sizes = np.bincount(lab.ravel())[1:]
    lumen = (lab == (np.argmax(sizes) + 1))
    spacing_zyx = np.array([sp[2], sp[1], sp[0]])        # (z,y,x) mm
    return lumen, spacing_zyx, float(sizes.max() / max(1, sizes.sum()))


def build_mesh(lumen, spacing_zyx, downsample=1):
    vol = lumen
    sp = spacing_zyx.copy()
    if downsample > 1:
        vol = lumen[::downsample, ::downsample, ::downsample]
        sp = spacing_zyx * downsample
    vol = np.pad(vol.astype(np.float32), 1, constant_values=0)
    verts, faces, _, _ = measure.marching_cubes(vol, level=0.5, spacing=tuple(sp))
    verts = verts - sp                                    # 抵消 pad 偏移
    tri = verts[faces]                                    # (F,3,3) mm (z,y,x)
    centroids = tri.mean(axis=1)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    return verts, faces, centroids, areas


# ----------------------------------------------------------------- 中心线
def _voxel_graph(mask, spacing, weight_fn=None):
    """26-邻接稀疏图; 节点=mask 内体素。weight_fn(dt)->每节点权重乘子(可空)。"""
    idx = np.full(mask.shape, -1, np.int64)
    coords = np.argwhere(mask)
    idx[tuple(coords.T)] = np.arange(len(coords))
    offs = [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            if not (dz == dy == dx == 0)]
    rows, cols, data = [], [], []
    dt = None
    if weight_fn is not None:
        dt = ndimage.distance_transform_edt(mask, sampling=spacing)
    for off in offs:
        step = np.array(off) * spacing
        elen = np.linalg.norm(step)
        shifted = np.roll(mask, off, (0, 1, 2))
        # roll 环绕需屏蔽越界, 简化: 用切片对齐
        sl_src, sl_dst = [], []
        for o in off:
            if o == 0: sl_src.append(slice(None)); sl_dst.append(slice(None))
            elif o == 1: sl_src.append(slice(0, -1)); sl_dst.append(slice(1, None))
            else: sl_src.append(slice(1, None)); sl_dst.append(slice(0, -1))
        src_m = mask[tuple(sl_src)] & mask[tuple(sl_dst)]
        si = idx[tuple(sl_src)][src_m]; di = idx[tuple(sl_dst)][src_m]
        w = np.full(si.shape, elen)
        if weight_fn is not None:
            wmul = weight_fn(dt)
            w = w * 0.5 * (wmul[tuple(sl_src)][src_m] + wmul[tuple(sl_dst)][src_m])
        rows.append(si); cols.append(di); data.append(w)
    rows = np.concatenate(rows); cols = np.concatenate(cols); data = np.concatenate(data)
    N = len(coords)
    g = csr_matrix((data, (rows, cols)), shape=(N, N))
    g = g.maximum(g.T)
    return g, coords, dt


def centerline(lumen, spacing_zyx, cl_downsample=3, off_center_penalty=6.0):
    # max-pool 降采样 (块内有腔即保留) 保住又细又绕的肠腔连通性;
    # stride 降采样会切断 1-3 体素宽的肠腔 → 中心线退化 (colon_002 bug)。
    from skimage.measure import block_reduce
    vol = block_reduce(lumen, (cl_downsample,) * 3, np.max).astype(bool)
    lab, n = ndimage.label(vol, structure=np.ones((3, 3, 3)))
    if n > 1:                                              # 防御: 仍碎则取最大块
        sizes = np.bincount(lab.ravel())[1:]
        vol = (lab == (np.argmax(sizes) + 1))
    sp = spacing_zyx * cl_downsample
    # 端点: 两次测地最远点 (无权图)
    g_geo, coords, _ = _voxel_graph(vol, sp, weight_fn=None)
    d0 = dijkstra(g_geo, indices=0, directed=False)
    A = int(np.argmax(np.where(np.isfinite(d0), d0, -1)))
    dA = dijkstra(g_geo, indices=A, directed=False)
    B = int(np.argmax(np.where(np.isfinite(dA), dA, -1)))
    # 路径: inverse-DT 加权, 让路径走腔中心
    def wfn(dt):
        dtn = dt / (dt.max() + 1e-9)
        return 1.0 + off_center_penalty * (1.0 - dtn)     # 离中心越远权越大
    g_w, coords, dt = _voxel_graph(vol, sp, weight_fn=wfn)
    dist, pred = dijkstra(g_w, indices=A, directed=False, return_predecessors=True)
    # 回溯 A->B
    path = []
    j = B
    while j != A and j >= 0:
        path.append(j); j = pred[j]
    path.append(A); path = path[::-1]
    pts_vox = coords[path]                                  # (P,3) 降采样体素
    pts_mm = pts_vox * sp                                   # mm (z,y,x), 原点一致(降采样按 stride)
    return pts_mm


def resample_path(pts_mm, step_mm=5.0):
    seg = np.linalg.norm(np.diff(pts_mm, axis=0), axis=1)
    arc = np.concatenate([[0], np.cumsum(seg)])
    total = arc[-1]
    targets = np.arange(0, total, step_mm)
    out = np.empty((len(targets), 3))
    for k, t in enumerate(targets):
        i = np.searchsorted(arc, t) - 1
        i = np.clip(i, 0, len(pts_mm) - 2)
        f = (t - arc[i]) / max(1e-9, seg[i])
        out[k] = pts_mm[i] * (1 - f) + pts_mm[i + 1] * f
    return out, total


# ----------------------------------------------------------------- 可见性
def cam_frame(forward):
    f = forward / (np.linalg.norm(forward) + 1e-9)
    up0 = np.array([1.0, 0, 0]) if abs(f[0]) < 0.9 else np.array([0, 1.0, 0])
    right = np.cross(f, up0); right /= (np.linalg.norm(right) + 1e-9)
    up = np.cross(right, f)
    return f, right, up


def visible_faces_from_pose(cam, forward, centroids, fov_deg, near, far, nbins=280):
    f, right, up = cam_frame(forward)
    rel = centroids - cam
    depth = rel @ f
    m = (depth > near) & (depth < far)
    if not m.any():
        return np.empty(0, np.int64)
    ridx = np.nonzero(m)[0]
    rr = rel[m]
    dz = depth[m]
    x = rr @ right; y = rr @ up
    ang = np.degrees(np.arctan2(np.sqrt(x * x + y * y), dz))   # 离光轴极角
    inside = ang <= (fov_deg / 2.0)
    if not inside.any():
        return np.empty(0, np.int64)
    ridx = ridx[inside]; dz = dz[inside]
    phi = np.arctan2(y[inside], x[inside])                      # 方位
    theta = np.radians(ang[inside])
    # 角度 bin (phi, theta) z-buffer: 每 bin 取最近面
    half = np.radians(fov_deg / 2.0)
    ti = np.clip((theta / half * (nbins - 1)).astype(int), 0, nbins - 1)
    pi = np.clip(((phi + np.pi) / (2 * np.pi) * (nbins - 1)).astype(int), 0, nbins - 1)
    binid = ti.astype(np.int64) * nbins + pi
    order = np.lexsort((dz, binid))                            # 同 bin 内按 depth 升序
    binid_s = binid[order]
    first = np.ones(len(binid_s), bool)
    first[1:] = binid_s[1:] != binid_s[:-1]                    # 每 bin 第一个=最近
    winners = ridx[order][first]
    return winners


def polar_depth_from_pose(cam, forward, centroids, fov_deg, near, far, nbins=280):
    """返回 [nbins,nbins] 极坐标深度图 (每 bin 最近面深度, 0=no-hit) + 每 bin 胜出面 idx (-1=空)。
    与 visible_faces_from_pose 同一可见性算子 → 深度输入与覆盖度 GT 天然一致 (红队 E0)。"""
    f, right, up = cam_frame(forward)
    rel = centroids - cam
    depth = rel @ f
    img = np.zeros((nbins, nbins), np.float32)
    win = np.full(nbins * nbins, -1, np.int64)
    m = (depth > near) & (depth < far)
    if not m.any():
        return img, win
    ridx = np.nonzero(m)[0]; rr = rel[m]; dz = depth[m]
    x = rr @ right; y = rr @ up
    ang = np.degrees(np.arctan2(np.sqrt(x * x + y * y), dz))
    inside = ang <= (fov_deg / 2.0)
    if not inside.any():
        return img, win
    ridx = ridx[inside]; dz = dz[inside]
    phi = np.arctan2(y[inside], x[inside]); theta = np.radians(ang[inside])
    half = np.radians(fov_deg / 2.0)
    ti = np.clip((theta / half * (nbins - 1)).astype(int), 0, nbins - 1)
    pi = np.clip(((phi + np.pi) / (2 * np.pi) * (nbins - 1)).astype(int), 0, nbins - 1)
    binid = ti.astype(np.int64) * nbins + pi
    order = np.lexsort((dz, binid))
    binid_s = binid[order]; dz_s = dz[order]; r_s = ridx[order]
    first = np.ones(len(binid_s), bool); first[1:] = binid_s[1:] != binid_s[:-1]
    bwin = binid_s[first]
    flat = img.ravel()
    flat[bwin] = dz_s[first]
    win[bwin] = r_s[first]
    return flat.reshape(nbins, nbins), win


def fly_through_poses(path_pts, directions):
    """生成 (pos, forward) 位姿序列 (与 fly_through_coverage 同). 供深度渲染/退化复用。"""
    out = []
    for direction in directions:
        pts = path_pts if direction == "ante" else path_pts[::-1]
        tang = np.gradient(pts, axis=0)
        for i in range(len(pts)):
            if np.linalg.norm(tang[i]) < 1e-6:
                continue
            out.append((pts[i], tang[i]))
    return out


def coverage_from_poses(centroids, areas, poses, fov_deg, near, far, nbins):
    """给定任意位姿集 (可退化), 用同一可见性算子算 seen 面积比。退化轨迹的 coverage target 用它。"""
    seen = np.zeros(len(centroids), bool)
    for cam, fwd in poses:
        w = visible_faces_from_pose(cam, fwd, centroids, fov_deg, near, far, nbins)
        seen[w] = True
    return areas[seen].sum() / areas.sum(), seen


def fly_through_coverage(centroids, areas, path_pts, cfg):
    seen = np.zeros(len(centroids), bool)
    poses = []
    # antegrade + retrograde
    for direction in cfg["directions"]:
        pts = path_pts if direction == "ante" else path_pts[::-1]
        tang = np.gradient(pts, axis=0)
        for i in range(len(pts)):
            fwd = tang[i] if direction == "ante" else tang[i]
            if np.linalg.norm(fwd) < 1e-6:
                continue
            w = visible_faces_from_pose(pts[i], fwd, centroids,
                                        cfg["fov_deg"], cfg["near_mm"], cfg["far_mm"],
                                        cfg["nbins"])
            seen[w] = True
            poses.append(pts[i])
    cov = areas[seen].sum() / areas.sum()
    return cov, seen, len(poses)


def per_segment_coverage(centroids, areas, path_pts, seen, n_seg=6):
    # 每面归到最近中心线点 → 按弧长分 n_seg 段
    from scipy.spatial import cKDTree
    seg = np.linalg.norm(np.diff(path_pts, axis=0), axis=1)
    arc = np.concatenate([[0], np.cumsum(seg)])
    tree = cKDTree(path_pts)
    _, nn = tree.query(centroids)
    face_arc = arc[nn]
    edges = np.linspace(0, arc[-1], n_seg + 1)
    out = []
    for s in range(n_seg):
        mask = (face_arc >= edges[s]) & (face_arc < edges[s + 1] + (1e-6 if s == n_seg - 1 else 0))
        if mask.sum() == 0:
            out.append(None); continue
        out.append(float(areas[mask & seen].sum() / areas[mask].sum()))
    return out


# ----------------------------------------------------------------- 渲染素材落盘
def save_render_npz(out_dir, name, verts, faces, centroids, areas, seen,
                    path_rs, seg_cov, cov, cfg, lumen=None):
    """落盘三图终稿渲染所需的真几何 + 真可见性 (供 scripts/render_three_figure_tiles.py).
    只存 engine 真算出来的量 (mesh/seen/中心线/一帧极坐标深度), 不编任何东西。
    npz 小 (~几 MB), 可同步到本地离线渲染, 不用搬 60GB 源 mask。"""
    from scipy.spatial import cKDTree
    # 代表性位姿的极坐标深度图 = 模型深度输入的真样子。
    # 在均匀抽样的若干位姿里选"看到面最多"的一帧 → 深度视图饱满 (不改任何 GT, 仅选展示帧)。
    poses = fly_through_poses(path_rs, cfg["directions"])
    if poses:
        cand = poses[::max(1, len(poses) // 24)]
        best_img, best_hit = None, -1.0
        for pcam, pfwd in cand:
            img, _ = polar_depth_from_pose(pcam, pfwd, centroids, cfg["fov_deg"],
                                           cfg["near_mm"], cfg["far_mm"], cfg["nbins"])
            hit = float((img > 0).mean())
            if hit > best_hit:
                best_hit, best_img, pcam_b, pfwd_b = hit, img, pcam, pfwd
        depth_img, pcam, pfwd = best_img, pcam_b, pfwd_b
    else:                                                   # 退化保护: 路径过短
        pcam = path_rs[0]
        pfwd = (path_rs[1] - path_rs[0]) if len(path_rs) > 1 else np.array([1.0, 0, 0])
        depth_img, _ = polar_depth_from_pose(pcam, pfwd, centroids, cfg["fov_deg"],
                                             cfg["near_mm"], cfg["far_mm"], cfg["nbins"])
    # 每面归到最近中心线点的弧长 (供按段/弧长给网格上色)
    segp = np.linalg.norm(np.diff(path_rs, axis=0), axis=1)
    arcp = np.concatenate([[0], np.cumsum(segp)])
    _, nn = cKDTree(path_rs).query(centroids)
    face_arc = arcp[nn]
    seg_arr = np.array([c if c is not None else np.nan for c in seg_cov], np.float32)
    # 全位姿数组 (供 A/B 退化忠实复算: B 漂移这些位姿重算 seen; 与 training/dataset.py 同源)
    poses_cam = np.asarray([p[0] for p in poses], np.float32) if poses else np.zeros((0, 3), np.float32)
    poses_fwd = np.asarray([p[1] for p in poses], np.float32) if poses else np.zeros((0, 3), np.float32)
    # CT 充气腔 mask 的最大强度投影 (M1 "CT mask" 槽用; 真分割, 非编)
    extra = {}
    if lumen is not None:
        extra["mask_mip_coronal"] = lumen.max(axis=1).astype(bool)   # z×x
        extra["mask_mip_axial"] = lumen.max(axis=0).astype(bool)     # y×x
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(out_dir, f"{name}_render.npz"),
        verts=verts.astype(np.float32), faces=faces.astype(np.int32),
        centroids=centroids.astype(np.float32), areas=areas.astype(np.float32),
        seen=seen.astype(bool), path_rs=path_rs.astype(np.float32),
        face_arc=face_arc.astype(np.float32), arc_total=np.float32(arcp[-1]),
        seg_coverage=seg_arr, depth_img=depth_img.astype(np.float32),
        pose_cam=np.asarray(pcam, np.float32), pose_fwd=np.asarray(pfwd, np.float32),
        poses_cam=poses_cam, poses_fwd=poses_fwd,
        coverage=np.float32(cov), name=name,
        cfg_json=json.dumps(cfg, ensure_ascii=False), **extra)


# ----------------------------------------------------------------- 主流程
def run(mask_path, out_dir, mesh_downsample, cl_downsample, cfg, save_npz=False,
        save_render=False):
    t0 = time.time()
    name = os.path.basename(mask_path).replace(".mha", "")
    lumen, sp, largest_frac = load_lumen(mask_path)
    verts, faces, centroids, areas = build_mesh(lumen, sp, mesh_downsample)
    # 水密性 (usable 判据): 每边须被恰 2 面共享
    _e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]])
    _e = np.sort(_e, axis=1)
    _, _cnt = np.unique(_e, axis=0, return_counts=True)
    boundary_edges = int((_cnt == 1).sum()); nonmanifold_edges = int((_cnt > 2).sum())
    watertight = bool(boundary_edges == 0 and nonmanifold_edges == 0)
    path_pts = centerline(lumen, sp, cl_downsample)
    path_rs, total_len = resample_path(path_pts, cfg["pose_step_mm"])
    # 中心线合法性: 真实结肠中心线远长于包围盒对角线(肠子盘绕);
    # 退化(端点塌缩→路径极短)必须 flag, 不能当真覆盖度发出。
    zz, yy, xx = np.where(lumen)
    bbox_diag = float(np.linalg.norm([(zz.max() - zz.min()) * sp[0],
                                      (yy.max() - yy.min()) * sp[1],
                                      (xx.max() - xx.min()) * sp[2]]))
    centerline_ok = bool(total_len >= max(50.0, 0.6 * bbox_diag))
    cov, seen, npose = fly_through_coverage(centroids, areas, path_rs, cfg)
    seg_cov = per_segment_coverage(centroids, areas, path_rs, seen, cfg["n_seg"])
    res = dict(case=name, lumen_largest_frac=round(largest_frac, 4),
               watertight=watertight, boundary_edges=boundary_edges,
               nonmanifold_edges=nonmanifold_edges,
               centerline_ok=centerline_ok, bbox_diag_cm=round(bbox_diag / 10.0, 1),
               n_faces=int(len(faces)), n_seen=int(seen.sum()),
               total_area_cm2=round(areas.sum() / 100.0, 1),
               centerline_len_cm=round(total_len / 10.0, 1), n_poses=npose,
               coverage=round(float(cov), 4),
               seg_coverage=[round(c, 3) if c is not None else None for c in seg_cov],
               cfg=cfg, mesh_downsample=mesh_downsample, sec=round(time.time() - t0, 1))
    os.makedirs(out_dir, exist_ok=True)
    if save_npz:
        np.savez_compressed(os.path.join(out_dir, f"{name}_seen.npz"),
                            faces=faces, seen=seen, areas=areas)
    if save_render:
        save_render_npz(out_dir, name, verts, faces, centroids, areas, seen,
                        path_rs, seg_cov, cov, cfg, lumen=lumen)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mask"); ap.add_argument("--out", default="results/coverage_gt")
    ap.add_argument("--mesh-downsample", type=int, default=2)
    ap.add_argument("--cl-downsample", type=int, default=3)
    ap.add_argument("--fov-deg", type=float, default=140.0)
    ap.add_argument("--pose-step-mm", type=float, default=5.0)
    ap.add_argument("--near-mm", type=float, default=1.0)
    ap.add_argument("--far-mm", type=float, default=60.0)
    ap.add_argument("--nbins", type=int, default=280)
    ap.add_argument("--n-seg", type=int, default=6)
    ap.add_argument("--directions", default="ante,retro")
    ap.add_argument("--save-npz", action="store_true")
    ap.add_argument("--save-render", action="store_true",
                    help="额外落盘 {name}_render.npz (三图终稿渲染素材: mesh+seen+中心线+深度)")
    a = ap.parse_args()
    cfg = dict(fov_deg=a.fov_deg, pose_step_mm=a.pose_step_mm, near_mm=a.near_mm,
               far_mm=a.far_mm, nbins=a.nbins, n_seg=a.n_seg,
               directions=a.directions.split(","))
    res = run(a.mask, a.out, a.mesh_downsample, a.cl_downsample, cfg, a.save_npz,
              a.save_render)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
