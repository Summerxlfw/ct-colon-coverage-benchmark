# 覆盖度 GT 引擎 (coverage_gt_engine.py) — v0

把 HQColon CT 充气结肠 mask 转成"虚拟内镜 fly-through 几何覆盖度真值"。P11 多几何
多几何 coverage-GT 底座的可执行实现。**不吃 GPU。**

## 管线

```
gas-filled lumen mask (.mha)
 → 最大连通块 → marching_cubes 水密腔面 (verts/faces/面积, mm, (z,y,x) frame)
 → 中心线 (降采样体 EDT + 两次测地最远点定端 + inverse-DT 加权 dijkstra 取路径)
 → 虚拟内镜相机沿中心线 (antegrade+retrograde, 可配 FOV/步长)
 → 每位姿"角度 z-buffer"可见性 (每角度 bin 取最近面 = 该方向看到的腔壁)
 → 累积 seen 面 → coverage = 已见面积/总面积 + per-segment(按弧长)
```

## 用法
```bash
python engine/coverage_gt_engine.py <mask.mha> --out results/coverage_gt \
  --fov-deg 140 --directions ante,retro --pose-step-mm 5 --mesh-downsample 2 [--save-npz]
```
关键参数: `--fov-deg`(内镜视场)、`--directions`(ante/retro=进/退镜)、`--pose-step-mm`(位姿密度)、
`--far-mm`(可视距)、`--mesh-downsample`、`--nbins`(角度分辨率)。退化这些 = 制造可控"漏诊"。

## v0 验证 (2026-06-12, 5 个 HQColon colon)

**单调性 sanity(colon_011, 观察越难覆盖度越低)= 通过**:
宽FOV双向 0.928 > 仅前向 0.902 > 稀疏 0.879 > 窄FOV70° 0.793 > 全难 0.537。
→ 引擎在真做几何可见性/遮挡,非返回常数。segment 2 一直最难(皱襞多)。

**跨几何(5 colon, baseline 配置 fov140/双向/step5)**:coverage 0.928–1.0(均 0.966),
中心线 91–135cm(真实结肠长度),面积 1963–3113 cm²,**7.6 s/colon**。见 `results/coverage_gt_pilot5.csv`。

## v0 已知局限(production 升级项, 非 blocker)
- 可见性是 **centroid 级角度 z-buffer 近似**(面只要 centroid 赢某角度 bin 即记 seen,不要求整面无遮挡)
  → 光滑充气好的肠子在最宽配置可饱和到 1.0(colon_003)。GT 一致且单调,够做底座;
  实验用带退化的部分轨迹不饱和。production 可升级**完整三角光栅化 + 真实内镜内参/光照**。
- 中心线 = 单条 A→B 最短路(inverse-DT 加权);未建 haustra 回退/retroflexion 复杂轨迹。
- 全 435 量产未跑(需解全 zip ~60GB);usable 几何数(防塌陷)待全量 meshability。
