# 上 GPU 前 4 道本地 CPU gate — 全 PASS (2026-06-12)

红队(wf_ed1a7aa5)强制:决胜 pilot 上 GPU 前先在本地过这 4 道,catch 最贵的失败模式。**全 PASS。**
服务器可直接复用这些脚本(数据在 WD-6TB scratch / OSF)。

| Gate | 脚本 | 验什么 | 结果 |
|---|---|---|---|
| 1 深度模态+一致性 | `gate1_depth_modality.py` | 引擎极坐标 z-buffer 出深度图; 从深度重算覆盖度 vs 引擎差 ≤±0.02 | ✓ 5/5 diff=0.0000, seen 100% 一致 (同算子), 深度[1,60]mm fill 0.16-0.18 |
| 2 config+manifest | `configs/make_split.py` + manifest | split/pilot membership 断言 | ✓ pilot_test⊆test44, holdout 不相交, 双 sha 锁 |
| 3 两族退化 | `gate3_degradation.py` | A 族(遮挡)外观可见单调降; B 族(漂移)期望误差单调增+外观无声 | ✓ 3/3 (抓出并修"漂移渲染相机"bug) |
| 4 bootstrap harness | `gate4_bootstrap.py` | 按病人配对 cluster bootstrap + verdict + 伪复制演示 | ✓ YES/NO/PARTIAL 正确; 几何聚类 CI 0.0038<0.0050 病人=伪复制 |

## 关键发现 / 给服务器的注

- **Gate1 一致性 = 精确 0**:深度输入与覆盖度 GT 用同一可见性算子(红队 E0 建议),无 train/target 几何不一致。深度张量 = 引擎 `polar_depth_from_pose` 的 [nbins,nbins],resample 到 [256,256]。
- **Gate3 修了真 bug**:原 Family-B 把漂移加在渲染相机上 → fill 漂移 0.39-0.51(外观可见,违 B 族目的)。修正模型:**网络看的深度帧从真位姿渲染(外观 σ-不变=结构性无声),漂移只腐蚀用于累积覆盖度的估计位姿**(可靠性头从位姿图残差读, 朴素从单帧外观读不到)。
- ⚠ **诚实记录(给服务器调 config)**:纯位姿漂移单独只产生 ~1% 覆盖度误差(13mm 漂移 → |err|≤0.009),且饱和肠子(cov=1.0)无 B 信号。**Family-B 需靠 `scale_drift` / 更大 σ / 结构化漂移拉开覆盖度误差谱**,`E001_decisive_pilot.yaml` 的 B 族范围服务器需按此调并重过 Gate3。
- **Gate4 确证红队**:同病人强相关双几何按几何聚类 CI 偏窄(假信心)→ **必须按病人(subject_id)聚类**。verdict 逻辑(YES/PARTIAL/NO/INCONCLUSIVE-UNDERPOWERED + 功率护栏 MDE=0.01)跑通。

引擎新增复用函数:`polar_depth_from_pose` / `fly_through_poses` / `coverage_from_poses`(`engine/coverage_gt_engine.py`)。
