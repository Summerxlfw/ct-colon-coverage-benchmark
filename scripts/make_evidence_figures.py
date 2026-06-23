#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P11 覆盖度 benchmark — F2/F3/F4/F5 证据图直出 (Python + 真实数据, 零补点)。
源 → 中间 figure-data CSV (可追溯) → vector PDF/SVG。所有数字来自已 recount 的结果文件。
用法: 在项目根运行 `python3 scripts/make_evidence_figures.py`。
"""
import csv
import json
import math
import os
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore", category=UserWarning)  # 屏蔽 Arial 缺失 fallback 警告

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from plotting_template import (  # noqa: E402
    plt, setup_figure, save, CATEGORICAL_PALETTE,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "04_figures", "figure_outputs")
DATA = os.path.join(OUT, "data")
os.makedirs(DATA, exist_ok=True)

HEADS = ["base", "mlp", "seq"]
HEAD_LABEL = {"base": "scalar", "mlp": "MLP", "seq": "sequence"}
MDE = 0.01


def spearman(x, y):
    n = len(x)
    def rank(v):
        idx = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[idx[j + 1]] == v[idx[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[idx[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    sx = math.sqrt(sum((v - mx) ** 2 for v in rx))
    sy = math.sqrt(sum((v - my) ** 2 for v in ry))
    rho = cov / (sx * sy) if sx * sy else float("nan")
    t = rho * math.sqrt((n - 2) / (1 - rho * rho)) if abs(rho) < 1 else float("inf")
    p = math.erfc(abs(t) / math.sqrt(2))
    return rho, p, n


def write_csv(name, header, rows):
    path = os.path.join(DATA, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


# ----------------------------------------------------------------------------
# F2 — appearance-silent family: true-coverage drift vs strength (evidence panel)
# 源: results/E013_gt_mesh_diag/bfamily_fullmesh_gt.csv (drift_full)
# ----------------------------------------------------------------------------
def fig_F2():
    rows = list(csv.DictReader(open(os.path.join(RES, "E013_gt_mesh_diag", "bfamily_fullmesh_gt.csv"))))
    by_s = defaultdict(list)
    for r in rows:
        by_s[int(r["strength"])].append(float(r["drift_full"]))
    strengths = sorted(by_s)
    mean_abs, frac_mde = [], []
    table = []
    for s in strengths:
        d = [abs(v) for v in by_s[s]]
        m = sum(d) / len(d)
        fr = sum(1 for v in d if v >= MDE) / len(d)
        mean_abs.append(m)
        frac_mde.append(fr)
        table.append([s, f"{m:.4f}", f"{fr:.4f}", len(d)])
    write_csv("F2_drift_by_strength.csv",
              ["strength", "mean_abs_drift", "frac_ge_MDE", "n_geometries"], table)

    fig, ax = setup_figure(width="single", aspect=1.4)
    c = CATEGORICAL_PALETTE
    ax.plot(strengths, mean_abs, "o-", color=c[0], label="mean |true-coverage drift|")
    ax.axhline(MDE, ls="--", lw=0.9, color="0.5")
    ax.text(0.05, MDE + 0.001, f"MDE = {MDE}", color="0.4", fontsize=7)
    ax.set_xlabel("degradation strength")
    ax.set_ylabel("mean |Δ true coverage|", color=c[0])
    ax.set_xticks(strengths)
    ax.tick_params(axis="y", labelcolor=c[0])
    ax.set_ylim(0, max(mean_abs) * 1.25)

    ax2 = ax.twinx()
    ax2.plot(strengths, [f * 100 for f in frac_mde], "s--", color=c[3],
             label="% geometries with |Δ| ≥ MDE")
    ax2.set_ylabel("% geometries |Δ| ≥ MDE", color=c[3])
    ax2.tick_params(axis="y", labelcolor=c[3])
    ax2.set_ylim(0, 100)

    lines = ax.get_lines()[:1] + ax2.get_lines()[:1]
    ax.legend(lines, [l.get_label() for l in lines], loc="upper left", frameon=False)
    ax.set_title("Appearance-silent degradation shifts true coverage", fontsize=9)
    save(fig, os.path.join(OUT, "F2_drift"))
    return mean_abs, frac_mde


# ----------------------------------------------------------------------------
# F3 — leaderboard: coverage MAE per method + patient-bootstrap CI
# 源: results/E014_baseline_expansion/recount.json (E010 5 方法的扩充版,
#     5 个旧方法行 byte-identical 复用 E010; 新增 5 方法吃同一 depth-space 输入)
# ----------------------------------------------------------------------------
def fig_F3():
    rc = json.load(open(os.path.join(RES, "E014_baseline_expansion", "recount.json")))
    tab = rc["table"]
    # kind: ref_pose (pose-given, 绿) / ref_floor (平凡/随机地板, 灰) / depth (depth-only 估计器, 蓝)
    disp = {
        "muhlethaler_geom":       ("Muhlethaler (pose-given geom.)", "ref_pose"),
        "oracle_upper":           ("Oracle (true pose)",             "ref_pose"),
        "const_lower":            ("Constant predictor",             "ref_floor"),
        "random":                 ("Random predictor",               "ref_floor"),
        "our_head":               ("Attention-pool head",            "depth"),
        "meanpool_cnn":           ("Mean-pool CNN",                   "depth"),
        "ridge_depthfeat":        ("Ridge (depth feats.)",           "depth"),
        "transformer_pool":       ("Transformer pool",               "depth"),
        "c2d2_depth_star":        ("C2D2-inspired depth *†",         "depth"),
        "visible_area_heuristic": ("Visible-area heuristic",         "depth"),
    }
    order = sorted(disp, key=lambda k: tab[k]["mae_macro"])  # ascending MAE
    rows_csv, labels, vals, los, his, kinds = [], [], [], [], [], []
    for k in order:
        mae = tab[k]["mae_macro"]
        lo, hi = tab[k]["ci"]
        labels.append(disp[k][0]); kinds.append(disp[k][1])
        vals.append(mae); los.append(mae - lo); his.append(hi - mae)
        # 3-dp figure-data CSV = 正文/表 reporting 精度 (figure_text_consistency 子串匹配 0.100/0.099 等需同精度)
        rows_csv.append([k, f"{mae:.3f}", f"{lo:.3f}", f"{hi:.3f}", tab[k]["n_patients"]])
    write_csv("F3_leaderboard_mae.csv",
              ["method", "mae_macro", "ci_lo", "ci_hi", "n_patients"], rows_csv)

    fig, ax = setup_figure(width="single", aspect=1.55)
    ypos = list(range(len(labels)))[::-1]
    cmap = {"ref_pose": "#55A868", "ref_floor": "#999999", "depth": "#4C72B0"}
    colors = [cmap[kd] for kd in kinds]
    ax.barh(ypos, vals, xerr=[los, his], color=colors, height=0.66,
            error_kw=dict(ecolor="0.3", elinewidth=0.9, capsize=2))
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("coverage MAE (patient-level macro)")
    ax.set_xlim(0, max(h + v for h, v in zip(his, vals)) * 1.18)
    # on-bar numeric labels dropped: exact MAE + CI live in Table 2 (figure_plan
    # rule "数值进 table"); the bars carry the pattern (pose-given≈0, depth-only≈const)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#55A868", label="pose-given reference"),
                       Patch(color="#999999", label="trivial / chance floor"),
                       Patch(color="#4C72B0", label="depth-only estimator")],
              loc="upper right", frameon=False, fontsize=7)
    save(fig, os.path.join(OUT, "F3_leaderboard"))
    return {k: tab[k]["mae_macro"] for k in order}


# ----------------------------------------------------------------------------
# F4 — negative result: full 3-family CI forest.
# 源: results/E011_fullmesh/ABLATION_three_family.md (3-family, verified recount)
#     + self-check B against results/E011_fullmesh/{head}/verdict.json
# ----------------------------------------------------------------------------
# (point, lo, hi) per head per family, from the verified ABLATION table
ABL = {
    "base": {"clean": (-0.0011, -0.0024, 0.0002), "A": (-0.0019, -0.0033, -0.0006), "B": (-0.0001, -0.0039, 0.0037)},
    "mlp":  {"clean": (0.0007, 0.0002, 0.0013),  "A": (0.0004, -0.0005, 0.0014),  "B": (-0.00004, -0.0009, 0.0008)},
    "seq":  {"clean": (-0.0019, -0.0045, 0.0007), "A": (-0.0015, -0.0042, 0.0013), "B": (-0.0020, -0.0045, 0.0005)},
}


def load_head_verdict(head):
    return json.load(open(os.path.join(RES, "E011_fullmesh", head, "verdict.json")))


def format_signed_milli(value):
    rounded = round(value, 1)
    if abs(rounded) == 0:
        rounded = 0.0
    return f"{rounded:+.1f}"


def _row(head, family):
    pt, lo, hi = ABL[head][family]
    return {
        "head": head,
        "head_label": HEAD_LABEL[head],
        "family": family,
        "point": pt,
        "ci_lo": lo,
        "ci_hi": hi,
        "ci_crosses_zero": lo <= 0 <= hi,
        "ci_within_mde": abs(lo) < MDE and abs(hi) < MDE,
        "meaningful_gain": pt >= MDE and lo > 0,
    }


def build_F4_primary_endpoint_payload():
    # self-check: B family must match verdict.json (within rounding)
    for h in HEADS:
        v = load_head_verdict(h)
        pt, lo, hi = ABL[h]["B"]
        assert abs(v["point_est_macro"] - pt) < 5e-4, f"{h} B point mismatch vs verdict.json"
        assert abs(v["ci"][0] - lo) < 5e-4 and abs(v["ci"][1] - hi) < 5e-4, f"{h} B CI mismatch"

    fams = ["clean", "A", "B"]
    forest_rows = [_row(h, fam) for fam in fams for h in HEADS]
    context_rows = [_row(h, fam) for h in HEADS for fam in fams]
    primary_rows = [_row(h, "B") for h in HEADS]
    return {
        "forest_rows": forest_rows,
        "primary_rows": primary_rows,
        "context_rows": context_rows,
        "summary": {
            "primary_all_ci_cross_zero": all(r["ci_crosses_zero"] for r in primary_rows),
            "primary_all_within_mde": all(r["ci_within_mde"] for r in primary_rows),
            "primary_all_ci_within_mde_band": all(
                abs(r["ci_lo"]) < MDE and abs(r["ci_hi"]) < MDE for r in primary_rows
            ),
            "any_cell_ge_mde": any(r["point"] >= MDE for r in context_rows),
            "all_primary_endpoint_cis_cross_zero": all(r["ci_crosses_zero"] for r in primary_rows),
        },
    }


def fig_F4():
    payload = build_F4_primary_endpoint_payload()
    fams = ["clean", "A", "B"]
    fam_label = {
        "clean": "Clean",
        "A": "Appearance-visible A",
        "B": "Appearance-silent B (primary)",
    }

    rows_csv = []
    for r in payload["context_rows"]:
        rows_csv.append([
            r["head"],
            r["family"],
            f"{r['point']:.5f}",
            f"{r['ci_lo']:.5f}",
            f"{r['ci_hi']:.5f}",
            int(r["ci_crosses_zero"]),
            int(r["ci_within_mde"]),
            int(r["meaningful_gain"]),
        ])
    write_csv("F4_aurc_diff_3family.csv",
              ["head", "family", "point", "ci_lo", "ci_hi",
               "ci_crosses_zero", "ci_within_mde", "meaningful_gain"], rows_csv)
    write_csv("F4_primary_endpoint_B.csv",
              ["head", "point", "ci_lo", "ci_hi", "ci_crosses_zero", "ci_within_mde"],
              [[r["head"], f"{r['point']:.5f}", f"{r['ci_lo']:.5f}", f"{r['ci_hi']:.5f}",
                int(r["ci_crosses_zero"]), int(r["ci_within_mde"])]
               for r in payload["primary_rows"]])

    forest_rows = payload["forest_rows"]
    y_positions = []
    y = len(forest_rows) + 2.0
    last_family = None
    family_centers = defaultdict(list)
    for r in forest_rows:
        if last_family is not None and r["family"] != last_family:
            y -= 0.7
        y_positions.append(y)
        family_centers[r["family"]].append(y)
        y -= 1.0
        last_family = r["family"]

    fig, ax = plt.subplots(figsize=(7.05, 4.55), dpi=300)
    ax.axvspan(-MDE, MDE, color="0.92", zorder=0, label=f"sub-MDE band (±{MDE})")
    ax.axvline(0, color="0.4", lw=0.9, zorder=1)
    ax.axvline(MDE, color="0.68", lw=0.75, ls=":", zorder=1)
    ax.axvline(-MDE, color="0.68", lw=0.75, ls=":", zorder=1)

    for yi, r in zip(y_positions, forest_rows):
        is_primary = r["family"] == "B"
        color = "#C44E52" if is_primary else "#4C72B0"
        alpha = 1.0 if is_primary else 0.78
        marker_size = 5.4 if is_primary else 4.6
        line_width = 1.35 if is_primary else 1.05
        ax.errorbar(r["point"], yi,
                    xerr=[[r["point"] - r["ci_lo"]], [r["ci_hi"] - r["point"]]],
                    fmt="o", color=color, ecolor=color, alpha=alpha,
                    elinewidth=line_width, capsize=3, markersize=marker_size, zorder=3)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([r["head_label"] for r in forest_rows])
    ax.set_xlabel("AURC difference (naive − reliability)\n[>0 favours reliability]")
    ax.set_xlim(-0.012, 0.012)
    ax.set_ylim(min(y_positions) - 1.0, max(y_positions) + 1.0)
    ax.set_title("Three-family reliability-gating ablation", fontsize=9.3)

    for fam in fams:
        ys = family_centers[fam]
        center = sum(ys) / len(ys)
        ax.text(-0.0117, center, fam_label[fam], va="center", ha="left",
                fontsize=7.5, color="#8C2D34" if fam == "B" else "0.28",
                fontweight="bold" if fam == "B" else "normal")
    ax.tick_params(axis="y", pad=45)
    ax.text(0.0115, min(y_positions) - 0.62,
            "No cell reaches +MDE; all primary-endpoint CIs cross 0.",
            ha="right", va="bottom", fontsize=7.4, color="0.32")
    ax.text(MDE, max(y_positions) + 0.55, "+MDE", ha="center", va="bottom",
            fontsize=6.7, color="0.42")
    ax.text(-MDE, max(y_positions) + 0.55, "-MDE", ha="center", va="bottom",
            fontsize=6.7, color="0.42")
    ax.grid(axis="x", color="0.88", lw=0.5)

    fig.tight_layout(pad=0.45)
    save(fig, os.path.join(OUT, "F4_negative_forest"))
    return payload


# ----------------------------------------------------------------------------
# F5 — mechanism: (a) pose-drift vs abs_err scatter (B, Spearman ~ 0)
#                 (b) abs_err per family = degradation-invariant estimation floor
# 源: results/E011_fullmesh/{head}/eval_test.csv
# ----------------------------------------------------------------------------
def fig_F5():
    # panel b: mean abs_err per family per head (recomputed from raw CSV)
    floor = {}
    fams = ["clean", "A", "B"]
    rows_csv_b = []
    base_B_drift, base_B_err = [], []
    for h in HEADS:
        rows = list(csv.DictReader(open(os.path.join(RES, "E011_fullmesh", h, "eval_test.csv"))))
        per_fam = {}
        for fam in fams:
            errs = [float(r["abs_err"]) for r in rows if r["family"] == fam]
            per_fam[fam] = sum(errs) / len(errs)
            rows_csv_b.append([h, fam, f"{per_fam[fam]:.3f}", len(errs)])
        floor[h] = per_fam
        if h == "base":
            for r in rows:
                if r["family"] == "B":
                    base_B_drift.append(float(r["x_rel_pose_drift"]))
                    base_B_err.append(float(r["abs_err"]))
    write_csv("F5b_abs_err_by_family.csv",
              ["head", "family", "mean_abs_err", "n_rows"], rows_csv_b)

    rho, p, n = spearman(base_B_drift, base_B_err)
    write_csv("F5a_scatter_B_base.csv",
              ["pose_drift", "abs_err"],
              [[f"{a:.6f}", f"{b:.6f}"] for a, b in zip(base_B_drift, base_B_err)])

    from plotting_template import DOUBLE_COLUMN_INCH
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_INCH, DOUBLE_COLUMN_INCH / 2.3), dpi=300)
    c = CATEGORICAL_PALETTE

    # (a) scatter
    ax = axes[0]
    ax.scatter(base_B_drift, base_B_err, s=6, alpha=0.22, color=c[0], edgecolors="none")
    ax.set_xlabel("reconstruction-reliability signal (pose-drift)")
    ax.set_ylabel("model |coverage error|")
    ax.set_title(f"(a) no measurable rank association\nSpearman = {rho:+.3f} (p = {p:.2f}, n = {n}); scalar head, B", fontsize=8.5)
    ax.set_ylim(0, max(base_B_err) * 1.05)

    # (b) estimation floor
    ax = axes[1]
    x = range(len(fams))
    width = 0.26
    for i, h in enumerate(HEADS):
        vals = [floor[h][f] for f in fams]
        ax.bar([xi + (i - 1) * width for xi in x], vals, width=width,
               color=CATEGORICAL_PALETTE[i], label=HEAD_LABEL[h])
    ax.axhline(0.075, ls="--", lw=0.9, color="0.45")
    ax.text(2.32, 0.076, "≈0.075 floor", fontsize=7, color="0.4", ha="right")
    ax.set_xticks(list(x))
    ax.set_xticklabels(["clean", "appear.-\nvisible (A)", "appear.-\nsilent (B)"])
    ax.set_ylabel("mean |coverage error|")
    ax.set_ylim(0, 0.10)
    ax.set_title("(b) error is a degradation-invariant floor", fontsize=8.5)
    ax.legend(loc="upper right", frameon=False, fontsize=7)

    fig.tight_layout(pad=0.4)
    save(fig, os.path.join(OUT, "F5_mechanism"))
    return rho, p, n, floor


if __name__ == "__main__":
    print("== F2 ==", fig_F2())
    print("== F3 ==", fig_F3())
    fig_F4()
    print("== F4 ==", "rendered; B self-check passed;", {h: ABL[h]["B"] for h in HEADS})
    print("== F5 ==", fig_F5()[:3])
    print("\noutputs ->", OUT)
    for f in sorted(os.listdir(OUT)):
        if f.endswith((".pdf", ".svg")):
            print("  ", f)
