#!/usr/bin/env python3
# 重做 Fig 2 的两个出问题板块为数据诚实的矢量图(供 user 拖进 PPT 替换):
#  (1) drift 曲线 panel —— marker 落真实 CSV 值, 保留真实形状(步长递减后 s3->s4 跳变),
#       修复原手画曲线 s=1 偏低 / s=3 偏高 / 被美化成光滑凸线 的问题(Critical 1)。
#  (2) A/B 覆盖网格 schematic —— A 两网格逐格全等(=coverage UNCHANGED),
#       B 两网格移位(≠coverage SHIFTS), 修复原 A 行画得不一样却标 UNCHANGED 的矛盾(Major 1)。
# 全部 pdf.fonttype=42 防 Type3; 矢量输出(无整页栅格, 修复 Major 2 的可编辑性)。
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams["pdf.fonttype"] = 42   # 防 Type 3
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.size"] = 10
plt.rcParams["svg.fonttype"] = "none"

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DATA = os.path.join(PROJ, "04_figures", "figure_outputs", "data", "F2_drift_by_strength.csv")
OUT = os.path.join(PROJ, "04_figures", "figure_outputs", "f2_rebuild")
os.makedirs(OUT, exist_ok=True)

ORANGE = "#E8923B"   # covered (seen)
GREY = "#5A6472"     # not covered (unseen)
BLUE = "#2E5C97"     # drift curve

# ---------------------------------------------------------------------------
# (1) drift 曲线 panel —— 真实 CSV 点, 真实形状
# ---------------------------------------------------------------------------
def curve_panel():
    rows = list(csv.DictReader(open(DATA)))
    s = [int(r["strength"]) for r in rows]
    drift = [float(r["mean_abs_drift"]) for r in rows]   # 0, 0.0082, 0.0154, 0.0211, 0.0364

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.plot(s, drift, "-o", color=BLUE, lw=2.0, ms=6.5, mfc=BLUE, mec="white", mew=1.0)
    # 端点真值标注(0.0364 -> 0.036)
    ax.annotate(f"{drift[-1]:.3f}", xy=(s[-1], drift[-1]), xytext=(s[-1]-0.05, drift[-1]+0.0022),
                ha="right", va="bottom", fontsize=9, color=BLUE)
    ax.axhline(0.01, ls="--", lw=0.9, color="0.55")
    ax.text(0.06, 0.0108, "MDE = 0.01", fontsize=7.5, color="0.4", va="bottom")
    ax.set_xlabel("degradation strength  $s$")
    ax.set_ylabel(r"mean $|\Delta$ true coverage$|$")
    ax.set_xticks(s)
    ax.set_xlim(-0.15, 4.25)
    ax.set_ylim(0, max(drift) * 1.18)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(OUT, f"F2_curve_clean.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return list(zip(s, drift))

# ---------------------------------------------------------------------------
# (2) A/B 覆盖网格 schematic —— A 全等, B 移位
# ---------------------------------------------------------------------------
def _staircase(n=10):
    """nominal 覆盖网格: 上左 unseen(grey) 阶梯, 下右 covered(orange)。返回 grid[row][col]=1 覆盖/0 未覆盖。"""
    # 每行 unseen 的列数(从上到下递减) —— 高覆盖(~75%)的阶梯
    unseen_per_row = [7, 6, 6, 5, 4, 3, 2, 1, 1, 0]
    grid = [[1] * n for _ in range(n)]
    for r in range(n):
        for c in range(unseen_per_row[r]):
            grid[r][c] = 0
    return grid

def _drift_grid(grid):
    """B 的 drifted 网格: 在边界上翻转少量 cell(modest shift), 体现 appearance-silent 真值移动。"""
    g = [row[:] for row in grid]
    # 沿阶梯边界翻几格(双向: 既有 covered->unseen 也有 unseen->covered, 与真实 drift 双向一致)
    flips = [(1, 6, 0), (2, 5, 0), (4, 3, 0),       # 原覆盖 -> 现未覆盖(后退)
             (6, 1, 1), (7, 0, 1), (8, 0, 1)]        # 原未覆盖 -> 现覆盖(前进)
    for r, c, val in flips:
        g[r][c] = val
    return g

def _draw_grid(ax, grid, x0, y0, cell=1.0, gap=0.06):
    n = len(grid)
    for r in range(n):
        for c in range(n):
            color = ORANGE if grid[r][c] else GREY
            ax.add_patch(Rectangle((x0 + c * cell, y0 + (n - 1 - r) * cell),
                                   cell - gap, cell - gap, facecolor=color, edgecolor="white", lw=0.4))

def covgrid_panel():
    nominal = _staircase()
    A_perturbed = [row[:] for row in nominal]       # A: 逐格全等(= UNCHANGED), 与 nominal 完全相同
    assert A_perturbed == nominal, "A 行两网格必须逐格全等"
    B_drifted = _drift_grid(nominal)                # B: 移位(≠ SHIFTS)
    assert B_drifted != nominal, "B 行 drifted 必须与 nominal 不同"

    n = len(nominal)
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.set_aspect("equal"); ax.axis("off")
    gw = n + 2.2   # 两网格间水平间距单位

    # A 行(上)
    yA = n + 2.0
    _draw_grid(ax, nominal, 0, yA)
    _draw_grid(ax, A_perturbed, gw, yA)
    ax.text(n/2, yA + n + 0.4, "nominal", ha="center", fontsize=10)
    ax.text(gw + n/2, yA + n + 0.4, "perturbed input", ha="center", fontsize=10)
    ax.text(2*gw + 0.3, yA + n/2, "$=$\ncoverage\nUNCHANGED", ha="left", va="center", fontsize=11, color=GREY)
    ax.text(-1.4, yA + n/2, "A", ha="center", va="center", fontsize=15, fontweight="bold", color="0.25")

    # B 行(下)
    yB = 0
    _draw_grid(ax, nominal, 0, yB)
    _draw_grid(ax, B_drifted, gw, yB)
    ax.text(n/2, yB + n + 0.4, "nominal", ha="center", fontsize=10)
    ax.text(gw + n/2, yB + n + 0.4, "drifted (same input)", ha="center", fontsize=10)
    ax.text(2*gw + 0.3, yB + n/2, r"$\neq$" + "\ncoverage\nSHIFTS", ha="left", va="center", fontsize=11, color=ORANGE)
    ax.text(-1.4, yB + n/2, "B", ha="center", va="center", fontsize=15, fontweight="bold", color="0.25")

    # 图例(横向分开, 防文字重叠)
    ax.add_patch(Rectangle((0, -1.9), 0.8, 0.8, facecolor=ORANGE, edgecolor="white"))
    ax.text(1.1, -1.5, "covered (seen)", va="center", fontsize=9)
    ax.add_patch(Rectangle((9.5, -1.9), 0.8, 0.8, facecolor=GREY, edgecolor="white"))
    ax.text(10.6, -1.5, "not covered (unseen)", va="center", fontsize=9)

    ax.set_xlim(-2.2, 2*gw + 6.5)
    ax.set_ylim(-2.2, 2*n + 3.2)
    fig.tight_layout(pad=0.2)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(OUT, f"F2_covgrid_clean.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    # 返回 A 两网格是否全等 + B 翻转格数, 供自检
    diff_B = sum(1 for r in range(n) for c in range(n) if nominal[r][c] != B_drifted[r][c])
    return (A_perturbed == nominal), diff_B

if __name__ == "__main__":
    pts = curve_panel()
    print("curve panel -> 真实 CSV 点:", [(s, round(d, 4)) for s, d in pts])
    a_equal, b_diff = covgrid_panel()
    print(f"covgrid panel -> A 两网格全等: {a_equal}; B drifted 翻转格数: {b_diff}")
    print("输出 ->", OUT)
    for f in sorted(os.listdir(OUT)):
        print("  ", f)
