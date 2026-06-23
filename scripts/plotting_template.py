#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P11 覆盖度 benchmark — Class A 证据图统一样式 (single source of truth)。
关键: pdf.fonttype=42 + ps.fonttype=42 防 IEEE/Elsevier Type-3 字体退稿
(SOP 模板漏设, 见 memory feedback_figure_matplotlib_type3_font_ieee)。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# === 字体 (Type-3 防御: 必设 42 = TrueType 内嵌) ===
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["font.size"] = 9
plt.rcParams["axes.titlesize"] = 10
plt.rcParams["axes.labelsize"] = 9
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["lines.linewidth"] = 1.5
plt.rcParams["lines.markersize"] = 4

# === Palette (corpus pastel CV) ===
CATEGORICAL_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
SEQUENTIAL_CMAP = "viridis"

SINGLE_COLUMN_INCH = 3.5
DOUBLE_COLUMN_INCH = 7.0


def setup_figure(width="single", aspect=1.5):
    w = SINGLE_COLUMN_INCH if width == "single" else DOUBLE_COLUMN_INCH
    h = w / aspect
    fig, ax = plt.subplots(figsize=(w, h), dpi=300)
    return fig, ax


def save(fig, stem):
    """导出 vector PDF + SVG (无栅格 final)。"""
    for ext in ("pdf", "svg"):
        fig.savefig(f"{stem}.{ext}", format=ext, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
