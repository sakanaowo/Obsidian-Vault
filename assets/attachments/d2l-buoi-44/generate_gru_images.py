"""
Tạo 5 hình minh họa cho Buổi 44 - GRU (Gated Recurrent Units)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np
import os

OUT = "/home/sakana/Code/Obsidian-Vault/assets/attachments/d2l-buoi-44"
os.makedirs(OUT, exist_ok=True)

# ─── Màu sắc ────────────────────────────────────────────────────────────────
C_BG      = "#0d1117"
C_PANEL   = "#161b22"
C_ARROW   = "#58a6ff"
C_GATE_S  = "#c9d1d9"   # sigmoid gate
C_NODE    = "#ff7b72"   # input node
C_STATE   = "#7ee787"   # hidden state / cell state
C_CAND    = "#d2a8ff"   # candidate
C_TEXT    = "#f0f6fc"
C_DIM     = "#8b949e"
C_RESET   = "#ffa657"   # reset gate
C_UPDATE  = "#79c0ff"   # update gate
C_BORDER  = "#30363d"

def draw_box(ax, x, y, w, h, label, sublabel=None,
             fc=C_PANEL, ec=C_BORDER, tc=C_TEXT, fontsize=10, bold=True):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.05",
                         facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=3)
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x, y + (0.06 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize,
            color=tc, fontweight=weight, zorder=4)
    if sublabel:
        ax.text(x, y - 0.18, sublabel,
                ha="center", va="center", fontsize=fontsize - 2,
                color=C_DIM, fontweight="normal", zorder=4)

def arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.5, style="->", label=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0"),
                zorder=2)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.12, label, ha="center", va="bottom",
                fontsize=7, color=C_DIM, zorder=5)

def dot_arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.5, label=None):
    arrow(ax, x1, y1, x2, y2, color=color, lw=lw, label=label)

# ═══════════════════════════════════════════════════════════════════════════════
# HÌNH 1: Hai cổng Reset Gate & Update Gate
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis("off")
ax.set_title("Fig 10.2.1 — Reset Gate & Update Gate trong GRU",
            fontsize=13, color=C_TEXT, fontweight="bold", pad=14)

# Input + Hidden State
draw_box(ax, 1.2, 3.5, 1.6, 1.0, "X_t", "input", fc="#21262d", ec=C_BORDER)
draw_box(ax, 1.2, 1.8, 1.6, 1.0, "H_{t-1}", "hidden state", fc="#21262d", ec=C_BORDER)

# Nối concat
arrow(ax, 2.0, 3.5, 2.8, 3.5, color=C_ARROW)
arrow(ax, 2.0, 1.8, 2.8, 1.8, color=C_ARROW)
draw_box(ax, 2.8, 2.65, 0.05, 1.7, "", fc="none", ec="none")

# Copy ra 2 nhánh (vẽ bằng đường rẽ)
# Nhánh Update Gate
arrow(ax, 2.85, 3.5, 4.5, 3.5, color=C_UPDATE)
draw_box(ax, 6.0, 3.5, 2.2, 1.1,
         "Update Gate", "Z_t = σ([X_t, H_{t-1}])",
         fc="#0d2d4a", ec=C_UPDATE, tc=C_UPDATE)

# Nhánh Reset Gate
arrow(ax, 2.85, 1.8, 4.5, 1.8, color=C_RESET)
draw_box(ax, 6.0, 1.8, 2.2, 1.1,
         "Reset Gate", "R_t = σ([X_t, H_{t-1}])",
         fc="#3d1f00", ec=C_RESET, tc=C_RESET)

# Outputs
arrow(ax, 7.1, 3.5, 8.5, 3.5, color=C_UPDATE, label="Z_t")
arrow(ax, 7.1, 1.8, 8.5, 1.8, color=C_RESET, label="R_t")

draw_box(ax, 9.0, 3.5, 1.5, 0.9, "Z_t", "(0,1)^h",
         fc="#0d2d4a", ec=C_UPDATE, tc=C_UPDATE)
draw_box(ax, 9.0, 1.8, 1.5, 0.9, "R_t", "(0,1)^h",
         fc="#3d1f00", ec=C_RESET, tc=C_RESET)

# Ghi chú
ax.text(6.0, 0.7,
        "Cả 2 cổng đều nhận [X_t, H_{t-1}] — sigmoid ép giá trị về (0,1)",
        ha="center", va="bottom", fontsize=9, color=C_DIM, style="italic")

# Legend
legend_items = [
    (C_UPDATE, "Update Gate (Z_t) — kiểm soát giữ hay thay thế"),
    (C_RESET, "Reset Gate (R_t) — kiểm soát quên hay nhớ"),
]
for i, (c, txt) in enumerate(legend_items):
    ax.add_patch(FancyBboxPatch((0.3, 5.3 + i * 0.55), 0.25, 0.3,
                                boxstyle="round,pad=0.02",
                                facecolor="none", edgecolor=c, lw=2))
    ax.text(0.65, 5.47 + i * 0.55, txt, fontsize=9, color=C_DIM, va="center")

plt.tight_layout()
fig.savefig(f"{OUT}/gru-1.png", dpi=150, bbox_inches="tight",
            facecolor=C_BG)
plt.close()
print(f"✓ gru-1.png")

# ═══════════════════════════════════════════════════════════════════════════════
# HÌNH 2: Candidate Hidden State
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)
ax.set_xlim(0, 13)
ax.set_ylim(0, 7)
ax.axis("off")
ax.set_title("Fig 10.2.2 — Candidate Hidden State trong GRU",
            fontsize=13, color=C_TEXT, fontweight="bold", pad=14)

# Input nodes
draw_box(ax, 1.0, 3.5, 1.5, 0.9, "X_t", fc="#21262d", ec=C_BORDER)
draw_box(ax, 1.0, 1.8, 1.5, 0.9, "H_{t-1}", fc="#21262d", ec=C_BORDER)

# Reset Gate
draw_box(ax, 3.8, 1.8, 1.6, 0.9, "R_t", "(0,1)^h",
         fc="#3d1f00", ec=C_RESET, tc=C_RESET)

# Multiply R * H
draw_box(ax, 5.8, 1.8, 0.7, 0.6, "⊙", "",
         fc=C_PANEL, ec=C_DIM, tc=C_TEXT, fontsize=14)
draw_box(ax, 5.8, 3.5, 0.7, 0.6, "X_t", "",
         fc=C_PANEL, ec=C_DIM, tc=C_TEXT, fontsize=9)

# Concatenate
draw_box(ax, 6.95, 2.65, 0.05, 1.7, "", fc="none", ec="none")

# Candidate tanh
draw_box(ax, 8.7, 2.65, 2.0, 1.2,
         "Candidate", "H̃_t = tanh([X_t, R⊙H]W)",
         fc="#1a1425", ec=C_CAND, tc=C_CAND)

# Output
arrow(ax, 9.7, 2.65, 11.2, 2.65, color=C_CAND, label="H̃_t")
draw_box(ax, 11.7, 2.65, 1.3, 0.9, "H̃_t", "(−1,1)^h",
         fc="#1a1425", ec=C_CAND, tc=C_CAND)

# Flow annotations
arrow(ax, 1.75, 3.5, 1.75, 4.2, color=C_DIM)
arrow(ax, 1.75, 4.2, 8.7, 4.2, color=C_DIM)
arrow(ax, 8.7, 4.2, 8.7, 3.25, color=C_DIM)

arrow(ax, 1.75, 1.8, 3.1, 1.8, color=C_RESET)
arrow(ax, 5.1, 1.8, 5.5, 1.8, color=C_DIM)
arrow(ax, 5.8, 1.5, 5.8, 1.5, color=C_DIM)

# Key insight
ax.text(6.5, 0.6,
        "Khi R_t ≈ 0: H̃_t phụ thuộc chủ yếu vào X_t (reset ký ức cũ) | "
        "Khi R_t ≈ 1: H̃_t khôi phục RNN thường",
        ha="center", va="bottom", fontsize=8.5, color=C_DIM, style="italic")

legend_items = [
    (C_RESET, "Reset Gate R_t điều khiển quên bao nhiêu ký ức cũ"),
    (C_CAND, "Candidate H̃_t = tanh(...) là nội dung mới tiềm năng"),
]
for i, (c, txt) in enumerate(legend_items):
    ax.add_patch(FancyBboxPatch((0.3, 5.3 + i * 0.55), 0.25, 0.3,
                                boxstyle="round,pad=0.02",
                                facecolor="none", edgecolor=c, lw=2))
    ax.text(0.65, 5.47 + i * 0.55, txt, fontsize=9, color=C_DIM, va="center")

plt.tight_layout()
fig.savefig(f"{OUT}/gru-2.png", dpi=150, bbox_inches="tight",
            facecolor=C_BG)
plt.close()
print(f"✓ gru-2.png")

# ═══════════════════════════════════════════════════════════════════════════════
# HÌNH 3: Hidden State Update (Update Gate)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)
ax.set_xlim(0, 13)
ax.set_ylim(0, 7)
ax.axis("off")
ax.set_title("Fig 10.2.3 — Hidden State Update trong GRU",
            fontsize=13, color=C_TEXT, fontweight="bold", pad=14)

# Left column: old state
draw_box(ax, 1.5, 4.5, 1.8, 1.0, "H_{t-1}", "old state",
         fc="#0d2d1a", ec=C_STATE, tc=C_STATE)

# Update gate
draw_box(ax, 1.5, 2.0, 1.6, 0.9, "Z_t", "(0,1)^h",
         fc="#0d2d4a", ec=C_UPDATE, tc=C_UPDATE)

# Candidate
draw_box(ax, 1.5, 0.5, 1.8, 0.9, "H̃_t", "candidate",
         fc="#1a1425", ec=C_CAND, tc=C_CAND)

# Multiply gates with states
# Z_t * H_{t-1}
draw_box(ax, 4.2, 4.5, 0.6, 0.6, "⊙", "",
         fc=C_PANEL, ec=C_DIM, tc=C_TEXT, fontsize=14)
# (1-Z_t) * H̃_t
draw_box(ax, 4.2, 0.5, 0.6, 0.6, "⊙", "",
         fc=C_PANEL, ec=C_DIM, tc=C_TEXT, fontsize=14)

# 1 - Z_t
draw_box(ax, 4.2, 2.0, 1.0, 0.8, "1 − Z_t", "",
         fc="#0d2d4a", ec=C_UPDATE, tc=C_UPDATE, fontsize=9)

# Arrows to multiply nodes
arrow(ax, 2.4, 4.5, 3.9, 4.5, color=C_STATE)
arrow(ax, 2.4, 2.0, 3.0, 2.0, color=C_UPDATE)
arrow(ax, 4.2, 1.7, 4.2, 0.8, color=C_DIM)

arrow(ax, 2.4, 0.5, 3.9, 0.5, color=C_CAND)

# Arrow 1-Z_t to candidate multiply
arrow(ax, 4.9, 2.0, 4.2, 0.8, color=C_UPDATE)

# Intermediate results
draw_box(ax, 6.0, 4.5, 1.2, 0.8, "Z_t ⊙ H", "",
         fc="#0d2d1a", ec=C_STATE, tc=C_STATE, fontsize=9)
draw_box(ax, 6.0, 0.5, 1.2, 0.8, "(1−Z)⊙H̃", "",
         fc="#1a1425", ec=C_CAND, tc=C_CAND, fontsize=8)

arrow(ax, 4.9, 4.5, 5.4, 4.5, color=C_STATE)
arrow(ax, 4.9, 0.5, 5.4, 0.5, color=C_CAND)

# Addition
draw_box(ax, 7.5, 2.5, 0.6, 0.6, "+", "",
         fc=C_PANEL, ec=C_DIM, tc=C_TEXT, fontsize=14)
arrow(ax, 6.6, 4.5, 7.2, 3.0, color=C_DIM, lw=1)
arrow(ax, 6.6, 0.5, 7.2, 2.0, color=C_DIM, lw=1)

# Output
arrow(ax, 7.8, 2.5, 9.5, 2.5, color=C_STATE, label="H_t")
draw_box(ax, 10.2, 2.5, 1.6, 1.0, "H_t", "new hidden state",
         fc="#0d2d1a", ec=C_STATE, tc=C_STATE)

# Key insight box
ax.add_patch(FancyBboxPatch((0.3, 5.8), 11.5, 0.9,
                             boxstyle="round,pad=0.1",
                             facecolor="#161b22", edgecolor=C_BORDER))
ax.text(6.05, 6.25,
        "H_t = Z_t ⊙ H_{t-1}  +  (1 − Z_t) ⊙ H̃_t",
        ha="center", va="center", fontsize=12, color=C_TEXT,
        fontweight="bold")
ax.text(6.05, 5.95,
        "Z_t → 1  : giữ nguyên H_{t-1}  (bỏ qua X_t)  |  "
        "Z_t → 0  : thay hoàn toàn bằng H̃_t",
        ha="center", va="center", fontsize=9, color=C_DIM)

legend_items = [
    (C_UPDATE, "Update Gate Z_t: cổng quyết định giữ bao nhiêu của trạng thái cũ"),
    (C_STATE,  "Trạng thái mới H_t là tổ hợp lồi của H_{t-1} và H̃_t"),
]
for i, (c, txt) in enumerate(legend_items):
    ax.add_patch(FancyBboxPatch((0.3, 5.1 + i * 0.55), 0.25, 0.3,
                                boxstyle="round,pad=0.02",
                                facecolor="none", edgecolor=c, lw=2))
    ax.text(0.65, 5.27 + i * 0.55, txt, fontsize=9, color=C_DIM, va="center")

plt.tight_layout()
fig.savefig(f"{OUT}/gru-3.png", dpi=150, bbox_inches="tight",
            facecolor=C_BG)
plt.close()
print(f"✓ gru-3.png")

# ═══════════════════════════════════════════════════════════════════════════════
# HÌNH 4: Kiến trúc hoàn chỉnh GRU
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(15, 9))
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)
ax.set_xlim(0, 15)
ax.set_ylim(0, 9)
ax.axis("off")
ax.set_title("Kiến trúc hoàn chỉnh GRU — từ đầu vào đến Hidden State",
            fontsize=13, color=C_TEXT, fontweight="bold", pad=14)

# ── INPUT LAYER ──
draw_box(ax, 1.0, 5.0, 1.3, 0.9, "X_t", fc="#21262d", ec=C_BORDER)
draw_box(ax, 1.0, 3.0, 1.3, 0.9, "H_{t-1}", fc="#21262d", ec=C_BORDER)

# ── CONCAT ──
draw_box(ax, 2.2, 4.0, 0.05, 2.0, "", fc="none", ec="none")

# ── UPDATE GATE ──
draw_box(ax, 3.5, 5.5, 1.5, 0.85, "σ", "Update Gate",
         fc="#0d2d4a", ec=C_UPDATE, tc=C_UPDATE, fontsize=10)

# ── RESET GATE ──
draw_box(ax, 3.5, 2.5, 1.5, 0.85, "σ", "Reset Gate",
         fc="#3d1f00", ec=C_RESET, tc=C_RESET, fontsize=10)

# ── ARROWS TO GATES ──
arrow(ax, 1.65, 5.0, 2.2, 5.5, color=C_UPDATE)
arrow(ax, 2.2, 4.0, 2.2, 5.5, color=C_UPDATE)
arrow(ax, 1.65, 3.0, 2.2, 2.5, color=C_RESET)
arrow(ax, 2.2, 4.0, 2.2, 2.5, color=C_RESET)

# ── UPDATE GATE OUTPUT ──
arrow(ax, 4.25, 5.5, 5.5, 5.5, color=C_UPDATE, label="Z_t")
draw_box(ax, 6.0, 5.5, 1.0, 0.75, "Z_t", "(0,1)^h",
         fc="#0d2d4a", ec=C_UPDATE, tc=C_UPDATE, fontsize=9)

# ── 1 - Z ──
draw_box(ax, 6.0, 4.0, 0.9, 0.65, "1 − Z_t", "",
         fc="#0d2d4a", ec=C_UPDATE, tc=C_UPDATE, fontsize=8)
arrow(ax, 6.5, 5.5, 6.0, 4.35, color=C_UPDATE)

# ── RESET GATE OUTPUT ──
arrow(ax, 4.25, 2.5, 5.5, 2.5, color=C_RESET, label="R_t")
draw_box(ax, 6.0, 2.5, 1.0, 0.75, "R_t", "(0,1)^h",
         fc="#3d1f00", ec=C_RESET, tc=C_RESET, fontsize=9)

# ── H̃_t CALCULATION ──
draw_box(ax, 8.5, 3.75, 0.55, 0.55, "⊙", "",
         fc=C_PANEL, ec=C_DIM, tc=C_TEXT, fontsize=13)
draw_box(ax, 8.5, 5.0, 0.55, 0.55, "X_t", "",
         fc=C_PANEL, ec=C_DIM, tc=C_TEXT, fontsize=8)

arrow(ax, 6.0, 2.5, 8.2, 3.75, color=C_RESET)
arrow(ax, 6.5, 5.5, 7.0, 5.0, color=C_DIM)
arrow(ax, 7.0, 5.0, 8.2, 5.0, color=C_DIM)
arrow(ax, 8.5, 4.05, 8.5, 4.55, color=C_DIM)

draw_box(ax, 10.2, 4.0, 1.7, 1.2,
         "tanh", "Candidate\nH̃_t",
         fc="#1a1425", ec=C_CAND, tc=C_CAND, fontsize=10)
arrow(ax, 9.3, 4.0, 9.35, 4.0, color=C_CAND)
arrow(ax, 8.5, 3.47, 9.3, 3.7, color=C_DIM)

# ── (1-Z) * H̃ ──
draw_box(ax, 12.0, 4.0, 0.6, 0.6, "⊙", "",
         fc=C_PANEL, ec=C_DIM, tc=C_TEXT, fontsize=13)
arrow(ax, 11.05, 4.0, 11.7, 4.0, color=C_CAND)
arrow(ax, 6.45, 4.0, 12.0, 4.35, color=C_UPDATE)

# ── Z * H_{t-1} ──
draw_box(ax, 10.0, 6.0, 0.6, 0.6, "⊙", "",
         fc=C_PANEL, ec=C_DIM, tc=C_TEXT, fontsize=13)
arrow(ax, 6.5, 5.5, 9.7, 6.0, color=C_UPDATE)
arrow(ax, 1.65, 3.0, 9.7, 6.0, color=C_STATE, lw=1)

# ── ADD ──
draw_box(ax, 13.0, 5.0, 0.55, 0.55, "+", "",
         fc=C_PANEL, ec=C_DIM, tc=C_TEXT, fontsize=14)
arrow(ax, 12.3, 6.0, 12.72, 5.35, color=C_DIM)
arrow(ax, 12.6, 4.0, 12.72, 4.65, color=C_DIM)

# ── OUTPUT ──
arrow(ax, 13.28, 5.0, 14.0, 5.0, color=C_STATE, label="H_t")
draw_box(ax, 14.3, 5.0, 1.1, 0.9, "H_t", "hidden state",
         fc="#0d2d1a", ec=C_STATE, tc=C_STATE, fontsize=9)

# ── SUMMARY BOX ──
ax.add_patch(FancyBboxPatch((0.2, 7.5), 14.5, 1.2,
                             boxstyle="round,pad=0.1",
                             facecolor="#161b22", edgecolor=C_BORDER))
ax.text(7.45, 8.05,
        "Z_t = σ(X_t W_xz + H_{t-1} W_hz)    "
        "R_t = σ(X_t W_xr + H_{t-1} W_hr)",
        ha="center", va="center", fontsize=9.5, color=C_TEXT)
ax.text(7.45, 7.7,
        "H̃_t = tanh(X_t W_xh + (R_t ⊙ H_{t-1}) W_hh)    "
        "H_t = Z_t ⊙ H_{t-1} + (1−Z_t) ⊙ H̃_t",
        ha="center", va="center", fontsize=9.5, color=C_TEXT,
        fontweight="bold")

# Legend
for i, (c, txt) in enumerate([
    (C_UPDATE, "Update Gate Z_t"),
    (C_RESET, "Reset Gate R_t"),
    (C_CAND, "Candidate H̃_t"),
    (C_STATE, "Hidden State H_t"),
]):
    bx = 0.2 + i * 3.6
    ax.add_patch(FancyBboxPatch((bx, 6.9), 0.25, 0.25,
                                boxstyle="round,pad=0.02",
                                facecolor="none", edgecolor=c, lw=2))
    ax.text(bx + 0.35, 7.03, txt, fontsize=8.5, color=C_DIM, va="center")

plt.tight_layout()
fig.savefig(f"{OUT}/gru-4.png", dpi=150, bbox_inches="tight",
            facecolor=C_BG)
plt.close()
print(f"✓ gru-4.png")

# ═══════════════════════════════════════════════════════════════════════════════
# HÌNH 5: So sánh RNN → GRU → LSTM
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 7))
fig.patch.set_facecolor(C_BG)
for ax_i in axes:
    ax_i.set_facecolor(C_BG)
    ax_i.set_xlim(0, 10)
    ax_i.set_ylim(0, 7)
    ax_i.axis("off")

# ── Panel A: Vanilla RNN ──
ax = axes[0]
ax.set_title("Vanilla RNN", fontsize=13, color=C_TEXT,
             fontweight="bold", pad=10)
draw_box(ax, 5, 5.0, 3.2, 1.1, "H_t = tanh(X_t W_xh\n+ H_{t-1} W_hh + b)",
         "", fc="#21262d", ec=C_NODE, tc=C_NODE, fontsize=8.5)
draw_box(ax, 2.0, 5.0, 1.5, 0.85, "X_t", fc="#21262d", ec=C_BORDER)
draw_box(ax, 2.0, 3.2, 1.5, 0.85, "H_{t-1}", fc="#21262d", ec=C_STATE)
arrow(ax, 3.25, 5.0, 3.4, 5.0, color=C_NODE)
arrow(ax, 3.25, 3.2, 3.4, 5.3, color=C_STATE)
arrow(ax, 6.6, 5.0, 7.5, 5.0, color=C_NODE)
draw_box(ax, 8.0, 5.0, 1.5, 0.85, "H_t", fc="#21262d", ec=C_STATE)
arrow(ax, 5, 4.45, 5, 4.0, color=C_STATE, lw=1.5, label="recurrence")
ax.annotate("", xy=(3.4, 3.62), xytext=(5, 4.0),
            arrowprops=dict(arrowstyle="->", color=C_STATE,
                            lw=1.5, connectionstyle="arc3,rad=0"))
ax.text(4.1, 3.5, "H_{t-1}", fontsize=8, color=C_STATE, ha="center")
ax.text(5, 1.5,
        "1 trạng thái\n1 tầng FC + tanh\nGradient: W_{hh}^k → biến mất",
        ha="center", va="center", fontsize=8.5, color=C_DIM,
        style="italic")

# ── Panel B: GRU ──
ax = axes[1]
ax.set_title("GRU", fontsize=13, color=C_UPDATE,
             fontweight="bold", pad=10)
draw_box(ax, 1.5, 5.2, 1.3, 0.8, "X_t", fc="#21262d", ec=C_BORDER)
draw_box(ax, 1.5, 3.0, 1.3, 0.8, "H_{t-1}", fc="#21262d", ec=C_STATE)
draw_box(ax, 4.2, 5.2, 1.3, 0.8, "Z_t", "σ",
         fc="#0d2d4a", ec=C_UPDATE, tc=C_UPDATE, fontsize=11)
draw_box(ax, 4.2, 3.0, 1.3, 0.8, "R_t", "σ",
         fc="#3d1f00", ec=C_RESET, tc=C_RESET, fontsize=11)
draw_box(ax, 6.5, 4.0, 1.5, 1.0, "H̃_t", "tanh",
         fc="#1a1425", ec=C_CAND, tc=C_CAND)
draw_box(ax, 8.5, 4.0, 1.3, 0.85, "H_t", fc="#0d2d1a", ec=C_STATE)
arrow(ax, 2.15, 5.2, 3.55, 5.2, color=C_UPDATE)
arrow(ax, 2.15, 3.0, 3.55, 3.0, color=C_UPDATE)
arrow(ax, 2.15, 3.0, 3.55, 3.0, color=C_RESET)
arrow(ax, 4.85, 4.0, 5.75, 4.0, color=C_CAND)
arrow(ax, 7.25, 4.0, 7.85, 4.0, color=C_STATE)
ax.annotate("", xy=(5.75, 4.6), xytext=(4.2, 4.8),
            arrowprops=dict(arrowstyle="->", color=C_STATE, lw=1))
ax.annotate("", xy=(5.75, 3.4), xytext=(4.2, 3.2),
            arrowprops=dict(arrowstyle="->", color=C_CAND, lw=1))
ax.text(5.0, 5.1, "Z_t⊙H", fontsize=7.5, color=C_STATE, ha="center")
ax.text(5.0, 2.9, "(1−Z)⊙H̃", fontsize=7.5, color=C_CAND, ha="center")
ax.text(5.0, 1.5,
        "2 cổng + 1 candidate\nCông thức cốt lõi:\n"
        "H_t = Z ⊙ H + (1−Z) ⊙ H̃",
        ha="center", va="center", fontsize=8.5, color=C_DIM,
        style="italic")

# ── Panel C: LSTM ──
ax = axes[2]
ax.set_title("LSTM", fontsize=13, color="#ff7b72",
             fontweight="bold", pad=10)
draw_box(ax, 1.5, 5.2, 1.3, 0.8, "X_t", fc="#21262d", ec=C_BORDER)
draw_box(ax, 1.5, 3.0, 1.3, 0.8, "H_{t-1}", fc="#21262d", ec=C_STATE)
draw_box(ax, 3.8, 5.8, 1.0, 0.7, "I_t", "σ",
         fc="#0d2d4a", ec=C_UPDATE, tc=C_UPDATE, fontsize=10)
draw_box(ax, 3.8, 5.0, 1.0, 0.7, "F_t", "σ",
         fc="#3d1f00", ec=C_RESET, tc=C_RESET, fontsize=10)
draw_box(ax, 3.8, 3.2, 1.0, 0.7, "O_t", "σ",
         fc="#2d4a0d", ec="#7ee787", tc="#7ee787", fontsize=10)
draw_box(ax, 5.5, 4.5, 1.3, 0.8, "C̃_t", "tanh",
         fc="#1a1425", ec=C_CAND, tc=C_CAND)
draw_box(ax, 7.0, 4.5, 1.3, 0.8, "C_t", fc="#1a0d2d", ec="#d2a8ff")
draw_box(ax, 8.8, 4.5, 1.3, 0.8, "H_t", fc="#0d2d1a", ec=C_STATE)
for src, dst, c in [
    (2.15, 3.8, C_UPDATE), (2.15, 3.8, C_RESET), (2.15, 3.8, "#7ee787"),
]:
    arrow(ax, 2.15, 5.2, 3.3, 5.5, color=C_UPDATE)
    arrow(ax, 2.15, 3.0, 3.3, 3.2, color=C_RESET)
arrow(ax, 3.8, 3.65, 5.5, 4.1, color=C_DIM)
arrow(ax, 5.5, 4.9, 5.5, 5.35, color=C_UPDATE)
arrow(ax, 6.15, 4.5, 6.35, 4.5, color=C_CAND)
arrow(ax, 3.8, 3.55, 5.5, 3.85, color=C_RESET)
arrow(ax, 7.15, 4.5, 8.15, 4.5, color="#d2a8ff")
arrow(ax, 3.8, 3.55, 6.5, 4.15, color=C_DIM)
arrow(ax, 8.15, 4.5, 8.8, 4.5, color=C_STATE)
arrow(ax, 7.65, 4.1, 8.15, 4.35, color="#7ee787")
ax.text(5.0, 1.5,
        "3 cổng + candidate\n+ Cell State C_t riêng\n"
        "C_t = F⊙C + I⊙C̃  |  H_t = O⊙tanh(C)",
        ha="center", va="center", fontsize=8.5, color=C_DIM,
        style="italic")

# ── Title row ──
fig.suptitle("Từ Vanilla RNN đến GRU đến LSTM",
             fontsize=14, color=C_TEXT, fontweight="bold", y=1.01)

# ── Bottom comparison table ──
table_data = [
    ["Đặc điểm",    "RNN",    "GRU",    "LSTM"],
    ["Số cổng",      "0",      "2",      "3"],
    ["Cell State",   "Không",  "Không",  "Có"],
    ["Trạng thái",  "H_t",     "H_t",    "H_t + C_t"],
    ["Tham số",     "dh+h²+h", "3(dh+h²+h)", "4(dh+h²+h)"],
    ["Biến mất\ngradient", "Nghiêm trọng", "Có", "Ít hơn"],
    ["Tốc độ",      "Nhanh nhất", "Trung bình", "Chậm nhất"],
]
for i, row in enumerate(table_data):
    for j, cell in enumerate(row):
        fc = "#161b22" if i == 0 else C_BG
        ec = C_BORDER if i == 0 else C_BORDER
        tc = C_TEXT if i == 0 else (C_UPDATE if j == 2 and i > 0 else
                                     C_NODE if j == 1 and i > 0 else C_DIM)
        fw = "bold" if i == 0 else "normal"
        ax = fig.add_axes([0.03 + j * 0.245, 0.0, 0.24, 0.22])
        ax.set_facecolor(fc)
        ax.add_patch(FancyBboxPatch((0, 0), 1, 1,
                                    boxstyle="round,pad=0.02",
                                    facecolor=fc, edgecolor=ec, lw=1))
        ax.text(0.5, 0.5, cell, ha="center", va="center",
                fontsize=8 if i > 0 else 9, color=tc,
                fontweight=fw, transform=ax.transAxes)
        ax.axis("off")

plt.tight_layout(rect=[0, 0.23, 1, 0.98])
fig.savefig(f"{OUT}/gru-5.png", dpi=150, bbox_inches="tight",
            facecolor=C_BG)
plt.close()
print(f"✓ gru-5.png")
print(f"\n✓ All 5 images saved to {OUT}/")
