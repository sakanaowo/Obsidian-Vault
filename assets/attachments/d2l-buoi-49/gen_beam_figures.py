#!/usr/bin/env python3
"""Generate SVG illustrations for Buổi 49 — 10.8 Beam Search"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = "/home/sakana/Code/Obsidian-Vault/assets/attachments/d2l-buoi-49"

# ─── Palette ────────────────────────────────────────────────────────────────
BLUE_DARK  = "#1565C0"
BLUE_LIGHT = "#BBDEFB"
GREEN_DARK = "#2E7D32"
GREEN_LIGHT= "#C8E6C9"
ORANGE     = "#E65100"
ORANGE_LT  = "#FFCCBC"
RED        = "#C62828"
RED_LIGHT  = "#FFCDD2"
PURPLE     = "#6A1B9A"
PURPLE_LT  = "#E1BEE7"
GREY       = "#757575"
GOLD       = "#F9A825"
GOLD_LT    = "#FFF9C4"
TEXT       = "#212121"

# ─────────────────────────────────────────────────────────────────────────────
# 1. greedy-search.svg — Greedy search (two variants)
# ─────────────────────────────────────────────────────────────────────────────
def make_greedy_search():
    """D2L Fig 10.8.1 and 10.8.2 side by side"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_alpha(0)
    for ax in (ax1, ax2):
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 6)
        ax.axis('off')

    # ── Left: Greedy = Good path (0.5 × 0.4 × 0.4 × 0.6 = 0.048) ──────────
    ax1.set_title("Greedy Search — Đi con đường tốt nhất mỗi bước\n"
                  "Output: A → B → C → <eos>  (P = 0.048)",
                  fontsize=11, fontweight='bold', color=TEXT, pad=10)

    tokens_l = ['A', 'B', 'C', '<eos>']
    x_l = [2, 5, 8, 11]
    # Probabilities at each step
    probs_l = [
        [0.5, 0.3, 0.15, 0.05],
        [0.4, 0.3, 0.2, 0.1],
        [0.4, 0.3, 0.2, 0.1],
        [0.6, 0.2, 0.1, 0.1],
    ]
    chosen_l = [0, 0, 0, 0]  # greedy always picks index 0

    for i, (x, tok) in enumerate(zip(x_l, tokens_l)):
        for j, (p, label) in enumerate(zip(probs_l[i], ['A', 'B', 'C', '<eos>'])):
            y = 4.5 - j * 0.65
            col = GREEN_DARK if j == chosen_l[i] else GREY
            bg  = GREEN_LIGHT if j == chosen_l[i] else "#F5F5F5"
            ax1.add_patch(FancyBboxPatch((x-0.45, y-0.2), 0.9, 0.4,
                                         boxstyle="round,pad=0.03",
                                         linewidth=1.5, edgecolor=col,
                                         facecolor=bg, zorder=2))
            ax1.text(x, y, f'{label}', ha='center', va='center',
                     fontsize=10, fontweight='bold', color=col, zorder=3)
            ax1.text(x+0.6, y, f'{p:.2f}', ha='left', va='center',
                     fontsize=8.5, color=col if j == chosen_l[i] else GREY, zorder=3)

    # Arrows for chosen path
    for i in range(len(x_l)-1):
        ax1.annotate("", xy=(x_l[i+1]-0.5, 4.15),
                     xytext=(x_l[i]+0.5, 4.15),
                     arrowprops=dict(arrowstyle="->", color=GREEN_DARK, lw=2.5))

    # Step labels
    for x, label in zip(x_l, ['Step 1', 'Step 2', 'Step 3', 'Step 4']):
        ax1.text(x, 5.4, label, ha='center', va='center',
                 fontsize=8, color=GREY, style='italic')

    ax1.text(6.5, 0.4, "Greedy luôn chọn token có P cao nhất → nhưng tích xác suất = 0.048",
             ha='center', fontsize=9, color=RED,
             bbox=dict(boxstyle="round,pad=0.3", facecolor=RED_LIGHT,
                       edgecolor=RED, linewidth=1.5))

    # ── Right: Non-greedy = Better total (0.5 × 0.3 × 0.6 × 0.6 = 0.054) ───
    ax2.set_title("Non-Greedy — Đổi hướng ở bước 2\n"
                  "Output: A → C → B → <eos>  (P = 0.054 > 0.048)",
                  fontsize=11, fontweight='bold', color=TEXT, pad=10)

    tokens_r = ['A', 'C', 'B', '<eos>']
    x_r = [2, 5, 8, 11]
    probs_r = [
        [0.5, 0.3, 0.15, 0.05],
        [0.2, 0.3, 0.4, 0.1],   # C=0.4 second highest at step 2
        [0.2, 0.3, 0.6, 0.1],   # B=0.6 highest at step 3
        [0.2, 0.2, 0.6, 0.1],
    ]
    chosen_r = [0, 2, 2, 2]  # A, then C (index 2), then B (index 2)

    for i, (x, tok) in enumerate(zip(x_r, tokens_r)):
        for j, (p, label) in enumerate(zip(probs_r[i], ['A', 'B', 'C', '<eos>'])):
            y = 4.5 - j * 0.65
            col = ORANGE if j == chosen_r[i] else GREY
            bg  = ORANGE_LT if j == chosen_r[i] else "#F5F5F5"
            ax2.add_patch(FancyBboxPatch((x-0.45, y-0.2), 0.9, 0.4,
                                         boxstyle="round,pad=0.03",
                                         linewidth=1.5, edgecolor=col,
                                         facecolor=bg, zorder=2))
            ax2.text(x, y, f'{label}', ha='center', va='center',
                     fontsize=10, fontweight='bold', color=col, zorder=3)
            ax2.text(x+0.6, y, f'{p:.2f}', ha='left', va='center',
                     fontsize=8.5, color=col if j == chosen_r[i] else GREY, zorder=3)

    for i in range(len(x_r)-1):
        ax2.annotate("", xy=(x_r[i+1]-0.5, 4.15),
                     xytext=(x_r[i]+0.5, 4.15),
                     arrowprops=dict(arrowstyle="->", color=ORANGE, lw=2.5))

    for x, label in zip(x_r, ['Step 1', 'Step 2', 'Step 3', 'Step 4']):
        ax2.text(x, 5.4, label, ha='center', va='center',
                 fontsize=8, color=GREY, style='italic')

    ax2.text(6.5, 0.4,
             "Chọn C (P=0.3) ở bước 2 → dẫn đến tích P = 0.054 > 0.048 ✗ Greedy!",
             ha='center', fontsize=9, color="#1B5E20",
             bbox=dict(boxstyle="round,pad=0.3", facecolor=GREEN_LIGHT,
                       edgecolor=GREEN_DARK, linewidth=1.5))

    plt.tight_layout(pad=0.5)
    plt.savefig(f"{OUT}/greedy-search.svg", dpi=150, bbox_inches='tight',
                facecolor='white', format='svg')
    plt.close()
    print(f"✓ {OUT}/greedy-search.svg")


# ─────────────────────────────────────────────────────────────────────────────
# 2. exhaustive-search.svg — Why exhaustive is impossible
# ─────────────────────────────────────────────────────────────────────────────
def make_exhaustive_search():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    fig.patch.set_alpha(0)

    ax.text(6, 5.5, "Exhaustive Search — Tìm kiếm toàn bộ không gian",
            ha='center', va='center', fontsize=13, fontweight='bold', color=TEXT)

    # Root
    ax.add_patch(FancyBboxPatch((5.3, 4.5), 1.4, 0.55,
                                 boxstyle="round,pad=0.1",
                                 linewidth=2, edgecolor=GREY,
                                 facecolor="#EEEEEE", zorder=2))
    ax.text(6, 4.775, "<start>", ha='center', va='center',
            fontsize=10, fontweight='bold', color=GREY, zorder=3)

    # Level 1
    x1 = [3, 6, 9]
    for x in x1:
        ax.annotate("", xy=(x, 4.5),
                    xytext=(6, 4.5),
                    arrowprops=dict(arrowstyle="->", color=GREY, lw=1.2))
        ax.add_patch(FancyBboxPatch((x-0.5, 4.0), 1.0, 0.45,
                                     boxstyle="round,pad=0.05",
                                     linewidth=1.2, edgecolor=BLUE_DARK,
                                     facecolor=BLUE_LIGHT, zorder=2))
        ax.text(x, 4.225, ["A", "B", "C"][x1.index(x)],
                ha='center', va='center', fontsize=9, fontweight='bold',
                color=BLUE_DARK, zorder=3)

    # Level 2 (branch from each level 1)
    x2_vals = [1.5, 2.5, 4.5, 5.5, 7.5, 8.5, 10.5, 11.5]
    labels2 = ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
    parents2 = [3, 3, 6, 6, 6, 6, 9, 9]
    for x2, lbl, par in zip(x2_vals, labels2, parents2):
        ax.annotate("", xy=(x2, 3.5),
                    xytext=(par, 4.0),
                    arrowprops=dict(arrowstyle="->", color=GREY, lw=0.8,
                                    alpha=0.6))
        ax.add_patch(FancyBboxPatch((x2-0.3, 3.1), 0.6, 0.38,
                                     boxstyle="round,pad=0.03",
                                     linewidth=1, edgecolor=GREY,
                                     facecolor="#F5F5F5", zorder=2))
        ax.text(x2, 3.29, lbl, ha='center', va='center',
                fontsize=8, color=GREY, zorder=3)

    # Level 3 (just a few shown)
    x3_vals = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5]
    for xi, x3 in enumerate(x3_vals):
        ax.annotate("", xy=(x3, 2.55),
                    xytext=(x3, 3.1),
                    arrowprops=dict(arrowstyle="->", color=GREY, lw=0.5,
                                    alpha=0.3))
        ax.add_patch(FancyBboxPatch((x3-0.2, 2.2), 0.4, 0.33,
                                     boxstyle="round,pad=0.02",
                                     linewidth=0.8, edgecolor=GREY,
                                     facecolor="#EEEEEE", zorder=2))
        ax.text(x3, 2.365, 'X', ha='center', va='center',
                fontsize=7, color=GREY, zorder=3)

    # Dots at bottom
    for x in np.linspace(0.3, 11.7, 20):
        ax.plot(x, 1.85, 'k.', markersize=2, alpha=0.3)

    ax.text(6, 1.6,
            "× ··· × ··· × ··· × ··· (Lớp tiếp theo: $|\mathcal{Y}|$ nhánh mới × mỗi nhánh cũ)",
            ha='center', va='center', fontsize=8, color=GREY, style='italic')

    # Complexity comparison box
    box_text = (
        "Chi phí: $\\mathcal{O}(|\\mathcal{Y}|^{T'})$\n"
        "Ví dụ: $|\\mathcal{Y}|=10{,}000$, $T'=10$\n"
        "→ $10{,}000^{10} = 10^{40}$ sequences\n"
        "→ Không thể tính toán được!"
    )
    ax.text(6, 0.85, box_text, ha='center', va='center', fontsize=9,
            color=RED,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=RED_LIGHT,
                      edgecolor=RED, linewidth=2),
            multialignment='center')

    plt.tight_layout(pad=0.3)
    plt.savefig(f"{OUT}/exhaustive-search.svg", dpi=150, bbox_inches='tight',
                facecolor='white', format='svg')
    plt.close()
    print(f"✓ {OUT}/exhaustive-search.svg")


# ─────────────────────────────────────────────────────────────────────────────
# 3. beam-search-comparison.svg — Greedy vs Beam vs Exhaustive
# ─────────────────────────────────────────────────────────────────────────────
def make_comparison():
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.axis('off')
    fig.patch.set_alpha(0)

    ax.text(6.5, 4.6, "Ba Chiến Lược Giải Mã Sequence",
            ha='center', va='center', fontsize=14, fontweight='bold', color=TEXT)

    strategies = [
        ("Greedy Search\n(k = 1)", "O(V·T')", "Rẻ nhất", "Tối ưu cục bộ\n→ Không tối ưu toàn cục",
         "#EF5350", "#FFCDD2"),
        ("Beam Search\n(k beams)", "O(k·V·T')", "Cân bằng", "Tìm kiếm k đường\n→ Đủ tốt trong thực tế",
         GOLD, GOLD_LT),
        ("Exhaustive Search", "O(V^{T'})", "Đắt nhất", "Tất cả đường đi\n→ Không khả thi",
         "#78909C", "#ECEFF1"),
    ]

    for i, (name, cost, label, desc, ec, fc) in enumerate(strategies):
        x = 1.5 + i * 4.0
        # Card background
        ax.add_patch(FancyBboxPatch((x-1.6, 0.4), 3.2, 3.6,
                                     boxstyle="round,pad=0.2",
                                     linewidth=2, edgecolor=ec,
                                     facecolor=fc, zorder=0, alpha=0.4))
        # Title
        ax.text(x, 3.7, name, ha='center', va='center',
                fontsize=10, fontweight='bold', color=TEXT, zorder=2)
        # Cost
        ax.text(x, 3.1, f"Chi phí:\n{cost}", ha='center', va='center',
                fontsize=8.5, color=TEXT, style='italic', zorder=2,
                multialignment='center')
        # Quality label
        ax.text(x, 2.5, f"⚡ {label}", ha='center', va='center',
                fontsize=9.5, fontweight='bold', color=ec, zorder=2)
        # Description
        ax.text(x, 1.8, desc, ha='center', va='center',
                fontsize=8, color="#424242", zorder=2,
                multialignment='center')
        # Draw beam paths for middle
        if i == 1:
            for bx in [x-0.5, x, x+0.5]:
                ax.annotate("", xy=(bx, 1.5),
                            xytext=(bx, 2.0),
                            arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.5))

    # Spectrum arrow
    ax.annotate("", xy=(12.2, 2.0), xytext=(0.8, 2.0),
                arrowprops=dict(arrowstyle="<->", color=GREY, lw=2))
    ax.text(6.5, 1.6, "← Chi phí tăng dần, Chất lượng tăng dần →",
            ha='center', fontsize=9, color=GREY, style='italic')

    plt.tight_layout(pad=0.3)
    plt.savefig(f"{OUT}/beam-search-comparison.svg", dpi=150, bbox_inches='tight',
                facecolor='white', format='svg')
    plt.close()
    print(f"✓ {OUT}/beam-search-comparison.svg")


if __name__ == "__main__":
    make_greedy_search()
    make_exhaustive_search()
    make_comparison()
    print("\nAll beam search figures generated!")
