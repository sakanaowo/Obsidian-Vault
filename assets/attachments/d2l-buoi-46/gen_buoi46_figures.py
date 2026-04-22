#!/usr/bin/env python3
"""
Generate SVG illustrations for Buổi 46 (Machine Translation & Dataset).
These replace the broken HTML files that were incorrectly named .svg.
"""

import subprocess
import sys

def check_matplotlib():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
        from matplotlib.patches import ArrowStyle
        return plt, mpatches, FancyArrowPatch, FancyBboxPatch
    except ImportError:
        print("Installing matplotlib...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "-q"])
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
        return plt, mpatches, FancyArrowPatch, FancyBboxPatch

plt, mpatches, FancyArrowPatch, FancyBboxPatch = check_matplotlib()

OUT = "/home/sakana/Code/Obsidian-Vault/assets/attachments/d2l-buoi-46"

# ─────────────────────────────────────────────────────────────────────────────
# 1. seq2seq.svg — Full Seq2Seq Architecture Overview
# ─────────────────────────────────────────────────────────────────────────────
def make_seq2seq():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis('off')
    fig.patch.set_alpha(0)

    # ── Colors ──────────────────────────────────────────────────────────────
    ENC_BG   = "#E3F2FD"   # light blue
    DEC_BG   = "#FFF3E0"   # light orange
    HID_BG   = "#E8F5E9"   # light green
    ARROW_C  = "#1976D2"   # blue arrows
    CTX_C    = "#388E3C"   # green for context
    ARROW2   = "#F57C00"   # orange arrows
    TEXT_C   = "#212121"
    LABEL_C  = "#1565C0"
    LABEL2   = "#E65100"

    # ── Title ───────────────────────────────────────────────────────────────
    ax.text(7, 6.6, "Sequence-to-Sequence (Seq2Seq) Architecture",
            ha='center', va='center', fontsize=14, fontweight='bold', color=TEXT_C)

    # ────────────────────────────────────────────────────────────────────────
    # ENCODER SECTION (top, y=3.8 to 5.8)
    # ────────────────────────────────────────────────────────────────────────
    # Background box
    enc_box = FancyBboxPatch((0.3, 3.7), 5.5, 2.1,
                             boxstyle="round,pad=0.1",
                             linewidth=1.5, edgecolor=LABEL_C,
                             facecolor=ENC_BG, zorder=0)
    ax.add_patch(enc_box)
    ax.text(3.05, 5.65, "ENCODER", ha='center', va='center',
            fontsize=11, fontweight='bold', color=LABEL_C)

    # Tokens in encoder
    enc_x = [0.9, 2.1, 3.3, 4.5, 5.7]
    enc_labels = [r'$x_1$', r'$x_2$', r'$x_3$', r'$...$', r'$x_T$']
    for x, lbl in zip(enc_x, enc_labels):
        rect = FancyBboxPatch((x-0.3, 4.1), 0.55, 0.7,
                               boxstyle="round,pad=0.05",
                               linewidth=1.2, edgecolor="#0D47A1",
                               facecolor="#BBDEFB", zorder=2)
        ax.add_patch(rect)
        ax.text(x-0.025, 4.45, lbl, ha='center', va='center',
                fontsize=11, fontweight='bold', color="#0D47A1", zorder=3)

    # Hidden states
    hid_x = [1.05, 2.25, 3.45, 4.65, 5.85]
    hid_labels = [r'$H_1$', r'$H_2$', r'$H_3$', r'$...$', r'$H_T$']
    for x, lbl in zip(hid_x, hid_labels):
        rect = FancyBboxPatch((x-0.3, 4.9), 0.55, 0.55,
                               boxstyle="round,pad=0.05",
                               linewidth=1.2, edgecolor="#1B5E20",
                               facecolor=HID_BG, zorder=2)
        ax.add_patch(rect)
        ax.text(x-0.025, 5.175, lbl, ha='center', va='center',
                fontsize=10, color="#1B5E20", zorder=3)

    # Arrows between encoder tokens
    for i in range(len(enc_x)-1):
        ax.annotate("", xy=(enc_x[i+1]-0.35, 4.45),
                   xytext=(enc_x[i]+0.3, 4.45),
                   arrowprops=dict(arrowstyle="->", color=ARROW_C, lw=1.5))

    # ────────────────────────────────────────────────────────────────────────
    # CONTEXT VECTOR (middle, y=2.9)
    # ────────────────────────────────────────────────────────────────────────
    ctx_rect = FancyBboxPatch((4.8, 2.9), 1.1, 0.65,
                               boxstyle="round,pad=0.1",
                               linewidth=2, edgecolor=CTX_C,
                               facecolor="#C8E6C9", zorder=2)
    ax.add_patch(ctx_rect)
    ax.text(5.35, 3.225, r'$C = H_T$', ha='center', va='center',
            fontsize=11, fontweight='bold', color=CTX_C, zorder=3)

    # Arrow: encoder final hidden → context
    ax.annotate("", xy=(5.3, 2.9),
                xytext=(5.85, 4.9),
                arrowprops=dict(arrowstyle="->", color=CTX_C, lw=2,
                                connectionstyle="arc3,rad=0.1"))

    # ────────────────────────────────────────────────────────────────────────
    # DECODER SECTION (bottom, y=0.5 to 2.8)
    # ────────────────────────────────────────────────────────────────────────
    dec_box = FancyBboxPatch((0.3, 0.5), 5.5, 2.3,
                             boxstyle="round,pad=0.1",
                             linewidth=1.5, edgecolor=LABEL2,
                             facecolor=DEC_BG, zorder=0)
    ax.add_patch(dec_box)
    ax.text(3.05, 2.65, "DECODER", ha='center', va='center',
            fontsize=11, fontweight='bold', color=LABEL2)

    # Tokens in decoder
    dec_x = [0.9, 2.1, 3.3, 4.5, 5.7]
    dec_labels = [r'$y_0$', r'$y_1$', r'$y_2$', r'$y_3$', r'$...$']
    for x, lbl in zip(dec_x, dec_labels):
        rect = FancyBboxPatch((x-0.3, 1.0), 0.55, 0.7,
                               boxstyle="round,pad=0.05",
                               linewidth=1.2, edgecolor="#BF360C",
                               facecolor="#FFCCBC", zorder=2)
        ax.add_patch(rect)
        ax.text(x-0.025, 1.35, lbl, ha='center', va='center',
                fontsize=11, fontweight='bold', color="#BF360C", zorder=3)

    # Hidden states in decoder
    for x in dec_x:
        rect = FancyBboxPatch((x-0.3, 1.85), 0.55, 0.55,
                               boxstyle="round,pad=0.05",
                               linewidth=1.2, edgecolor="#1B5E20",
                               facecolor=HID_BG, zorder=2)
        ax.add_patch(rect)

    # s_0, s_1, s_2 labels
    for i, x in enumerate(dec_x):
        ax.text(x-0.025, 2.125, f'$s_{i}$', ha='center', va='center',
                fontsize=10, color="#1B5E20", zorder=3)

    # Arrows between decoder tokens
    for i in range(len(dec_x)-1):
        ax.annotate("", xy=(dec_x[i+1]-0.35, 1.35),
                   xytext=(dec_x[i]+0.3, 1.35),
                   arrowprops=dict(arrowstyle="->", color=ARROW2, lw=1.5))

    # Arrow: context → decoder initial state
    ax.annotate("", xy=(0.9+0.3, 2.42),
                xytext=(4.85, 3.2),
                arrowprops=dict(arrowstyle="->", color=CTX_C, lw=2,
                                connectionstyle="arc3,rad=-0.2"))

    # ────────────────────────────────────────────────────────────────────────
    # OUTPUT PREDICTIONS (right side, y=0.5 to 2.8)
    # ────────────────────────────────────────────────────────────────────────
    ax.text(10.5, 2.65, "Predictions", ha='center', va='center',
            fontsize=11, fontweight='bold', color=TEXT_C)

    pred_x = [9.5, 10.5, 11.5]
    pred_labels = [r'$\hat{y}_1$', r'$\hat{y}_2$', r'$\hat{y}_3$']
    for x, lbl in zip(pred_x, pred_labels):
        rect = FancyBboxPatch((x-0.35, 1.8), 0.65, 0.6,
                               boxstyle="round,pad=0.05",
                               linewidth=1.2, edgecolor="#6A1B9A",
                               facecolor="#E1BEE7", zorder=2)
        ax.add_patch(rect)
        ax.text(x-0.025, 2.1, lbl, ha='center', va='center',
                fontsize=11, fontweight='bold', color="#6A1B9A", zorder=3)

    # Arrows from decoder to predictions
    for i, px in enumerate(pred_x):
        ax.annotate("", xy=(px-0.05, 1.8),
                    xytext=(dec_x[i+1]-0.05, 2.42),
                    arrowprops=dict(arrowstyle="->", color="#7B1FA2", lw=1.5))

    # ────────────────────────────────────────────────────────────────────────
    # LEGEND
    # ────────────────────────────────────────────────────────────────────────
    ax.text(10.5, 5.65, "Input Sequence", ha='center', va='center',
            fontsize=9, color=LABEL_C)
    rect = FancyBboxPatch((9.9, 5.2), 1.2, 0.35,
                           boxstyle="round,pad=0.05",
                           linewidth=1, edgecolor="#0D47A1",
                           facecolor="#BBDEFB", zorder=2)
    ax.add_patch(rect)

    ax.text(10.5, 4.6, "Encoder\nHidden", ha='center', va='center',
            fontsize=9, color="#1B5E20")
    rect = FancyBboxPatch((9.9, 4.1), 1.2, 0.35,
                           boxstyle="round,pad=0.05",
                           linewidth=1, edgecolor="#1B5E20",
                           facecolor=HID_BG, zorder=2)
    ax.add_patch(rect)

    ax.text(10.5, 3.5, "Context\nVector C", ha='center', va='center',
            fontsize=9, color=CTX_C)
    rect = FancyBboxPatch((9.9, 3.0), 1.2, 0.35,
                           boxstyle="round,pad=0.05",
                           linewidth=1, edgecolor=CTX_C,
                           facecolor="#C8E6C9", zorder=2)
    ax.add_patch(rect)

    ax.text(10.5, 2.0, "Predictions", ha='center', va='center',
            fontsize=9, color="#6A1B9A")
    rect = FancyBboxPatch((9.9, 1.5), 1.2, 0.35,
                           boxstyle="round,pad=0.05",
                           linewidth=1, edgecolor="#6A1B9A",
                           facecolor="#E1BEE7", zorder=2)
    ax.add_patch(rect)

    # ────────────────────────────────────────────────────────────────────────
    # Stage labels
    # ────────────────────────────────────────────────────────────────────────
    ax.text(3.05, 3.55, "Stage 1: Encode source → context",
            ha='center', va='center', fontsize=9, color=LABEL_C,
            style='italic')
    ax.text(3.05, 0.35, "Stage 2: Decode autoregressively (Teacher Forcing during training)",
            ha='center', va='center', fontsize=9, color=LABEL2,
            style='italic')

    plt.tight_layout(pad=0.3)
    plt.savefig(f"{OUT}/seq2seq.svg", dpi=150, bbox_inches='tight',
                facecolor='white', format='svg')
    plt.close()
    print(f"✓ {OUT}/seq2seq.svg")


# ─────────────────────────────────────────────────────────────────────────────
# 2. encoder.svg — Detailed Encoder
# ─────────────────────────────────────────────────────────────────────────────
def make_encoder():
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis('off')
    fig.patch.set_alpha(0)

    BLUE_DARK   = "#0D47A1"
    BLUE_MED    = "#1976D2"
    BLUE_LIGHT  = "#BBDEFB"
    GREEN_DARK  = "#1B5E20"
    GREEN_MED   = "#388E3C"
    GREEN_LIGHT = "#C8E6C9"
    ORANGE      = "#E65100"
    TEXT        = "#212121"

    # Title
    ax.text(6.5, 5.6, "Encoder — Processing the Source Sequence",
            ha='center', va='center', fontsize=13, fontweight='bold', color=TEXT)

    # Number of timesteps
    n_steps = 6
    start_x = 1.5
    step_w  = 1.6

    for i in range(n_steps):
        x = start_x + i * step_w

        # Token box (input word)
        if i == n_steps - 1:
            token_lbl = r'$x_T$'
            token_bg  = "#FFF9C4"
            token_ec  = ORANGE
        elif i == n_steps - 2:
            token_lbl = r'$x_{T-1}$'
            token_bg  = BLUE_LIGHT
            token_ec  = BLUE_DARK
        else:
            token_lbl = f'$x_{{{i+1}}}$'
            token_bg  = BLUE_LIGHT
            token_ec  = BLUE_DARK

        rect = FancyBboxPatch((x-0.35, 3.6), 0.65, 0.65,
                               boxstyle="round,pad=0.05",
                               linewidth=1.5, edgecolor=token_ec,
                               facecolor=token_bg, zorder=2)
        ax.add_patch(rect)
        ax.text(x-0.025, 3.925, token_lbl, ha='center', va='center',
                fontsize=11, fontweight='bold', color=token_ec, zorder=3)

        # RNN cell box
        cell_rect = FancyBboxPatch((x-0.45, 2.0), 0.85, 1.4,
                                    boxstyle="round,pad=0.1",
                                    linewidth=1.5, edgecolor=BLUE_MED,
                                    facecolor="#E3F2FD", zorder=2)
        ax.add_patch(cell_rect)
        ax.text(x-0.025, 2.7, 'RNN\nCell', ha='center', va='center',
                fontsize=9, color=BLUE_MED, fontweight='bold', zorder=3)

        # Hidden state box
        hid_rect = FancyBboxPatch((x-0.45, 1.0), 0.85, 0.8,
                                   boxstyle="round,pad=0.05",
                                   linewidth=1.5, edgecolor=GREEN_DARK,
                                   facecolor=GREEN_LIGHT, zorder=2)
        ax.add_patch(hid_rect)
        ax.text(x-0.025, 1.4, f'$H_{{{i+1}}}$', ha='center', va='center',
                fontsize=10, color=GREEN_DARK, fontweight='bold', zorder=3)

        # Arrow: token → RNN cell
        ax.annotate("", xy=(x-0.025, 3.6),
                    xytext=(x-0.025, 3.3),
                    arrowprops=dict(arrowstyle="->", color=BLUE_MED, lw=1.5))
        ax.annotate("", xy=(x-0.025, 2.0),
                    xytext=(x-0.025, 2.1),
                    arrowprops=dict(arrowstyle="->", color=BLUE_MED, lw=1.5))

        # Arrow: RNN cell → hidden state
        ax.annotate("", xy=(x-0.025, 1.8),
                    xytext=(x-0.025, 2.3),
                    arrowprops=dict(arrowstyle="->", color=GREEN_MED, lw=1.5))

        # Arrow between steps (hidden state flow)
        if i < n_steps - 1:
            ax.annotate("", xy=(start_x+(i+1)*step_w-0.5, 1.4),
                        xytext=(x+0.45, 1.4),
                        arrowprops=dict(arrowstyle="->", color=GREEN_MED, lw=2))

    # Final hidden state → Context Vector
    last_x = start_x + (n_steps-1)*step_w
    ctx_rect = FancyBboxPatch((last_x+1.0, 0.2), 1.5, 0.65,
                               boxstyle="round,pad=0.1",
                               linewidth=2, edgecolor=GREEN_MED,
                               facecolor=GREEN_LIGHT, zorder=2)
    ax.add_patch(ctx_rect)
    ax.text(last_x+1.75, 0.525, r'$C = H_T$', ha='center', va='center',
            fontsize=11, fontweight='bold', color=GREEN_DARK, zorder=3)

    ax.annotate("", xy=(last_x+1.0, 0.525),
                xytext=(last_x+0.45, 1.0),
                arrowprops=dict(arrowstyle="->", color=GREEN_MED, lw=2,
                                connectionstyle="arc3,rad=0.1"))

    # Annotations
    ax.annotate("RNN unrolls over time",
                xy=(start_x+2*step_w, 2.7),
                xytext=(start_x+2*step_w, 5.0),
                ha='center',
                arrowprops=dict(arrowstyle="->", color="#757575",
                                connectionstyle="arc3,rad=0.1"),
                fontsize=9, color="#757575", style='italic')

    ax.annotate("Final hidden state\n= Context Vector",
                xy=(last_x+1.75, 0.2),
                xytext=(last_x+2.8, 0.8),
                ha='center',
                arrowprops=dict(arrowstyle="->", color=GREEN_MED,
                                connectionstyle="arc3,rad=-0.2"),
                fontsize=9, color=GREEN_DARK, fontweight='bold')

    plt.tight_layout(pad=0.3)
    plt.savefig(f"{OUT}/encoder.svg", dpi=150, bbox_inches='tight',
                facecolor='white', format='svg')
    plt.close()
    print(f"✓ {OUT}/encoder.svg")


# ─────────────────────────────────────────────────────────────────────────────
# 3. decoder.svg — Detailed Decoder
# ─────────────────────────────────────────────────────────────────────────────
def make_decoder():
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis('off')
    fig.patch.set_alpha(0)

    BLUE_DARK   = "#0D47A1"
    BLUE_MED    = "#1976D2"
    BLUE_LIGHT  = "#BBDEFB"
    GREEN_DARK  = "#1B5E20"
    GREEN_MED   = "#388E3C"
    GREEN_LIGHT = "#C8E6C9"
    PURPLE_DARK = "#4A148C"
    PURPLE_MED  = "#7B1FA2"
    PURPLE_LIGHT= "#E1BEE7"
    ORANGE      = "#E65100"
    ORANGE_LIGHT= "#FFCCBC"
    TEXT        = "#212121"

    # Title
    ax.text(6.5, 5.6, "Decoder — Autoregressive Generation",
            ha='center', va='center', fontsize=13, fontweight='bold', color=TEXT)

    # Number of decoder steps
    n_steps = 5
    start_x = 1.5
    step_w  = 1.7

    for i in range(n_steps):
        x = start_x + i * step_w

        # Input token (y_{t-1})
        if i == 0:
            in_bg, in_ec = "#FFF9C4", ORANGE
            in_lbl = r'$y_0$ (<bos>)'
        else:
            in_bg, in_ec = ORANGE_LIGHT, "#BF360C"
            in_lbl = f'$y_{{{i}}}$'

        rect = FancyBboxPatch((x-0.4, 3.6), 0.75, 0.65,
                               boxstyle="round,pad=0.05",
                               linewidth=1.5, edgecolor=in_ec,
                               facecolor=in_bg, zorder=2)
        ax.add_patch(rect)
        ax.text(x-0.025, 3.925, in_lbl, ha='center', va='center',
                fontsize=9, fontweight='bold', color=in_ec, zorder=3)

        # RNN cell
        cell_rect = FancyBboxPatch((x-0.5, 2.0), 0.9, 1.4,
                                    boxstyle="round,pad=0.1",
                                    linewidth=1.5, edgecolor=PURPLE_MED,
                                    facecolor="#F3E5F5", zorder=2)
        ax.add_patch(cell_rect)
        ax.text(x-0.05, 2.7, 'RNN\nCell', ha='center', va='center',
                fontsize=9, color=PURPLE_MED, fontweight='bold', zorder=3)

        # Hidden state
        hid_rect = FancyBboxPatch((x-0.5, 1.0), 0.9, 0.8,
                                   boxstyle="round,pad=0.05",
                                   linewidth=1.5, edgecolor=GREEN_DARK,
                                   facecolor=GREEN_LIGHT, zorder=2)
        ax.add_patch(hid_rect)
        ax.text(x-0.05, 1.4, f'$s_{{{i}}}$', ha='center', va='center',
                fontsize=10, color=GREEN_DARK, fontweight='bold', zorder=3)

        # Arrow: token → RNN
        ax.annotate("", xy=(x-0.025, 3.6),
                    xytext=(x-0.025, 3.3),
                    arrowprops=dict(arrowstyle="->", color=in_ec, lw=1.5))

        # Arrow: RNN → hidden
        ax.annotate("", xy=(x-0.05, 1.8),
                    xytext=(x-0.05, 2.3),
                    arrowprops=dict(arrowstyle="->", color=GREEN_MED, lw=1.5))

        # Arrow between steps
        if i < n_steps - 1:
            ax.annotate("", xy=(start_x+(i+1)*step_w-0.55, 1.4),
                        xytext=(x+0.45, 1.4),
                        arrowprops=dict(arrowstyle="->", color=GREEN_MED, lw=2))

        # Prediction output
        pred_rect = FancyBboxPatch((x-0.35, 0.1), 0.65, 0.7,
                                    boxstyle="round,pad=0.05",
                                    linewidth=1.5, edgecolor=PURPLE_DARK,
                                    facecolor=PURPLE_LIGHT, zorder=2)
        ax.add_patch(pred_rect)
        ax.text(x-0.025, 0.45, f'$\\hat{{y}}_{{{i+1}}}$', ha='center', va='center',
                fontsize=10, fontweight='bold', color=PURPLE_DARK, zorder=3)

        # Arrow: hidden → prediction
        ax.annotate("", xy=(x-0.025, 0.8),
                    xytext=(x-0.025, 1.0),
                    arrowprops=dict(arrowstyle="->", color=PURPLE_MED, lw=1.5))

        # Arrow: prediction → next token input (teacher forcing / loop)
        if i < n_steps - 1:
            ax.annotate("", xy=(start_x+(i+1)*step_w-0.45, 3.6),
                        xytext=(x-0.025, 0.1),
                        arrowprops=dict(arrowstyle="->", color="#757575", lw=1,
                                        connectionstyle="arc3,rad=-0.25",
                                        linestyle='dashed'))

    # Context vector input (for first step)
    ctx_rect = FancyBboxPatch((start_x-0.1, 2.15), 1.0, 0.9,
                               boxstyle="round,pad=0.1",
                               linewidth=2, edgecolor=GREEN_MED,
                               facecolor=GREEN_LIGHT, zorder=2)
    ax.add_patch(ctx_rect)
    ax.text(start_x+0.4, 2.6, r'$C$', ha='center', va='center',
            fontsize=12, fontweight='bold', color=GREEN_DARK, zorder=3)

    ax.annotate("", xy=(start_x-0.05, 3.3),
                xytext=(start_x+0.4, 3.05),
                arrowprops=dict(arrowstyle="->", color=GREEN_MED, lw=1.5,
                                connectionstyle="arc3,rad=0.3"))

    # Note box
    ax.text(12.0, 5.2,
            "Training:\nTeacher Forcing\n(ground truth → input)\n\nInference:\nAutoregressive\n(prediction → input)",
            ha='center', va='center', fontsize=8,
            color="#5D4037",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFFDE7",
                      edgecolor="#F9A825", linewidth=1.5))

    plt.tight_layout(pad=0.3)
    plt.savefig(f"{OUT}/decoder.svg", dpi=150, bbox_inches='tight',
                facecolor='white', format='svg')
    plt.close()
    print(f"✓ {OUT}/decoder.svg")


# ─────────────────────────────────────────────────────────────────────────────
# 4. mt-seq2seq.svg — Seq2Seq with Attention
# ─────────────────────────────────────────────────────────────────────────────
def make_mt_seq2seq():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis('off')
    fig.patch.set_alpha(0)

    BLUE_DARK   = "#0D47A1"
    BLUE_MED    = "#1976D2"
    BLUE_LIGHT  = "#BBDEFB"
    GREEN_DARK  = "#1B5E20"
    GREEN_MED   = "#388E3C"
    GREEN_LIGHT = "#C8E6C9"
    PURPLE_MED  = "#7B1FA2"
    PURPLE_LIGHT= "#E1BEE7"
    RED_MED     = "#D32F2F"
    RED_LIGHT   = "#FFCDD2"
    ORANGE      = "#E65100"
    TEXT        = "#212121"

    ax.text(7, 6.6, "Seq2Seq with Attention — Dynamic Context per Decoder Step",
            ha='center', va='center', fontsize=13, fontweight='bold', color=TEXT)

    # ── ENCODER ──────────────────────────────────────────────────────────────
    enc_box = FancyBboxPatch((0.3, 4.2), 6.0, 2.3,
                             boxstyle="round,pad=0.1",
                             linewidth=1.5, edgecolor=BLUE_MED,
                             facecolor="#E3F2FD", zorder=0)
    ax.add_patch(enc_box)
    ax.text(3.3, 6.35, "ENCODER", ha='center', va='center',
            fontsize=11, fontweight='bold', color=BLUE_MED)

    enc_xs = [0.9, 2.0, 3.1, 4.2, 5.3]
    enc_labels = [r'$x_1$', r'$x_2$', r'$x_3$', r'$...$', r'$x_T$']
    for x, lbl in zip(enc_xs, enc_labels):
        rect = FancyBboxPatch((x-0.3, 4.5), 0.55, 0.55,
                               boxstyle="round,pad=0.05",
                               linewidth=1.2, edgecolor=BLUE_DARK,
                               facecolor=BLUE_LIGHT, zorder=2)
        ax.add_patch(rect)
        ax.text(x-0.025, 4.775, lbl, ha='center', va='center',
                fontsize=10, fontweight='bold', color=BLUE_DARK, zorder=3)

        # Hidden state
        rect2 = FancyBboxPatch((x-0.3, 5.2), 0.55, 0.6,
                                boxstyle="round,pad=0.05",
                                linewidth=1.2, edgecolor=GREEN_DARK,
                                facecolor=GREEN_LIGHT, zorder=2)
        ax.add_patch(rect2)
        ax.text(x-0.025, 5.5, f'$H_{enc_xs.index(x)+1}$', ha='center', va='center',
                fontsize=9, color=GREEN_DARK, fontweight='bold', zorder=3)

    for i in range(len(enc_xs)-1):
        ax.annotate("", xy=(enc_xs[i+1]-0.35, 4.775),
                   xytext=(enc_xs[i]+0.3, 4.775),
                   arrowprops=dict(arrowstyle="->", color=BLUE_MED, lw=1.5))

    # Arrow from last hidden state (don't use as context; all H are kept)
    ax.annotate("All $H_i$ kept\nfor attention",
                xy=(enc_xs[-1]-0.025, 5.8),
                xytext=(enc_xs[-1]+1.2, 6.0),
                ha='center', fontsize=8, color=GREEN_DARK, style='italic',
                arrowprops=dict(arrowstyle="->", color=GREEN_MED,
                                connectionstyle="arc3,rad=-0.1"))

    # ── DECODER ─────────────────────────────────────────────────────────────
    dec_box = FancyBboxPatch((0.3, 0.5), 6.0, 2.5,
                             boxstyle="round,pad=0.1",
                             linewidth=1.5, edgecolor=PURPLE_MED,
                             facecolor="#F3E5F5", zorder=0)
    ax.add_patch(dec_box)
    ax.text(3.3, 2.85, "DECODER", ha='center', va='center',
            fontsize=11, fontweight='bold', color=PURPLE_MED)

    dec_xs = [1.0, 2.2, 3.4, 4.6, 5.8]
    dec_labels = [r'$y_0$', r'$y_1$', r'$y_2$', r'$y_3$', r'$...$']
    for i, (x, lbl) in enumerate(zip(dec_xs, dec_labels)):
        if i == 0:
            bg, ec = "#FFF9C4", ORANGE
        else:
            bg, ec = "#FFCCBC", "#BF360C"
        rect = FancyBboxPatch((x-0.3, 1.0), 0.55, 0.55,
                               boxstyle="round,pad=0.05",
                               linewidth=1.2, edgecolor=ec,
                               facecolor=bg, zorder=2)
        ax.add_patch(rect)
        ax.text(x-0.025, 1.275, lbl, ha='center', va='center',
                fontsize=10, fontweight='bold', color=ec, zorder=3)

        # Decoder hidden state
        rect2 = FancyBboxPatch((x-0.3, 1.7), 0.55, 0.6,
                                boxstyle="round,pad=0.05",
                                linewidth=1.2, edgecolor=GREEN_DARK,
                                facecolor=GREEN_LIGHT, zorder=2)
        ax.add_patch(rect2)
        ax.text(x-0.025, 2.0, f'$s_{i}$', ha='center', va='center',
                fontsize=9, color=GREEN_DARK, fontweight='bold', zorder=3)

        # Prediction
        pred_rect = FancyBboxPatch((x-0.3, 2.45), 0.55, 0.5,
                                    boxstyle="round,pad=0.05",
                                    linewidth=1.2, edgecolor=PURPLE_MED,
                                    facecolor=PURPLE_LIGHT, zorder=2)
        ax.add_patch(pred_rect)
        ax.text(x-0.025, 2.7, f'$\\hat{{y}}_{i+1}$', ha='center', va='center',
                fontsize=9, fontweight='bold', color=PURPLE_MED, zorder=3)

    for i in range(len(dec_xs)-1):
        ax.annotate("", xy=(dec_xs[i+1]-0.35, 1.275),
                   xytext=(dec_xs[i]+0.3, 1.275),
                   arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.5))

    # ── ATTENTION MECHANISM (center column) ───────────────────────────────────
    ax.text(8.0, 5.5, "Attention\nMechanism", ha='center', va='center',
            fontsize=10, fontweight='bold', color="#5D4037",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF9C4",
                      edgecolor="#F9A825", linewidth=1.5))

    att_x = 8.0
    att_y_top = 4.8
    att_h = 3.2

    # Attention weights block
    att_rect = FancyBboxPatch((att_x-0.7, att_y_top - att_h), 1.4, att_h,
                               boxstyle="round,pad=0.1",
                               linewidth=1.5, edgecolor="#F9A825",
                               facecolor="#FFFDE7", zorder=2)
    ax.add_patch(att_rect)
    ax.text(att_x, att_y_top - 0.15,
            r"$C_t = \sum_i \alpha_{t,i} H_i$",
            ha="center", va="center", fontsize=8, color="#5D4037",
            fontweight="bold", zorder=3)

    # Draw attention score bars (simplified)
    bar_x = att_x - 0.5
    bar_h_max = att_h - 0.5
    weights = [0.1, 0.15, 0.4, 0.25, 0.1]
    enc_labels_short = [r'$H_1$', r'$H_2$', r'$H_3$', r'$H_{...}$', r'$H_T$']
    for j, (w, el) in enumerate(zip(weights, enc_labels_short)):
        bh = w * bar_h_max * 2.5
        bar = FancyBboxPatch((bar_x, att_y_top - att_h + 0.1 + j * (bar_h_max/5)),
                             0.35, bh,
                             boxstyle="round,pad=0.02",
                             linewidth=0.8, edgecolor=GREEN_MED,
                             facecolor=GREEN_LIGHT, zorder=3)
        ax.add_patch(bar)
        ax.text(bar_x + 0.5, att_y_top - att_h + 0.1 + j*(bar_h_max/5) + bh/2,
                el, ha='left', va='center', fontsize=7, color=GREEN_DARK)

    # Arrows: encoder H_i → attention
    for i, ex in enumerate(enc_xs):
        ax.annotate("", xy=(att_x-0.7, att_y_top - att_h + 0.1 + i*(att_h/5) + 0.3),
                    xytext=(ex+0.3, 5.2),
                    arrowprops=dict(arrowstyle="->", color=GREEN_MED, lw=0.8,
                                    alpha=0.6))

    # Arrows: decoder s_t → attention
    for i, dx in enumerate(dec_xs):
        ax.annotate("", xy=(att_x, att_y_top - att_h + 0.1 + i*(att_h/5) + 0.3),
                    xytext=(dx, 2.3),
                    arrowprops=dict(arrowstyle="->", color=PURPLE_MED, lw=0.8,
                                    alpha=0.6, connectionstyle="arc3,rad=-0.1"))

    # Arrow: attention → decoder context
    ax.annotate("", xy=(dec_xs[0]-0.3, 2.0),
                xytext=(att_x+0.7, att_y_top - att_h + 0.5),
                arrowprops=dict(arrowstyle="->", color=RED_MED, lw=2,
                                connectionstyle="arc3,rad=0.2"))

    ax.text(att_x+0.5, 2.0, r'$C_t$', ha='left', va='center',
            fontsize=10, fontweight='bold', color=RED_MED)

    plt.tight_layout(pad=0.3)
    plt.savefig(f"{OUT}/mt-seq2seq.svg", dpi=150, bbox_inches='tight',
                facecolor='white', format='svg')
    plt.close()
    print(f"✓ {OUT}/mt-seq2seq.svg")


# ─────────────────────────────────────────────────────────────────────────────
# 5. mt-transformer.svg — Evolution from RNN Seq2Seq to Transformer
# ─────────────────────────────────────────────────────────────────────────────
def make_mt_transformer():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis('off')
    fig.patch.set_alpha(0)

    BLUE_MED  = "#1976D2"
    BLUE_LIGHT= "#BBDEFB"
    GREEN_MED = "#388E3C"
    GREEN_LIGHT="#C8E6C9"
    PURPLE_MED= "#7B1FA2"
    PURPLE_LIGHT="#E1BEE7"
    ORANGE    = "#E65100"
    TEAL_MED  = "#00796B"
    TEAL_LIGHT= "#B2DFDB"
    TEXT      = "#212121"
    ARROW_C   = "#757575"

    ax.text(6.5, 6.55, "Evolution: RNN Seq2Seq → Transformer",
            ha='center', va='center', fontsize=13, fontweight='bold', color=TEXT)

    # ── LEFT: RNN Seq2Seq ─────────────────────────────────────────────────────
    left_box = FancyBboxPatch((0.2, 0.3), 5.6, 5.8,
                              boxstyle="round,pad=0.15",
                              linewidth=1.5, edgecolor=BLUE_MED,
                              facecolor="#E3F2FD", zorder=0)
    ax.add_patch(left_box)
    ax.text(3.0, 5.9, "RNN Seq2Seq (2014)", ha='center', va='center',
            fontsize=11, fontweight='bold', color=BLUE_MED)

    # Encoder RNN
    ax.text(1.0, 5.0, "Encoder", ha='left', va='center',
            fontsize=9, color=BLUE_MED, fontweight='bold')
    for i, x in enumerate([0.8, 1.7, 2.6]):
        rect = FancyBboxPatch((x-0.3, 4.1), 0.55, 0.65,
                               boxstyle="round,pad=0.05",
                               linewidth=1, edgecolor=BLUE_MED,
                               facecolor=BLUE_LIGHT, zorder=2)
        ax.add_patch(rect)
        ax.text(x-0.025, 4.425, f'$x_{i+1}$', ha='center', va='center',
                fontsize=9, color=BLUE_MED, zorder=3)

        # RNN cell
        crect = FancyBboxPatch((x-0.35, 3.0), 0.65, 0.9,
                                boxstyle="round,pad=0.05",
                                linewidth=1, edgecolor="#0D47A1",
                                facecolor="#E3F2FD", zorder=2)
        ax.add_patch(crect)
        ax.text(x-0.025, 3.45, 'RNN', ha='center', va='center',
                fontsize=8, color="#0D47A1", fontweight='bold', zorder=3)

        # H state
        hrect = FancyBboxPatch((x-0.35, 2.3), 0.65, 0.55,
                                boxstyle="round,pad=0.05",
                                linewidth=1, edgecolor=GREEN_MED,
                                facecolor=GREEN_LIGHT, zorder=2)
        ax.add_patch(hrect)
        ax.text(x-0.025, 2.575, f'$H_{i+1}$', ha='center', va='center',
                fontsize=8, color=GREEN_MED, fontweight='bold', zorder=3)

        if i < 2:
            ax.annotate("", xy=(x+0.7-0.35, 2.575),
                        xytext=(x+0.3, 2.575),
                        arrowprops=dict(arrowstyle="->", color=GREEN_MED, lw=1.2))

    # Context arrow
    ax.annotate("", xy=(3.0, 2.3), xytext=(2.95, 2.85),
                arrowprops=dict(arrowstyle="->", color=GREEN_MED, lw=2))

    ctx_rect = FancyBboxPatch((2.5, 1.75), 1.0, 0.45,
                               boxstyle="round,pad=0.05",
                               linewidth=1.5, edgecolor=GREEN_MED,
                               facecolor=GREEN_LIGHT, zorder=2)
    ax.add_patch(ctx_rect)
    ax.text(3.0, 1.975, r'$C$', ha='center', va='center',
            fontsize=9, fontweight='bold', color=GREEN_MED, zorder=3)
    ax.text(3.0, 1.55, "bottleneck",
            ha='center', va='center', fontsize=7, color=GREEN_MED, style='italic')

    # Decoder RNN
    ax.text(1.0, 1.1, "Decoder", ha='left', va='center',
            fontsize=9, color=PURPLE_MED, fontweight='bold')
    for i, x in enumerate([0.8, 1.7, 2.6]):
        rect = FancyBboxPatch((x-0.3, 0.55), 0.55, 0.55,
                               boxstyle="round,pad=0.05",
                               linewidth=1, edgecolor=PURPLE_MED,
                               facecolor=PURPLE_LIGHT, zorder=2)
        ax.add_patch(rect)
        ax.text(x-0.025, 0.825, f'$y_i$', ha='center', va='center',
                fontsize=9, color=PURPLE_MED, zorder=3)

    ax.annotate("", xy=(3.0, 0.55),
                xytext=(3.0, 1.75),
                arrowprops=dict(arrowstyle="->", color=GREEN_MED, lw=1.5))

    # Limitation note
    ax.text(3.0, 4.5, "⚠ Fixed context C\nLong sequences suffer",
            ha='center', va='center', fontsize=7.5, color="#B71C1C",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFCDD2",
                      edgecolor="#E57373", linewidth=1))

    # ── ARROW between boxes ───────────────────────────────────────────────────
    ax.annotate("2017\nTransformer", xy=(6.6, 3.5),
                xytext=(5.85, 3.5),
                ha='center', fontsize=9, fontweight='bold', color=ARROW_C,
                arrowprops=dict(arrowstyle="->", color=ARROW_C, lw=2))

    # ── RIGHT: Transformer ────────────────────────────────────────────────────
    right_box = FancyBboxPatch((6.7, 0.3), 5.8, 5.8,
                               boxstyle="round,pad=0.15",
                               linewidth=1.5, edgecolor=TEAL_MED,
                               facecolor="#E0F2F1", zorder=0)
    ax.add_patch(right_box)
    ax.text(9.6, 5.9, "Transformer (2017)", ha='center', va='center',
            fontsize=11, fontweight='bold', color=TEAL_MED)

    # Encoder (self-attention stack)
    ax.text(7.3, 5.0, "Encoder", ha='left', va='center',
            fontsize=9, color=TEAL_MED, fontweight='bold')
    enc_layers = ["Self-\nAttention", "Feed\nForward", "Self-\nAttention"]
    for i, (x, lbl) in enumerate(zip([7.5, 8.5, 9.5], enc_layers)):
        rect = FancyBboxPatch((x-0.4, 3.9), 0.75, 0.9,
                               boxstyle="round,pad=0.05",
                               linewidth=1, edgecolor=TEAL_MED,
                               facecolor=TEAL_LIGHT, zorder=2)
        ax.add_patch(rect)
        ax.text(x-0.025, 4.35, lbl, ha='center', va='center',
                fontsize=7.5, color=TEAL_MED, fontweight='bold', zorder=3)
        if i < 2:
            ax.annotate("", xy=(x+0.4, 4.35),
                        xytext=(x+0.4, 4.35),
                        arrowprops=dict(arrowstyle="->", color=TEAL_MED, lw=1.2))

    # Encoder input tokens
    for x in [7.5, 8.5, 9.5]:
        rect = FancyBboxPatch((x-0.3, 4.95), 0.55, 0.55,
                               boxstyle="round,pad=0.05",
                               linewidth=1, edgecolor=TEAL_MED,
                               facecolor="#B2DFDB", zorder=2)
        ax.add_patch(rect)
        ax.text(x-0.025, 5.225, '$x_i$', ha='center', va='center',
                fontsize=9, color=TEAL_MED, fontweight='bold', zorder=3)
        ax.annotate("", xy=(x-0.025, 4.95),
                    xytext=(x-0.025, 5.05),
                    arrowprops=dict(arrowstyle="->", color=TEAL_MED, lw=1))

    # Decoder (cross-attention stack)
    ax.text(7.3, 3.5, "Decoder", ha='left', va='center',
            fontsize=9, color=PURPLE_MED, fontweight='bold')
    dec_layers = ["Self-\nAttention", "Cross-\nAttention", "Feed\nForward"]
    for i, (x, lbl) in enumerate(zip([7.5, 8.5, 9.5], dec_layers)):
        rect = FancyBboxPatch((x-0.4, 2.1), 0.75, 0.9,
                               boxstyle="round,pad=0.05",
                               linewidth=1, edgecolor=PURPLE_MED,
                               facecolor=PURPLE_LIGHT, zorder=2)
        ax.add_patch(rect)
        ax.text(x-0.025, 2.55, lbl, ha='center', va='center',
                fontsize=7.5, color=PURPLE_MED, fontweight='bold', zorder=3)

    # Decoder input tokens
    for x in [7.5, 8.5, 9.5]:
        rect = FancyBboxPatch((x-0.3, 1.1), 0.55, 0.55,
                               boxstyle="round,pad=0.05",
                               linewidth=1, edgecolor=PURPLE_MED,
                               facecolor="#F3E5F5", zorder=2)
        ax.add_patch(rect)
        ax.text(x-0.025, 1.375, '$y_i$', ha='center', va='center',
                fontsize=9, color=PURPLE_MED, fontweight='bold', zorder=3)

    # Encoder → Decoder cross-attention arrow
    ax.annotate("Cross-Attention", xy=(8.5, 2.1),
                xytext=(9.5, 3.9),
                ha='center', fontsize=7.5, color=PURPLE_MED,
                arrowprops=dict(arrowstyle="<->", color=PURPLE_MED, lw=1.5,
                                connectionstyle="arc3,rad=0"))

    # Output tokens
    for x in [10.5, 11.2]:
        rect = FancyBboxPatch((x-0.3, 0.55), 0.55, 0.55,
                               boxstyle="round,pad=0.05",
                               linewidth=1, edgecolor=ORANGE,
                               facecolor="#FFCCBC", zorder=2)
        ax.add_patch(rect)
        ax.text(x-0.025, 0.825, f'$\\hat{{y}}_{x-10}$', ha='center', va='center',
                fontsize=9, color=ORANGE, fontweight='bold', zorder=3)

    ax.annotate("", xy=(10.5-0.025, 1.1),
                xytext=(9.5-0.025, 2.1),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))

    # Key advantage note
    ax.text(9.6, 1.8, "✓ No bottleneck\n✓ Parallel (no recurrence)\n✓ Long-range dependencies",
            ha='center', va='center', fontsize=7.5, color="#1B5E20",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#C8E6C9",
                      edgecolor=GREEN_MED, linewidth=1))

    plt.tight_layout(pad=0.3)
    plt.savefig(f"{OUT}/mt-transformer.svg", dpi=150, bbox_inches='tight',
                facecolor='white', format='svg')
    plt.close()
    print(f"✓ {OUT}/mt-transformer.svg")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating SVG illustrations for Buổi 46...")
    make_seq2seq()
    make_encoder()
    make_decoder()
    make_mt_seq2seq()
    make_mt_transformer()
    print("\nAll illustrations generated successfully!")
