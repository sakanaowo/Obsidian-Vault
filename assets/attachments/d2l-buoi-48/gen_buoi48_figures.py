"""
Gen figures for Buổi 48 - 10.7 Sequence-to-Sequence Learning
D2L Chapter 10.7
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
OUTPUT_DIR = '/home/sakana/Code/Obsidian-Vault/assets/attachments/d2l-buoi-48/'

# =========================================================================
# Fig 1: seq2seq-layers.svg - Layers in RNN Encoder-Decoder (D2L Fig 10.7.2)
# =========================================================================
def fig_seq2seq_layers():
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Fig 10.7.2 — Layers trong RNN Encoder-Decoder Model', fontsize=14, fontweight='bold', pad=15)

    # ===== ENCODER (top half) =====
    # Title
    ax.text(1.2, 7.2, 'ENCODER', fontsize=12, fontweight='bold', color='#1565C0')
    ax.text(1.2, 6.85, '(Multi-layer GRU)', fontsize=10, color='gray')

    # Embedding box
    box1 = FancyBboxPatch((0.5, 6.1), 2.2, 0.65,
                          boxstyle="round,pad=0.05", facecolor='#BBDEFB', edgecolor='#1565C0', linewidth=1.5)
    ax.add_patch(box1)
    ax.text(1.6, 6.43, 'Embedding', fontsize=10, ha='center', va='center')
    ax.text(1.6, 6.15, '(vocab → embed)', fontsize=8, ha='center', va='center', color='gray')

    # GRU box
    box2 = FancyBboxPatch((3.5, 6.1), 2.2, 0.65,
                          boxstyle="round,pad=0.05", facecolor='#90CAF9', edgecolor='#1565C0', linewidth=1.5)
    ax.add_patch(box2)
    ax.text(4.6, 6.43, 'GRU', fontsize=10, ha='center', va='center')
    ax.text(4.6, 6.15, '(embed_size → h)', fontsize=8, ha='center', va='center', color='gray')

    # Final state box
    box3 = FancyBboxPatch((6.5, 6.1), 1.8, 0.65,
                          boxstyle="round,pad=0.05", facecolor='#64B5F6', edgecolor='#1565C0', linewidth=2)
    ax.add_patch(box3)
    ax.text(7.4, 6.43, 'h_T', fontsize=11, ha='center', va='center', fontweight='bold')
    ax.text(7.4, 6.15, '(context)', fontsize=8, ha='center', va='center', color='gray')

    # Arrow: input → embedding
    ax.annotate('', xy=(0.5, 6.42), xytext=(-0.1, 6.42),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text(0.1, 6.6, 'x_1...x_T\n(batch,T)', fontsize=8, ha='center')

    # Arrow: embedding → GRU
    ax.annotate('', xy=(3.5, 6.42), xytext=(2.7, 6.42),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text(3.1, 6.65, 'emb', fontsize=8, ha='center', color='gray')

    # Arrow: GRU → h_T
    ax.annotate('', xy=(6.5, 6.42), xytext=(5.7, 6.42),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text(6.1, 6.65, 'h_1...h_T', fontsize=8, ha='center', color='gray')

    # GRU layers indicators
    for i in range(2):
        ax.text(4.7, 5.85 - i*0.35, f'Layer {i+1}', fontsize=8, ha='center', color='#0D47A1')
        rect = patches.Rectangle((4.3, 5.72 - i*0.35), 0.6, 0.12,
                                 linewidth=1, edgecolor='#1565C0', facecolor='#E3F2FD')
        ax.add_patch(rect)

    # ===== DECODER (bottom half) =====
    ax.text(9.2, 7.2, 'DECODER', fontsize=12, fontweight='bold', color='#2E7D32')
    ax.text(9.2, 6.85, '(Multi-layer GRU)', fontsize=10, color='gray')

    # Embedding box
    box4 = FancyBboxPatch((8.5, 6.1), 2.2, 0.65,
                          boxstyle="round,pad=0.05", facecolor='#C8E6C9', edgecolor='#2E7D32', linewidth=1.5)
    ax.add_patch(box4)
    ax.text(9.6, 6.43, 'Embedding', fontsize=10, ha='center', va='center')
    ax.text(9.6, 6.15, '(vocab → embed)', fontsize=8, ha='center', va='center', color='gray')

    # Concat box
    box5 = FancyBboxPatch((11.5, 6.1), 2.0, 0.65,
                          boxstyle="round,pad=0.05", facecolor='#FFF9C4', edgecolor='#F9A825', linewidth=1.5)
    ax.add_patch(box5)
    ax.text(12.5, 6.43, 'Concat', fontsize=10, ha='center', va='center')
    ax.text(12.5, 6.15, '[emb; C]', fontsize=8, ha='center', va='center', color='gray')

    # Arrow: h_T → concat
    ax.annotate('', xy=(12.7, 6.2), xytext=(8.3, 6.2),
                arrowprops=dict(arrowstyle='->', color='#FF6F00', lw=2,
                              connectionstyle='arc3,rad=0.2'))
    ax.text(10.0, 6.75, 'C (context)', fontsize=8, ha='center', color='#FF6F00', fontweight='bold')

    # Decoder GRU
    box6 = FancyBboxPatch((8.5, 5.1), 2.2, 0.65,
                          boxstyle="round,pad=0.05", facecolor='#A5D6A7', edgecolor='#2E7D32', linewidth=1.5)
    ax.add_patch(box6)
    ax.text(9.6, 5.43, 'GRU', fontsize=10, ha='center', va='center')
    ax.text(9.6, 5.15, '(embed+hidden → h)', fontsize=8, ha='center', va='center', color='gray')

    # Arrow: concat → GRU
    ax.annotate('', xy=(9.6, 5.75), xytext=(9.6, 6.1),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text(9.9, 5.95, '×T', fontsize=8, ha='center', color='gray')

    # Dense (output projection)
    box7 = FancyBboxPatch((8.5, 4.1), 2.2, 0.65,
                          boxstyle="round,pad=0.05", facecolor='#FFCC80', edgecolor='#E65100', linewidth=1.5)
    ax.add_patch(box7)
    ax.text(9.6, 4.43, 'Dense', fontsize=10, ha='center', va='center')
    ax.text(9.6, 4.15, '(hidden → vocab)', fontsize=8, ha='center', va='center', color='gray')

    # Arrow: GRU → Dense
    ax.annotate('', xy=(9.6, 4.75), xytext=(9.6, 5.1),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Arrow: Dense → output
    ax.annotate('', xy=(11.5, 4.42), xytext=(10.7, 4.42),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text(11.0, 4.6, 'logits', fontsize=8, ha='center', color='gray')

    # ===== CONNECTOR: Encoder → Decoder =====
    ax.annotate('', xy=(8.3, 6.42), xytext=(7.4, 6.42),
                arrowprops=dict(arrowstyle='->', color='#FF6F00', lw=2))
    ax.text(7.85, 6.6, 'h_T', fontsize=9, ha='center', color='#FF6F00', fontweight='bold')

    # ===== SIDE LABELS =====
    # Shape labels
    ax.text(0.3, 5.9, 'Input:\n(batch,T)', fontsize=8, ha='center', color='#616161')
    ax.text(2.5, 5.9, '(T,B,E)', fontsize=8, ha='center', color='#616161')
    ax.text(5.5, 5.9, '(T,B,h)', fontsize=8, ha='center', color='#616161')
    ax.text(8.3, 5.9, '(B,T)', fontsize=8, ha='center', color='#616161')
    ax.text(11.3, 5.9, '(T,B,E+h)', fontsize=8, ha='center', color='#616161')
    ax.text(13.8, 5.9, '(T,B,h)', fontsize=8, ha='center', color='#616161')
    ax.text(11.8, 3.9, '(B,T,V)', fontsize=8, ha='center', color='#616161')

    # Legend
    legend_items = [
        mpatches.Patch(facecolor='#BBDEFB', edgecolor='#1565C0', label='Encoder'),
        mpatches.Patch(facecolor='#C8E6C9', edgecolor='#2E7D32', label='Decoder'),
        mpatches.Patch(facecolor='#FFF9C4', edgecolor='#F9A825', label='Concat'),
        mpatches.Patch(facecolor='#FFCC80', edgecolor='#E65100', label='Output'),
    ]
    ax.legend(handles=legend_items, loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'seq2seq-layers.svg', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved seq2seq-layers.svg")


# =========================================================================
# Fig 2: teacher_forcing.svg - Teacher Forcing vs Free-running
# =========================================================================
def fig_teacher_forcing():
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Teacher Forcing vs Free-running — Hai Chiến lược Training Decoder', fontsize=14, fontweight='bold')

    # ===== LEFT: Teacher Forcing =====
    ax = axes[0]
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Teacher Forcing (Training)', fontsize=12, fontweight='bold', color='#2E7D32', pad=10)

    # Source context
    rect_s = FancyBboxPatch((0.2, 8.5), 3.5, 0.8,
                             boxstyle="round,pad=0.1", facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=2)
    ax.add_patch(rect_s)
    ax.text(2.0, 8.9, 'ENCODER', fontsize=10, fontweight='bold', ha='center')
    ax.text(2.0, 8.65, 'I → love → you → h_T', fontsize=9, ha='center', color='gray')
    ax.annotate('', xy=(7.5, 8.3), xytext=(3.7, 8.3),
                arrowprops=dict(arrowstyle='->', color='#FF6F00', lw=2))
    ax.text(5.7, 8.5, 'h_T', fontsize=9, ha='center', color='#FF6F00', fontweight='bold')

    # Decoder boxes
    dec_labels = ['<bos>', 'je', "suis", 'content', '.']
    dec_gts = ['je', "suis", 'content', '.', '<eos>']
    colors = ['#C8E6C9', '#A5D6A7', '#A5D6A7', '#A5D6A7', '#A5D6A7']

    for i, (inp, gt, col) in enumerate(zip(dec_labels, dec_gts, colors)):
        x = 4.5 + i * 1.4

        # Decoder box
        rect = FancyBboxPatch((x, 6.0), 1.2, 0.8,
                              boxstyle="round,pad=0.05", facecolor=col, edgecolor='#2E7D32', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.6, 6.4, f'input={inp}', fontsize=8, ha='center', va='center')

        # GT label (above)
        rect_gt = FancyBboxPatch((x, 7.2), 1.2, 0.5,
                                 boxstyle="round,pad=0.05", facecolor='#FFF9C4', edgecolor='#F9A825', linewidth=1.5)
        ax.add_patch(rect_gt)
        ax.text(x + 0.6, 7.45, f'GT={gt}', fontsize=8, ha='center', va='center', fontweight='bold')

        # Arrow from GT to next decoder
        if i < 4:
            ax.annotate('', xy=(x+1.4, 6.8), xytext=(x+1.2, 7.2),
                        arrowprops=dict(arrowstyle='->', color='#FF6F00', lw=1.5))
            ax.text(x+1.3, 7.05, 'GT', fontsize=7, ha='center', color='#FF6F00')

        # Prediction arrow
        ax.annotate('', xy=(x + 0.6, 5.6), xytext=(x + 0.6, 5.95),
                    arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=1))
        ax.text(x + 0.6, 5.4, f'p={gt}', fontsize=7, ha='center', color='#D32F2F')

    # Legend
    ax.text(8.5, 6.9, 'GT: Ground Truth', fontsize=9, color='#FF6F00', fontweight='bold')
    ax.text(8.5, 6.5, '→ Input của step tiếp', fontsize=9, color='#FF6F00')
    ax.text(8.5, 5.5, '→ Loss = CE(p, GT)', fontsize=9, color='#D32F2F')
    ax.text(8.5, 4.5, '✓ Gradient ổn định\n✓ Hội tụ nhanh\n✗ Exposure bias', fontsize=9, va='top')

    # ===== RIGHT: Free-running =====
    ax = axes[1]
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Free-running (Training khó)', fontsize=12, fontweight='bold', color='#C62828', pad=10)

    # Source context
    rect_s = FancyBboxPatch((0.2, 8.5), 3.5, 0.8,
                             boxstyle="round,pad=0.1", facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=2)
    ax.add_patch(rect_s)
    ax.text(2.0, 8.9, 'ENCODER', fontsize=10, fontweight='bold', ha='center')
    ax.text(2.0, 8.65, 'I → love → you → h_T', fontsize=9, ha='center', color='gray')
    ax.annotate('', xy=(7.5, 8.3), xytext=(3.7, 8.3),
                arrowprops=dict(arrowstyle='->', color='#FF6F00', lw=2))
    ax.text(5.7, 8.5, 'h_T', fontsize=9, ha='center', color='#FF6F00', fontweight='bold')

    # Decoder boxes
    dec_preds = ['<bos>', 'je', "aime", '!', '<eos>']
    dec_gts = ['je', "suis", 'content', '.', '<eos>']

    for i, (inp, pred, gt, col) in enumerate(zip(dec_labels, dec_preds, dec_gts, colors)):
        x = 4.5 + i * 1.4
        correct = pred == gt

        rect = FancyBboxPatch((x, 6.0), 1.2, 0.8,
                              boxstyle="round,pad=0.05", facecolor=col if correct else '#FFCDD2',
                              edgecolor='#C62828' if not correct else '#2E7D32', linewidth=2 if not correct else 1.5)
        ax.add_patch(rect)
        ax.text(x + 0.6, 6.4, f'input={inp}', fontsize=8, ha='center', va='center')

        # Predicted label (above)
        rect_gt = FancyBboxPatch((x, 7.2), 1.2, 0.5,
                                 boxstyle="round,pad=0.05", facecolor='#FFCDD2', edgecolor='#C62828', linewidth=1.5)
        ax.add_patch(rect_gt)
        ax.text(x + 0.6, 7.45, f'pred={pred}', fontsize=8, ha='center', va='center', fontweight='bold',
               color='#C62828')

        # Arrow: predicted to next input
        if i < 4:
            ax.annotate('', xy=(x+1.4, 6.8), xytext=(x+1.2, 7.2),
                        arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.5))
            ax.text(x+1.3, 7.05, 'pred', fontsize=7, ha='center', color='#C62828')

        ax.annotate('', xy=(x + 0.6, 5.6), xytext=(x + 0.6, 5.95),
                    arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=1))
        ax.text(x + 0.6, 5.4, f'GT={gt}', fontsize=7, ha='center', color='#388E3C')

    ax.text(8.5, 6.9, 'pred: Predicted', fontsize=9, color='#C62828', fontweight='bold')
    ax.text(8.5, 6.5, '→ Input = prev pred', fontsize=9, color='#C62828')
    ax.text(8.5, 5.5, '→ Loss = CE(p, GT)', fontsize=9, color='#D32F2F')
    ax.text(8.5, 4.5, '✗ Gradient unstable\n✗ Có thể diverge\n✓ Giống inference thật', fontsize=9, va='top', color='#C62828')

    # Arrows between decoder steps
    for i in range(4):
        x = 4.5 + i * 1.4
        ax.annotate('', xy=(x+1.4, 6.4), xytext=(x+1.2, 6.4),
                    arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1, linestyle='dashed'))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'teacher_forcing.svg', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved teacher_forcing.svg")


# =========================================================================
# Fig 3: greedy_decoding.svg - Greedy Decoding Process
# =========================================================================
def fig_greedy_decoding():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')
    ax.set_title('Greedy Decoding — Sinh từng token bằng argmax', fontsize=14, fontweight='bold', pad=15)

    # Source
    rect_src = FancyBboxPatch((0.3, 7.5), 4.0, 1.2,
                               boxstyle="round,pad=0.1", facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=2)
    ax.add_patch(rect_src)
    ax.text(2.3, 8.35, 'ENCODER', fontsize=11, fontweight='bold', ha='center')
    ax.text(2.3, 8.0, '"I love you"', fontsize=10, ha='center', style='italic')

    # Encoder → context
    ax.annotate('', xy=(5.0, 8.0), xytext=(4.3, 8.0),
                arrowprops=dict(arrowstyle='->', color='#FF6F00', lw=2))
    ax.text(4.6, 8.2, 'h_T', fontsize=9, ha='center', color='#FF6F00', fontweight='bold')

    # Context
    rect_ctx = FancyBboxPatch((5.0, 7.5), 1.5, 1.2,
                              boxstyle="round,pad=0.1", facecolor='#FFF9C4', edgecolor='#F9A825', linewidth=2)
    ax.add_patch(rect_ctx)
    ax.text(5.75, 8.35, 'C', fontsize=12, fontweight='bold', ha='center')
    ax.text(5.75, 8.0, '(cố định)', fontsize=8, ha='center', color='gray')

    # Decoder steps
    steps = [
        {'step': 1, 'inp': '<bos>', 'probs': [('je', 0.45), ('il', 0.20), ('elle', 0.15), ('un', 0.10), ('...', 0.10)], 'pred': 'je', 'correct': True},
        {'step': 2, 'inp': 'je', 'probs': [("aime", 0.55), ('suis', 0.20), ('sont', 0.10), ('...', 0.15)], 'pred': 'aime', 'correct': False},
        {'step': 3, 'inp': "aime", 'probs': [('.', 0.40), ('<eos>', 0.30), ('...', 0.30)], 'pred': '.', 'correct': False},
        {'step': 4, 'inp': '.', 'probs': [('<eos>', 0.50), ('...', 0.50)], 'pred': '<eos>', 'correct': True},
    ]

    step_x_start = 7.5
    step_w = 1.4
    step_gap = 0.1

    for i, s in enumerate(steps):
        x = step_x_start + i * (step_w + step_gap)
        is_eos = s['pred'] == '<eos>'

        # Decoder box
        bg = '#C8E6C9' if s['correct'] else '#FFCDD2'
        edge = '#2E7D32' if s['correct'] else '#C62828'
        rect = FancyBboxPatch((x, 6.5), step_w, 1.5,
                              boxstyle="round,pad=0.08", facecolor=bg, edgecolor=edge, linewidth=2 if not s['correct'] else 1.5)
        ax.add_patch(rect)
        ax.text(x + step_w/2, 7.55, f'Step {s["step"]}', fontsize=9, ha='center', fontweight='bold')
        ax.text(x + step_w/2, 7.25, f'in={s["inp"]}', fontsize=8, ha='center', color='#37474F')
        ax.text(x + step_w/2, 6.9, f'→ {s["pred"]}', fontsize=9, ha='center', fontweight='bold', color=edge)

        # Probability bars
        bar_y = 5.5 - i * 0.9
        bar_h = 0.35
        for j, (token, prob) in enumerate(s['probs'][:3]):
            bar_x = x + 0.05
            bar_width = (step_w - 0.1) * prob / max(s['probs'][0][1], 0.01)
            rect_bar = patches.Rectangle((bar_x, bar_y - j * (bar_h + 0.03)),
                                        bar_width, bar_h,
                                        linewidth=0.5, edgecolor='gray', facecolor='#90CAF9')
            ax.add_patch(rect_bar)
            ax.text(bar_x + bar_width + 0.05, bar_y - j*(bar_h+0.03) + bar_h/2,
                   f'{token}={prob:.2f}', fontsize=7, va='center', color='#0D47A1')

        # "ARGMAX" label
        ax.text(x + step_w/2, 4.7, 'argmax', fontsize=7, ha='center', color='#D32F2F', fontstyle='italic')

        # Arrow to next step
        if i < 3:
            ax.annotate('', xy=(x+step_w+step_gap, 7.05), xytext=(x+step_w, 7.05),
                        arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.5))

    # Arrow: context → each step
    for i in range(4):
        x = step_x_start + i * (step_w + step_gap) + step_w/2
        ax.annotate('', xy=(x, 7.8), xytext=(6.5, 8.0),
                    arrowprops=dict(arrowstyle='->', color='#FF6F00', lw=1,
                                  connectionstyle=f'arc3,rad=-0.15'))
        if i > 0:
            ax.annotate('', xy=(x, 8.0), xytext=(x, 7.8),
                        arrowprops=dict(arrowstyle='->', color='#FF6F00', lw=0.5))

    # Output
    ax.annotate('', xy=(13.2, 7.05), xytext=(step_x_start + 3*(step_w+step_gap) + step_w, 7.05),
                arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=2))
    ax.text(13.5, 7.3, 'Output:', fontsize=9, fontweight='bold')
    ax.text(13.5, 7.0, '"je aime ."', fontsize=10, style='italic', color='#2E7D32')
    ax.text(13.5, 6.6, '(≠ ref)', fontsize=8, color='#C62828')

    # Bottleneck annotation
    ax.annotate('', xy=(6.0, 5.5), xytext=(5.75, 6.0),
                arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.5))
    ax.text(4.0, 5.2, 'CỐ ĐỊNH!', fontsize=9, ha='center', color='#C62828', fontweight='bold')
    ax.text(4.0, 4.9, 'Decoder không thể "nhìn lại"\ncâu nguồn một cách chọn lọc', fontsize=8, ha='center', color='#C62828')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'greedy_decoding.svg', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved greedy_decoding.svg")


# =========================================================================
# Fig 4: bleu_score.svg - BLEU Score Computation
# =========================================================================
def fig_bleu_score():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')
    ax.set_title('BLEU Score Computation — Chi tiết từng bước', fontsize=14, fontweight='bold', pad=15)

    # Reference
    rect_ref = FancyBboxPatch((0.3, 7.0), 4.5, 1.5,
                               boxstyle="round,pad=0.1", facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=2)
    ax.add_patch(rect_ref)
    ax.text(2.55, 8.0, 'REFERENCE', fontsize=11, fontweight='bold', ha='center', color='#2E7D32')
    ax.text(2.55, 7.6, 'il  est  calme  .', fontsize=11, ha='center', style='italic')
    ax.text(2.55, 7.25, '(độ dài r = 4)', fontsize=9, ha='center', color='gray')

    # Predicted
    rect_pred = FancyBboxPatch((0.3, 5.0), 4.5, 1.5,
                                boxstyle="round,pad=0.1", facecolor='#FFEBEE', edgecolor='#C62828', linewidth=2)
    ax.add_patch(rect_pred)
    ax.text(2.55, 6.0, 'PREDICTED', fontsize=11, fontweight='bold', ha='center', color='#C62828')
    ax.text(2.55, 5.6, 'elle  court  .', fontsize=11, ha='center', style='italic')
    ax.text(2.55, 5.25, '(độ dài c = 3)', fontsize=9, ha='center', color='gray')

    # Step 1: Token matching
    ax.text(5.5, 8.3, 'Bước 1: Precision n-grams', fontsize=11, fontweight='bold', ha='center')

    # n-gram table
    headers = ['n', 'Pred n-grams', 'Ref n-grams', 'Matches', 'p_n']
    col_x = [5.5, 6.8, 8.3, 10.0, 11.5]
    col_w = [0.8, 1.2, 1.2, 1.0, 1.0]
    row_h = 0.4

    # Header
    for h, cx in zip(headers, col_x):
        ax.text(cx, 7.9, h, fontsize=9, ha='center', fontweight='bold')
    ax.plot([5.1, 12.5], [7.75, 7.75], color='black', lw=1)

    data = [
        ('1', 'elle, court, .', 'il, est, calme, .', '1 (.)', '1/3=0.333'),
        ('2', 'elle court, court .', 'il est, est calme, calme .', '0', '0/2=0.000'),
        ('3', 'elle court .', 'il est calme .', '0', '0/1=0.000'),
    ]

    for i, row in enumerate(data):
        y = 7.4 - i * row_h
        for val, cx in zip(row, col_x):
            color = '#C62828' if '0' in val and i > 0 else 'black'
            ax.text(cx, y, val, fontsize=8, ha='center', va='center', color=color)

    # Step 2: Brevity Penalty
    ax.text(5.5, 6.1, 'Bước 2: Brevity Penalty', fontsize=11, fontweight='bold', ha='center')

    rect_bp = FancyBboxPatch((5.2, 4.5), 7.3, 1.3,
                              boxstyle="round,pad=0.1", facecolor='#FFF9C4', edgecolor='#F9A825', linewidth=1.5)
    ax.add_patch(rect_bp)
    ax.text(8.85, 5.5, 'c = 3, r = 4,  c < r', fontsize=10, ha='center')
    ax.text(8.85, 5.15, 'BP = exp(1 - r/c) = exp(1 - 4/3) = exp(-0.333) = 0.716', fontsize=9, ha='center')
    ax.text(8.85, 4.8, '→ Phạt ngắn!', fontsize=9, ha='center', color='#C62828', fontweight='bold')

    # Step 3: BLEU
    ax.text(5.5, 3.8, 'Bước 3: Tính BLEU', fontsize=11, fontweight='bold', ha='center')

    rect_bleu = FancyBboxPatch((5.2, 2.2), 7.3, 1.3,
                               boxstyle="round,pad=0.1", facecolor='#FFCDD2', edgecolor='#C62828', linewidth=2)
    ax.add_patch(rect_bleu)
    ax.text(8.85, 3.2, 'BLEU = BP × p₁^(1/2) × p₂^(1/4) × p₃^(1/8)', fontsize=10, ha='center', fontweight='bold')
    ax.text(8.85, 2.85, '= 0.716 × (0.333)^0.5 × 0 × 0', fontsize=10, ha='center')
    ax.text(8.85, 2.5, '= 0.716 × 0.577 × 0 × 0 = 0.000', fontsize=12, ha='center', color='#C62828', fontweight='bold')

    # Good example
    ax.text(13.0, 8.3, 'Ví dụ tốt:', fontsize=10, ha='center', fontweight='bold', color='#2E7D32')

    rect_good = FancyBboxPatch((12.5, 5.0), 1.3, 3.2,
                               boxstyle="round,pad=0.1", facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=2)
    ax.add_patch(rect_good)
    ax.text(13.15, 7.8, 'go .', fontsize=9, ha='center', fontweight='bold')
    ax.text(13.15, 7.5, '↓', fontsize=9, ha='center')
    ax.text(13.15, 7.2, 'va !', fontsize=9, ha='center', style='italic')
    ax.text(13.15, 6.8, 'BLEU=1.0', fontsize=10, ha='center', color='#2E7D32', fontweight='bold')
    ax.text(13.15, 6.3, 'i lost .', fontsize=9, ha='center', fontweight='bold')
    ax.text(13.15, 6.0, '↓', fontsize=9, ha='center')
    ax.text(13.15, 5.7, "j'ai perdu .", fontsize=9, ha='center', style='italic')
    ax.text(13.15, 5.3, 'BLEU=1.0', fontsize=10, ha='center', color='#2E7D32', fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'bleu_score.svg', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved bleu_score.svg")


# =========================================================================
# Fig 5: masked_loss.svg - Masked Loss Visualization
# =========================================================================
def fig_masked_loss():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Masked Loss — Bỏ qua Padding Tokens trong Cross-Entropy', fontsize=14, fontweight='bold', pad=15)

    # Batch example
    ax.text(1.0, 7.3, 'Batch Example (batch_size=3, num_steps=6)', fontsize=11, fontweight='bold', ha='left')

    # Sentence boxes
    sentences = [
        {'tokens': ['je', 'suis', '<eos>', '<pad>', '<pad>', '<pad>'], 'valid': 3, 'color': '#E8F5E9'},
        {'tokens': ['il', 'est', 'calme', '<eos>', '<pad>', '<pad>'], 'valid': 4, 'color': '#E8F5E9'},
        {'tokens': ['merci', '<eos>', '<pad>', '<pad>', '<pad>', '<pad>'], 'valid': 2, 'color': '#E8F5E9'},
    ]

    for i, sent in enumerate(sentences):
        y = 6.2 - i * 1.0
        ax.text(0.5, y + 0.2, f'Sentence {i+1}:', fontsize=9, ha='left')

        for j, tok in enumerate(sent['tokens']):
            x = 2.5 + j * 1.0
            is_pad = tok == '<pad>'
            is_valid = j < sent['valid']

            bg = '#FFCDD2' if is_pad else '#C8E6C9' if is_valid else '#E3F2FD'
            edge = '#C62828' if is_pad else '#2E7D32'

            rect = FancyBboxPatch((x, y - 0.25), 0.85, 0.5,
                                  boxstyle="round,pad=0.05", facecolor=bg, edgecolor=edge, linewidth=1.5)
            ax.add_patch(rect)
            ax.text(x + 0.42, y, tok, fontsize=8, ha='center', va='center',
                   fontweight='bold' if is_pad else 'normal',
                   color='#C62828' if is_pad else 'black')

        ax.text(9.2, y, f'valid_len={sent["valid"]}', fontsize=8, ha='left', color='#2E7D32')

    # Loss computation
    ax.text(0.5, 3.5, 'Loss Computation:', fontsize=11, fontweight='bold')

    # Loss vector
    ax.text(0.5, 3.0, 'Loss per position:', fontsize=9)
    loss_vals = ['2.1', '0.3', '1.5', '0.0', '0.0', '0.0',  # sentence 1
                 '0.5', '0.1', '0.2', '1.8', '0.0', '0.0',  # sentence 2
                 '0.2', '0.0', '0.0', '0.0', '0.0', '0.0']  # sentence 3
    pad_idx = [3, 4, 5, 9, 10, 11, 13, 14, 15, 16, 17]  # 0-indexed from start of flat array

    flat_y = 2.5
    for j, lv in enumerate(loss_vals):
        x = 0.5 + j * 0.6
        is_pad = j in pad_idx
        bg = '#FFCDD2' if is_pad else '#FFCC80'
        ax.text(x + 0.3, flat_y, lv, fontsize=7, ha='center',
               color='#C62828' if is_pad else '#E65100', fontweight='bold' if not is_pad else 'normal')

    # Mask
    ax.text(0.5, 1.8, 'Mask (Y != <pad>):', fontsize=9)
    for j in range(len(loss_vals)):
        x = 0.5 + j * 0.6
        is_pad = j in pad_idx
        bg = '#FFCDD2' if is_pad else '#C8E6C9'
        rect = patches.Rectangle((x, 1.4), 0.55, 0.3, facecolor=bg, edgecolor='gray', lw=0.5)
        ax.add_patch(rect)
        ax.text(x + 0.27, 1.55, '0' if is_pad else '1', fontsize=8, ha='center', va='center',
               color='#C62828' if is_pad else '#2E7D32', fontweight='bold')

    # Formula
    rect_formula = FancyBboxPatch((0.3, 0.3), 8.5, 0.9,
                                   boxstyle="round,pad=0.1", facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=2)
    ax.add_patch(rect_formula)
    ax.text(4.5, 0.85, 'Loss = Σ(l × mask) / Σ(mask)', fontsize=12, ha='center', fontweight='bold')
    ax.text(4.5, 0.5, '= (valid losses summed) / (number of non-padding tokens)', fontsize=9, ha='center', color='gray')

    # Legend
    legend_x = 9.5
    legend_items = [
        mpatches.Patch(facecolor='#C8E6C9', edgecolor='#2E7D32', label='Valid token'),
        mpatches.Patch(facecolor='#FFCDD2', edgecolor='#C62828', label='<pad> token'),
        mpatches.Patch(facecolor='#FFCC80', edgecolor='#E65100', label='CE loss value'),
    ]
    ax.legend(handles=legend_items, loc='upper right', fontsize=9)

    # Count display
    ax.text(legend_x, 2.8, f'Valid tokens: 9', fontsize=9, color='#2E7D32', fontweight='bold')
    ax.text(legend_x, 2.5, f'Pad tokens: {len(loss_vals) - 9}', fontsize=9, color='#C62828', fontweight='bold')
    ax.text(legend_x, 2.0, f'Effective batch\nsize for loss:\n9 tokens', fontsize=9, va='top')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'masked_loss.svg', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved masked_loss.svg")


# =========================================================================
# Fig 6: seq2seq_architecture.svg - Full Seq2Seq Architecture Overview
# =========================================================================
def fig_seq2seq_overview():
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('Hình 10.7.1 — Sequence-to-Sequence Learning với RNN Encoder và RNN Decoder',
                 fontsize=13, fontweight='bold', pad=12)

    # ENCODER
    ax.text(1.5, 5.4, 'ENCODER', fontsize=12, fontweight='bold', color='#1565C0', ha='center')
    enc_steps = ['x₁', 'x₂', '...', 'xₜ', '...', 'x_T']
    for i, s in enumerate(enc_steps):
        x = 0.5 + i * 0.95
        rect = patches.Rectangle((x, 4.4), 0.8, 0.5, facecolor='#BBDEFB', edgecolor='#1565C0', lw=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.4, 4.65, s, fontsize=9, ha='center', va='center')

    # RNN cell (encoder)
    for i in range(len(enc_steps)):
        x = 0.9 + i * 0.95
        ax.annotate('', xy=(x, 4.1), xytext=(x, 4.4),
                    arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1))
        rect = patches.Rectangle((x, 3.7), 0.8, 0.4, facecolor='#90CAF9', edgecolor='#1565C0', lw=1)
        ax.add_patch(rect)
        ax.text(x + 0.4, 3.9, 'RNN', fontsize=7, ha='center', va='center')

    # Hidden states
    for i in range(len(enc_steps)):
        x = 0.9 + i * 0.95
        ax.annotate('', xy=(x, 3.4), xytext=(x, 3.7),
                    arrowprops=dict(arrowstyle='->', color='#64B5F6', lw=1))
        ax.text(x + 0.4, 3.25, f'h{i+1}', fontsize=7, ha='center', va='center', color='#1565C0')

    # Context vector
    ax.annotate('', xy=(6.8, 2.9), xytext=(6.0, 3.25),
                arrowprops=dict(arrowstyle='->', color='#FF6F00', lw=2))
    rect_ctx = FancyBboxPatch((6.8, 2.3), 1.2, 0.6,
                              boxstyle="round,pad=0.05", facecolor='#FFF9C4', edgecolor='#FF6F00', linewidth=2)
    ax.add_patch(rect_ctx)
    ax.text(7.4, 2.6, 'C', fontsize=12, fontweight='bold', ha='center', va='center')
    ax.text(7.4, 2.15, '= h_T', fontsize=8, ha='center', color='gray')

    # DECODER
    ax.text(9.5, 5.4, 'DECODER', fontsize=12, fontweight='bold', color='#2E7D32', ha='center')
    dec_steps = ['y₁', 'y₂', 'y₃', '...', 'yₜ₋₁', 'yₜ']
    dec_xs = [8.5, 9.5, 10.5, 11.5, 12.5, 13.5]
    for i, (s, x) in enumerate(zip(dec_steps, dec_xs)):
        rect = patches.Rectangle((x, 4.4), 0.8, 0.5, facecolor='#C8E6C9', edgecolor='#2E7D32', lw=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.4, 4.65, s, fontsize=9, ha='center', va='center')

    # RNN cell (decoder)
    for i, x in enumerate(dec_xs):
        ax.annotate('', xy=(x, 4.1), xytext=(x, 4.4),
                    arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1))
        rect = patches.Rectangle((x, 3.7), 0.8, 0.4, facecolor='#A5D6A7', edgecolor='#2E7D32', lw=1)
        ax.add_patch(rect)
        ax.text(x + 0.4, 3.9, 'RNN', fontsize=7, ha='center', va='center')

    # Hidden states decoder
    for i, x in enumerate(dec_xs):
        ax.annotate('', xy=(x, 3.4), xytext=(x, 3.7),
                    arrowprops=dict(arrowstyle='->', color='#81C784', lw=1))
        ax.text(x + 0.4, 3.25, f's{i+1}', fontsize=7, ha='center', va='center', color='#2E7D32')

    # Output
    for i, x in enumerate(dec_xs):
        ax.annotate('', xy=(x, 2.9), xytext=(x, 3.4),
                    arrowprops=dict(arrowstyle='->', color='#388E3C', lw=1))
        ax.text(x + 0.4, 2.75, f'ŷ{i+1}', fontsize=8, ha='center', va='center', color='#388E3C')

    # BOS token
    rect_bos = FancyBboxPatch((7.7, 4.4), 0.7, 0.5,
                              boxstyle="round,pad=0.05", facecolor='#FFF9C4', edgecolor='#F9A825', lw=1.5)
    ax.add_patch(rect_bos)
    ax.text(8.05, 4.65, '<bos>', fontsize=8, ha='center', va='center')
    ax.annotate('', xy=(8.5, 4.65), xytext=(8.4, 4.65),
                arrowprops=dict(arrowstyle='->', color='#F9A825', lw=1))

    # EOS token (output)
    rect_eos = FancyBboxPatch((14.3, 2.5), 0.7, 0.5,
                              boxstyle="round,pad=0.05", facecolor='#FFF9C4', edgecolor='#F9A825', lw=1.5)
    ax.add_patch(rect_eos)
    ax.text(14.65, 2.75, '<eos>', fontsize=8, ha='center', va='center')
    ax.annotate('', xy=(14.65, 2.9), xytext=(14.3, 2.9),
                arrowprops=dict(arrowstyle='->', color='#F9A825', lw=1))

    # Context → each decoder step
    for i, x in enumerate(dec_xs):
        ax.annotate('', xy=(x + 0.4, 4.1), xytext=(7.4, 2.9),
                    arrowprops=dict(arrowstyle='->', color='#FF6F00', lw=1,
                                  connectionstyle=f'arc3,rad=0.3'))

    # Legend
    items = [
        mpatches.Patch(facecolor='#BBDEFB', edgecolor='#1565C0', label='Encoder'),
        mpatches.Patch(facecolor='#C8E6C9', edgecolor='#2E7D32', label='Decoder'),
        mpatches.Patch(facecolor='#FFF9C4', edgecolor='#FF6F00', label='Context/Tokens'),
    ]
    ax.legend(handles=items, loc='lower left', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'seq2seq_architecture.svg', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved seq2seq_architecture.svg")


# =========================================================================
# Run all
# =========================================================================
if __name__ == '__main__':
    print("Generating figures for Buổi 48 - 10.7 Sequence-to-Sequence Learning...")
    fig_seq2seq_layers()
    fig_teacher_forcing()
    fig_greedy_decoding()
    fig_bleu_score()
    fig_masked_loss()
    fig_seq2seq_overview()
    print("\nAll figures saved to:", OUTPUT_DIR)
