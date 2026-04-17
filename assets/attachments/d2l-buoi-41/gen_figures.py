"""Generate figures for Buổi 41 — D2L 9.6 Concise Implementation of RNNs."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams["font.family"] = ["DejaVu Sans"]
plt.rcParams["font.size"] = 11

SAVE_DIR = "/home/sakana/Code/Obsidian-Vault/assets/attachments/d2l-buoi-41"


def fig1_scratch_vs_highlevel():
    """Compare scratch RNN vs nn.RNN architecture side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))

    # === Left: Scratch Implementation ===
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 12)
    ax1.set_title(
        "FROM SCRATCH (Buoi 40)", fontsize=14, fontweight="bold", color="#c0392b"
    )
    ax1.axis("off")

    # Boxes for scratch
    scratch_boxes = [
        (1, 10, 8, 1.2, "#fadbd8", "RNNLMScratch", "#c0392b"),
        (1.5, 8.2, 3.2, 1.0, "#f5cba7", "RNNScratch\nW_xh, W_hh, b_h", "#e67e22"),
        (5.3, 8.2, 3.5, 1.0, "#f5cba7", "Output Layer\nW_hq, b_q (manual)", "#e67e22"),
        (1.5, 6.2, 3.2, 1.0, "#d5f5e3", "one_hot()\nF.one_hot(X.T, |V|)", "#27ae60"),
        (5.3, 6.2, 3.5, 1.0, "#d5f5e3", "output_layer()\nH @ W_hq + b_q", "#27ae60"),
        (
            1.5,
            4.0,
            7.3,
            1.0,
            "#d6eaf8",
            "forward: for X in inputs:\n  state = tanh(X@W_xh + state@W_hh + b_h)",
            "#2980b9",
        ),
        (
            1.5,
            2.0,
            7.3,
            1.0,
            "#f9e79f",
            "clip_gradients(): manual global norm clipping",
            "#f39c12",
        ),
        (
            1.5,
            0.5,
            7.3,
            0.8,
            "#e8daef",
            "Tong params: W_xh + W_hh + b_h + W_hq + b_q\n= d*h + h*h + h + h*q + q",
            "#8e44ad",
        ),
    ]

    for x, y, w, h, color, text, edge in scratch_boxes:
        rect = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor=edge,
            linewidth=1.5,
        )
        ax1.add_patch(rect)
        ax1.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    # Arrows
    ax1.annotate(
        "",
        xy=(3.1, 8.2),
        xytext=(3.1, 7.4),
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.5),
    )
    ax1.annotate(
        "",
        xy=(7.0, 8.2),
        xytext=(7.0, 7.4),
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.5),
    )
    ax1.annotate(
        "",
        xy=(5, 6.2),
        xytext=(5, 5.2),
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.5),
    )
    ax1.annotate(
        "",
        xy=(5, 4.0),
        xytext=(5, 3.2),
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.5),
    )

    # === Right: High-level API ===
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 12)
    ax2.set_title(
        "HIGH-LEVEL API (Buoi 41)", fontsize=14, fontweight="bold", color="#2980b9"
    )
    ax2.axis("off")

    hl_boxes = [
        (1, 10, 8, 1.2, "#d6eaf8", "RNNLM", "#2980b9"),
        (1.5, 8.2, 3.2, 1.0, "#aed6f1", "RNN\nnn.RNN(num_inputs, h)", "#2471a3"),
        (5.3, 8.2, 3.5, 1.0, "#aed6f1", "Output Layer\nnn.LazyLinear(|V|)", "#2471a3"),
        (1.5, 6.2, 3.2, 1.0, "#d5f5e3", "one_hot()\n(ke thua tu Scratch)", "#27ae60"),
        (
            5.3,
            6.2,
            3.5,
            1.0,
            "#d5f5e3",
            "output_layer()\nself.linear(hiddens)",
            "#27ae60",
        ),
        (
            1.5,
            4.0,
            7.3,
            1.0,
            "#aed6f1",
            "forward: nn.RNN xu ly noi bo\n(cuDNN optimized, khong can viet loop)",
            "#2471a3",
        ),
        (
            1.5,
            2.0,
            7.3,
            1.0,
            "#f9e79f",
            "Trainer(gradient_clip_val=1)\nauto clip by framework",
            "#f39c12",
        ),
        (
            1.5,
            0.5,
            7.3,
            0.8,
            "#e8daef",
            "Tong params: TUONG TU scratch\nnhung nn.RNN quan ly tu dong",
            "#8e44ad",
        ),
    ]

    for x, y, w, h, color, text, edge in hl_boxes:
        rect = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor=edge,
            linewidth=1.5,
        )
        ax2.add_patch(rect)
        ax2.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    ax2.annotate(
        "",
        xy=(3.1, 8.2),
        xytext=(3.1, 7.4),
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.5),
    )
    ax2.annotate(
        "",
        xy=(7.0, 8.2),
        xytext=(7.0, 7.4),
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.5),
    )
    ax2.annotate(
        "",
        xy=(5, 6.2),
        xytext=(5, 5.2),
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.5),
    )
    ax2.annotate(
        "",
        xy=(5, 4.0),
        xytext=(5, 3.2),
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.5),
    )

    # Highlight differences
    for x, y, w, h in [
        (1.5, 8.2, 3.2, 1.0),
        (5.3, 8.2, 3.5, 1.0),
        (1.5, 4.0, 7.3, 1.0),
        (1.5, 2.0, 7.3, 1.0),
    ]:
        rect = mpatches.FancyBboxPatch(
            (x - 0.08, y - 0.08),
            w + 0.16,
            h + 0.16,
            boxstyle="round,pad=0.05",
            facecolor="none",
            edgecolor="#e74c3c",
            linewidth=2,
            linestyle="--",
        )
        ax2.add_patch(rect)

    ax2.text(
        9.5,
        7.5,
        "THAY DOI\n(vien do)",
        fontsize=8,
        color="#e74c3c",
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round", facecolor="#fadbd8", edgecolor="#e74c3c"),
    )

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/scratch_vs_highlevel.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved scratch_vs_highlevel.png")


def fig2_nn_rnn_internals():
    """Show what happens inside nn.RNN — cuDNN optimization."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title(
        "Ben trong nn.RNN: Tu Python loop -> cuDNN kernel",
        fontsize=14,
        fontweight="bold",
    )

    # Left side: Python loop (scratch)
    ax.text(
        2.5,
        9.3,
        "SCRATCH (Python loop)",
        fontsize=12,
        fontweight="bold",
        ha="center",
        color="#c0392b",
    )

    for i, t in enumerate(range(3)):
        y = 7.5 - i * 2.2
        # Step box
        rect = mpatches.FancyBboxPatch(
            (0.5, y),
            4,
            1.5,
            boxstyle="round,pad=0.1",
            facecolor="#fadbd8",
            edgecolor="#c0392b",
            linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(
            2.5, y + 1.0, f"Step t={t}", fontsize=10, fontweight="bold", ha="center"
        )
        ax.text(
            2.5,
            y + 0.4,
            f"state = tanh(X@W_xh\n  + state@W_hh + b_h)",
            fontsize=8,
            ha="center",
            va="center",
        )
        if i < 2:
            ax.annotate(
                "",
                xy=(2.5, y),
                xytext=(2.5, y - 0.5),
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2),
            )
            ax.text(
                3.5, y - 0.35, "serial", fontsize=8, color="#c0392b", fontstyle="italic"
            )

    # Arrow between sides
    ax.annotate(
        "",
        xy=(7.5, 5.5),
        xytext=(5.5, 5.5),
        arrowprops=dict(arrowstyle="->", color="#2980b9", lw=3),
    )
    ax.text(
        6.5,
        6.0,
        "nn.RNN\nthay the",
        fontsize=10,
        ha="center",
        fontweight="bold",
        color="#2980b9",
    )

    # Right side: cuDNN fused kernel
    ax.text(
        10.5,
        9.3,
        "nn.RNN (cuDNN kernel)",
        fontsize=12,
        fontweight="bold",
        ha="center",
        color="#2980b9",
    )

    # Single fused box
    rect = mpatches.FancyBboxPatch(
        (7.5, 2.8),
        6,
        5.5,
        boxstyle="round,pad=0.15",
        facecolor="#d6eaf8",
        edgecolor="#2980b9",
        linewidth=2,
    )
    ax.add_patch(rect)

    benefits = [
        "1. Fused CUDA kernel — 1 GPU call thay vi T calls",
        "2. Memory-efficient: khong luu intermediate Python objects",
        "3. Parallelized matrix ops trong moi step",
        "4. Optimized memory access patterns (coalesced)",
        "5. Support bidirectional, multi-layer, dropout",
    ]
    for i, b in enumerate(benefits):
        ax.text(10.5, 7.3 - i * 0.9, b, fontsize=9, ha="center", va="center")

    # Bottom: API comparison
    rect = mpatches.FancyBboxPatch(
        (0.5, 0.3),
        13,
        1.5,
        boxstyle="round,pad=0.1",
        facecolor="#fef9e7",
        edgecolor="#f39c12",
        linewidth=1.5,
    )
    ax.add_patch(rect)
    ax.text(
        7,
        1.5,
        "API COMPARISON",
        fontsize=11,
        fontweight="bold",
        ha="center",
        color="#f39c12",
    )
    ax.text(
        3.5,
        0.8,
        "Scratch: outputs, state = rnn(inputs, state)\n  inputs: (T, batch, |V|)  |  outputs: list of T tensors",
        fontsize=8,
        ha="center",
        family="monospace",
    )
    ax.text(
        10.5,
        0.8,
        "nn.RNN: output, h_n = rnn(inputs, h_0)\n  inputs: (T, batch, |V|)  |  output: (T, batch, h)",
        fontsize=8,
        ha="center",
        family="monospace",
    )

    plt.savefig(f"{SAVE_DIR}/nn_rnn_internals.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved nn_rnn_internals.png")


def fig3_code_comparison():
    """Side-by-side code diff — what changes, what stays."""
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 12)
    ax.axis("off")
    ax.set_title(
        "So sanh Code: 4 thay doi chinh tu Scratch -> High-level",
        fontsize=14,
        fontweight="bold",
    )

    changes = [
        {
            "title": "1. RNN Core",
            "scratch": "class RNNScratch(nn.Module):\n  W_xh = nn.Parameter(...)\n  W_hh = nn.Parameter(...)\n  b_h = nn.Parameter(...)\n  def forward: for X in inputs...",
            "highlevel": "class RNN(nn.Module):\n  self.rnn = nn.RNN(\n    num_inputs, num_hiddens)\n  def forward:\n    return self.rnn(inputs, H)",
            "color_s": "#fadbd8",
            "color_h": "#d6eaf8",
        },
        {
            "title": "2. Output Layer",
            "scratch": "W_hq = nn.Parameter(...)\nb_q = nn.Parameter(...)\ndef output_layer(rnn_outputs):\n  [H @ W_hq + b_q\n   for H in rnn_outputs]",
            "highlevel": "self.linear = nn.LazyLinear(\n  vocab_size)\ndef output_layer(hiddens):\n  return self.linear(\n    hiddens).swapaxes(0,1)",
            "color_s": "#fadbd8",
            "color_h": "#d6eaf8",
        },
        {
            "title": "3. Return format",
            "scratch": "rnn_outputs: list of T tensors\n  moi tensor shape (batch, h)\n\nCan stack/concat thu cong",
            "highlevel": "output: tensor (T, batch, h)\n  da stack san\n  + h_n: (1, batch, h)\n  [last hidden state]",
            "color_s": "#fef9e7",
            "color_h": "#d5f5e3",
        },
        {
            "title": "4. Gradient Clipping",
            "scratch": "clip_gradients(model, theta)\n  norm = sqrt(sum(p.grad**2))\n  if norm > theta:\n    p.grad *= theta/norm\n  [MANUAL]",
            "highlevel": "Trainer(\n  gradient_clip_val=1)\n\n[AUTOMATIC by framework]\n\ntorch.nn.utils.\n  clip_grad_norm_(...)",
            "color_s": "#fef9e7",
            "color_h": "#d5f5e3",
        },
    ]

    for i, ch in enumerate(changes):
        y = 10.5 - i * 2.7

        # Title
        ax.text(
            7.5,
            y + 0.6,
            ch["title"],
            fontsize=12,
            fontweight="bold",
            ha="center",
            bbox=dict(boxstyle="round", facecolor="#ecf0f1", edgecolor="gray"),
        )

        # Scratch box
        rect = mpatches.FancyBboxPatch(
            (0.3, y - 1.5),
            6.5,
            2.0,
            boxstyle="round,pad=0.1",
            facecolor=ch["color_s"],
            edgecolor="#c0392b",
            linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(
            3.55,
            y - 0.5,
            ch["scratch"],
            fontsize=7.5,
            ha="center",
            va="center",
            family="monospace",
        )

        # Arrow
        ax.annotate(
            "",
            xy=(7.8, y - 0.5),
            xytext=(7.1, y - 0.5),
            arrowprops=dict(arrowstyle="->", color="#27ae60", lw=2.5),
        )

        # High-level box
        rect = mpatches.FancyBboxPatch(
            (8.2, y - 1.5),
            6.5,
            2.0,
            boxstyle="round,pad=0.1",
            facecolor=ch["color_h"],
            edgecolor="#2980b9",
            linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(
            11.45,
            y - 0.5,
            ch["highlevel"],
            fontsize=7.5,
            ha="center",
            va="center",
            family="monospace",
        )

    # Legend
    ax.text(
        3.55,
        0.1,
        "SCRATCH (manual)",
        fontsize=10,
        fontweight="bold",
        ha="center",
        color="#c0392b",
    )
    ax.text(
        11.45,
        0.1,
        "HIGH-LEVEL (nn.RNN)",
        fontsize=10,
        fontweight="bold",
        ha="center",
        color="#2980b9",
    )

    plt.savefig(f"{SAVE_DIR}/code_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved code_comparison.png")


def fig4_output_shape_flow():
    """Show shape differences in data flow between scratch and nn.RNN."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    for ax, title, color, shapes, notes in [
        (
            ax1,
            "SCRATCH: Data Flow & Shapes",
            "#c0392b",
            [
                ("X\n(batch, T)", 0.5),
                ("one_hot\n(T, batch, |V|)", 2.5),
                ("RNN loop\nlist of T x\n(batch, h)", 5.0),
                ("output_layer\nlist -> stack\n(batch, T, |V|)", 7.8),
                ("loss\nreshape(-1, |V|)\nscalar", 10.5),
            ],
            [
                "F.one_hot(X.T, |V|)",
                "for X in inputs:\n  state = tanh(...)",
                "[H@W_hq+b_q\n for H in outputs]",
                "CE(reshape, Y)",
            ],
        ),
        (
            ax2,
            "HIGH-LEVEL: Data Flow & Shapes",
            "#2980b9",
            [
                ("X\n(batch, T)", 0.5),
                ("one_hot\n(T, batch, |V|)", 2.5),
                ("nn.RNN\noutput:\n(T, batch, h)", 5.0),
                ("nn.LazyLinear\nswapaxes\n(batch, T, |V|)", 7.8),
                ("loss\nreshape(-1, |V|)\nscalar", 10.5),
            ],
            [
                "F.one_hot(X.T, |V|)",
                "self.rnn(inputs, H)",
                "self.linear(hiddens)\n.swapaxes(0,1)",
                "CE(reshape, Y)",
            ],
        ),
    ]:
        ax.set_xlim(0, 13)
        ax.set_ylim(0, 3.5)
        ax.axis("off")
        ax.set_title(title, fontsize=12, fontweight="bold", color=color)

        bg_color = "#fadbd8" if color == "#c0392b" else "#d6eaf8"
        for i, (label, x) in enumerate(shapes):
            rect = mpatches.FancyBboxPatch(
                (x, 0.8),
                1.8,
                2.0,
                boxstyle="round,pad=0.1",
                facecolor=bg_color,
                edgecolor=color,
                linewidth=1.5,
            )
            ax.add_patch(rect)
            ax.text(
                x + 0.9,
                1.8,
                label,
                fontsize=8,
                ha="center",
                va="center",
                fontweight="bold",
            )
            if i < len(notes):
                if i < len(shapes) - 1:
                    next_x = shapes[i + 1][1]
                    ax.annotate(
                        "",
                        xy=(next_x, 1.8),
                        xytext=(x + 1.8, 1.8),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                    )
                    mid = (x + 1.8 + next_x) / 2
                    ax.text(
                        mid,
                        2.6,
                        notes[i],
                        fontsize=6.5,
                        ha="center",
                        va="center",
                        fontstyle="italic",
                        color="gray",
                    )

    # Highlight difference
    for ax_ref in [ax2]:
        for x in [5.0, 7.8]:
            rect = mpatches.FancyBboxPatch(
                (x - 0.1, 0.7),
                2.0,
                2.2,
                boxstyle="round,pad=0.05",
                facecolor="none",
                edgecolor="#e74c3c",
                linewidth=2,
                linestyle="--",
            )
            ax_ref.add_patch(rect)

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/output_shape_flow.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved output_shape_flow.png")


if __name__ == "__main__":
    fig1_scratch_vs_highlevel()
    fig2_nn_rnn_internals()
    fig3_code_comparison()
    fig4_output_shape_flow()
    print("All figures generated!")
