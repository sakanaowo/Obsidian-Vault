"""Tạo hình minh họa cho Buổi 40 — 9.5 RNN Implementation from Scratch."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = "/home/sakana/Code/Obsidian-Vault/assets/attachments/d2l-buoi-40/"


# ============================================================
# Hình 1: One-Hot Encoding cho Character-Level LM
# ============================================================
def fig1_onehot():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # === Panel A: Token indices → One-hot vectors ===
    ax = axes[0]
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.5, 6.5)
    ax.set_aspect("equal")
    ax.set_title("Token Index → One-Hot Vector", fontsize=13, fontweight="bold")

    vocab = ["a", "c", "e", "h", "i", "m", "n"]
    word = "machine"
    indices = [5, 1, 2, 3, 4, 6, 2]  # m=5, a=1, c=2, h=3, i=4, n=6, e=2

    # Draw one-hot matrix
    for t, (ch, idx) in enumerate(zip(word, indices)):
        for v in range(7):
            color = "#FF6B6B" if v == idx else "#F0F0F0"
            rect = mpatches.FancyBboxPatch(
                (t - 0.35, 6 - v - 0.35),
                0.7,
                0.7,
                boxstyle="round,pad=0.05",
                facecolor=color,
                edgecolor="gray",
                linewidth=0.5,
            )
            ax.add_patch(rect)
            val = "1" if v == idx else "0"
            ax.text(
                t,
                6 - v,
                val,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold" if v == idx else "normal",
                color="white" if v == idx else "#999",
            )

        # char label on top
        ax.text(
            t,
            7.0,
            f"'{ch}'\nidx={idx}",
            ha="center",
            va="center",
            fontsize=9,
            color="#333",
        )

    # vocab labels on left
    for v in range(7):
        ax.text(
            -1.0, 6 - v, f"'{vocab[v]}' ({v})", ha="center", va="center", fontsize=9
        )

    ax.set_xlim(-1.8, 7)
    ax.set_ylim(-1, 8)
    ax.axis("off")

    # === Panel B: Shape transformation ===
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Shape: (batch, T) → (T, batch, |V|)", fontsize=13, fontweight="bold")

    # Input shape
    box_props = dict(
        boxstyle="round,pad=0.6", facecolor="#E8F4FD", edgecolor="#2196F3", linewidth=2
    )
    ax.text(
        2,
        6.5,
        "Input X\n(batch_size, num_steps)\nVD: (2, 7)",
        fontsize=11,
        ha="center",
        va="center",
        bbox=box_props,
    )

    # Arrow
    ax.annotate(
        "",
        xy=(2, 4.8),
        xytext=(2, 5.5),
        arrowprops=dict(arrowstyle="->", lw=2, color="#FF9800"),
    )
    ax.text(
        3.8,
        5.15,
        "F.one_hot() + transpose",
        fontsize=10,
        color="#FF9800",
        fontweight="bold",
    )

    # Output shape
    box_props2 = dict(
        boxstyle="round,pad=0.6", facecolor="#FFF3E0", edgecolor="#FF9800", linewidth=2
    )
    ax.text(
        2,
        3.8,
        "One-Hot Tensor\n(num_steps, batch_size, vocab_size)\nVD: (7, 2, 28)",
        fontsize=11,
        ha="center",
        va="center",
        bbox=box_props2,
    )

    # Arrow to RNN
    ax.annotate(
        "",
        xy=(2, 2.5),
        xytext=(2, 3.0),
        arrowprops=dict(arrowstyle="->", lw=2, color="#4CAF50"),
    )
    ax.text(3.5, 2.75, "Loop qua dim 0\n(tung time step)", fontsize=10, color="#4CAF50")

    # RNN box
    box_props3 = dict(
        boxstyle="round,pad=0.6", facecolor="#E8F5E9", edgecolor="#4CAF50", linewidth=2
    )
    ax.text(
        2,
        1.5,
        "RNN Forward\nfor X in inputs:\n  state = tanh(X@W_xh + state@W_hh + b_h)",
        fontsize=10,
        ha="center",
        va="center",
        bbox=box_props3,
        family="monospace",
    )

    # Why transpose?
    box_note = dict(
        boxstyle="round,pad=0.4",
        facecolor="#FFFDE7",
        edgecolor="#FFC107",
        linewidth=1.5,
    )
    ax.text(
        7.5,
        5.5,
        "Tai sao transpose?\nDe loop theo time step\n(dim 0 = num_steps)\nthuan tien cho\nRNN forward",
        fontsize=9,
        ha="center",
        va="center",
        bbox=box_note,
    )

    plt.tight_layout()
    path = OUT + "onehot_encoding.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# Hình 2: Gradient Clipping Visualization
# ============================================================
def fig2_gradient_clipping():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # === Panel A: Gradient clipping concept ===
    ax = axes[0]
    ax.set_title(
        "Gradient Clipping: Chieu gradient len hinh cau", fontsize=12, fontweight="bold"
    )

    theta = 1.0  # clip threshold
    circle = plt.Circle(
        (0, 0),
        theta,
        fill=False,
        color="#4CAF50",
        linewidth=2,
        linestyle="--",
        label=f"||g|| = theta = {theta}",
    )
    ax.add_patch(circle)

    # Original gradients (some inside, some outside the ball)
    gradients = [
        (0.3, 0.5, "#2196F3", "g1: ||g||<theta\n(giu nguyen)"),
        (1.5, 2.0, "#F44336", "g2: ||g||>theta\n(bi cat)"),
        (-1.8, 1.2, "#FF9800", "g3: ||g||>theta\n(bi cat)"),
        (-0.4, -0.6, "#2196F3", "g4: ||g||<theta\n(giu nguyen)"),
    ]

    for gx, gy, color, label in gradients:
        norm = np.sqrt(gx**2 + gy**2)
        ax.annotate(
            "",
            xy=(gx, gy),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", lw=2, color=color, alpha=0.3),
        )

        if norm > theta:
            # Clipped version
            scale = theta / norm
            cgx, cgy = gx * scale, gy * scale
            ax.annotate(
                "",
                xy=(cgx, cgy),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", lw=2.5, color=color),
            )
            # Dashed line from clipped to original
            ax.plot([cgx, gx], [cgy, gy], "--", color=color, alpha=0.3, linewidth=1)
            ax.plot(gx, gy, "x", color=color, markersize=8, alpha=0.4)
        else:
            ax.annotate(
                "",
                xy=(gx, gy),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", lw=2.5, color=color),
            )

        ax.text(gx + 0.1, gy + 0.15, label, fontsize=8, color=color)

    ax.set_xlim(-2.8, 2.8)
    ax.set_ylim(-2.0, 2.8)
    ax.set_aspect("equal")
    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.axvline(x=0, color="gray", linewidth=0.5)
    ax.grid(True, alpha=0.2)

    # Legend
    ax.text(
        0,
        -1.5,
        r"$\mathbf{g} \leftarrow \min\left(1, \frac{\theta}{\|\mathbf{g}\|}\right) \cdot \mathbf{g}$",
        fontsize=14,
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="orange"
        ),
    )

    # === Panel B: Why gradient clipping is needed for RNN ===
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Tai sao RNN can Gradient Clipping?", fontsize=12, fontweight="bold")

    # Chain of W_hh multiplications
    y_pos = 6.5
    colors_chain = ["#E3F2FD", "#BBDEFB", "#90CAF9", "#64B5F6", "#42A5F5"]
    for i in range(5):
        x = 1 + i * 1.8
        box = dict(
            boxstyle="round,pad=0.3",
            facecolor=colors_chain[i],
            edgecolor="#1565C0",
            linewidth=1.5,
        )
        ax.text(
            x,
            y_pos,
            f"W_hh",
            fontsize=10,
            ha="center",
            va="center",
            bbox=box,
            fontweight="bold",
        )
        if i < 4:
            ax.annotate(
                "",
                xy=(x + 0.7, y_pos),
                xytext=(x + 1.1, y_pos),
                arrowprops=dict(arrowstyle="<-", lw=1.5, color="#1565C0"),
            )

    ax.text(
        5,
        7.3,
        "Backprop qua T time steps = nhan lien tiep T lan W_hh",
        fontsize=10,
        ha="center",
        color="#1565C0",
    )

    # Two scenarios
    # Exploding
    box_exp = dict(
        boxstyle="round,pad=0.4", facecolor="#FFEBEE", edgecolor="#F44336", linewidth=2
    )
    ax.text(
        2.5,
        4.5,
        "||W_hh|| > 1\nGradient BUNG NO\n(exploding)",
        fontsize=10,
        ha="center",
        va="center",
        bbox=box_exp,
        color="#C62828",
        fontweight="bold",
    )

    # Vanishing
    box_van = dict(
        boxstyle="round,pad=0.4", facecolor="#E8F5E9", edgecolor="#4CAF50", linewidth=2
    )
    ax.text(
        7.5,
        4.5,
        "||W_hh|| < 1\nGradient TRIET TIEU\n(vanishing)",
        fontsize=10,
        ha="center",
        va="center",
        bbox=box_van,
        color="#2E7D32",
        fontweight="bold",
    )

    # Solution
    box_sol = dict(
        boxstyle="round,pad=0.5", facecolor="#FFF8E1", edgecolor="#FF8F00", linewidth=2
    )
    ax.text(
        2.5,
        2.0,
        "Gradient Clipping\nGiai quyet EXPLODING\n(cat gradient khi qua lon)",
        fontsize=10,
        ha="center",
        va="center",
        bbox=box_sol,
        fontweight="bold",
    )

    box_sol2 = dict(
        boxstyle="round,pad=0.5", facecolor="#F3E5F5", edgecolor="#7B1FA2", linewidth=2
    )
    ax.text(
        7.5,
        2.0,
        "LSTM / GRU\nGiai quyet VANISHING\n(gating mechanisms)",
        fontsize=10,
        ha="center",
        va="center",
        bbox=box_sol2,
        fontweight="bold",
    )

    # Arrows
    ax.annotate(
        "",
        xy=(2.5, 2.8),
        xytext=(2.5, 3.7),
        arrowprops=dict(arrowstyle="->", lw=2, color="#FF8F00"),
    )
    ax.annotate(
        "",
        xy=(7.5, 2.8),
        xytext=(7.5, 3.7),
        arrowprops=dict(arrowstyle="->", lw=2, color="#7B1FA2"),
    )

    plt.tight_layout()
    path = OUT + "gradient_clipping.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# Hình 3: Full RNN Language Model Pipeline
# ============================================================
def fig3_rnn_lm_pipeline():
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title(
        "RNN Language Model Pipeline (from scratch)", fontsize=14, fontweight="bold"
    )

    # Input tokens
    tokens = ["m", "a", "c", "h", "i", "n"]
    targets = ["a", "c", "h", "i", "n", "e"]
    T = len(tokens)

    x_start = 1.5
    spacing = 2.2

    for t in range(T):
        x = x_start + t * spacing

        # Layer 1: Token input
        box_in = dict(
            boxstyle="round,pad=0.3",
            facecolor="#E8EAF6",
            edgecolor="#3F51B5",
            linewidth=1.5,
        )
        ax.text(
            x,
            0.5,
            f"'{tokens[t]}'",
            fontsize=11,
            ha="center",
            va="center",
            bbox=box_in,
            fontweight="bold",
        )

        # Arrow: input → one-hot
        ax.annotate(
            "",
            xy=(x, 1.2),
            xytext=(x, 0.9),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="#3F51B5"),
        )

        # Layer 2: One-hot
        box_oh = dict(
            boxstyle="round,pad=0.25",
            facecolor="#FFF3E0",
            edgecolor="#FF9800",
            linewidth=1.5,
        )
        ax.text(
            x,
            1.6,
            f"one-hot\n(|V|,)",
            fontsize=8,
            ha="center",
            va="center",
            bbox=box_oh,
        )

        # Arrow: one-hot → RNN
        ax.annotate(
            "",
            xy=(x, 2.3),
            xytext=(x, 2.0),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="#FF9800"),
        )

        # Layer 3: RNN cell
        box_rnn = dict(
            boxstyle="round,pad=0.35",
            facecolor="#E8F5E9",
            edgecolor="#4CAF50",
            linewidth=2,
        )
        ax.text(
            x,
            3.0,
            f"RNN\nH_{t+1}",
            fontsize=10,
            ha="center",
            va="center",
            bbox=box_rnn,
            fontweight="bold",
        )

        # Horizontal arrows between RNN cells (hidden state)
        if t < T - 1:
            x_next = x_start + (t + 1) * spacing
            ax.annotate(
                "",
                xy=(x_next - 0.7, 3.0),
                xytext=(x + 0.7, 3.0),
                arrowprops=dict(arrowstyle="->", lw=2, color="#F44336"),
            )
            ax.text(
                (x + x_next) / 2, 3.35, "H_t", fontsize=8, ha="center", color="#F44336"
            )

        # Arrow: RNN → output layer
        ax.annotate(
            "",
            xy=(x, 3.9),
            xytext=(x, 3.6),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="#4CAF50"),
        )

        # Layer 4: Output layer (W_hq)
        box_out = dict(
            boxstyle="round,pad=0.25",
            facecolor="#FCE4EC",
            edgecolor="#E91E63",
            linewidth=1.5,
        )
        ax.text(
            x,
            4.3,
            f"H@W_hq+b_q\n(|V|,)",
            fontsize=8,
            ha="center",
            va="center",
            bbox=box_out,
        )

        # Arrow: output → softmax
        ax.annotate(
            "",
            xy=(x, 5.0),
            xytext=(x, 4.7),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="#E91E63"),
        )

        # Layer 5: Softmax + prediction
        box_sm = dict(
            boxstyle="round,pad=0.25",
            facecolor="#F3E5F5",
            edgecolor="#9C27B0",
            linewidth=1.5,
        )
        ax.text(
            x,
            5.4,
            f"softmax\nP(next)",
            fontsize=8,
            ha="center",
            va="center",
            bbox=box_sm,
        )

        # Target
        box_tgt = dict(
            boxstyle="round,pad=0.3",
            facecolor="#FFCDD2",
            edgecolor="#F44336",
            linewidth=1.5,
        )
        ax.text(
            x,
            6.3,
            f"target: '{targets[t]}'",
            fontsize=9,
            ha="center",
            va="center",
            bbox=box_tgt,
            fontweight="bold",
        )

        # Arrow prediction → target (loss)
        ax.annotate(
            "",
            xy=(x, 6.0),
            xytext=(x, 5.7),
            arrowprops=dict(
                arrowstyle="->", lw=1.5, color="#9C27B0", linestyle="dashed"
            ),
        )

    # H_0 arrow
    box_h0 = dict(
        boxstyle="round,pad=0.3",
        facecolor="#ECEFF1",
        edgecolor="#607D8B",
        linewidth=1.5,
    )
    ax.text(0.3, 3.0, "H_0\n= 0", fontsize=9, ha="center", va="center", bbox=box_h0)
    ax.annotate(
        "",
        xy=(x_start - 0.7, 3.0),
        xytext=(0.7, 3.0),
        arrowprops=dict(arrowstyle="->", lw=2, color="#F44336"),
    )

    # Labels on left
    labels_y = [
        (0.5, "Input token", "#3F51B5"),
        (1.6, "One-hot\nencoding", "#FF9800"),
        (3.0, "RNN cell\n(tanh)", "#4CAF50"),
        (4.3, "Output\nlayer", "#E91E63"),
        (5.4, "Softmax", "#9C27B0"),
        (6.3, "Cross-entropy\nloss", "#F44336"),
    ]

    # Loss annotation
    ax.text(
        14.5,
        5.85,
        "Loss = CE(P, target)\n= -log P(target)",
        fontsize=10,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF9C4", edgecolor="#F9A825"),
    )

    plt.tight_layout()
    path = OUT + "rnn_lm_pipeline.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# Hình 4: Decoding Process (Warm-up + Generation)
# ============================================================
def fig4_decoding():
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title(
        "Decoding: Warm-up Phase + Generation Phase", fontsize=14, fontweight="bold"
    )

    # Prefix: "it ha" → Generate: "s the time "
    prefix = ["i", "t", " ", "h", "a"]
    generated = ["s", " ", "t", "h", "e"]

    x_start = 0.8
    spacing = 1.5

    # Warm-up phase
    for t, ch in enumerate(prefix):
        x = x_start + t * spacing

        # Input
        box_in = dict(
            boxstyle="round,pad=0.2",
            facecolor="#E3F2FD",
            edgecolor="#1565C0",
            linewidth=1.5,
        )
        ax.text(
            x,
            1.0,
            f"'{ch}'",
            fontsize=11,
            ha="center",
            va="center",
            bbox=box_in,
            fontweight="bold",
        )

        # RNN cell
        box_rnn = dict(
            boxstyle="round,pad=0.3",
            facecolor="#BBDEFB",
            edgecolor="#1565C0",
            linewidth=2,
        )
        ax.text(
            x,
            2.5,
            "RNN",
            fontsize=10,
            ha="center",
            va="center",
            bbox=box_rnn,
            fontweight="bold",
        )

        ax.annotate(
            "",
            xy=(x, 2.0),
            xytext=(x, 1.4),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="#1565C0"),
        )

        # Hidden state arrow
        if t < len(prefix) - 1:
            x_next = x_start + (t + 1) * spacing
            ax.annotate(
                "",
                xy=(x_next - 0.55, 2.5),
                xytext=(x + 0.55, 2.5),
                arrowprops=dict(arrowstyle="->", lw=2, color="#F44336"),
            )

        # Output (discarded during warm-up, except last)
        if t < len(prefix) - 1:
            ax.text(
                x,
                3.5,
                "X",
                fontsize=12,
                ha="center",
                va="center",
                color="#BDBDBD",
                fontweight="bold",
            )
            ax.text(x, 3.9, "(bo qua)", fontsize=7, ha="center", color="#BDBDBD")
        else:
            # Last prefix char → first prediction
            ax.annotate(
                "",
                xy=(x, 3.3),
                xytext=(x, 2.9),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#4CAF50"),
            )
            box_pred = dict(
                boxstyle="round,pad=0.2",
                facecolor="#C8E6C9",
                edgecolor="#4CAF50",
                linewidth=1.5,
            )
            ax.text(
                x,
                3.7,
                f"→ '{generated[0]}'",
                fontsize=10,
                ha="center",
                va="center",
                bbox=box_pred,
                fontweight="bold",
            )

    # Warm-up label
    warmup_x_mid = x_start + (len(prefix) - 1) * spacing / 2
    ax.annotate(
        "",
        xy=(x_start - 0.3, 0.3),
        xytext=(x_start + (len(prefix) - 1) * spacing + 0.3, 0.3),
        arrowprops=dict(arrowstyle="<->", lw=1.5, color="#1565C0"),
    )
    ax.text(
        warmup_x_mid,
        0.0,
        "WARM-UP PHASE\n(nap prefix, khong output)",
        fontsize=9,
        ha="center",
        va="center",
        color="#1565C0",
        fontweight="bold",
    )

    # Generation phase
    gen_start_x = x_start + len(prefix) * spacing

    # Arrow from last warm-up to first gen
    last_warmup_x = x_start + (len(prefix) - 1) * spacing
    ax.annotate(
        "",
        xy=(gen_start_x - 0.55, 2.5),
        xytext=(last_warmup_x + 0.55, 2.5),
        arrowprops=dict(arrowstyle="->", lw=2, color="#F44336"),
    )

    for t, ch in enumerate(generated):
        x = gen_start_x + t * spacing

        # Input (from previous prediction)
        box_in = dict(
            boxstyle="round,pad=0.2",
            facecolor="#FFF3E0",
            edgecolor="#FF6F00",
            linewidth=1.5,
        )
        prev_ch = generated[t - 1] if t > 0 else generated[0]
        if t == 0:
            prev_ch = generated[0]  # 's' (predicted from warm-up)
        ax.text(
            x,
            1.0,
            f"'{prev_ch}'",
            fontsize=11,
            ha="center",
            va="center",
            bbox=box_in,
            fontweight="bold",
        )

        # RNN cell
        box_rnn = dict(
            boxstyle="round,pad=0.3",
            facecolor="#FFE0B2",
            edgecolor="#FF6F00",
            linewidth=2,
        )
        ax.text(
            x,
            2.5,
            "RNN",
            fontsize=10,
            ha="center",
            va="center",
            bbox=box_rnn,
            fontweight="bold",
        )

        ax.annotate(
            "",
            xy=(x, 2.0),
            xytext=(x, 1.4),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="#FF6F00"),
        )

        # Hidden state arrow
        if t < len(generated) - 1:
            x_next = gen_start_x + (t + 1) * spacing
            ax.annotate(
                "",
                xy=(x_next - 0.55, 2.5),
                xytext=(x + 0.55, 2.5),
                arrowprops=dict(arrowstyle="->", lw=2, color="#F44336"),
            )

        # Prediction
        ax.annotate(
            "",
            xy=(x, 3.3),
            xytext=(x, 2.9),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="#4CAF50"),
        )
        if t < len(generated) - 1:
            next_ch = generated[t + 1]
            box_pred = dict(
                boxstyle="round,pad=0.2",
                facecolor="#C8E6C9",
                edgecolor="#4CAF50",
                linewidth=1.5,
            )
            ax.text(
                x,
                3.7,
                f"→ '{next_ch}'",
                fontsize=10,
                ha="center",
                va="center",
                bbox=box_pred,
                fontweight="bold",
            )
            # Feedback arrow (prediction → next input)
            ax.annotate(
                "",
                xy=(gen_start_x + (t + 1) * spacing, 1.4),
                xytext=(x + 0.3, 3.5),
                arrowprops=dict(
                    arrowstyle="->",
                    lw=1,
                    color="#4CAF50",
                    connectionstyle="arc3,rad=0.3",
                    linestyle="dashed",
                ),
            )
        else:
            box_pred = dict(
                boxstyle="round,pad=0.2",
                facecolor="#C8E6C9",
                edgecolor="#4CAF50",
                linewidth=1.5,
            )
            ax.text(
                x,
                3.7,
                f"→ ...",
                fontsize=10,
                ha="center",
                va="center",
                bbox=box_pred,
                fontweight="bold",
            )

    # Generation label
    gen_mid_x = gen_start_x + (len(generated) - 1) * spacing / 2
    ax.annotate(
        "",
        xy=(gen_start_x - 0.3, 0.3),
        xytext=(gen_start_x + (len(generated) - 1) * spacing + 0.3, 0.3),
        arrowprops=dict(arrowstyle="<->", lw=1.5, color="#FF6F00"),
    )
    ax.text(
        gen_mid_x,
        0.0,
        "GENERATION PHASE\n(output = argmax, dung lam input tiep)",
        fontsize=9,
        ha="center",
        va="center",
        color="#FF6F00",
        fontweight="bold",
    )

    # Result
    result_text = "Ket qua: 'it has the ...'"
    box_res = dict(
        boxstyle="round,pad=0.4", facecolor="#F1F8E9", edgecolor="#558B2F", linewidth=2
    )
    ax.text(
        8,
        5.2,
        result_text,
        fontsize=12,
        ha="center",
        va="center",
        bbox=box_res,
        fontweight="bold",
    )

    # Dividing line
    div_x = (last_warmup_x + gen_start_x) / 2
    ax.axvline(
        x=div_x,
        ymin=0.05,
        ymax=0.85,
        color="gray",
        linewidth=1.5,
        linestyle=":",
        alpha=0.5,
    )

    plt.tight_layout()
    path = OUT + "decoding_process.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# Hình 5: Training Loop Flow
# ============================================================
def fig5_training_loop():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Training Loop cua RNN Language Model", fontsize=14, fontweight="bold")

    steps = [
        (
            6,
            9.0,
            "1. Lay minibatch (X, Y)\nX: (batch, num_steps)\nY: (batch, num_steps)",
            "#E3F2FD",
            "#1565C0",
        ),
        (
            6,
            7.5,
            "2. One-hot encode\nX → (num_steps, batch, |V|)",
            "#FFF3E0",
            "#FF9800",
        ),
        (
            6,
            6.0,
            "3. RNN Forward\nfor X_t in inputs:\n  H = tanh(X_t@W_xh + H@W_hh + b_h)",
            "#E8F5E9",
            "#4CAF50",
        ),
        (6, 4.5, "4. Output Layer\nO_t = H_t @ W_hq + b_q", "#FCE4EC", "#E91E63"),
        (
            6,
            3.2,
            "5. Tinh Loss\nloss = CrossEntropy(O, Y)\nppl = exp(loss)",
            "#FFF9C4",
            "#F9A825",
        ),
        (
            6,
            1.8,
            "6. Backward + Gradient Clipping\nloss.backward()\nif ||g|| > theta: g *= theta/||g||",
            "#FFEBEE",
            "#F44336",
        ),
        (6, 0.5, "7. Update params\nparams -= lr * clipped_grad", "#F3E5F5", "#7B1FA2"),
    ]

    for x, y, text, bg, edge in steps:
        box = dict(boxstyle="round,pad=0.4", facecolor=bg, edgecolor=edge, linewidth=2)
        ax.text(
            x,
            y,
            text,
            fontsize=10,
            ha="center",
            va="center",
            bbox=box,
            family="monospace",
        )

    # Arrows between steps
    for i in range(len(steps) - 1):
        y_from = steps[i][1] - 0.45
        y_to = steps[i + 1][1] + 0.45
        ax.annotate(
            "",
            xy=(6, y_to),
            xytext=(6, y_from),
            arrowprops=dict(arrowstyle="->", lw=2, color="#455A64"),
        )

    # Highlight gradient clipping
    box_note = dict(
        boxstyle="round,pad=0.3",
        facecolor="#FFCDD2",
        edgecolor="#D32F2F",
        linewidth=1.5,
    )
    ax.text(
        10.5,
        1.8,
        "QUAN TRONG!\nKhong co grad clip\n→ gradient bung no\n→ loss = NaN",
        fontsize=9,
        ha="center",
        va="center",
        bbox=box_note,
        color="#C62828",
        fontweight="bold",
    )
    ax.annotate(
        "",
        xy=(8.5, 1.8),
        xytext=(9.3, 1.8),
        arrowprops=dict(arrowstyle="<-", lw=1.5, color="#D32F2F"),
    )

    plt.tight_layout()
    path = OUT + "training_loop.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    fig1_onehot()
    fig2_gradient_clipping()
    fig3_rnn_lm_pipeline()
    fig4_decoding()
    fig5_training_loop()
    print("Done — all 5 figures generated.")
