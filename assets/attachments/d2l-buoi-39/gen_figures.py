import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

outdir = os.path.dirname(os.path.abspath(__file__))

# ==========================================================================
# FIGURE 1: MLP vs RNN comparison
# ==========================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Left: MLP (no hidden state) ---
ax = axes[0]
ax.set_xlim(-1, 5)
ax.set_ylim(-0.5, 5)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("MLP (Khong co Hidden State)", fontsize=14, fontweight="bold", pad=15)

for i, label in enumerate(["$x_1$", "$x_2$", "$x_3$"]):
    circle = plt.Circle((i + 1, 0.5), 0.35, color="#3498DB", ec="black", lw=1.5)
    ax.add_patch(circle)
    ax.text(
        i + 1,
        0.5,
        label,
        ha="center",
        va="center",
        fontsize=12,
        color="white",
        fontweight="bold",
    )

for i in range(4):
    circle = plt.Circle((i + 0.5, 2.5), 0.35, color="#F39C12", ec="black", lw=1.5)
    ax.add_patch(circle)
    ax.text(
        i + 0.5,
        2.5,
        f"$h_{i+1}$",
        ha="center",
        va="center",
        fontsize=11,
        color="white",
        fontweight="bold",
    )

circle = plt.Circle((2, 4.2), 0.35, color="#E74C3C", ec="black", lw=1.5)
ax.add_patch(circle)
ax.text(
    2,
    4.2,
    "$O$",
    ha="center",
    va="center",
    fontsize=13,
    color="white",
    fontweight="bold",
)

for i in range(3):
    for j in range(4):
        ax.annotate(
            "",
            xy=(j + 0.5, 2.15),
            xytext=(i + 1, 0.85),
            arrowprops=dict(arrowstyle="->", color="gray", alpha=0.4, lw=0.8),
        )

for j in range(4):
    ax.annotate(
        "",
        xy=(2, 3.85),
        xytext=(j + 0.5, 2.85),
        arrowprops=dict(arrowstyle="->", color="gray", alpha=0.4, lw=0.8),
    )

ax.text(0.3, 1.5, "$W_{xh}$", fontsize=12, color="#2C3E50", fontstyle="italic")
ax.text(3.3, 3.3, "$W_{hq}$", fontsize=12, color="#2C3E50", fontstyle="italic")
ax.text(
    -0.5, 0.5, "Input\nLayer", fontsize=10, ha="center", va="center", color="#7F8C8D"
)
ax.text(
    -0.5, 2.5, "Hidden\nLayer", fontsize=10, ha="center", va="center", color="#7F8C8D"
)
ax.text(
    -0.5, 4.2, "Output\nLayer", fontsize=10, ha="center", va="center", color="#7F8C8D"
)
ax.text(
    2,
    -0.3,
    "$H = \\phi(X W_{xh} + b_h)$\n$O = H W_{hq} + b_q$",
    fontsize=11,
    ha="center",
    va="center",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#ECF0F1", edgecolor="#BDC3C7"),
)

# --- Right: RNN (with hidden state) ---
ax = axes[1]
ax.set_xlim(-1, 7)
ax.set_ylim(-1, 5)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("RNN (Co Hidden State)", fontsize=14, fontweight="bold", pad=15)

positions = [(1, 0), (3.5, 0), (6, 0)]
labels_x = ["$X_{t-1}$", "$X_t$", "$X_{t+1}$"]
labels_h = ["$H_{t-1}$", "$H_t$", "$H_{t+1}$"]
labels_o = ["$O_{t-1}$", "$O_t$", "$O_{t+1}$"]

for idx, (px, py) in enumerate(positions):
    circle = plt.Circle((px, 0.5), 0.35, color="#3498DB", ec="black", lw=1.5)
    ax.add_patch(circle)
    ax.text(
        px,
        0.5,
        labels_x[idx],
        ha="center",
        va="center",
        fontsize=10,
        color="white",
        fontweight="bold",
    )

    circle = plt.Circle((px, 2.5), 0.4, color="#F39C12", ec="black", lw=2)
    ax.add_patch(circle)
    ax.text(
        px,
        2.5,
        labels_h[idx],
        ha="center",
        va="center",
        fontsize=10,
        color="white",
        fontweight="bold",
    )

    circle = plt.Circle((px, 4.2), 0.35, color="#E74C3C", ec="black", lw=1.5)
    ax.add_patch(circle)
    ax.text(
        px,
        4.2,
        labels_o[idx],
        ha="center",
        va="center",
        fontsize=10,
        color="white",
        fontweight="bold",
    )

    ax.annotate(
        "",
        xy=(px, 2.1),
        xytext=(px, 0.85),
        arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.5),
    )
    ax.annotate(
        "",
        xy=(px, 3.85),
        xytext=(px, 2.9),
        arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.5),
    )

for idx in range(len(positions) - 1):
    ax.annotate(
        "",
        xy=(positions[idx + 1][0] - 0.4, 2.5),
        xytext=(positions[idx][0] + 0.4, 2.5),
        arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=2.5),
    )

ax.text(2.25, 3.05, "$W_{hh}$", fontsize=12, color="#E74C3C", fontweight="bold")
ax.text(4.75, 3.05, "$W_{hh}$", fontsize=12, color="#E74C3C", fontweight="bold")
ax.text(0.3, 1.5, "$W_{xh}$", fontsize=11, color="#2C3E50", fontstyle="italic")
ax.text(5.3, 3.5, "$W_{hq}$", fontsize=11, color="#2C3E50", fontstyle="italic")

for idx, (px, _) in enumerate(positions):
    ax.text(
        px,
        -0.3,
        f't = {["t-1","t","t+1"][idx]}',
        fontsize=10,
        ha="center",
        color="#7F8C8D",
    )

ax.text(
    3.5,
    -0.8,
    "Khac biet: $H_t = \\phi(X_t W_{xh} + H_{t-1} W_{hh} + b_h)$",
    fontsize=10,
    ha="center",
    va="center",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#FADBD8", edgecolor="#E74C3C"),
)

plt.tight_layout()
plt.savefig(
    os.path.join(outdir, "mlp_vs_rnn.png"),
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print("Figure 1 saved: mlp_vs_rnn.png")

# ==========================================================================
# FIGURE 2: RNN Unrolled over time
# ==========================================================================
fig, ax = plt.subplots(figsize=(18, 7))
ax.set_xlim(-2, 18)
ax.set_ylim(-1.5, 7)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title(
    "RNN Trien khai theo thoi gian (Unrolling)", fontsize=16, fontweight="bold", pad=20
)

T = 5
spacing = 3.2

for t in range(T):
    px = t * spacing + 1

    rect = mpatches.FancyBboxPatch(
        (px - 0.6, 0),
        1.2,
        0.8,
        boxstyle="round,pad=0.1",
        facecolor="#3498DB",
        edgecolor="black",
        lw=1.5,
    )
    ax.add_patch(rect)
    ax.text(
        px,
        0.4,
        f"$X_{{{t+1}}}$",
        ha="center",
        va="center",
        fontsize=12,
        color="white",
        fontweight="bold",
    )

    rect = mpatches.FancyBboxPatch(
        (px - 0.7, 2.5),
        1.4,
        1.0,
        boxstyle="round,pad=0.1",
        facecolor="#F39C12",
        edgecolor="black",
        lw=2,
    )
    ax.add_patch(rect)
    ax.text(
        px,
        3.0,
        f"$H_{{{t+1}}}$",
        ha="center",
        va="center",
        fontsize=13,
        color="white",
        fontweight="bold",
    )

    rect = mpatches.FancyBboxPatch(
        (px - 0.6, 5.0),
        1.2,
        0.8,
        boxstyle="round,pad=0.1",
        facecolor="#E74C3C",
        edgecolor="black",
        lw=1.5,
    )
    ax.add_patch(rect)
    ax.text(
        px,
        5.4,
        f"$O_{{{t+1}}}$",
        ha="center",
        va="center",
        fontsize=12,
        color="white",
        fontweight="bold",
    )

    ax.annotate(
        "",
        xy=(px, 2.5),
        xytext=(px, 0.8),
        arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.5),
    )
    ax.text(px + 0.3, 1.6, "$W_{xh}$", fontsize=9, color="#2C3E50", fontstyle="italic")

    ax.annotate(
        "",
        xy=(px, 5.0),
        xytext=(px, 3.5),
        arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.5),
    )
    ax.text(px + 0.3, 4.3, "$W_{hq}$", fontsize=9, color="#2C3E50", fontstyle="italic")

    ax.text(
        px,
        -0.5,
        f"t = {t+1}",
        fontsize=11,
        ha="center",
        color="#7F8C8D",
        fontweight="bold",
    )

for t in range(T - 1):
    px1 = t * spacing + 1 + 0.7
    px2 = (t + 1) * spacing + 1 - 0.7
    ax.annotate(
        "",
        xy=(px2, 3.0),
        xytext=(px1, 3.0),
        arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=2.5),
    )
    mid = (px1 + px2) / 2
    ax.text(
        mid,
        3.5,
        "$W_{hh}$",
        fontsize=10,
        ha="center",
        color="#E74C3C",
        fontweight="bold",
    )

ax.annotate(
    "",
    xy=(1 - 0.7, 3.0),
    xytext=(-1.2, 3.0),
    arrowprops=dict(arrowstyle="->", color="#9B59B6", lw=2, linestyle="dashed"),
)
ax.text(
    -1.8,
    3.0,
    "$H_0$\n(init=0)",
    fontsize=10,
    ha="center",
    color="#9B59B6",
    fontweight="bold",
)

ax.text(
    8,
    -1.2,
    "Trong so (params) CHIA SE qua moi time step (weight sharing)",
    fontsize=11,
    ha="center",
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#D5F5E3", edgecolor="#27AE60"),
)

plt.savefig(
    os.path.join(outdir, "rnn_unrolled.png"),
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print("Figure 2 saved: rnn_unrolled.png")

# ==========================================================================
# FIGURE 3: Concatenation trick visualization
# ==========================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

ax = axes[0]
ax.set_xlim(-0.5, 10)
ax.set_ylim(-0.5, 6)
ax.axis("off")
ax.set_title("Cach 1: Nhan rieng roi cong", fontsize=13, fontweight="bold")

rect = mpatches.FancyBboxPatch(
    (0, 3),
    1.2,
    2.5,
    boxstyle="round,pad=0.1",
    facecolor="#3498DB",
    alpha=0.3,
    edgecolor="#3498DB",
    lw=2,
)
ax.add_patch(rect)
ax.text(
    0.6,
    4.25,
    "$X_t$\n(n x d)",
    ha="center",
    va="center",
    fontsize=10,
    fontweight="bold",
)

rect = mpatches.FancyBboxPatch(
    (1.8, 3.5),
    2,
    1.5,
    boxstyle="round,pad=0.1",
    facecolor="#3498DB",
    alpha=0.3,
    edgecolor="#3498DB",
    lw=2,
)
ax.add_patch(rect)
ax.text(
    2.8,
    4.25,
    "$W_{xh}$\n(d x h)",
    ha="center",
    va="center",
    fontsize=10,
    fontweight="bold",
)

ax.text(4.5, 4.25, "+", fontsize=20, ha="center", va="center", fontweight="bold")

rect = mpatches.FancyBboxPatch(
    (5, 3),
    1.5,
    2.5,
    boxstyle="round,pad=0.1",
    facecolor="#F39C12",
    alpha=0.3,
    edgecolor="#F39C12",
    lw=2,
)
ax.add_patch(rect)
ax.text(
    5.75,
    4.25,
    "$H_{t-1}$\n(n x h)",
    ha="center",
    va="center",
    fontsize=10,
    fontweight="bold",
)

rect = mpatches.FancyBboxPatch(
    (7, 3.5),
    2,
    1.5,
    boxstyle="round,pad=0.1",
    facecolor="#F39C12",
    alpha=0.3,
    edgecolor="#F39C12",
    lw=2,
)
ax.add_patch(rect)
ax.text(
    8,
    4.25,
    "$W_{hh}$\n(h x h)",
    ha="center",
    va="center",
    fontsize=10,
    fontweight="bold",
)

ax.text(
    5,
    1.5,
    "= $X_t W_{xh} + H_{t-1} W_{hh}$    shape: (n x h)",
    fontsize=12,
    ha="center",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#ECF0F1", edgecolor="#BDC3C7"),
)
ax.text(
    5,
    0.3,
    "2 phep nhan ma tran + 1 phep cong",
    fontsize=10,
    ha="center",
    color="#E74C3C",
)

ax = axes[1]
ax.set_xlim(-0.5, 10)
ax.set_ylim(-0.5, 6)
ax.axis("off")
ax.set_title("Cach 2: Ghep noi (Concatenation)", fontsize=13, fontweight="bold")

rect1 = mpatches.FancyBboxPatch(
    (0, 3),
    1.2,
    2.5,
    boxstyle="round,pad=0.1",
    facecolor="#3498DB",
    alpha=0.3,
    edgecolor="#3498DB",
    lw=2,
)
ax.add_patch(rect1)
ax.text(
    0.6,
    4.5,
    "$X_t$",
    ha="center",
    va="center",
    fontsize=10,
    fontweight="bold",
    color="#3498DB",
)

rect2 = mpatches.FancyBboxPatch(
    (1.2, 3),
    1.5,
    2.5,
    boxstyle="round,pad=0.1",
    facecolor="#F39C12",
    alpha=0.3,
    edgecolor="#F39C12",
    lw=2,
)
ax.add_patch(rect2)
ax.text(
    1.95,
    4.5,
    "$H_{t-1}$",
    ha="center",
    va="center",
    fontsize=10,
    fontweight="bold",
    color="#F39C12",
)
ax.text(1.35, 3.5, "n x (d+h)", ha="center", va="center", fontsize=9, color="#7F8C8D")

ax.text(3.3, 4.25, "@", fontsize=18, ha="center", va="center", fontweight="bold")

rect3 = mpatches.FancyBboxPatch(
    (4, 3.5),
    2,
    0.75,
    boxstyle="round,pad=0.1",
    facecolor="#3498DB",
    alpha=0.3,
    edgecolor="#3498DB",
    lw=2,
)
ax.add_patch(rect3)
ax.text(
    5,
    3.88,
    "$W_{xh}$",
    ha="center",
    va="center",
    fontsize=10,
    fontweight="bold",
    color="#3498DB",
)

rect4 = mpatches.FancyBboxPatch(
    (4, 4.25),
    2,
    0.75,
    boxstyle="round,pad=0.1",
    facecolor="#F39C12",
    alpha=0.3,
    edgecolor="#F39C12",
    lw=2,
)
ax.add_patch(rect4)
ax.text(
    5,
    4.63,
    "$W_{hh}$",
    ha="center",
    va="center",
    fontsize=10,
    fontweight="bold",
    color="#F39C12",
)
ax.text(5, 3.1, "(d+h) x h", ha="center", va="center", fontsize=9, color="#7F8C8D")

ax.text(
    5,
    1.5,
    "= $[X_t, H_{t-1}] \\cdot [W_{xh}; W_{hh}]$    shape: (n x h)",
    fontsize=12,
    ha="center",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#D5F5E3", edgecolor="#27AE60"),
)
ax.text(
    5,
    0.3,
    "Chi 1 phep nhan ma tran -> Nhanh hon!",
    fontsize=10,
    ha="center",
    color="#27AE60",
    fontweight="bold",
)

plt.tight_layout()
plt.savefig(
    os.path.join(outdir, "concat_trick.png"),
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print("Figure 3 saved: concat_trick.png")

# ==========================================================================
# FIGURE 4: Character-level language model with RNN
# ==========================================================================
fig, ax = plt.subplots(figsize=(18, 8))
ax.set_xlim(-1, 20)
ax.set_ylim(-2, 8.5)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title(
    'RNN Character-Level Language Model\nInput: "machin" -> Target: "achine"',
    fontsize=14,
    fontweight="bold",
    pad=15,
)

chars_in = ["m", "a", "c", "h", "i", "n"]
chars_out = ["a", "c", "h", "i", "n", "e"]
spacing = 3

for t, (ch_in, ch_out) in enumerate(zip(chars_in, chars_out)):
    px = t * spacing + 1.5

    rect = mpatches.FancyBboxPatch(
        (px - 0.5, 0),
        1.0,
        0.8,
        boxstyle="round,pad=0.1",
        facecolor="#3498DB",
        edgecolor="black",
        lw=1.5,
    )
    ax.add_patch(rect)
    ax.text(
        px,
        0.4,
        f'"{ch_in}"',
        ha="center",
        va="center",
        fontsize=14,
        color="white",
        fontweight="bold",
    )
    ax.text(px, -0.3, f"$X_{{{t+1}}}$", fontsize=10, ha="center", color="#7F8C8D")

    circle = plt.Circle((px, 3.0), 0.6, color="#F39C12", ec="black", lw=2)
    ax.add_patch(circle)
    ax.text(
        px,
        3.0,
        f"$H_{{{t+1}}}$",
        ha="center",
        va="center",
        fontsize=12,
        color="white",
        fontweight="bold",
    )

    rect = mpatches.FancyBboxPatch(
        (px - 0.5, 5.2),
        1.0,
        0.8,
        boxstyle="round,pad=0.1",
        facecolor="#9B59B6",
        edgecolor="black",
        lw=1.5,
    )
    ax.add_patch(rect)
    ax.text(px, 5.6, "softmax", ha="center", va="center", fontsize=9, color="white")

    rect = mpatches.FancyBboxPatch(
        (px - 0.5, 6.8),
        1.0,
        0.8,
        boxstyle="round,pad=0.1",
        facecolor="#27AE60",
        edgecolor="black",
        lw=1.5,
    )
    ax.add_patch(rect)
    ax.text(
        px,
        7.2,
        f'"{ch_out}"',
        ha="center",
        va="center",
        fontsize=14,
        color="white",
        fontweight="bold",
    )
    ax.text(px, 7.9, "target", fontsize=9, ha="center", color="#7F8C8D")

    ax.annotate(
        "",
        xy=(px, 2.4),
        xytext=(px, 0.8),
        arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.3),
    )
    ax.annotate(
        "",
        xy=(px, 5.2),
        xytext=(px, 3.6),
        arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.3),
    )
    ax.annotate(
        "",
        xy=(px, 6.8),
        xytext=(px, 6.0),
        arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.3),
    )

for t in range(len(chars_in) - 1):
    px1 = t * spacing + 1.5 + 0.6
    px2 = (t + 1) * spacing + 1.5 - 0.6
    ax.annotate(
        "",
        xy=(px2, 3.0),
        xytext=(px1, 3.0),
        arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=2.5),
    )

ax.annotate(
    "",
    xy=(1.5 - 0.6, 3.0),
    xytext=(0, 3.0),
    arrowprops=dict(arrowstyle="->", color="#9B59B6", lw=2, linestyle="dashed"),
)
ax.text(
    -0.5, 3.0, "$H_0=0$", fontsize=10, ha="center", color="#9B59B6", fontweight="bold"
)

ax.text(
    10,
    -1.3,
    "Loss = Cross-Entropy tai moi time step: $L = -\\frac{1}{T}\\sum_{t=1}^{T}\\log P(y_t | H_t)$",
    fontsize=11,
    ha="center",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#FADBD8", edgecolor="#E74C3C"),
)

plt.savefig(
    os.path.join(outdir, "rnn_char_lm.png"),
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print("Figure 4 saved: rnn_char_lm.png")

# ==========================================================================
# FIGURE 5: N-gram explosion vs RNN parameter efficiency
# ==========================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
vocab_sizes = [100, 500, 1000, 5000, 10000]
ns = [2, 3, 4, 5]
colors_list = ["#3498DB", "#E67E22", "#E74C3C", "#9B59B6"]

for n, color in zip(ns, colors_list):
    params = [v**n for v in vocab_sizes]
    ax.semilogy(
        vocab_sizes,
        params,
        "o-",
        color=color,
        lw=2,
        markersize=6,
        label=f"{n}-gram ($|V|^{{{n}}}$)",
    )

ax.set_xlabel("Kich thuoc Vocabulary ($|V|$)", fontsize=12)
ax.set_ylabel("So luong parameters", fontsize=12)
ax.set_title("N-gram: Parameters tang theo ham mu", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.axhline(y=1e9, color="red", linestyle="--", alpha=0.5)
ax.text(5500, 2e9, "1 ty params", fontsize=10, color="red", alpha=0.7)

ax = axes[1]
seq_lengths = [10, 50, 100, 500, 1000, 5000]
d = 256
h = 512
q = 100
rnn_params = d * h + h * h + h + h * q + q
rnn_params_list = [rnn_params] * len(seq_lengths)

ax.plot(
    seq_lengths,
    rnn_params_list,
    "o-",
    color="#27AE60",
    lw=3,
    markersize=8,
    label=f"RNN (d={d}, h={h})\n= {rnn_params:,} params",
)
ax.set_xlabel("Do dai chuoi (T)", fontsize=12)
ax.set_ylabel("So luong parameters RNN", fontsize=12)
ax.set_title(
    "RNN: Parameters KHONG TANG theo do dai chuoi", fontsize=13, fontweight="bold"
)
ax.legend(fontsize=11, loc="center right")
ax.grid(True, alpha=0.3)
ax.set_ylim(0, rnn_params * 2)

breakdown = f"$W_{{xh}}$: {d}x{h} = {d*h:,}\n$W_{{hh}}$: {h}x{h} = {h*h:,}\n$W_{{hq}}$: {h}x{q} = {h*q:,}\nTotal: {rnn_params:,}"
ax.text(
    3000,
    rnn_params * 1.5,
    breakdown,
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#D5F5E3", edgecolor="#27AE60"),
    va="center",
)

plt.tight_layout()
plt.savefig(
    os.path.join(outdir, "ngram_vs_rnn_params.png"),
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print("Figure 5 saved: ngram_vs_rnn_params.png")

print("\nAll 5 figures generated successfully!")
