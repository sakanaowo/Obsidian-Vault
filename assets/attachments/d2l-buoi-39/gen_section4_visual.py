import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

plt.rcParams["font.family"] = "DejaVu Sans"

fig = plt.figure(figsize=(16, 8), dpi=150)

# ===== Left panel: one RNN step =====
ax1 = fig.add_axes([0.04, 0.12, 0.44, 0.78])
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis("off")

ax1.text(
    5,
    9.5,
    "Mot time step cua RNN",
    ha="center",
    fontsize=18,
    weight="bold",
    color="#1f2937",
)

# Boxes
boxes = [
    (0.8, 4.5, 1.8, 1.2, "#DBEAFE", "X_t\nInput hien tai"),
    (0.8, 2.0, 1.8, 1.2, "#FCE7F3", "H_{t-1}\nTri nho cu"),
    (4.0, 3.25, 2.2, 1.8, "#DCFCE7", "H_t\nTri nho moi"),
    (7.1, 3.4, 1.8, 1.5, "#FEF3C7", "O_t\nDu doan"),
]

for x, y, w, h, color, label in boxes:
    ax1.add_patch(
        Rectangle((x, y), w, h, facecolor=color, edgecolor="#334155", linewidth=2)
    )
    ax1.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=12,
        weight="bold",
    )

# Arrows with labels
arrows = [
    ((2.6, 5.1), (4.0, 4.5), "W_xh\nDoc input"),
    ((2.6, 2.6), (4.0, 3.8), "W_hh\nTruyen tri nho"),
    ((6.2, 4.1), (7.1, 4.1), "W_hq\nRa output"),
]

for start, end, label in arrows:
    ax1.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=20,
            linewidth=2.5,
            color="#2563EB",
        )
    )
    mx = (start[0] + end[0]) / 2
    my = (start[1] + end[1]) / 2
    ax1.text(
        mx,
        my + 0.35,
        label,
        ha="center",
        va="center",
        fontsize=11,
        color="#1D4ED8",
        weight="bold",
    )

# Formula
ax1.text(
    5,
    1.0,
    r"$H_t = \tanh(X_t W_{xh} + H_{t-1} W_{hh} + b_h)$",
    ha="center",
    fontsize=16,
    color="#111827",
)

# ===== Right panel: weight sharing =====
ax2 = fig.add_axes([0.53, 0.12, 0.43, 0.78])
ax2.set_xlim(0, 14)
ax2.set_ylim(0, 10)
ax2.axis("off")

ax2.text(
    7,
    9.5,
    "Weight sharing theo thoi gian",
    ha="center",
    fontsize=18,
    weight="bold",
    color="#1f2937",
)

x_positions = [1.0, 5.0, 9.0]
labels = ["t=1", "t=2", "t=3"]
for xp, lab in zip(x_positions, labels):
    ax2.text(xp + 1.5, 8.5, lab, ha="center", fontsize=13, weight="bold")
    ax2.add_patch(
        Rectangle(
            (xp, 5.8), 1.4, 0.9, facecolor="#DBEAFE", edgecolor="#334155", linewidth=1.8
        )
    )
    ax2.text(xp + 0.7, 6.25, "X", ha="center", va="center", fontsize=12, weight="bold")
    ax2.add_patch(
        Rectangle(
            (xp + 1.0, 3.9),
            1.9,
            1.3,
            facecolor="#DCFCE7",
            edgecolor="#334155",
            linewidth=1.8,
        )
    )
    ax2.text(
        xp + 1.95,
        4.55,
        "Hidden\nstate",
        ha="center",
        va="center",
        fontsize=11,
        weight="bold",
    )
    ax2.add_patch(
        Rectangle(
            (xp + 1.4, 1.7),
            1.2,
            0.8,
            facecolor="#FEF3C7",
            edgecolor="#334155",
            linewidth=1.8,
        )
    )
    ax2.text(xp + 2.0, 2.1, "O", ha="center", va="center", fontsize=11, weight="bold")

    ax2.add_patch(
        FancyArrowPatch(
            (xp + 0.7, 5.8),
            (xp + 1.6, 5.2),
            arrowstyle="->",
            mutation_scale=16,
            linewidth=2,
            color="#2563EB",
        )
    )
    ax2.text(xp + 1.0, 5.45, "W_xh", fontsize=10, color="#1D4ED8", weight="bold")
    ax2.add_patch(
        FancyArrowPatch(
            (xp + 1.95, 3.9),
            (xp + 2.0, 2.5),
            arrowstyle="->",
            mutation_scale=16,
            linewidth=2,
            color="#D97706",
        )
    )
    ax2.text(xp + 2.15, 3.0, "W_hq", fontsize=10, color="#B45309", weight="bold")

# recurrent arrows between hidden states
for i in range(2):
    start_x = x_positions[i] + 2.9
    end_x = x_positions[i + 1] + 1.0
    ax2.add_patch(
        FancyArrowPatch(
            (start_x, 4.55),
            (end_x, 4.55),
            arrowstyle="->",
            mutation_scale=18,
            linewidth=2.5,
            color="#DC2626",
        )
    )
    ax2.text(
        (start_x + end_x) / 2,
        4.9,
        "W_hh (giong nhau)",
        ha="center",
        fontsize=10,
        color="#B91C1C",
        weight="bold",
    )

# bottom note
ax2.text(
    7,
    0.8,
    "Chuoi dai hon khong tao them ma tran moi\nchi lap lai cung W_xh, W_hh, W_hq",
    ha="center",
    fontsize=13,
    color="#065F46",
    weight="bold",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#ECFDF5", edgecolor="#10B981"),
)

fig.suptitle(
    "Truc quan hoa Phan 4 - Unrolling va Weight Sharing trong RNN",
    fontsize=20,
    weight="bold",
    y=0.98,
)

out = "/home/sakana/Code/Obsidian-Vault/assets/attachments/d2l-buoi-39/section4_weight_sharing_explained.png"
plt.savefig(out, bbox_inches="tight")
print(out)
