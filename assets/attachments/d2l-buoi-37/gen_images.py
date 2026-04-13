"""Generate illustrations for Buổi 37 — Working with Sequences."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 11

OUT = "/home/sakana/Code/Obsidian-Vault/assets/attachments/d2l-buoi-37"


# ============================================================
# 1. Sequence Problem Types
# ============================================================
def fig_sequence_types():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Các dạng bài toán Sequence", fontsize=16, fontweight="bold", y=0.98)

    colors = {
        "input": "#3498DB",
        "output": "#E74C3C",
        "arrow": "#2C3E50",
        "bg": "#ECF0F1",
    }

    # --- (a) Sequence → Fixed ---
    ax = axes[0, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.set_title(
        "(a) Sequence --> Fixed\n(Sentiment Classification)",
        fontsize=11,
        fontweight="bold",
    )
    for i, w in enumerate(["This", "movie", "is", "great", "!"]):
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (i * 1.5 + 0.5, 2.2),
                1.2,
                0.8,
                boxstyle="round,pad=0.1",
                fc=colors["input"],
                ec="white",
                alpha=0.85,
            )
        )
        ax.text(
            i * 1.5 + 1.1,
            2.6,
            w,
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            fontweight="bold",
        )
    ax.annotate(
        "",
        xy=(5, 1.5),
        xytext=(4, 2.2),
        arrowprops=dict(arrowstyle="->", color=colors["arrow"], lw=2),
    )
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (4, 0.5),
            2.5,
            0.9,
            boxstyle="round,pad=0.15",
            fc=colors["output"],
            ec="white",
            alpha=0.85,
        )
    )
    ax.text(
        5.25,
        0.95,
        "Positive",
        ha="center",
        va="center",
        color="white",
        fontsize=11,
        fontweight="bold",
    )
    ax.axis("off")

    # --- (b) Fixed → Sequence ---
    ax = axes[0, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.set_title(
        "(b) Fixed --> Sequence\n(Image Captioning)", fontsize=11, fontweight="bold"
    )
    ax.add_patch(
        mpatches.FancyBboxPatch((0.5, 1.5), 2.5, 2),
    )
    ax.add_patch(
        mpatches.Rectangle(
            (0.5, 1.5), 2.5, 2, fc=colors["input"], ec="white", alpha=0.85
        )
    )
    ax.text(
        1.75,
        2.5,
        "IMAGE",
        ha="center",
        va="center",
        color="white",
        fontsize=12,
        fontweight="bold",
    )
    ax.annotate(
        "",
        xy=(4, 2.5),
        xytext=(3, 2.5),
        arrowprops=dict(arrowstyle="->", color=colors["arrow"], lw=2),
    )
    for i, w in enumerate(["A", "cat", "on", "mat"]):
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (4 + i * 1.4, 2),
                1.2,
                0.8,
                boxstyle="round,pad=0.1",
                fc=colors["output"],
                ec="white",
                alpha=0.85,
            )
        )
        ax.text(
            4 + i * 1.4 + 0.6,
            2.4,
            w,
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            fontweight="bold",
        )
    ax.axis("off")

    # --- (c) Aligned Seq → Seq ---
    ax = axes[1, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.set_title(
        "(c) Aligned Seq --> Seq\n(POS Tagging)", fontsize=11, fontweight="bold"
    )
    words = ["The", "cat", "sat", "down"]
    tags = ["DET", "NOUN", "VERB", "ADV"]
    for i, (w, t) in enumerate(zip(words, tags)):
        x = i * 2.2 + 0.5
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, 2.2),
                1.8,
                0.8,
                boxstyle="round,pad=0.1",
                fc=colors["input"],
                ec="white",
                alpha=0.85,
            )
        )
        ax.text(
            x + 0.9,
            2.6,
            w,
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold",
        )
        ax.annotate(
            "",
            xy=(x + 0.9, 1.5),
            xytext=(x + 0.9, 2.2),
            arrowprops=dict(arrowstyle="->", color=colors["arrow"], lw=1.5),
        )
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, 0.5),
                1.8,
                0.8,
                boxstyle="round,pad=0.1",
                fc=colors["output"],
                ec="white",
                alpha=0.85,
            )
        )
        ax.text(
            x + 0.9,
            0.9,
            t,
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold",
        )
    ax.axis("off")

    # --- (d) Unaligned Seq → Seq ---
    ax = axes[1, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.set_title(
        "(d) Unaligned Seq --> Seq\n(Machine Translation)",
        fontsize=11,
        fontweight="bold",
    )
    src = ["I", "love", "cats"]
    tgt = ["Toi", "yeu", "meo"]
    for i, w in enumerate(src):
        x = i * 1.8 + 0.3
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, 2.5),
                1.5,
                0.7,
                boxstyle="round,pad=0.1",
                fc=colors["input"],
                ec="white",
                alpha=0.85,
            )
        )
        ax.text(
            x + 0.75,
            2.85,
            w,
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold",
        )
    ax.annotate(
        "",
        xy=(5.5, 2.0),
        xytext=(3.5, 2.0),
        arrowprops=dict(arrowstyle="->", color=colors["arrow"], lw=2),
    )
    ax.text(
        4.5,
        2.15,
        "Encoder\n-Decoder",
        ha="center",
        va="bottom",
        fontsize=8,
        color=colors["arrow"],
    )
    for i, w in enumerate(tgt):
        x = i * 1.8 + 0.3
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, 0.5),
                1.5,
                0.7,
                boxstyle="round,pad=0.1",
                fc=colors["output"],
                ec="white",
                alpha=0.85,
            )
        )
        ax.text(
            x + 0.75,
            0.85,
            w,
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold",
        )

    ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(
        f"{OUT}/sequence_problem_types.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()
    print("OK: sequence_problem_types.png")


# ============================================================
# 2. Autoregressive vs Latent Autoregressive
# ============================================================
def fig_autoregressive_models():
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    fig.suptitle(
        "Hai chiến lược xử lý Sequence", fontsize=15, fontweight="bold", y=1.02
    )

    c_in = "#3498DB"
    c_pred = "#E74C3C"
    c_hidden = "#F39C12"
    c_arrow = "#2C3E50"

    # --- (a) Fixed Window (tau=4) ---
    ax = axes[0]
    ax.set_xlim(-0.5, 11)
    ax.set_ylim(-0.5, 4)
    ax.set_title(
        "(a) Fixed Window (tau=4)\nSo x dau vao CO DINH", fontsize=11, fontweight="bold"
    )

    # Input window
    for i in range(4):
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (i * 2, 2.5),
                1.5,
                0.8,
                boxstyle="round,pad=0.1",
                fc=c_in,
                alpha=0.85,
                ec="white",
            )
        )
        ax.text(
            i * 2 + 0.75,
            2.9,
            f"$x_{{t-{4-i}}}$",
            ha="center",
            va="center",
            color="white",
            fontsize=11,
            fontweight="bold",
        )

    # Arrow to prediction
    ax.annotate(
        "",
        xy=(5, 1.5),
        xytext=(4, 2.5),
        arrowprops=dict(arrowstyle="->", color=c_arrow, lw=2),
    )
    ax.text(5.5, 2.0, "f()", fontsize=12, fontweight="bold", color=c_arrow)

    # Prediction
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (4, 0.5),
            2,
            0.8,
            boxstyle="round,pad=0.15",
            fc=c_pred,
            alpha=0.85,
            ec="white",
        )
    )
    ax.text(
        5,
        0.9,
        "$\\hat{x}_t$",
        ha="center",
        va="center",
        color="white",
        fontsize=13,
        fontweight="bold",
    )

    # Annotation
    ax.text(
        5,
        -0.2,
        "Input = $[x_{t-4}, x_{t-3}, x_{t-2}, x_{t-1}]$\nLuon co DUNG 4 features",
        ha="center",
        fontsize=9,
        style="italic",
        color="#555",
    )

    ax.axis("off")

    # --- (b) Latent Autoregressive ---
    ax = axes[1]
    ax.set_xlim(-0.5, 12)
    ax.set_ylim(-0.5, 4.5)
    ax.set_title(
        "(b) Latent Autoregressive\nTrang thai an (hidden state) TOM TAT lich su",
        fontsize=11,
        fontweight="bold",
    )

    # Time steps
    for i in range(4):
        t_label = f"t-{3-i}" if i < 3 else "t"
        x = i * 3

        # Input
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, 0.5),
                1.5,
                0.7,
                boxstyle="round,pad=0.1",
                fc=c_in,
                alpha=0.85,
                ec="white",
            )
        )
        ax.text(
            x + 0.75,
            0.85,
            f"$x_{{{t_label}}}$",
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold",
        )

        # Hidden state
        ax.add_patch(
            mpatches.Circle((x + 0.75, 2.5), 0.5, fc=c_hidden, alpha=0.85, ec="white")
        )
        ax.text(
            x + 0.75,
            2.5,
            f"$h_{{{t_label}}}$",
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold",
        )

        # x → h arrow
        ax.annotate(
            "",
            xy=(x + 0.75, 2.0),
            xytext=(x + 0.75, 1.2),
            arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.5),
        )

        # h → h arrow (recurrent)
        if i < 3:
            ax.annotate(
                "",
                xy=((i + 1) * 3 + 0.25, 2.5),
                xytext=(x + 1.25, 2.5),
                arrowprops=dict(arrowstyle="->", color=c_hidden, lw=2),
            )

        # h → prediction at last step
        if i == 3:
            ax.add_patch(
                mpatches.FancyBboxPatch(
                    (x - 0.25, 3.5),
                    2,
                    0.7,
                    boxstyle="round,pad=0.1",
                    fc=c_pred,
                    alpha=0.85,
                    ec="white",
                )
            )
            ax.text(
                x + 0.75,
                3.85,
                "$\\hat{x}_{t+1}$",
                ha="center",
                va="center",
                color="white",
                fontsize=11,
                fontweight="bold",
            )
            ax.annotate(
                "",
                xy=(x + 0.75, 3.5),
                xytext=(x + 0.75, 3.0),
                arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.5),
            )

    ax.text(
        5.5,
        -0.2,
        "$h_t = g(h_{t-1}, x_{t-1})$ -- cap nhat LIEN TUC\nKhong can CO DINH so features",
        ha="center",
        fontsize=9,
        style="italic",
        color="#555",
    )
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        f"{OUT}/autoregressive_models.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()
    print("OK: autoregressive_models.png")


# ============================================================
# 3. K-step prediction error accumulation
# ============================================================
def fig_k_step_prediction():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- (a) Synthetic data + predictions ---
    ax = axes[0]
    T = 1000
    tau = 4
    num_train = 600
    np.random.seed(42)
    time = np.arange(1, T + 1, dtype=np.float32)
    x = np.sin(0.01 * time) + np.random.randn(T) * 0.2

    # Simple autoregressive prediction (simulated)
    # 1-step: just use sin as approximation
    onestep = np.sin(0.01 * time)

    # Multi-step: diverges
    multistep = np.copy(x)
    for i in range(num_train + tau, T):
        # Simple extrapolation that decays
        window = multistep[i - tau : i]
        multistep[i] = np.mean(window) * 0.98  # Simulated decay

    ax.plot(
        time[tau : num_train + tau],
        x[tau : num_train + tau],
        "b-",
        alpha=0.3,
        linewidth=0.8,
        label="Du lieu goc (train)",
    )
    ax.plot(
        time[num_train + tau :],
        x[num_train + tau :],
        color="#888",
        alpha=0.3,
        linewidth=0.8,
        label="Du lieu goc (test)",
    )
    ax.plot(
        time[tau:],
        onestep[tau:],
        "g-",
        linewidth=1.2,
        alpha=0.7,
        label="Du doan 1-buoc (tot!)",
    )
    ax.plot(
        time[num_train + tau :],
        multistep[num_train + tau :],
        "r-",
        linewidth=2,
        label="Du doan nhieu-buoc (te!)",
    )
    ax.axvline(
        x=num_train + tau,
        color="black",
        linestyle="--",
        alpha=0.5,
        label=f"Ranh gioi train/test (t={num_train+tau})",
    )
    ax.set_xlabel("Thoi gian (t)", fontsize=11)
    ax.set_ylabel("x", fontsize=11)
    ax.set_title(
        "(a) 1-step vs Multi-step Prediction\nSai so TICH LUY khi du doan xa",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(1, T)

    # --- (b) Error accumulation diagram ---
    ax = axes[1]
    steps = np.arange(1, 65)
    # Error grows roughly exponentially
    error = 0.05 * (1.08**steps)
    error_bounded = np.minimum(error, 1.0)

    ax.fill_between(steps, 0, error_bounded, alpha=0.3, color="#E74C3C")
    ax.plot(steps, error_bounded, "r-", linewidth=2, label="Sai so tich luy")
    ax.axhline(
        y=0.15, color="green", linestyle="--", alpha=0.7, label="Nguong chap nhan duoc"
    )

    # Annotations
    ax.annotate(
        "Du doan tin cay\n(1-4 buoc)",
        xy=(2, 0.08),
        fontsize=10,
        color="green",
        fontweight="bold",
    )
    ax.annotate(
        "Sai so tang THEO HAM MU\n--> du doan vo nghia",
        xy=(30, 0.75),
        fontsize=10,
        color="red",
        fontweight="bold",
        ha="center",
    )

    # k=1,4,16,64 markers
    for k, c in [(1, "green"), (4, "#F39C12"), (16, "#E67E22"), (64, "red")]:
        idx = k - 1
        ax.plot(k, error_bounded[idx], "o", color=c, markersize=10, zorder=5)
        ax.annotate(
            f"k={k}",
            xy=(k, error_bounded[idx]),
            xytext=(k + 3, error_bounded[idx] + 0.05),
            fontsize=9,
            fontweight="bold",
            color=c,
            arrowprops=dict(arrowstyle="->", color=c, lw=1),
        )

    ax.set_xlabel("So buoc du doan (k)", fontsize=11)
    ax.set_ylabel("Sai so (error)", fontsize=11)
    ax.set_title(
        "(b) Tich luy sai so theo so buoc\nGiong nhu du bao thoi tiet",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.set_xlim(0, 68)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(
        f"{OUT}/k_step_prediction.png", dpi=150, bbox_inches="tight", facecolor="white"
    )
    plt.close()
    print("OK: k_step_prediction.png")


# ============================================================
# 4. Markov Models diagram
# ============================================================
def fig_markov_models():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        "Markov Models: Lich su can thiet de du doan",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    c_active = "#3498DB"
    c_inactive = "#BDC3C7"
    c_pred = "#E74C3C"

    for ax_idx, (tau, title) in enumerate(
        [
            (1, "Bac 1: chi can $x_{t-1}$"),
            (3, "Bac 3: can $x_{t-3}, x_{t-2}, x_{t-1}$"),
            (None, "Day du: can TOAN BO lich su"),
        ]
    ):
        ax = axes[ax_idx]
        ax.set_xlim(-0.5, 12)
        ax.set_ylim(-0.5, 3)
        ax.set_title(title, fontsize=11, fontweight="bold")

        n_steps = 6
        for i in range(n_steps):
            # Determine color
            if tau is None:
                c = c_active  # All active
            elif i >= n_steps - 1 - tau and i < n_steps - 1:
                c = c_active
            elif i == n_steps - 1:
                c = c_pred
            else:
                c = c_inactive

            x = i * 2
            ax.add_patch(
                mpatches.FancyBboxPatch(
                    (x, 1),
                    1.4,
                    0.8,
                    boxstyle="round,pad=0.1",
                    fc=c,
                    alpha=0.85,
                    ec="white",
                )
            )
            label = f"$x_{{{i+1}}}$" if i < n_steps - 1 else "$\\hat{x}_t$"
            ax.text(
                x + 0.7,
                1.4,
                label,
                ha="center",
                va="center",
                color="white",
                fontsize=10,
                fontweight="bold",
            )

            # Arrows
            if i > 0:
                if tau is None or (i >= n_steps - 1 - tau and i <= n_steps - 1):
                    ax.annotate(
                        "",
                        xy=(x, 1.4),
                        xytext=(x - 0.6, 1.4),
                        arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.2),
                    )

        # Legend colors
        if ax_idx == 0:
            ax.add_patch(
                mpatches.FancyBboxPatch(
                    (0, 0),
                    0.6,
                    0.4,
                    boxstyle="round,pad=0.05",
                    fc=c_inactive,
                    alpha=0.7,
                )
            )
            ax.text(0.8, 0.2, "= Bo qua", fontsize=8, va="center")
            ax.add_patch(
                mpatches.FancyBboxPatch(
                    (3.5, 0),
                    0.6,
                    0.4,
                    boxstyle="round,pad=0.05",
                    fc=c_active,
                    alpha=0.7,
                )
            )
            ax.text(4.3, 0.2, "= Su dung", fontsize=8, va="center")
            ax.add_patch(
                mpatches.FancyBboxPatch(
                    (7, 0), 0.6, 0.4, boxstyle="round,pad=0.05", fc=c_pred, alpha=0.7
                )
            )
            ax.text(7.8, 0.2, "= Du doan", fontsize=8, va="center")

        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        f"{OUT}/markov_models.png", dpi=150, bbox_inches="tight", facecolor="white"
    )
    plt.close()
    print("OK: markov_models.png")


# ============================================================
# 5. Chain Rule Decomposition
# ============================================================
def fig_chain_rule():
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(-0.5, 14)
    ax.set_ylim(-0.5, 5)
    ax.set_title(
        "Chain Rule: Phan ra xac suat chuoi thanh tich cac xac suat co dieu kien",
        fontsize=13,
        fontweight="bold",
    )

    c_token = "#3498DB"
    c_context = "#95A5A6"
    c_arrow = "#2C3E50"

    # Show P(x1, x2, x3, x4) decomposition
    tokens = ["$x_1$", "$x_2$", "$x_3$", "$x_4$", "...", "$x_T$"]
    probs = [
        "$P(x_1)$",
        "$P(x_2|x_1)$",
        "$P(x_3|x_1,x_2)$",
        "$P(x_4|x_1,...,x_3)$",
        "...",
        "$P(x_T|x_1,...,x_{T-1})$",
    ]

    for i, (tok, prob) in enumerate(zip(tokens, probs)):
        x = i * 2.3
        # Token box
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, 2.8),
                1.8,
                0.8,
                boxstyle="round,pad=0.1",
                fc=c_token,
                alpha=0.85,
                ec="white",
            )
        )
        ax.text(
            x + 0.9,
            3.2,
            tok,
            ha="center",
            va="center",
            color="white",
            fontsize=11,
            fontweight="bold",
        )

        # Conditional probability
        ax.annotate(
            "",
            xy=(x + 0.9, 2.2),
            xytext=(x + 0.9, 2.8),
            arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.2),
        )
        ax.text(
            x + 0.9,
            1.7,
            prob,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=c_arrow,
        )

        # Context arrows (from previous tokens)
        if i > 0 and i < 5:
            for j in range(max(0, i - 2), i):
                ax.annotate(
                    "",
                    xy=(x + 0.3, 3.0),
                    xytext=(j * 2.3 + 1.6, 3.4),
                    arrowprops=dict(
                        arrowstyle="->", color=c_context, lw=0.8, alpha=0.4
                    ),
                )

    # Bottom formula
    ax.text(
        6.5,
        0.5,
        r"$P(x_1, ..., x_T) = P(x_1) \times \prod_{t=2}^{T} P(x_t | x_{t-1}, ..., x_1)$",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FEF9E7", ec="#F39C12", lw=2),
    )

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(
        f"{OUT}/chain_rule_decomposition.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()
    print("OK: chain_rule_decomposition.png")


if __name__ == "__main__":
    fig_sequence_types()
    fig_autoregressive_models()
    fig_k_step_prediction()
    fig_markov_models()
    fig_chain_rule()
    print("\nAll images generated successfully!")
