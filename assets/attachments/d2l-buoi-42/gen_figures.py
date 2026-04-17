#!/usr/bin/env python3
"""Generate all figures for Buổi 42 — D2L 9.7 BPTT."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent
matplotlib.rcParams["font.size"] = 11


# ======================================================================
# Figure 1: Computational Graph of BPTT (RNN unrolled through 3 steps)
# ======================================================================
def fig1_bptt_graph():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(-2, 16)
    ax.set_ylim(-1, 11.5)
    ax.axis("off")
    ax.set_title(
        "Computational Graph: RNN Unrolled qua 3 Time Steps\n"
        "Forward (xanh) — Backward BPTT (đỏ nét đứt)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    c_fwd = "#2196F3"
    c_bwd = "#E53935"

    def box(x, y, w, h, txt, fc, fs=10, bld=False):
        r = mpatches.FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.12",
            fc=fc,
            ec="black",
            lw=1.4,
        )
        ax.add_patch(r)
        ax.text(
            x,
            y,
            txt,
            ha="center",
            va="center",
            fontsize=fs,
            fontweight="bold" if bld else "normal",
        )

    def circ(x, y, r, txt, fc="#FFF9C4"):
        c = plt.Circle((x, y), r, fc=fc, ec="black", lw=1.4)
        ax.add_patch(c)
        ax.text(x, y, txt, ha="center", va="center", fontsize=9, fontweight="bold")

    fwd = dict(arrowstyle="->", color=c_fwd, lw=2, mutation_scale=14)
    bwd = dict(
        arrowstyle="->", color=c_bwd, lw=2, linestyle="dashed", mutation_scale=14
    )

    ts = [3, 8, 13]
    labels_x = ["$x_1$", "$x_2$", "$x_3$"]
    labels_h = ["$h_1$", "$h_2$", "$h_3$"]
    labels_o = ["$o_1$", "$o_2$", "$o_3$"]
    labels_l = ["$\\ell_1$", "$\\ell_2$", "$\\ell_3$"]

    # Time labels
    for t, lb in zip(ts, ["t = 1", "t = 2", "t = 3"]):
        ax.text(t, 10.8, lb, ha="center", fontsize=12, fontweight="bold", color="#555")

    # h_0
    box(-0.8, 6, 1.1, 0.7, "$h_0$", "#E0E0E0", fs=11)

    for i, t in enumerate(ts):
        # x_t
        box(t, 10, 1.1, 0.7, labels_x[i], "#E3F2FD", fs=11)
        # h_t
        box(t, 6, 1.1, 0.7, labels_h[i], "#E8F5E9", fs=11, bld=True)
        # o_t
        box(t, 3.5, 1.1, 0.7, labels_o[i], "#E3F2FD", fs=11)
        # l_t
        box(t, 1.2, 1.0, 0.6, labels_l[i], "#FFCDD2", fs=10)
        # f circle (hidden computation)
        circ(t, 8, 0.38, "f")
        # g circle (output computation)
        circ(t, 4.8, 0.32, "g")

        # Forward: x -> f
        ax.annotate("", xy=(t, 8.38), xytext=(t, 9.3), arrowprops=fwd)
        # Forward: f -> h
        ax.annotate("", xy=(t, 6.35), xytext=(t, 7.62), arrowprops=fwd)
        # Forward: h -> g
        ax.annotate("", xy=(t, 5.12), xytext=(t, 5.65), arrowprops=fwd)
        # Forward: g -> o
        ax.annotate("", xy=(t, 3.85), xytext=(t, 4.48), arrowprops=fwd)
        # Forward: o -> l
        ax.annotate("", xy=(t, 1.5), xytext=(t, 3.15), arrowprops=fwd)

        # Backward: l -> o -> h (offset right)
        off = 0.25
        ax.annotate("", xy=(t + off, 3.15), xytext=(t + off, 1.5), arrowprops=bwd)
        ax.annotate("", xy=(t + off, 5.65), xytext=(t + off, 3.85), arrowprops=bwd)

    # Forward h -> h (horizontal)
    ax.annotate("", xy=(2.45, 6), xytext=(0.0, 6), arrowprops=fwd)
    ax.annotate("", xy=(7.45, 6), xytext=(3.55, 6), arrowprops=fwd)
    ax.annotate("", xy=(12.45, 6), xytext=(8.55, 6), arrowprops=fwd)

    # Backward h <- h (horizontal, slightly above)
    ax.annotate("", xy=(3.55, 6.55), xytext=(7.45, 6.55), arrowprops=bwd)
    ax.annotate("", xy=(8.55, 6.55), xytext=(12.45, 6.55), arrowprops=bwd)

    # Labels on arrows
    for t in ts:
        ax.text(t - 1.5, 5.5, "$W_{hx}$", fontsize=8, color="#666", ha="center")
    ax.text(5.5, 6.9, "$W_{hh}$", fontsize=8, color="#666", ha="center")
    ax.text(10.5, 6.9, "$W_{hh}$", fontsize=8, color="#666", ha="center")
    for t in ts:
        ax.text(t + 0.6, 4.5, "$W_{qh}$", fontsize=8, color="#666", ha="center")

    # Key insight annotation
    ax.text(
        8,
        -0.3,
        "Gradient $\\partial L / \\partial W_{hh}$ phải truyền ngược qua MỌI time step\n"
        "→ tích $\\prod W_{hh}^T$ gây vanishing hoặc exploding gradient",
        ha="center",
        fontsize=10,
        style="italic",
        color="#888",
        bbox=dict(boxstyle="round,pad=0.4", fc="#FFF8E1", ec="#FFB300", lw=1),
    )

    # Legend
    leg = [
        plt.Line2D([0], [0], color=c_fwd, lw=2, label="Forward pass"),
        plt.Line2D(
            [0], [0], color=c_bwd, lw=2, ls="dashed", label="Backward pass (BPTT)"
        ),
        mpatches.Patch(fc="#FFF9C4", ec="black", label="Operator (f, g)"),
    ]
    ax.legend(handles=leg, loc="upper right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(
        OUT / "bptt_computational_graph.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print(f'Saved: {OUT / "bptt_computational_graph.png"}')


# ======================================================================
# Figure 2: Three truncation strategies comparison
# ======================================================================
def fig2_truncation_strategies():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "3 Chien luoc Truncation cho BPTT", fontsize=14, fontweight="bold", y=1.02
    )

    T = 10
    strategies = [
        ("Full Computation", "Truyen nguoc\ntoan bo T steps", None),
        ("Truncated BPTT (tau=3)", "Chi truyen nguoc\ntau steps gan nhat", 3),
        (
            "Randomized Truncation",
            "Truncate ngau nhien\n(correct in expectation)",
            "random",
        ),
    ]

    np.random.seed(42)

    for idx, (title, desc, tau) in enumerate(strategies):
        ax = axes[idx]
        ax.set_xlim(-0.5, T + 0.5)
        ax.set_ylim(-0.8, 2.5)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Time step $t$", fontsize=9)
        ax.set_xticks(range(T + 1))
        ax.set_yticks([])

        # Draw hidden state nodes
        for t in range(T + 1):
            color = "#4CAF50" if t == T else "#90CAF9"
            circle = plt.Circle((t, 1), 0.3, fc=color, ec="black", lw=1.2, zorder=3)
            ax.add_patch(circle)
            ax.text(
                t, 1, f"$h_{{{t}}}$", ha="center", va="center", fontsize=7, zorder=4
            )

        # Forward arrows (all the same)
        for t in range(T):
            ax.annotate(
                "",
                xy=(t + 0.7, 1),
                xytext=(t + 0.3, 1),
                arrowprops=dict(
                    arrowstyle="->", color="#2196F3", lw=1.5, mutation_scale=10
                ),
            )

        # Backward gradient flow
        if tau is None:  # Full
            for t in range(T, 0, -1):
                ax.annotate(
                    "",
                    xy=(t - 0.7, 1.6),
                    xytext=(t - 0.3, 1.6),
                    arrowprops=dict(
                        arrowstyle="->",
                        color="#E53935",
                        lw=1.5,
                        ls="dashed",
                        mutation_scale=10,
                    ),
                )
            ax.text(
                T / 2,
                2.2,
                "O(T) thoi gian, O(T) bo nho",
                ha="center",
                fontsize=8,
                color="#E53935",
                bbox=dict(fc="#FFEBEE", ec="#E53935", boxstyle="round,pad=0.2"),
            )

        elif tau == 3:  # Truncated
            for t in range(T, max(0, T - 3), -1):
                ax.annotate(
                    "",
                    xy=(t - 0.7, 1.6),
                    xytext=(t - 0.3, 1.6),
                    arrowprops=dict(
                        arrowstyle="->",
                        color="#E53935",
                        lw=1.5,
                        ls="dashed",
                        mutation_scale=10,
                    ),
                )
            # Gray out the rest
            for t in range(max(0, T - 3), 0, -1):
                ax.annotate(
                    "",
                    xy=(t - 0.7, 1.6),
                    xytext=(t - 0.3, 1.6),
                    arrowprops=dict(
                        arrowstyle="->",
                        color="#BDBDBD",
                        lw=1,
                        ls="dotted",
                        mutation_scale=8,
                    ),
                )
            # tau bracket
            ax.annotate(
                "",
                xy=(T - 3 + 0.5, 2.1),
                xytext=(T - 0.5, 2.1),
                arrowprops=dict(arrowstyle="<->", color="#FF6F00", lw=1.5),
            )
            ax.text(
                T - 1.5,
                2.3,
                r"$\tau = 3$",
                ha="center",
                fontsize=9,
                color="#FF6F00",
                fontweight="bold",
            )

        else:  # Randomized
            # Draw random truncation points for illustration
            trunc_point = np.random.geometric(p=0.3)
            trunc_point = min(trunc_point, T)
            for t in range(T, max(0, T - trunc_point), -1):
                ax.annotate(
                    "",
                    xy=(t - 0.7, 1.6),
                    xytext=(t - 0.3, 1.6),
                    arrowprops=dict(
                        arrowstyle="->",
                        color="#E53935",
                        lw=1.5,
                        ls="dashed",
                        mutation_scale=10,
                    ),
                )
            for t in range(max(0, T - trunc_point), 0, -1):
                ax.annotate(
                    "",
                    xy=(t - 0.7, 1.6),
                    xytext=(t - 0.3, 1.6),
                    arrowprops=dict(
                        arrowstyle="->",
                        color="#BDBDBD",
                        lw=1,
                        ls="dotted",
                        mutation_scale=8,
                    ),
                )
            # "?" marks to show randomness
            ax.text(
                T / 2,
                2.2,
                "$\\xi_t \\sim$ Geometric → E[correct]\nnhung variance cao",
                ha="center",
                fontsize=8,
                color="#7B1FA2",
                bbox=dict(fc="#F3E5F5", ec="#7B1FA2", boxstyle="round,pad=0.2"),
            )

        ax.text(
            T / 2, -0.5, desc, ha="center", fontsize=8, color="#666", style="italic"
        )

    fig.tight_layout()
    fig.savefig(
        OUT / "truncation_strategies.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print(f'Saved: {OUT / "truncation_strategies.png"}')


# ======================================================================
# Figure 3: Vanishing / Exploding gradient — eigenvalue effect
# ======================================================================
def fig3_vanishing_exploding():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Vanishing vs Exploding Gradient: Anh huong cua Eigenvalue $\\lambda$\n"
        "cua $W_{hh}$ len gradient qua $k$ time steps",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )

    k = np.arange(0, 30)

    # Left: vanishing
    ax = axes[0]
    for lam, ls, c in [
        (0.9, "-", "#1565C0"),
        (0.7, "--", "#42A5F5"),
        (0.5, "-.", "#90CAF9"),
    ]:
        ax.plot(k, lam**k, ls, color=c, lw=2, label=f"$\\lambda = {lam}$")
    ax.set_title(
        "Vanishing ($|\\lambda| < 1$)", fontsize=12, fontweight="bold", color="#1565C0"
    )
    ax.set_xlabel("So buoc thoi gian $k$", fontsize=10)
    ax.set_ylabel("$|\\lambda|^k$ (ty le gradient)", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.1)
    ax.axhline(y=0, color="gray", lw=0.5)
    ax.grid(True, alpha=0.3)
    ax.text(
        15,
        0.55,
        "Gradient\n→ 0\n(mat thong tin\ntir xa)",
        ha="center",
        fontsize=9,
        color="#1565C0",
        style="italic",
        bbox=dict(fc="#E3F2FD", ec="#1565C0", boxstyle="round,pad=0.3"),
    )

    # Right: exploding
    ax = axes[1]
    for lam, ls, c in [
        (1.1, "-", "#C62828"),
        (1.3, "--", "#EF5350"),
        (1.5, "-.", "#FF8A80"),
    ]:
        vals = lam**k
        ax.plot(k, vals, ls, color=c, lw=2, label=f"$\\lambda = {lam}$")
    ax.set_title(
        "Exploding ($|\\lambda| > 1$)", fontsize=12, fontweight="bold", color="#C62828"
    )
    ax.set_xlabel("So buoc thoi gian $k$", fontsize=10)
    ax.set_ylabel("$|\\lambda|^k$ (ty le gradient)", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.text(
        15,
        1e4,
        "Gradient\n→ inf\n(model phat no)",
        ha="center",
        fontsize=9,
        color="#C62828",
        style="italic",
        bbox=dict(fc="#FFEBEE", ec="#C62828", boxstyle="round,pad=0.3"),
    )

    fig.tight_layout()
    fig.savefig(
        OUT / "vanishing_exploding_gradient.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print(f'Saved: {OUT / "vanishing_exploding_gradient.png"}')


# ======================================================================
# Figure 4: Gradient chain formula visual
# ======================================================================
def fig4_gradient_chain():
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.set_xlim(-1, 16)
    ax.set_ylim(-1, 7)
    ax.axis("off")
    ax.set_title(
        "Chuoi nhan Gradient trong BPTT\n"
        "$\\partial h_t / \\partial w_h$ phu thuoc vao TOAN BO lich su",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    # Draw chain: h_T depends on h_{T-1} depends on ... h_1
    xs = [1, 4, 7, 10, 13]
    labels = ["$h_1$", "$h_2$", "$h_3$", "$\\cdots$", "$h_T$"]
    colors = ["#E8F5E9", "#E8F5E9", "#E8F5E9", "#FAFAFA", "#C8E6C9"]

    for x, lb, fc in zip(xs, labels, colors):
        rect = mpatches.FancyBboxPatch(
            (x - 0.6, 3.8),
            1.2,
            0.8,
            boxstyle="round,pad=0.1",
            fc=fc,
            ec="black",
            lw=1.5,
        )
        ax.add_patch(rect)
        ax.text(x, 4.2, lb, ha="center", va="center", fontsize=12, fontweight="bold")

    # Arrows between them with labels
    for i in range(len(xs) - 1):
        if i == 3:
            continue
        ax.annotate(
            "",
            xy=(xs[i + 1] - 0.65, 4.2),
            xytext=(xs[i] + 0.65, 4.2),
            arrowprops=dict(arrowstyle="->", color="#2196F3", lw=2, mutation_scale=14),
        )
    ax.annotate(
        "",
        xy=(xs[3] - 0.6, 4.2),
        xytext=(xs[2] + 0.65, 4.2),
        arrowprops=dict(
            arrowstyle="->", color="#BDBDBD", lw=2, mutation_scale=14, ls="dotted"
        ),
    )
    ax.annotate(
        "",
        xy=(xs[4] - 0.65, 4.2),
        xytext=(xs[3] + 0.6, 4.2),
        arrowprops=dict(
            arrowstyle="->", color="#BDBDBD", lw=2, mutation_scale=14, ls="dotted"
        ),
    )

    # Jacobian labels
    for i, x_mid in enumerate([2.5, 5.5]):
        ax.text(
            x_mid,
            4.8,
            f"$\\partial h_{{{i+2}}} / \\partial h_{{{i+1}}}$",
            ha="center",
            fontsize=9,
            color="#E53935",
        )

    # The key equation
    ax.text(
        8,
        2.2,
        r"$\frac{\partial h_T}{\partial w_h} = \sum_{t=1}^{T}"
        r"\left(\prod_{i=t+1}^{T} \frac{\partial h_i}{\partial h_{i-1}}\right)"
        r"\frac{\partial h_t}{\partial w_h}$",
        ha="center",
        fontsize=15,
        bbox=dict(fc="#FFF3E0", ec="#FF6F00", boxstyle="round,pad=0.5", lw=2),
    )

    # Explanation
    ax.text(
        8,
        0.5,
        "Moi so hang trong tong chua TICH cua cac Jacobian\n"
        "→ Neu $\\|\\partial h_i / \\partial h_{i-1}\\| < 1$ → tich → 0 (vanishing)\n"
        "→ Neu $\\|\\partial h_i / \\partial h_{i-1}\\| > 1$ → tich → $\\infty$ (exploding)",
        ha="center",
        fontsize=10,
        color="#555",
        bbox=dict(fc="#FAFAFA", ec="#BDBDBD", boxstyle="round,pad=0.4"),
    )

    # Direct dependency arrows from each h_t going down to w_h
    ax.text(
        8,
        6.3,
        "$w_h$ duoc chia se (shared) tai MOI time step → gradient la TONG cua T so hang",
        ha="center",
        fontsize=10,
        color="#1565C0",
        style="italic",
    )

    fig.tight_layout()
    fig.savefig(
        OUT / "gradient_chain.png", dpi=150, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)
    print(f'Saved: {OUT / "gradient_chain.png"}')


# ======================================================================
# Figure 5: detach_() effect in truncated BPTT
# ======================================================================
def fig5_detach():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Truncated BPTT voi detach_() trong code",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    for idx, (title, detach_at) in enumerate(
        [
            ("Khong detach → Full BPTT", None),
            ("detach_() moi num_steps → Truncated BPTT", 3),
        ]
    ):
        ax = axes[idx]
        ax.set_xlim(-0.5, 8.5)
        ax.set_ylim(-0.5, 3)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(range(9))
        ax.set_xlabel("Time step", fontsize=9)
        ax.set_yticks([])

        for t in range(9):
            fc = "#C8E6C9" if (detach_at and t % detach_at == 0) else "#90CAF9"
            circle = plt.Circle((t, 1.5), 0.3, fc=fc, ec="black", lw=1.2, zorder=3)
            ax.add_patch(circle)
            ax.text(
                t, 1.5, f"$h_{{{t}}}$", ha="center", va="center", fontsize=7, zorder=4
            )

            if t > 0:
                ax.annotate(
                    "",
                    xy=(t - 0.7, 1.5),
                    xytext=(t - 0.3, 1.5),
                    arrowprops=dict(
                        arrowstyle="<-", color="#2196F3", lw=1.5, mutation_scale=10
                    ),
                )

        # Backward flow
        if detach_at is None:
            for t in range(8, 0, -1):
                ax.annotate(
                    "",
                    xy=(t - 0.7, 2.1),
                    xytext=(t - 0.3, 2.1),
                    arrowprops=dict(
                        arrowstyle="->",
                        color="#E53935",
                        lw=1.5,
                        ls="dashed",
                        mutation_scale=10,
                    ),
                )
        else:
            for t in range(8, 0, -1):
                if t % detach_at == 0:
                    # Stop here
                    continue
                ax.annotate(
                    "",
                    xy=(t - 0.7, 2.1),
                    xytext=(t - 0.3, 2.1),
                    arrowprops=dict(
                        arrowstyle="->",
                        color="#E53935",
                        lw=1.5,
                        ls="dashed",
                        mutation_scale=10,
                    ),
                )
            # Mark detach points
            for t in range(0, 9, detach_at):
                ax.text(
                    t,
                    0.7,
                    "detach_()",
                    fontsize=7,
                    ha="center",
                    color="#4CAF50",
                    fontweight="bold",
                    rotation=45,
                )

    fig.tight_layout()
    fig.savefig(
        OUT / "detach_truncated_bptt.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print(f'Saved: {OUT / "detach_truncated_bptt.png"}')


if __name__ == "__main__":
    fig1_bptt_graph()
    fig2_truncation_strategies()
    fig3_vanishing_exploding()
    fig4_gradient_chain()
    fig5_detach()
    print("Done — all 5 figures generated.")
