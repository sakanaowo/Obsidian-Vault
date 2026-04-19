"""Generate LSTM figures for Buổi 43."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 150

OUT = Path(__file__).parent

# ============================================================
# Figure 1: LSTM Cell Architecture — Full data flow
# ============================================================
def fig1_lstm_cell():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(-1, 13)
    ax.set_ylim(-1, 9)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Kiến trúc một LSTM Cell', fontsize=16, fontweight='bold', pad=15)

    # Colors
    c_forget = '#e74c3c'
    c_input = '#2ecc71'
    c_output = '#3498db'
    c_candidate = '#f39c12'
    c_cell = '#9b59b6'
    c_arrow = '#2c3e50'

    # --- Cell state line (top highway) ---
    ax.annotate('', xy=(11.5, 7), xytext=(0.5, 7),
                arrowprops=dict(arrowstyle='->', color=c_cell, lw=3))
    ax.text(6, 7.5, 'Trạng thái ô nhớ  $C_t$  (đường cao tốc)', ha='center',
            fontsize=13, color=c_cell, fontweight='bold')

    # --- Forget gate ---
    gate_y = 4.5
    ax.add_patch(plt.Rectangle((1.5, gate_y - 0.6), 2, 1.2, fc=c_forget, alpha=0.3, ec=c_forget, lw=2, zorder=5))
    ax.text(2.5, gate_y, '$\\sigma$', ha='center', va='center', fontsize=14, fontweight='bold', color=c_forget)
    ax.text(2.5, gate_y - 1.2, 'Cổng quên\n$F_t$', ha='center', va='center', fontsize=10, color=c_forget)

    # Multiply symbol for forget gate
    ax.plot(2.5, 7, 'o', ms=18, mfc='white', mec=c_forget, mew=2, zorder=6)
    ax.text(2.5, 7, '×', ha='center', va='center', fontsize=14, fontweight='bold', color=c_forget)
    ax.annotate('', xy=(2.5, 6.7), xytext=(2.5, gate_y + 0.6),
                arrowprops=dict(arrowstyle='->', color=c_forget, lw=1.5))

    # --- Input gate ---
    ax.add_patch(plt.Rectangle((4.5, gate_y - 0.6), 2, 1.2, fc=c_input, alpha=0.3, ec=c_input, lw=2, zorder=5))
    ax.text(5.5, gate_y, '$\\sigma$', ha='center', va='center', fontsize=14, fontweight='bold', color=c_input)
    ax.text(5.5, gate_y - 1.2, 'Cổng đầu vào\n$I_t$', ha='center', va='center', fontsize=10, color=c_input)

    # Candidate
    ax.add_patch(plt.Rectangle((4.5, 2 - 0.6), 2, 1.2, fc=c_candidate, alpha=0.3, ec=c_candidate, lw=2, zorder=5))
    ax.text(5.5, 2, 'tanh', ha='center', va='center', fontsize=12, fontweight='bold', color=c_candidate)
    ax.text(5.5, 0.6, 'Ứng viên\n$\\tilde{C}_t$', ha='center', va='center', fontsize=10, color=c_candidate)

    # Multiply symbol for input gate
    ax.plot(5.5, 5.8, 'o', ms=14, mfc='white', mec=c_input, mew=2, zorder=6)
    ax.text(5.5, 5.8, '×', ha='center', va='center', fontsize=12, fontweight='bold', color=c_input)
    ax.annotate('', xy=(5.5, 5.5), xytext=(5.5, gate_y + 0.6),
                arrowprops=dict(arrowstyle='->', color=c_input, lw=1.5))
    ax.annotate('', xy=(5.5, 5.5), xytext=(5.5, 2.6),
                arrowprops=dict(arrowstyle='->', color=c_candidate, lw=1.5, ls='--'))

    # Plus symbol on cell state line
    ax.plot(5.5, 7, 'o', ms=18, mfc='white', mec=c_cell, mew=2, zorder=6)
    ax.text(5.5, 7, '+', ha='center', va='center', fontsize=16, fontweight='bold', color=c_cell)
    ax.annotate('', xy=(5.5, 6.7), xytext=(5.5, 6.1),
                arrowprops=dict(arrowstyle='->', color=c_input, lw=1.5))

    # --- Output gate ---
    ax.add_patch(plt.Rectangle((8, gate_y - 0.6), 2, 1.2, fc=c_output, alpha=0.3, ec=c_output, lw=2, zorder=5))
    ax.text(9, gate_y, '$\\sigma$', ha='center', va='center', fontsize=14, fontweight='bold', color=c_output)
    ax.text(9, gate_y - 1.2, 'Cổng đầu ra\n$O_t$', ha='center', va='center', fontsize=10, color=c_output)

    # tanh on cell state
    ax.add_patch(plt.Rectangle((8, 5.7), 2, 1.0, fc=c_cell, alpha=0.2, ec=c_cell, lw=1.5, zorder=5))
    ax.text(9, 6.2, 'tanh', ha='center', va='center', fontsize=11, color=c_cell)
    # Arrow from cell state down to tanh
    ax.annotate('', xy=(9, 6.7), xytext=(9, 7),
                arrowprops=dict(arrowstyle='->', color=c_cell, lw=1.5))

    # Multiply for output
    ax.plot(9, 5.2, 'o', ms=14, mfc='white', mec=c_output, mew=2, zorder=6)
    ax.text(9, 5.2, '×', ha='center', va='center', fontsize=12, fontweight='bold', color=c_output)
    ax.annotate('', xy=(9, 5.5), xytext=(9, 5.7),
                arrowprops=dict(arrowstyle='->', color=c_cell, lw=1.5))
    ax.annotate('', xy=(9, 5.5), xytext=(9, gate_y + 0.6),
                arrowprops=dict(arrowstyle='->', color=c_output, lw=1.5))

    # H_t output
    ax.annotate('', xy=(11.5, 5.2), xytext=(9.3, 5.2),
                arrowprops=dict(arrowstyle='->', color=c_arrow, lw=2))
    ax.text(11.8, 5.2, '$H_t$', fontsize=14, fontweight='bold', va='center')

    # Inputs
    ax.text(-0.5, 7, '$C_{t-1}$', fontsize=14, fontweight='bold', va='center', ha='right')
    ax.text(-0.5, 3, '$H_{t-1}$\n$X_t$', fontsize=12, va='center', ha='right', color=c_arrow)

    # Input arrows to gates
    for gx in [2.5, 5.5, 9]:
        ax.annotate('', xy=(gx, gate_y - 0.6), xytext=(gx, 3.2),
                    arrowprops=dict(arrowstyle='->', color=c_arrow, lw=1, ls=':'))
    ax.annotate('', xy=(5.5, 1.4), xytext=(5.5, 3.2),
                arrowprops=dict(arrowstyle='->', color=c_arrow, lw=1, ls=':'))

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=c_forget, alpha=0.3, edgecolor=c_forget, label='Cổng quên ($F_t$)'),
        mpatches.Patch(facecolor=c_input, alpha=0.3, edgecolor=c_input, label='Cổng đầu vào ($I_t$)'),
        mpatches.Patch(facecolor=c_output, alpha=0.3, edgecolor=c_output, label='Cổng đầu ra ($O_t$)'),
        mpatches.Patch(facecolor=c_candidate, alpha=0.3, edgecolor=c_candidate, label='Ứng viên ($\\tilde{C}_t$)'),
        mpatches.Patch(facecolor=c_cell, alpha=0.3, edgecolor=c_cell, label='Trạng thái ô nhớ ($C_t$)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(OUT / 'lstm_cell_architecture.png', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("OK: lstm_cell_architecture.png")

# ============================================================
# Figure 2: Gate behavior — sigmoid as switch
# ============================================================
def fig2_gate_behavior():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    x = np.linspace(-6, 6, 300)
    sigmoid = 1 / (1 + np.exp(-x))
    tanh_x = np.tanh(x)

    # Forget gate
    ax = axes[0]
    ax.fill_between(x, sigmoid, alpha=0.3, color='#e74c3c')
    ax.plot(x, sigmoid, color='#e74c3c', lw=2)
    ax.axhline(0.5, ls='--', color='gray', alpha=0.5)
    ax.axhline(0, ls='-', color='gray', alpha=0.3)
    ax.axhline(1, ls='-', color='gray', alpha=0.3)
    ax.set_title('Cổng quên $F_t = \\sigma(...)$', fontweight='bold', color='#e74c3c')
    ax.set_xlabel('Giá trị đầu vào')
    ax.set_ylabel('Xác suất giữ lại')
    ax.annotate('Gần 1 = GIỮ LẠI\nký ức cũ', xy=(4, 0.98), fontsize=9,
                ha='center', color='#e74c3c', fontweight='bold')
    ax.annotate('Gần 0 = XÓA\nký ức cũ', xy=(-4, 0.02), fontsize=9,
                ha='center', color='#e74c3c', fontweight='bold')

    # Input gate
    ax = axes[1]
    ax.fill_between(x, sigmoid, alpha=0.3, color='#2ecc71')
    ax.plot(x, sigmoid, color='#2ecc71', lw=2)
    ax.plot(x, tanh_x * 0.5 + 0.5, color='#f39c12', lw=2, ls='--', label='tanh (ứng viên)')
    ax.axhline(0.5, ls='--', color='gray', alpha=0.5)
    ax.set_title('Cổng đầu vào $I_t = \\sigma(...)$', fontweight='bold', color='#2ecc71')
    ax.set_xlabel('Giá trị đầu vào')
    ax.set_ylabel('Xác suất ghi mới')
    ax.annotate('Gần 1 = GHI NHẬN\nthông tin mới', xy=(4, 0.98), fontsize=9,
                ha='center', color='#2ecc71', fontweight='bold')
    ax.annotate('Gần 0 = BỎ QUA\nthông tin mới', xy=(-4, 0.02), fontsize=9,
                ha='center', color='#2ecc71', fontweight='bold')
    ax.legend(fontsize=8, loc='center right')

    # Output gate
    ax = axes[2]
    ax.fill_between(x, sigmoid, alpha=0.3, color='#3498db')
    ax.plot(x, sigmoid, color='#3498db', lw=2)
    ax.axhline(0.5, ls='--', color='gray', alpha=0.5)
    ax.set_title('Cổng đầu ra $O_t = \\sigma(...)$', fontweight='bold', color='#3498db')
    ax.set_xlabel('Giá trị đầu vào')
    ax.set_ylabel('Xác suất xuất ra')
    ax.annotate('Gần 1 = XUẤT RA\ncho layer tiếp', xy=(4, 0.98), fontsize=9,
                ha='center', color='#3498db', fontweight='bold')
    ax.annotate('Gần 0 = GIỮ BÍ MẬT\nkhông xuất', xy=(-4, 0.02), fontsize=9,
                ha='center', color='#3498db', fontweight='bold')

    for ax in axes:
        ax.set_ylim(-0.1, 1.15)
        ax.grid(True, alpha=0.2)

    fig.suptitle('Ba cổng LSTM — Tất cả đều dùng sigmoid (0 đến 1) để "đóng/mở"',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / 'lstm_gate_behavior.png', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("OK: lstm_gate_behavior.png")

# ============================================================
# Figure 3: Cell state update — step by step
# ============================================================
def fig3_cell_state_update():
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Simulated values
    C_prev = np.array([0.8, -0.3, 0.5, 0.1, -0.7])
    F = np.array([0.9, 0.1, 0.8, 0.3, 0.95])
    I = np.array([0.2, 0.85, 0.1, 0.9, 0.05])
    C_tilde = np.array([0.6, 0.4, -0.2, 0.7, -0.5])
    C_new = F * C_prev + I * C_tilde
    labels = [f'd{i+1}' for i in range(5)]

    # Step 1: C_prev
    ax = axes[0]
    bars = ax.bar(labels, C_prev, color='#9b59b6', alpha=0.7)
    ax.set_title('Bước 1: $C_{t-1}$\n(ký ức cũ)', fontweight='bold')
    ax.set_ylim(-1, 1)
    ax.axhline(0, color='gray', lw=0.5)
    ax.grid(axis='y', alpha=0.2)

    # Step 2: F * C_prev
    ax = axes[1]
    kept = F * C_prev
    colors = ['#e74c3c' if f < 0.5 else '#27ae60' for f in F]
    ax.bar(labels, kept, color=colors, alpha=0.7)
    for i, f in enumerate(F):
        ax.text(i, kept[i] + 0.05 * np.sign(kept[i]), f'F={f:.1f}',
                ha='center', fontsize=8, color='gray')
    ax.set_title('Bước 2: $F_t \\odot C_{t-1}$\n(quên bớt)', fontweight='bold')
    ax.set_ylim(-1, 1)
    ax.axhline(0, color='gray', lw=0.5)
    ax.grid(axis='y', alpha=0.2)

    # Step 3: I * C_tilde
    ax = axes[2]
    new_info = I * C_tilde
    ax.bar(labels, new_info, color='#f39c12', alpha=0.7)
    for i, (ig, ct) in enumerate(zip(I, C_tilde)):
        ax.text(i, new_info[i] + 0.05 * np.sign(new_info[i]),
                f'I={ig:.1f}', ha='center', fontsize=8, color='gray')
    ax.set_title('Bước 3: $I_t \\odot \\tilde{C}_t$\n(thông tin mới)', fontweight='bold')
    ax.set_ylim(-1, 1)
    ax.axhline(0, color='gray', lw=0.5)
    ax.grid(axis='y', alpha=0.2)

    # Step 4: C_new
    ax = axes[3]
    ax.bar(labels, C_new, color='#8e44ad', alpha=0.7)
    ax.set_title('Bước 4: $C_t = F \\odot C_{t-1} + I \\odot \\tilde{C}_t$\n(ký ức mới)', fontweight='bold')
    ax.set_ylim(-1, 1)
    ax.axhline(0, color='gray', lw=0.5)
    ax.grid(axis='y', alpha=0.2)

    fig.suptitle('Quá trình cập nhật trạng thái ô nhớ — từng bước', fontsize=14, fontweight='bold', y=1.05)
    fig.tight_layout()
    fig.savefig(OUT / 'lstm_cell_state_update.png', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("OK: lstm_cell_state_update.png")

# ============================================================
# Figure 4: RNN vs LSTM gradient flow comparison
# ============================================================
def fig4_gradient_flow():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    T = 20
    steps = np.arange(1, T + 1)

    # Vanilla RNN
    ax = axes[0]
    lambda_vals = [0.7, 0.9, 1.0, 1.1, 1.3]
    for lam in lambda_vals:
        grad = lam ** steps
        ax.semilogy(steps, grad, lw=2, label=f'$|\\lambda|={lam}$')
    ax.axhline(1, ls='--', color='gray', alpha=0.5)
    ax.set_title('Vanilla RNN: gradient $\\propto |\\lambda|^k$', fontweight='bold', fontsize=12)
    ax.set_xlabel('Khoảng cách thời gian (k)')
    ax.set_ylabel('Độ lớn gradient (log)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(1e-8, 1e8)
    ax.fill_between(steps, 1e-8, 1e-2, alpha=0.1, color='blue', label='_nolegend_')
    ax.fill_between(steps, 1e2, 1e8, alpha=0.1, color='red', label='_nolegend_')
    ax.text(15, 1e-5, 'Vanishing', fontsize=11, color='blue', fontweight='bold', ha='center')
    ax.text(15, 1e5, 'Exploding', fontsize=11, color='red', fontweight='bold', ha='center')

    # LSTM
    ax = axes[1]
    # LSTM with forget gate ~1 keeps gradient stable
    for f_val, label, color in [(0.99, 'F gần 1 (nhớ lâu)', '#2ecc71'),
                                  (0.9, 'F = 0.9', '#f39c12'),
                                  (0.5, 'F = 0.5', '#e74c3c')]:
        grad_lstm = f_val ** steps
        ax.semilogy(steps, grad_lstm, lw=2, label=label, color=color)

    ax.axhline(1, ls='--', color='gray', alpha=0.5)
    ax.set_title('LSTM: gradient qua đường $C_t$\n(cổng quên kiểm soát)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Khoảng cách thời gian (k)')
    ax.set_ylabel('Độ lớn gradient (log)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(1e-8, 1e8)
    ax.text(10, 2, 'Gradient ổn định\nkhi F gần 1', fontsize=11,
            color='#2ecc71', fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    fig.suptitle('So sánh: Vanilla RNN (không kiểm soát) vs LSTM (cổng kiểm soát gradient)',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / 'rnn_vs_lstm_gradient.png', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("OK: rnn_vs_lstm_gradient.png")

# ============================================================
# Figure 5: LSTM formulas summary — visual equation sheet
# ============================================================
def fig5_formulas():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)

    title_props = dict(fontsize=13, fontweight='bold', va='center')
    eq_props = dict(fontsize=14, va='center', fontfamily='serif')

    y = 7.2
    ax.text(5, y, 'Tóm tắt công thức LSTM', fontsize=16, fontweight='bold',
            ha='center', va='center')

    rows = [
        ('Cổng quên', '#e74c3c',
         '$F_t = \\sigma(X_t W_{xf} + H_{t-1} W_{hf} + b_f)$',
         'Giữ hay xóa ký ức cũ?'),
        ('Cổng đầu vào', '#2ecc71',
         '$I_t = \\sigma(X_t W_{xi} + H_{t-1} W_{hi} + b_i)$',
         'Ghi nhận bao nhiêu thông tin mới?'),
        ('Ứng viên', '#f39c12',
         '$\\tilde{C}_t = \\tanh(X_t W_{xc} + H_{t-1} W_{hc} + b_c)$',
         'Thông tin mới là gì?'),
        ('Trạng thái ô nhớ', '#9b59b6',
         '$C_t = F_t \\odot C_{t-1} + I_t \\odot \\tilde{C}_t$',
         'Kết hợp cũ + mới'),
        ('Cổng đầu ra', '#3498db',
         '$O_t = \\sigma(X_t W_{xo} + H_{t-1} W_{ho} + b_o)$',
         'Xuất bao nhiêu cho layer tiếp?'),
        ('Trạng thái ẩn', '#2c3e50',
         '$H_t = O_t \\odot \\tanh(C_t)$',
         'Output cuối cùng'),
    ]

    y = 6.3
    for name, color, formula, desc in rows:
        ax.add_patch(plt.Rectangle((0.2, y - 0.4), 2.2, 0.8, fc=color, alpha=0.15,
                                     ec=color, lw=2, zorder=5))
        ax.text(1.3, y, name, ha='center', color=color, **title_props)
        ax.text(4.2, y, formula, ha='left', **eq_props)
        ax.text(9.5, y, desc, ha='right', fontsize=10, va='center', color='gray', style='italic')
        y -= 1.05

    fig.tight_layout()
    fig.savefig(OUT / 'lstm_formulas_summary.png', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("OK: lstm_formulas_summary.png")


if __name__ == '__main__':
    fig1_lstm_cell()
    fig2_gate_behavior()
    fig3_cell_state_update()
    fig4_gradient_flow()
    fig5_formulas()
    print("All figures generated!")
