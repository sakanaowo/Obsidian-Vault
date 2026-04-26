"""
D2L Buổi 51 - Attention Pooling by Similarity (Nadaraya-Watson)
Tạo ảnh minh họa cho 11.2 Attention Pooling by Similarity
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

plt.rcParams['font.family'] = 'DejaVu Sans'

# ============================================================
# 1. Kernels comparison - 4 kernel shapes
# ============================================================
def gaussian(x, sigma=1.0):
    return np.exp(-x**2 / (2 * sigma**2))

def boxcar(x):
    return (np.abs(x) < 1.0).astype(float)

def constant(x):
    return np.ones_like(x)

def epanechikov(x):
    return np.maximum(1 - np.abs(x), 0.0)

x = np.arange(-2.5, 2.5, 0.01)

fig, axes = plt.subplots(1, 4, sharey=True, figsize=(13, 3.0))
plt.rcParams['font.size'] = 10

kernels = [(gaussian, 'Gaussian\n' + r'$\exp(-\frac{1}{2}\|q-k\|^2)$'),
            (boxcar, 'Boxcar\n' + r'$\mathbb{1}(\|q-k\| \leq 1)$'),
            (epanechikov, 'Epanechikov\n' + r'$\max(0, 1-\|q-k\|)$'),
            (constant, 'Constant\n' + r'$1$')]

colors = ['#1565C0', '#2E7D32', '#E65100', '#7B1FA2']

for ax, (kernel_fn, title), color in zip(axes, kernels, colors):
    y = kernel_fn(x) if kernel_fn != gaussian else gaussian(x)
    ax.plot(x, y, color=color, lw=2.5)
    ax.fill_between(x, y, alpha=0.15, color=color)
    ax.axhline(y=0, color='gray', lw=0.8, ls='--')
    ax.axvline(x=0, color='gray', lw=0.8, ls='--')
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel(title, fontsize=9)
    ax.set_xlim(-2.5, 2.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle('Fig. 11.2.1  Common Kernels in Nadaraya-Watson Regression', fontsize=11, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('/home/sakana/Code/Obsidian-Vault/assets/attachments/d2l-buoi-51/kernel-shapes.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: kernel-shapes.png")

# ============================================================
# 2. Nadaraya-Watson Regression - 4 kernels vs ground truth
# ============================================================
np.random.seed(42)
torch.manual_seed(42)

n = 40
x_train_np = np.sort(np.random.rand(n) * 5)
y_train_np = 2 * np.sin(x_train_np) + x_train_np + np.random.randn(n)
x_val_np = np.arange(0, 5, 0.05)
y_val_np = 2 * np.sin(x_val_np) + x_val_np

def nadaraya_watson(x_train, y_train, x_val, kernel_fn):
    dists = x_train.reshape((-1, 1)) - x_val.reshape((1, -1))
    k = kernel_fn(dists)
    k = np.maximum(k, 0)
    attention_w = k / (k.sum(axis=0) + 1e-8)
    y_hat = y_train @ attention_w
    return y_hat, attention_w

fig, axes = plt.subplots(1, 4, sharey=True, figsize=(13, 3.0))
plt.rcParams['font.size'] = 10

for ax, (kernel_fn, name), color in zip(axes, kernels, colors):
    y_hat, _ = nadaraya_watson(x_train_np, y_train_np, x_val_np, kernel_fn)
    ax.scatter(x_train_np, y_train_np, s=25, c='#424242', alpha=0.6, zorder=5, label='Data')
    ax.plot(x_val_np, y_val_np, 'm--', lw=1.5, label='True $f(x)$', zorder=3)
    ax.plot(x_val_np, y_hat, color=color, lw=2.5, label='NW Estimate', zorder=4)
    ax.set_xlabel(name.split('\n')[0], fontsize=9)
    ax.set_xlim(0, 5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if ax == axes[0]:
        ax.legend(fontsize=7, loc='upper left')

fig.suptitle('Fig. 11.2.2  Nadaraya-Watson Regression Estimates', fontsize=11, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('/home/sakana/Code/Obsidian-Vault/assets/attachments/d2l-buoi-51/nw-regression-comparison.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: nw-regression-comparison.png")

# ============================================================
# 3. Attention weights heatmap - 4 kernels
# ============================================================
x_train_t = torch.tensor(x_train_np, dtype=torch.float32)
x_val_t = torch.tensor(x_val_np, dtype=torch.float32)
y_train_t = torch.tensor(y_train_np, dtype=torch.float32)

fig, axes = plt.subplots(1, 4, figsize=(13, 3.0))
plt.rcParams['font.size'] = 10

for ax, (kernel_fn, name), color in zip(axes, kernels, colors):
    dists = x_train_t.reshape((-1, 1)) - x_val_t.reshape((1, -1))
    k = kernel_fn(dists.numpy())
    k = np.maximum(k, 0)
    attention_w = k / (k.sum(axis=0) + 1e-8)
    im = ax.imshow(attention_w, aspect='auto', cmap='Reds', vmin=0, vmax=0.25)
    ax.set_yticks([])
    ax.set_xticks([0, len(x_val_np)//2, len(x_val_np)-1])
    ax.set_xticklabels([f'{x_val_np[0]:.1f}', f'{x_val_np[len(x_val_np)//2]:.1f}', f'{x_val_np[-1]:.1f}'])
    ax.set_xlabel(f'Queries ($q$)\n{name.split(chr(10))[0]}', fontsize=8)

# Colorbar
fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.90, 0.25, 0.02, 0.5])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Attention Weight', fontsize=9)

fig.text(0.02, 0.02, 'Keys ($k_i$) = Training $x_i$', fontsize=8, style='italic')
fig.suptitle('Fig. 11.2.3  Attention Weights Heatmaps (4 Kernels)', fontsize=11, fontweight='bold', y=1.05)
plt.savefig('/home/sakana/Code/Obsidian-Vault/assets/attachments/d2l-buoi-51/nw-attention-weights.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: nw-attention-weights.png")

# ============================================================
# 4. Gaussian width comparison - sigma effect
# ============================================================
fig, axes = plt.subplots(1, 4, sharey=True, figsize=(13, 3.0))
plt.rcParams['font.size'] = 10

sigmas = [0.1, 0.2, 0.5, 1.0]
sigma_colors = ['#D32F2F', '#F57C00', '#1976D2', '#388E3C']

for ax, sigma, scolor in zip(axes, sigmas, sigma_colors):
    kernel_fn = lambda x, s=sigma: np.exp(-x**2 / (2 * s**2))
    y_hat, _ = nadaraya_watson(x_train_np, y_train_np, x_val_np, kernel_fn)
    ax.scatter(x_train_np, y_train_np, s=20, c='#424242', alpha=0.5, zorder=5)
    ax.plot(x_val_np, y_val_np, 'm--', lw=1.2, label='True')
    ax.plot(x_val_np, y_hat, color=scolor, lw=2.5, label=f'$\\sigma={sigma}$')
    ax.set_xlabel(f'$\\sigma = {sigma}$', fontsize=9)
    ax.set_xlim(0, 5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if ax == axes[0]:
        ax.legend(fontsize=7, loc='upper left')

fig.suptitle('Fig. 11.2.4  Gaussian Kernel Width Effect on NW Estimates', fontsize=11, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('/home/sakana/Code/Obsidian-Vault/assets/attachments/d2l-buoi-51/nw-gaussian-width.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: nw-gaussian-width.png")

# ============================================================
# 5. Attention as nearest-neighbor diagram
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(-0.5, 5.5)
ax.set_ylim(-2, 12)
plt.rcParams['font.size'] = 10

# Plot ground truth
x_smooth = np.arange(0, 5, 0.01)
y_smooth = 2 * np.sin(x_smooth) + x_smooth
ax.plot(x_smooth, y_smooth, 'm--', lw=2, label='True $f(x) = 2\\sin(x) + x$', zorder=2)

# Training data
ax.scatter(x_train_np, y_train_np, s=50, c='#1565C0', alpha=0.8, zorder=5, label='Training data $(x_i, y_i)$')

# Query point
q_example = 2.0
ax.axvline(x=q_example, color='#E65100', lw=2, ls=':', zorder=1, label=f'Query $q = {q_example}$')

# Show attention contribution lines
dists_ex = x_train_np - q_example
kernel_ex = np.exp(-dists_ex**2 / (2 * 0.5**2))
weights_ex = kernel_ex / kernel_ex.sum()

for i in range(n):
    alpha_i = weights_ex[i]
    if alpha_i > 0.01:
        lw = 0.5 + alpha_i * 4
        color_intensity = alpha_i / weights_ex.max()
        ax.plot([q_example, x_train_np[i]], [6.5, y_train_np[i]],
                color='#E65100', lw=lw, alpha=0.3 + color_intensity * 0.5)

ax.annotate('', xy=(q_example, 7), xytext=(q_example, 0),
    arrowprops=dict(arrowstyle='->', color='#E65100', lw=2))
ax.text(q_example, 7.3, f'$f(q)$\n= weighted\naverage', fontsize=9, ha='center', color='#E65100', fontweight='bold')

ax.set_xlabel('$x$', fontsize=11)
ax.set_ylabel('$y$', fontsize=11)
ax.set_title('Fig. 11.2.5  Nadaraya-Watson: Query at $x=2.0$ — Line Width = Attention Weight',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('/home/sakana/Code/Obsidian-Vault/assets/attachments/d2l-buoi-51/nw-query-diagram.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: nw-query-diagram.png")

# ============================================================
# 6. Attention = NW Diagram (How it connects to QKV)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)

ax.text(6, 7.6, 'Fig. 11.2.6  Attention Mechanism = Learned Nadaraya-Watson', fontsize=11, fontweight='bold', ha='center')

# NW Estimator box (left)
ax.add_patch(plt.Rectangle((0.3, 2.5), 5.5, 4.8, fill=True, fc='#E3F2FD', ec='#1565C0', lw=2.5, zorder=2))
ax.text(3.05, 6.8, 'Nadaraya-Watson (11.2)', fontsize=11, ha='center', fontweight='bold', color='#0D47A1')

nw_items = [
    ('Query $q$', '= $x$ (vị trí cần predict)', '#1565C0'),
    ('Keys $k_i$', '= $x_i$ (training features)', '#2E7D32'),
    ('Values $v_i$', '= $y_i$ (training labels)', '#E65100'),
    ('Kernel $\\alpha(q,k_i)$', '= $K(q - k_i)$ (similarity)', '#7B1FA2'),
    ('Output $f(q)$', '= $\\sum_i v_i \\cdot \\frac{\\alpha}{\\sum\\alpha}$', '#C62828'),
]
for i, (label, desc, color) in enumerate(nw_items):
    y_pos = 6.0 - i * 0.72
    ax.text(0.6, y_pos, label, fontsize=9.5, color=color, fontweight='bold', va='center')
    ax.text(3.8, y_pos, desc, fontsize=9.5, color='#263238', va='center')

# Arrow
ax.annotate('', xy=(5.9, 5.0), xytext=(5.85, 5.0),
    arrowprops=dict(arrowstyle='->', color='#C62828', lw=3))
ax.text(6.1, 5.4, 'Learn\nfrom\ndata', fontsize=9, ha='left', color='#C62828', fontweight='bold')

# Learned Attention box (right)
ax.add_patch(mpatches.FancyBboxPatch((6.2, 2.5), 5.5, 4.8,
    boxstyle='round,pad=0.1', fc='#F3E5F5', ec='#7B1FA2', lw=2.5, zorder=2))
ax.text(8.95, 6.8, 'Learned Attention (11.3+)', fontsize=11, ha='center', fontweight='bold', color='#4A148C')

la_items = [
    ('Query $q$', '= $W_Q h$ (learnable projection)', '#1565C0'),
    ('Keys $k_i$', '= $W_K h_i$ (learnable projection)', '#2E7D32'),
    ('Values $v_i$', '= $W_V h_i$ (learnable projection)', '#E65100'),
    ('Scoring $a(q,k_i)$', '= $q^T k_i$ or MLP (learned)', '#7B1FA2'),
    ('Output', '= $\\sum_i \\text{softmax}(a) \\cdot v_i$', '#C62828'),
]
for i, (label, desc, color) in enumerate(la_items):
    y_pos = 6.0 - i * 0.72
    ax.text(6.5, y_pos, label, fontsize=9.5, color=color, fontweight='bold', va='center')
    ax.text(9.7, y_pos, desc, fontsize=9.5, color='#263238', va='center')

# Key insight box at bottom
ax.add_patch(plt.Rectangle((0.3, 0.2), 11.4, 1.9, fill=True, fc='#FFF9C4', ec='#F9A825', lw=2, zorder=2))
ax.text(6, 1.7, 'Key Insight', fontsize=10, ha='center', fontweight='bold', color='#E65100')
ax.text(6, 1.1,
    'NW = hand-crafted kernel (fixed, no training) | Learned Attention = kernel learned from data\n'
    'Both follow: Output = $\\sum_i$ AttentionWeight$_i$ $\\times$ Value$_i$  |  AttentionWeight$_i$ = $\\frac{\\alpha(q,k_i)}{\\sum_j \\alpha(q,k_j)}$',
    fontsize=9, ha='center', va='center', color='#263238', style='italic',
    bbox=dict(boxstyle='round,pad=0.2', fc='#FFF9C4', alpha=0))

plt.tight_layout()
plt.savefig('/home/sakana/Code/Obsidian-Vault/assets/attachments/d2l-buoi-51/nw-vs-learned-attention.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: nw-vs-learned-attention.png")

print("\nAll visualizations for Buổi 51 saved successfully.")
