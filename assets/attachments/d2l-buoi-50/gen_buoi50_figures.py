"""
D2L Buổi 50 - Attention Mechanism Visualization
Tạo ảnh minh họa cho 11.1 Queries, Keys, and Values
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

def show_heatmaps_simple(matrices, xlabel, ylabel, titles=None, figsize=(3, 3), cmap='Reds'):
    """Custom heatmap that works standalone."""
    num_rows, num_cols, _, _ = matrices.shape
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            arr = matrix.detach().numpy() if hasattr(matrix, 'detach') else np.array(matrix)
            pcm = ax.imshow(arr, cmap=cmap, vmin=0, vmax=1)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(pcm, ax=axes, shrink=0.6)
    return fig

# ============================================================
# 1. Attention Mechanism Flow Diagram
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)
ax.axis('off')
ax.set_title('Attention Mechanism: Queries, Keys, and Values', fontsize=14, fontweight='bold', pad=15)

ax.text(1.5, 4.5, 'Query\n(q)', fontsize=12, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', fc='#BBDEFB', ec='#1565C0', lw=2))
ax.text(1.5, 2.5, 'Key\n(k_i)', fontsize=12, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', fc='#C8E6C9', ec='#2E7D32', lw=2))
ax.text(1.5, 0.8, 'Value\n(v_i)', fontsize=12, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', fc='#FFCCBC', ec='#E65100', lw=2))

ax.add_patch(mpatches.FancyBboxPatch((3.5, 3.3), 2.5, 1.2, boxstyle='round,pad=0.1', fc='#E1BEE7', ec='#7B1FA2', lw=2))
ax.text(4.75, 3.9, 'Compatibility\na(q, k_i)', fontsize=11, ha='center', va='center')
ax.text(4.75, 3.4, 'dot product / MLP', fontsize=9, ha='center', va='center', color='gray')

ax.add_patch(mpatches.FancyBboxPatch((6.8, 3.3), 2.2, 1.2, boxstyle='round,pad=0.1', fc='#FFCCBC', ec='#BF360C', lw=2))
ax.text(7.9, 3.9, 'Softmax\n+ Normalize', fontsize=11, ha='center', va='center')
ax.text(7.9, 3.4, 'alpha_i = exp(a_i) / sum', fontsize=9, ha='center', va='center', color='gray')

ax.add_patch(mpatches.FancyBboxPatch((9.8, 2.8), 2, 1.8, boxstyle='round,pad=0.1', fc='#B2DFDB', ec='#00838F', lw=2))
ax.text(10.8, 3.9, 'Weighted Sum', fontsize=11, ha='center', va='center')
ax.text(10.8, 3.3, 'sum_i alpha_i * v_i', fontsize=10, ha='center', va='center', color='gray')

ax.add_patch(mpatches.FancyBboxPatch((9.8, 0.3), 2, 1.6, boxstyle='round,pad=0.1', fc='#DCEDC8', ec='#33691E', lw=2.5))
ax.text(10.8, 1.2, 'Output\n(context)', fontsize=12, ha='center', va='center', fontweight='bold')

arrow_style = dict(arrowstyle='->', color='#424242', lw=1.5)
ax.annotate('', xy=(3.4, 4.0), xytext=(2.7, 4.0), arrowprops=arrow_style)
ax.annotate('', xy=(6.0, 3.9), xytext=(6.0, 2.9), arrowprops=dict(arrowstyle='->', color='#424242', lw=1.5))
ax.annotate('', xy=(9.7, 3.9), xytext=(9.1, 3.9), arrowprops=arrow_style)
ax.annotate('', xy=(10.8, 2.8), xytext=(10.8, 1.9), arrowprops=arrow_style)
ax.annotate('', xy=(3.4, 2.0), xytext=(2.7, 2.0), arrowprops=arrow_style)
ax.annotate('', xy=(10.3, 1.0), xytext=(10.3, 0.7), arrowprops=dict(arrowstyle='->', color='#424242', lw=1.5, linestyle='dashed'))
ax.text(11.2, 1.0, 'v_i goes\ndirectly', fontsize=8, ha='left', color='#616161')

ax.text(5.0, 4.5, 'attention\nweights', fontsize=9, ha='center', color='#7B1FA2')
ax.text(8.5, 4.5, 'normalized\nweights', fontsize=9, ha='center', color='#BF360C')

plt.tight_layout()
plt.savefig('/home/sakana/Code/Obsidian-Vault/assets/attachments/d2l-buoi-50/attention-flow.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: attention-flow.png")

# ============================================================
# 2. Heatmap: 3 special cases
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

# Identity
mat_identity = torch.eye(8).reshape(1, 1, 8, 8)
pcm = axes[0].imshow(mat_identity.squeeze().numpy(), cmap='Reds', vmin=0, vmax=1)
axes[0].set_title('Identity (hard)\nAll weight on matching key', fontsize=10)
axes[0].set_xlabel('Keys')
axes[0].set_ylabel('Queries')
axes[0].set_xticks([])
axes[0].set_yticks([])

# Uniform
mat_uniform = torch.ones(8, 8) / 8
pcm = axes[1].imshow(mat_uniform.numpy(), cmap='Reds', vmin=0, vmax=1)
axes[1].set_title('Uniform (average pooling)\nEqual weight on all', fontsize=10)
axes[1].set_xlabel('Keys')
axes[1].set_xticks([])
axes[1].set_yticks([])

# Softmax of dot-product
torch.manual_seed(42)
Q_sim = torch.randn(6, 8)
K_sim = torch.randn(6, 8)
scores = Q_sim @ K_sim.T
weights = torch.softmax(scores, dim=1)
pcm = axes[2].imshow(weights.numpy(), cmap='Reds', vmin=0, vmax=1)
axes[2].set_title('Softmax (learned similarity)\nLearned soft matching', fontsize=10)
axes[2].set_xlabel('Keys')
axes[2].set_xticks([])
axes[2].set_yticks([])

for ax in axes:
    ax.set_ylabel('Queries')

fig.colorbar(pcm, ax=axes, shrink=0.7)
fig.suptitle('Three Special Cases of Attention Weights', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/home/sakana/Code/Obsidian-Vault/assets/attachments/d2l-buoi-50/attention-special-cases.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: attention-special-cases.png")

# ============================================================
# 3. Database analogy
# ============================================================
fig, ax = plt.subplots(figsize=(12, 5))
ax.axis('off')
ax.set_title('Attention as a "Smart Database"', fontsize=13, fontweight='bold', pad=10)

# Traditional DB
ax.add_patch(mpatches.FancyBboxPatch((0.05, 0.1), 5.0, 5.5, boxstyle='round,pad=0.2', fc='#ECEFF1', ec='#607D8B', lw=2))
ax.text(2.55, 5.3, 'Traditional Database', fontsize=11, ha='center', fontweight='bold')
ax.text(2.55, 4.8, 'EXACT match only', fontsize=10, ha='center', color='#C62828')

headers = [('Key', 'Value'), ('Zhang', 'Aston'), ('Lipton', 'Zachary'), ('Li', 'Mu'), ('Smola', 'Alex'), ('Hu', 'Rachel')]
y_start = 4.1
row_h = 0.6
for i, (k, v) in enumerate(headers):
    fc = '#CFD8DC' if i == 0 else '#FFFFFF'
    ax.add_patch(mpatches.FancyBboxPatch((0.3, y_start - i*row_h), 2.2, 0.45, boxstyle='round,pad=0.05', fc=fc, ec='#90A4AE', lw=1))
    ax.add_patch(mpatches.FancyBboxPatch((2.7, y_start - i*row_h), 2.2, 0.45, boxstyle='round,pad=0.05', fc=fc, ec='#90A4AE', lw=1))
    ax.text(1.4, y_start - i*row_h + 0.22, k, fontsize=9, ha='center', va='center')
    ax.text(3.8, y_start - i*row_h + 0.22, v, fontsize=9, ha='center', va='center')

ax.text(2.55, 0.5, 'Query "Li" -> Returns "Mu"\nNo exact match -> No result', fontsize=9, ha='center', va='center', style='italic', color='#616161')

# Arrow
ax.annotate('', xy=(5.3, 2.8), xytext=(5.1, 2.8), arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2))
ax.text(5.2, 3.2, 'soft\nmatch', fontsize=8, ha='center', color='#1565C0')

# Attention DB
ax.add_patch(mpatches.FancyBboxPatch((5.5, 0.1), 6.3, 5.5, boxstyle='round,pad=0.2', fc='#E8EAF6', ec='#3949AB', lw=2))
ax.text(8.65, 5.3, 'Attention Database', fontsize=11, ha='center', fontweight='bold')
ax.text(8.65, 4.8, 'APPROXIMATE match (weighted)', fontsize=10, ha='center', color='#1B5E20')

headers2 = [('Key', 'Value', 'alpha'), ('Zhang', 'Aston', '0.05'), ('Lipton', 'Zachary', '0.15'),
            ('Li', 'Mu', '0.70'), ('Smola', 'Alex', '0.07'), ('Hu', 'Rachel', '0.03')]
y_start2 = 4.1
row_h2 = 0.62
for i, (k, v, w) in enumerate(headers2):
    fc = '#C5CAE9' if i == 0 else '#FFFFFF'
    ax.add_patch(mpatches.FancyBboxPatch((5.7, y_start2 - i*row_h2), 1.8, 0.5, boxstyle='round,pad=0.05', fc=fc, ec='#7986CB', lw=1))
    ax.add_patch(mpatches.FancyBboxPatch((7.7, y_start2 - i*row_h2), 1.8, 0.5, boxstyle='round,pad=0.05', fc=fc, ec='#7986CB', lw=1))
    ax.add_patch(mpatches.FancyBboxPatch((9.7, y_start2 - i*row_h2), 1.8, 0.5, boxstyle='round,pad=0.05', fc=fc, ec='#7986CB', lw=1))
    ax.text(6.6, y_start2 - i*row_h2 + 0.25, k, fontsize=9, ha='center', va='center')
    ax.text(8.6, y_start2 - i*row_h2 + 0.25, v, fontsize=9, ha='center', va='center')
    weight_color = '#1B5E20' if i > 0 else '#424242'
    ax.text(10.6, y_start2 - i*row_h2 + 0.25, w, fontsize=9, ha='center', va='center', color=weight_color, fontweight='bold' if i > 0 else 'normal')

ax.text(8.65, 0.4, 'Query "Li" -> "Mu"(0.70) + "Zachary"(0.15) + "Alex"(0.07) + ...\nOutput = 0.70*"Mu" + 0.15*"Zachary" + 0.07*"Alex" + 0.03*"Rachel"',
        fontsize=9, ha='center', va='center', style='italic', color='#1B5E20')

plt.tight_layout()
plt.savefig('/home/sakana/Code/Obsidian-Vault/assets/attachments/d2l-buoi-50/attention-database-analogy.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: attention-database-analogy.png")

# ============================================================
# 4. Special cases comparison table
# ============================================================
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')

cases = [
    (r'\alpha(q, k_i) >= 0 (nonnegative)', r'sum_i alpha = 1 (convex combination)'),
    (r'Exact match: one alpha=1, rest 0', r'alpha_i in {0, 1}'),
    (r'All equal: alpha_i = 1/m', r'm = number of keys'),
    (r'alpha = softmax(a(q,k_i))', r'Differentiable, nonnegative'),
]
row_labels = ['Nonnegative', 'Hard attention', 'Uniform pooling', 'Softmax attention']

ax.set_title('Special Cases of Attention Weights alpha(q, k_i)', fontsize=12, fontweight='bold', pad=10)
col_labels = ['Case', 'Condition', 'Constraint']

table_data = []
for i, case in enumerate(cases):
    table_data.append([row_labels[i], case[0], case[1]])

table = ax.table(cellText=table_data, colLabels=col_labels,
                 loc='center', cellLoc='center', colWidths=[0.22, 0.42, 0.36])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 2.0)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#1565C0')
        cell.set_text_props(color='white', fontweight='bold')
    else:
        colors = ['#E3F2FD', '#E8F5E9', '#FFF3E0', '#F3E5F5']
        cell.set_facecolor(colors[(row-1) % 4])

plt.tight_layout()
plt.savefig('/home/sakana/Code/Obsidian-Vault/assets/attachments/d2l-buoi-50/attention-special-cases-table.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: attention-special-cases-table.png")

print("\nAll visualizations saved successfully.")
