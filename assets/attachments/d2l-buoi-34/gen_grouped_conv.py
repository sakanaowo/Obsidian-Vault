import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# ===== LEFT: Standard Convolution =====
ax = axes[0]
ax.set_xlim(-1, 11)
ax.set_ylim(-1, 9)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Standard Convolution\n$\mathcal{O}(c_i \\cdot c_o)$', fontsize=16, fontweight='bold', pad=15)

# Input channels (c_i = 4)
colors_in = ['#FF6B6B', '#FFA07A', '#FFD93D', '#6BCB77']
for i, c in enumerate(colors_in):
    rect = mpatches.FancyBboxPatch((0.5, 7 - i*1.5), 1.5, 1.0, 
                                     boxstyle="round,pad=0.05", facecolor=c, edgecolor='#333', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(1.25, 7.5 - i*1.5, f'$c_{{i{i+1}}}$', ha='center', va='center', fontsize=11, fontweight='bold')

ax.text(1.25, 8.5, 'Input\n($c_i$ channels)', ha='center', va='center', fontsize=12, fontweight='bold')

# Kernel matrix (full connection)
colors_out = ['#4ECDC4', '#45B7D1', '#A78BFA', '#F472B6']
for i in range(4):  # input
    for j in range(4):  # output
        ax.annotate('', xy=(5.5, 7.5 - j*1.5), xytext=(2.2, 7.5 - i*1.5),
                    arrowprops=dict(arrowstyle='->', color='#666', lw=0.8, alpha=0.4))

# Kernel box
kernel_rect = mpatches.FancyBboxPatch((4, 1.5), 3, 6, boxstyle="round,pad=0.1",
                                        facecolor='#FFF3E0', edgecolor='#FF9800', linewidth=2, alpha=0.5)
ax.add_patch(kernel_rect)
ax.text(5.5, 4.5, 'Kernel\n$c_i \\times c_o$\nmatrix', ha='center', va='center', fontsize=12, 
        fontweight='bold', color='#E65100')

# Output channels
for i, c in enumerate(colors_out):
    rect = mpatches.FancyBboxPatch((8.5, 7 - i*1.5), 1.5, 1.0,
                                     boxstyle="round,pad=0.05", facecolor=c, edgecolor='#333', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(9.25, 7.5 - i*1.5, f'$c_{{o{i+1}}}$', ha='center', va='center', fontsize=11, fontweight='bold')

ax.text(9.25, 8.5, 'Output\n($c_o$ channels)', ha='center', va='center', fontsize=12, fontweight='bold')

# Cost annotation
ax.text(5.5, 0.3, 'Chi phi: $\mathcal{O}(c_i \\cdot c_o \\cdot k^2)$\nMoi input lien ket voi MOI output', 
        ha='center', va='center', fontsize=11, color='#C62828',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor='#EF5350'))

# ===== RIGHT: Grouped Convolution (g=2) =====
ax = axes[1]
ax.set_xlim(-1, 11)
ax.set_ylim(-1, 9)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Grouped Convolution ($g=2$)\n$\mathcal{O}(c_i \\cdot c_o / g)$', fontsize=16, fontweight='bold', pad=15)

# Input channels - Group 1
for i, c in enumerate(['#FF6B6B', '#FFA07A']):
    rect = mpatches.FancyBboxPatch((0.5, 7 - i*1.5), 1.5, 1.0,
                                     boxstyle="round,pad=0.05", facecolor=c, edgecolor='#333', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(1.25, 7.5 - i*1.5, f'$c_{{i{i+1}}}$', ha='center', va='center', fontsize=11, fontweight='bold')

# Input channels - Group 2
for i, c in enumerate(['#FFD93D', '#6BCB77']):
    rect = mpatches.FancyBboxPatch((0.5, 3.5 - i*1.5), 1.5, 1.0,
                                     boxstyle="round,pad=0.05", facecolor=c, edgecolor='#333', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(1.25, 4.0 - i*1.5, f'$c_{{i{i+3}}}$', ha='center', va='center', fontsize=11, fontweight='bold')

ax.text(1.25, 8.5, 'Input\n($c_i$ channels)', ha='center', va='center', fontsize=12, fontweight='bold')

# Group 1 kernel
g1_rect = mpatches.FancyBboxPatch((4, 5.5), 3, 2.5, boxstyle="round,pad=0.1",
                                    facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2)
ax.add_patch(g1_rect)
ax.text(5.5, 6.75, 'Nhom 1\n$\\frac{c_i}{g} \\times \\frac{c_o}{g}$', ha='center', va='center', fontsize=10, 
        fontweight='bold', color='#0D47A1')

# Group 2 kernel
g2_rect = mpatches.FancyBboxPatch((4, 1.5), 3, 2.5, boxstyle="round,pad=0.1",
                                    facecolor='#E8F5E9', edgecolor='#388E3C', linewidth=2)
ax.add_patch(g2_rect)
ax.text(5.5, 2.75, 'Nhom 2\n$\\frac{c_i}{g} \\times \\frac{c_o}{g}$', ha='center', va='center', fontsize=10, 
        fontweight='bold', color='#1B5E20')

# Arrows Group 1
for i in range(2):
    for j in range(2):
        ax.annotate('', xy=(5.2, 7.0 - j*0.8), xytext=(2.2, 7.5 - i*1.5),
                    arrowprops=dict(arrowstyle='->', color='#1976D2', lw=1.2))

# Arrows Group 2
for i in range(2):
    for j in range(2):
        ax.annotate('', xy=(5.2, 3.0 - j*0.8), xytext=(2.2, 4.0 - i*1.5),
                    arrowprops=dict(arrowstyle='->', color='#388E3C', lw=1.2))

# Output channels
for i, c in enumerate(['#4ECDC4', '#45B7D1']):
    rect = mpatches.FancyBboxPatch((8.5, 7 - i*1.5), 1.5, 1.0,
                                     boxstyle="round,pad=0.05", facecolor=c, edgecolor='#333', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(9.25, 7.5 - i*1.5, f'$c_{{o{i+1}}}$', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.annotate('', xy=(8.4, 7.5 - i*1.5), xytext=(7.2, 7.0 - i*0.8),
                arrowprops=dict(arrowstyle='->', color='#1976D2', lw=1.2))

for i, c in enumerate(['#A78BFA', '#F472B6']):
    rect = mpatches.FancyBboxPatch((8.5, 3.5 - i*1.5), 1.5, 1.0,
                                     boxstyle="round,pad=0.05", facecolor=c, edgecolor='#333', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(9.25, 4.0 - i*1.5, f'$c_{{o{i+3}}}$', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.annotate('', xy=(8.4, 4.0 - i*1.5), xytext=(7.2, 3.0 - i*0.8),
                arrowprops=dict(arrowstyle='->', color='#388E3C', lw=1.2))

ax.text(9.25, 8.5, 'Output\n($c_o$ channels)', ha='center', va='center', fontsize=12, fontweight='bold')

# Cost annotation
ax.text(5.5, 0.0, 'Chi phi: $\mathcal{O}(c_i \\cdot c_o \\cdot k^2 / g)$\nMoi nhom xu ly DOC LAP -- giam $g$ lan!', 
        ha='center', va='center', fontsize=11, color='#1B5E20',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor='#66BB6A'))

# Divider
ax.plot([0.3, 10.5], [4.75, 4.75], '--', color='#888', lw=1, alpha=0.5)

plt.tight_layout()
plt.savefig('/home/sakana/Code/Obsidian-Vault/assets/attachments/d2l-buoi-34/grouped_convolution.png', 
            dpi=200, bbox_inches='tight', facecolor='white')
print("OK: grouped_convolution.png")
