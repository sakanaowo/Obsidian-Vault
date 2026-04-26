"""
D2L Buổi 50 - Attention Mechanism Visualization (Updated)
Tạo ảnh minh họa cho 11.1 Queries, Keys, and Values
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
import torch

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

# ============================================================
# 1. D2L Figure 11.1.1 — The Attention Mechanism (clean, D2L style)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis('off')
ax.set_title('Fig. 11.1.1  The Attention Mechanism', fontsize=13, fontweight='bold', pad=12)

# --- INPUT SIDE ---
# Query
ax.add_patch(mpatches.FancyBboxPatch((0.3, 5.5), 2.5, 1.2,
    boxstyle='round,pad=0.15', fc='#BBDEFB', ec='#1565C0', lw=2.5))
ax.text(1.55, 6.1, 'Query\n(q)', fontsize=13, ha='center', va='center',
        fontweight='bold', color='#0D47A1')

# Keys
ax.add_patch(mpatches.FancyBboxPatch((0.3, 3.3), 2.5, 1.2,
    boxstyle='round,pad=0.15', fc='#C8E6C9', ec='#2E7D32', lw=2.5))
ax.text(1.55, 3.9, 'Keys\n(k_1, ..., k_m)', fontsize=13, ha='center', va='center',
        fontweight='bold', color='#1B5E20')

# Values
ax.add_patch(mpatches.FancyBboxPatch((0.3, 1.0), 2.5, 1.2,
    boxstyle='round,pad=0.15', fc='#FFCCBC', ec='#E65100', lw=2.5))
ax.text(1.55, 1.7, 'Values\n(v_1, ..., v_m)', fontsize=13, ha='center', va='center',
        fontweight='bold', color='#BF360C')

# --- MIDDLE STAGE ---
# Compatibility function
ax.add_patch(mpatches.FancyBboxPatch((4.0, 4.3), 2.5, 1.6,
    boxstyle='round,pad=0.15', fc='#E1BEE7', ec='#6A1B9A', lw=2.5))
ax.text(5.25, 5.1, 'Compatibility\nfunction', fontsize=12, ha='center', va='center',
        fontweight='bold', color='#4A148C')
ax.text(5.25, 4.6, 'a(q, k_i)', fontsize=11, ha='center', va='center',
        color='#6A1B9A', style='italic')

# Softmax
ax.add_patch(mpatches.FancyBboxPatch((7.5, 4.3), 2.0, 1.6,
    boxstyle='round,pad=0.15', fc='#F8BBD0', ec='#AD1457', lw=2.5))
ax.text(8.5, 5.1, 'Softmax', fontsize=12, ha='center', va='center',
        fontweight='bold', color='#880E4F')
ax.text(8.5, 4.6, '(Eq. 11.1.3)', fontsize=10, ha='center', va='center',
        color='#AD1457', style='italic')

# Attention Weights (below softmax)
ax.add_patch(mpatches.FancyBboxPatch((7.5, 2.0), 2.0, 1.4,
    boxstyle='round,pad=0.15', fc='#FFF9C4', ec='#F9A825', lw=2.0))
ax.text(8.5, 2.7, 'Attention\nWeights', fontsize=11, ha='center', va='center',
        fontweight='bold', color='#F57F17')
ax.text(8.5, 2.2, r'alpha_1,...,alpha_m', fontsize=10, ha='center', va='center',
        color='#F57F17', style='italic')

# Arrow: weights go to output
ax.annotate('', xy=(8.5, 1.95), xytext=(8.5, 2.0),
    arrowprops=dict(arrowstyle='->', color='#F9A825', lw=1.5))

# --- OUTPUT ---
ax.add_patch(mpatches.FancyBboxPatch((10.5, 3.8), 1.3, 1.6,
    boxstyle='round,pad=0.15', fc='#DCEDC8', ec='#33691E', lw=3.0))
ax.text(11.15, 4.8, 'Output', fontsize=12, ha='center', va='center',
        fontweight='bold', color='#1B5E20')
ax.text(11.15, 4.2, r'sum_i alpha_i v_i', fontsize=10, ha='center', va='center',
        color='#33691E', style='italic')

# --- ARROWS ---
arrow_kw = dict(arrowstyle='->', color='#424242', lw=1.8, connectionstyle='arc3,rad=0.0')
arrow_kw2 = dict(arrowstyle='->', color='#424242', lw=1.8, connectionstyle='arc3,rad=0.0')

# Q to Compatibility
ax.annotate('', xy=(4.0, 5.0), xytext=(2.8, 5.0),
    arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2.0))
ax.text(3.4, 5.3, 'q', fontsize=11, color='#1565C0', style='italic', ha='center')

# Keys to Compatibility
ax.annotate('', xy=(4.0, 4.3), xytext=(2.8, 3.9),
    arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=2.0))
ax.text(3.4, 4.05, 'k_i', fontsize=11, color='#2E7D32', style='italic', ha='center')

# Compatibility to Softmax
ax.annotate('', xy=(7.5, 5.0), xytext=(6.5, 5.0),
    arrowprops=dict(arrowstyle='->', color='#6A1B9A', lw=2.0))
ax.text(7.0, 5.3, 'a', fontsize=11, color='#6A1B9A', style='italic', ha='center')

# Keys to Values (dashed)
ax.annotate('', xy=(2.8, 2.6), xytext=(2.8, 3.3),
    arrowprops=dict(arrowstyle='->', color='#E65100', lw=1.5, linestyle='dashed'))

# Values to Weighted Sum
ax.annotate('', xy=(8.0, 2.8), xytext=(2.8, 1.6),
    arrowprops=dict(arrowstyle='->', color='#E65100', lw=1.8,
                   connectionstyle='arc3,rad=0.1'))
ax.text(5.2, 1.3, 'v_i', fontsize=11, color='#E65100', style='italic', ha='center')

# Softmax to Weighted Sum
ax.annotate('', xy=(8.5, 3.4), xytext=(8.5, 4.3),
    arrowprops=dict(arrowstyle='->', color='#AD1457', lw=1.8))

# Weighted Sum to Output
ax.annotate('', xy=(10.5, 4.6), xytext=(9.5, 2.7),
    arrowprops=dict(arrowstyle='->', color='#33691E', lw=2.0))

# Annotation
ax.text(0.15, 7.0, '(Eq. 11.1.1)', fontsize=10, color='#616161', style='italic', ha='left')

plt.tight_layout()
plt.savefig('/home/sakana/Code/Obsidian-Vault/assets/attachments/d2l-buoi-50/d2l-fig-11-1-1.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: d2l-fig-11-1-1.png")

# ============================================================
# 2. Attention Weight Special Cases — clean 4-panel
# ============================================================
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
plt.rcParams['font.size'] = 10

# (a) Hard attention (one-hot)
mat_hard = torch.zeros(6, 6)
mat_hard[2, 2] = 1.0  # Query 2 only attends to Key 2
axes[0].imshow(mat_hard.numpy(), cmap='Reds', vmin=0, vmax=1)
axes[0].set_title('(a) Hard Attention\n(alpha_i in {0,1})', fontsize=10, fontweight='bold')
axes[0].set_xlabel('Keys')
axes[0].set_ylabel('Queries')
axes[0].set_xticks([])
axes[0].set_yticks([])

# (b) Softmax with strong peak
mat_softmax_strong = torch.zeros(6, 6)
for i in range(6):
    scores = torch.tensor([-abs(i-j)*2 if j != i else 3.0 for j in range(6)], dtype=torch.float32)
    mat_softmax_strong[i] = torch.softmax(scores, dim=0)
axes[1].imshow(mat_softmax_strong.numpy(), cmap='Reds', vmin=0, vmax=1)
axes[1].set_title('(b) Softmax (sharp peak)\nConcentrated weights', fontsize=10, fontweight='bold')
axes[1].set_xlabel('Keys')
axes[1].set_xticks([])
axes[1].set_yticks([])

# (c) Softmax with soft distribution
mat_softmax_soft = torch.zeros(6, 6)
for i in range(6):
    scores = torch.tensor([-0.3*abs(i-j) for j in range(6)], dtype=torch.float32)
    mat_softmax_soft[i] = torch.softmax(scores, dim=0)
axes[2].imshow(mat_softmax_soft.numpy(), cmap='Reds', vmin=0, vmax=1)
axes[2].set_title('(c) Softmax (soft spread)\nDiffuse weights', fontsize=10, fontweight='bold')
axes[2].set_xlabel('Keys')
axes[2].set_xticks([])
axes[2].set_yticks([])

# (d) Uniform
mat_uniform = torch.ones(6, 6) / 6.0
axes[3].imshow(mat_uniform.numpy(), cmap='Reds', vmin=0, vmax=1)
axes[3].set_title('(d) Uniform\n(alpha_i = 1/m)', fontsize=10, fontweight='bold')
axes[3].set_xlabel('Keys')
axes[3].set_xticks([])
axes[3].set_yticks([])

fig.suptitle('Fig. Attention Weight Special Cases', fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/home/sakana/Code/Obsidian-Vault/assets/attachments/d2l-buoi-50/attention-special-cases.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: attention-special-cases.png")

# ============================================================
# 3. Database Analogy — Traditional vs Attention
# ============================================================
fig, ax = plt.subplots(figsize=(13, 5))
ax.axis('off')
ax.set_title('Fig. 2  Traditional Database vs Attention Mechanism', fontsize=12, fontweight='bold', pad=10)

# Left panel: Traditional DB
ax.add_patch(mpatches.FancyBboxPatch((0.05, 0.05), 5.7, 5.6,
    boxstyle='round,pad=0.2', fc='#ECEFF1', ec='#607D8B', lw=2.5))
ax.text(2.9, 5.35, 'Traditional Database', fontsize=12, ha='center', fontweight='bold', color='#455A64')
ax.text(2.9, 4.9, 'EXACT match only', fontsize=10, ha='center', color='#C62828', fontweight='bold')

db_rows = [('Key (k_i)', 'Value (v_i)'),
           ('"Zhang"', '"Aston"'), ('"Lipton"', '"Zachary"'),
           ('"Li"', '"Mu"'), ('"Smola"', '"Alex"'),
           ('"Hu"', '"Rachel"'), ('"Werness"', '"Brent"')]
y_start = 4.3
rh = 0.54
for i, (k, v) in enumerate(db_rows):
    fc = '#90A4AE' if i == 0 else '#FFFFFF'
    fw = 'bold' if i == 0 else 'normal'
    ax.add_patch(mpatches.FancyBboxPatch((0.3, y_start - i*rh), 2.5, 0.42,
        boxstyle='round,pad=0.05', fc=fc, ec='#78909C', lw=1))
    ax.add_patch(mpatches.FancyBboxPatch((3.0, y_start - i*rh), 2.5, 0.42,
        boxstyle='round,pad=0.05', fc=fc, ec='#78909C', lw=1))
    ax.text(1.55, y_start - i*rh + 0.21, k, fontsize=9, ha='center', va='center',
            fontweight=fw, color='#263238' if i == 0 else '#212121')
    ax.text(4.25, y_start - i*rh + 0.21, v, fontsize=9, ha='center', va='center',
            fontweight=fw, color='#263238' if i == 0 else '#212121')

ax.text(2.9, 0.5,
    'Query "Li" -> Returns "Mu"\nQuery "Lipt" -> No result!',
    fontsize=9, ha='center', va='center', style='italic', color='#616161',
    bbox=dict(boxstyle='round', fc='#FFEBEE', ec='#EF9A9A', lw=1))

# Arrow
ax.annotate('', xy=(6.0, 2.8), xytext=(5.8, 2.8),
    arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2.5))
ax.text(5.9, 3.2, 'soft\nmatch', fontsize=9, ha='center', color='#1565C0', fontweight='bold')

# Right panel: Attention DB
ax.add_patch(mpatches.FancyBboxPatch((6.2, 0.05), 6.5, 5.6,
    boxstyle='round,pad=0.2', fc='#E8EAF6', ec='#3949AB', lw=2.5))
ax.text(9.45, 5.35, 'Attention Database', fontsize=12, ha='center', fontweight='bold', color='#283593')
ax.text(9.45, 4.9, 'APPROXIMATE weighted match', fontsize=10, ha='center', color='#1B5E20', fontweight='bold')

att_rows = [('Key (k_i)', 'Value (v_i)', 'alpha_i'),
            ('"Zhang"', '"Aston"', '0.05'),
            ('"Lipton"', '"Zachary"', '0.15'),
            ('"Li"', '"Mu"', '0.70'),
            ('"Smola"', '"Alex"', '0.07'),
            ('"Hu"', '"Rachel"', '0.03')]
y_start2 = 4.3
rh2 = 0.60
for i, (k, v, a) in enumerate(att_rows):
    fc = '#7986CB' if i == 0 else '#FFFFFF'
    fw = 'bold' if i == 0 else 'normal'
    ax.add_patch(mpatches.FancyBboxPatch((6.4, y_start2 - i*rh2), 1.8, 0.48,
        boxstyle='round,pad=0.05', fc=fc, ec='#5C6BC0', lw=1))
    ax.add_patch(mpatches.FancyBboxPatch((8.4, y_start2 - i*rh2), 1.8, 0.48,
        boxstyle='round,pad=0.05', fc=fc, ec='#5C6BC0', lw=1))
    ax.add_patch(mpatches.FancyBboxPatch((10.4, y_start2 - i*rh2), 2.1, 0.48,
        boxstyle='round,pad=0.05', fc=fc, ec='#5C6BC0', lw=1))
    ax.text(7.3, y_start2 - i*rh2 + 0.24, k, fontsize=9, ha='center', va='center',
            fontweight=fw, color='#1A237E' if i == 0 else '#212121')
    ax.text(9.3, y_start2 - i*rh2 + 0.24, v, fontsize=9, ha='center', va='center',
            fontweight=fw, color='#1A237E' if i == 0 else '#212121')
    ac = '#1A237E' if i == 0 else '#1B5E20'
    ax.text(11.45, y_start2 - i*rh2 + 0.24, a, fontsize=9, ha='center', va='center',
            fontweight=fw if i > 0 else 'normal', color=ac)

ax.text(9.45, 0.5,
    'Query "Li" ->\n"0.70*Mu + 0.15*Zachary + 0.07*Alex + 0.03*Rachel"',
    fontsize=9, ha='center', va='center', style='italic', color='#1B5E20',
    bbox=dict(boxstyle='round', fc='#E8F5E9', ec='#A5D6A7', lw=1))

plt.tight_layout()
plt.savefig('/home/sakana/Code/Obsidian-Vault/assets/attachments/d2l-buoi-50/attention-database-analogy.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: attention-database-analogy.png")

print("\nAll D2L-style visualizations saved.")
