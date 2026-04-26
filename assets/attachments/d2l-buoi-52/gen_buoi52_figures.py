"""
D2L Buổi 52 - Attention Scoring Functions
Tạo ảnh minh họa cho 11.3 Attention Scoring Functions
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

# ============================================================
# 1. Fig 11.3.1 — Attention Scoring & Pooling Flow
# ============================================================
fig, ax = plt.subplots(figsize=(13, 7))
ax.set_xlim(0, 13)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_title('Fig. 11.3.1  Attention Scoring Function and Pooling', fontsize=13, fontweight='bold', pad=12)

# --- INPUT ---
# Queries
ax.add_patch(mpatches.FancyBboxPatch((0.3, 5.5), 2.2, 1.3,
    boxstyle='round,pad=0.15', fc='#BBDEFB', ec='#1565C0', lw=2.5))
ax.text(1.4, 6.15, 'Queries\n(Q)', fontsize=12, ha='center', va='center',
        fontweight='bold', color='#0D47A1')

# Keys
ax.add_patch(mpatches.FancyBboxPatch((0.3, 3.5), 2.2, 1.3,
    boxstyle='round,pad=0.15', fc='#C8E6C9', ec='#2E7D32', lw=2.5))
ax.text(1.4, 4.15, 'Keys\n(K)', fontsize=12, ha='center', va='center',
        fontweight='bold', color='#1B5E20')

# Values
ax.add_patch(mpatches.FancyBboxPatch((0.3, 1.5), 2.2, 1.3,
    boxstyle='round,pad=0.15', fc='#FFCCBC', ec='#E65100', lw=2.5))
ax.text(1.4, 2.15, 'Values\n(V)', fontsize=12, ha='center', va='center',
        fontweight='bold', color='#BF360C')

# --- SCORING ---
# Attention scoring box
ax.add_patch(mpatches.FancyBboxPatch((3.5, 3.8), 3.5, 2.2,
    boxstyle='round,pad=0.15', fc='#E1BEE7', ec='#6A1B9A', lw=2.5))
ax.text(5.25, 5.6, 'Attention Scoring\nFunction a(q, k_i)', fontsize=12, ha='center',
        va='center', fontweight='bold', color='#4A148C')
ax.text(5.25, 4.85, 'a(q,k_i) = q^T k_i / sqrt(d)', fontsize=10, ha='center',
        va='center', color='#6A1B9A', style='italic')
ax.text(5.25, 4.35, '(or additive: tanh)', fontsize=9, ha='center',
        va='center', color='#7B1FA2', style='italic')

# --- SOFTMAX ---
ax.add_patch(mpatches.FancyBboxPatch((7.8, 3.8), 2.0, 2.2,
    boxstyle='round,pad=0.15', fc='#F8BBD0', ec='#AD1457', lw=2.5))
ax.text(8.8, 5.6, 'Softmax', fontsize=12, ha='center', va='center',
        fontweight='bold', color='#880E4F')
ax.text(8.8, 4.9, 'alpha_i =', fontsize=10, ha='center', va='center', color='#AD1457')
ax.text(8.8, 4.4, 'exp(a)/sum(exp)', fontsize=9, ha='center', va='center',
        color='#AD1457', style='italic')

# --- ATTENTION WEIGHTS ---
ax.add_patch(mpatches.FancyBboxPatch((7.8, 1.5), 2.0, 1.6,
    boxstyle='round,pad=0.15', fc='#FFF9C4', ec='#F9A825', lw=2.0))
ax.text(8.8, 2.3, 'Attention\nWeights', fontsize=11, ha='center', va='center',
        fontweight='bold', color='#F57F17')
ax.text(8.8, 1.75, 'sum = 1', fontsize=10, ha='center', va='center',
        color='#F57F17', style='italic')

ax.annotate('', xy=(8.8, 3.1), xytext=(8.8, 3.8),
    arrowprops=dict(arrowstyle='->', color='#F9A825', lw=1.8))

# --- OUTPUT ---
ax.add_patch(mpatches.FancyBboxPatch((10.8, 3.8), 1.8, 2.2,
    boxstyle='round,pad=0.15', fc='#DCEDC8', ec='#33691E', lw=3.0))
ax.text(11.7, 5.6, 'Output', fontsize=12, ha='center', va='center',
        fontweight='bold', color='#1B5E20')
ax.text(11.7, 4.9, 'sum_i', fontsize=10, ha='center', va='center', color='#33691E', style='italic')
ax.text(11.7, 4.4, 'alpha_i * v_i', fontsize=10, ha='center', va='center',
        color='#33691E', style='italic')

# --- ARROWS ---
# Q -> Scoring
ax.annotate('', xy=(3.5, 5.7), xytext=(2.5, 5.7),
    arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2.0))
ax.text(3.0, 6.1, 'q', fontsize=11, color='#1565C0', style='italic', ha='center')

# K -> Scoring
ax.annotate('', xy=(3.5, 4.7), xytext=(2.5, 4.3),
    arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=2.0))
ax.text(3.0, 4.3, 'k_i', fontsize=11, color='#2E7D32', style='italic', ha='center')

# Scoring -> Softmax
ax.annotate('', xy=(7.8, 5.0), xytext=(7.0, 5.0),
    arrowprops=dict(arrowstyle='->', color='#6A1B9A', lw=2.0))

# Softmax -> Weights
ax.annotate('', xy=(8.8, 3.1), xytext=(8.8, 3.8),
    arrowprops=dict(arrowstyle='->', color='#AD1457', lw=1.8))

# K -> V (dashed, keep track)
ax.annotate('', xy=(2.5, 2.3), xytext=(2.5, 3.5),
    arrowprops=dict(arrowstyle='->', color='#E65100', lw=1.5, linestyle='dashed'))

# Values -> Output
ax.annotate('', xy=(10.5, 4.9), xytext=(2.5, 2.1),
    arrowprops=dict(arrowstyle='->', color='#E65100', lw=1.8, connectionstyle='arc3,rad=0.05'))

# Weights -> Output
ax.annotate('', xy=(10.8, 4.9), xytext=(9.8, 2.3),
    arrowprops=dict(arrowstyle='->', color='#33691E', lw=2.0))

plt.tight_layout()
plt.savefig('/home/sakana/Code/Obsidian-Vault/assets/attachments/d2l-buoi-52/d2l-fig-11-3-1.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: d2l-fig-11-3-1.png")

# ============================================================
# 2. Scaled Dot Product — why divide by sqrt(d)?
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
plt.rcParams['font.size'] = 10

# Left: dot product variance vs d
d_values = np.arange(1, 513)
variances = d_values.astype(float)  # Var of dot product of i.i.d. unit variance vectors = d

ax = axes[0]
ax.plot(d_values, variances, color='#1565C0', lw=2.5)
ax.axhline(y=1, color='red', lw=1.5, ls='--', label='Variance = 1 (stable)')
ax.fill_between(d_values, variances, alpha=0.1, color='#1565C0')
ax.set_xlabel('Vector dimension d', fontsize=11)
ax.set_ylabel('Variance of dot product', fontsize=11)
ax.set_title('Dot Product Variance vs Dimension d\n'
             '(q, k_i ~ i.i.d. N(0,1))', fontsize=10, fontweight='bold')
ax.set_xlim(1, 512)
ax.set_ylim(0, 520)
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.3)

# Right: softmax with/without scaling
np.random.seed(42)
d = 64
q = np.random.randn(d) / np.sqrt(d)  # already scaled
k = np.random.randn(5, d) / np.sqrt(d)
scores_raw = k @ q
scores_scaled = scores_raw / np.sqrt(d)

x_pos = np.arange(5)

ax = axes[1]
width = 0.35
bars1 = ax.bar(x_pos - width/2, scores_raw, width, label='Raw dot product',
               color='#C62828', alpha=0.8)
bars2 = ax.bar(x_pos + width/2, scores_scaled, width, label='Scaled (÷√d)',
               color='#1565C0', alpha=0.8)
ax.set_xlabel('Key index', fontsize=11)
ax.set_ylabel('Attention score', fontsize=11)
ax.set_title('Raw vs Scaled Dot Product Scores\n'
             'd=64, q,k ~ N(0,1/√d)', fontsize=10, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'k_{i}' for i in range(5)])
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.3, axis='y')

fig.suptitle('Fig. 11.3.2  Why Scaled Dot Product Attention?', fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/home/sakana/Code/Obsidian-Vault/assets/attachments/d2l-buoi-52/scaled-dot-product.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: scaled-dot-product.png")

# ============================================================
# 3. Masked Softmax visualization
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
plt.rcParams['font.size'] = 10

# Case 1: valid_lens = [2, 3], 2D tensor inputs
# Row 0, query 0: valid_len=2 → weights for keys 0,1; keys 2,3 masked
# Row 0, query 1: valid_len=2 → weights for keys 0,1; keys 2,3 masked
# Row 1, query 0: valid_len=3 → weights for keys 0,1,2; key 3 masked
# Row 1, query 1: valid_len=3 → weights for keys 0,1,2; key 3 masked

# Simulate attention weights for 2 batch, 2 queries, 4 keys
weights = np.array([
    # batch 0
    [[0.45, 0.55, 0.0, 0.0],
     [0.38, 0.62, 0.0, 0.0]],
    # batch 1
    [[0.30, 0.35, 0.35, 0.0],
     [0.28, 0.32, 0.40, 0.0]]
])

im = axes[0].imshow(weights[:, 0, :], cmap='Reds', vmin=0, vmax=0.7, aspect='auto')
axes[0].set_title('Attention Weights\nAfter Masked Softmax', fontsize=10, fontweight='bold')
axes[0].set_xlabel('Keys (index)')
axes[0].set_ylabel('Queries (batch, query)')
axes[0].set_xticks([0, 1, 2, 3])
axes[0].set_xticklabels(['k_0', 'k_1', 'k_2', 'k_3'])
axes[0].set_yticks([0, 1])
axes[0].set_yticklabels(['b0,q0', 'b0,q1'], fontsize=8)
for r in range(2):
    for c in range(4):
        val = weights[0, r, c]
        axes[0].text(c, r, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color='white' if val > 0.3 else 'black')

# Add valid length indicators
axes[0].axvline(x=1.5, color='black', lw=2, ls='--', alpha=0.5)
axes[0].axvline(x=2.5, color='black', lw=2, ls='--', alpha=0.5)

# Annotation
axes[0].text(0.5, -0.22, 'valid\nlen=2', ha='center', fontsize=7, color='#1565C0',
             transform=axes[0].transAxes)
axes[0].text(0.75, -0.22, 'valid\nlen=3', ha='center', fontsize=7, color='#2E7D32',
             transform=axes[0].transAxes)

# Case 2: Raw scores before masking
raw_scores = np.array([
    [[0.5, 1.2, 0.8, 0.3],
     [0.2, 1.5, 0.7, 0.4]],
    [[0.1, 0.2, 0.3, 0.0],
     [0.2, 0.3, 0.4, 0.0]]
])
axes[1].imshow(raw_scores[0], cmap='Blues', aspect='auto')
axes[1].set_title('Raw Scores\nBefore Masking', fontsize=10, fontweight='bold')
axes[1].set_xlabel('Keys (index)')
axes[1].set_xticks([0, 1, 2, 3])
axes[1].set_xticklabels(['k_0', 'k_1', 'k_2', 'k_3'])
axes[1].set_yticks([0, 1])
axes[1].set_yticklabels(['b0,q0', 'b0,q1'], fontsize=8)
for r in range(2):
    for c in range(4):
        val = raw_scores[0, r, c]
        axes[1].text(c, r, f'{val:.1f}', ha='center', va='center', fontsize=7,
                    color='white' if val > 0.8 else 'black')

# Case 3: After applying -1e6 mask
masked_scores = raw_scores.copy()
masked_scores[0, :, 2] = -1e6  # key 2 masked for batch 0
masked_scores[0, :, 3] = -1e6  # key 3 masked for batch 0
axes[2].imshow(masked_scores[0], cmap='Blues', aspect='auto', vmin=-2, vmax=2)
axes[2].set_title('Scores After Masking\n(-1e6 for padded tokens)', fontsize=10, fontweight='bold')
axes[2].set_xlabel('Keys (index)')
axes[2].set_xticks([0, 1, 2, 3])
axes[2].set_xticklabels(['k_0', 'k_1', 'k_2', 'k_3'])
axes[2].set_yticks([0, 1])
axes[2].set_yticklabels(['b0,q0', 'b0,q1'], fontsize=8)
for r in range(2):
    for c in range(4):
        val = masked_scores[0, r, c]
        if val < 0:
            axes[2].text(c, r, f'-1e6', ha='center', va='center', fontsize=6, color='white')
        else:
            axes[2].text(c, r, f'{val:.1f}', ha='center', va='center', fontsize=7,
                        color='white' if val > 1.0 else 'black')

fig.suptitle('Fig. 11.3.3  Masked Softmax Operation', fontsize=12, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('/home/sakana/Code/Obsidian-Vault/assets/attachments/d2l-buoi-52/masked-softmax.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: masked-softmax.png")

# ============================================================
# 4. Dot Product vs Additive Attention — Architecture comparison
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
plt.rcParams['font.size'] = 10

# --- Dot Product Attention ---
ax = axes[0]
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_title('Dot Product Attention (Eq. 11.3.2)', fontsize=12, fontweight='bold',
             color='#1565C0', pad=8)

# Q box
ax.add_patch(mpatches.FancyBboxPatch((0.5, 5.5), 1.8, 1.2,
    boxstyle='round,pad=0.1', fc='#BBDEFB', ec='#1565C0', lw=2))
ax.text(1.4, 6.1, 'Q\n(n×d)', fontsize=10, ha='center', va='center', color='#0D47A1', fontweight='bold')

# K box
ax.add_patch(mpatches.FancyBboxPatch((0.5, 3.5), 1.8, 1.2,
    boxstyle='round,pad=0.1', fc='#C8E6C9', ec='#2E7D32', lw=2))
ax.text(1.4, 4.1, 'K\n(m×d)', fontsize=10, ha='center', va='center', color='#1B5E20', fontweight='bold')

# V box
ax.add_patch(mpatches.FancyBboxPatch((0.5, 1.3), 1.8, 1.2,
    boxstyle='round,pad=0.1', fc='#FFCCBC', ec='#E65100', lw=2))
ax.text(1.4, 1.9, 'V\n(m×v)', fontsize=10, ha='center', va='center', color='#BF360C', fontweight='bold')

# K^T box
ax.add_patch(mpatches.FancyBboxPatch((3.2, 3.5), 1.5, 1.2,
    boxstyle='round,pad=0.1', fc='#E8F5E9', ec='#2E7D32', lw=1.5))
ax.text(3.95, 4.1, 'K^T', fontsize=10, ha='center', va='center', color='#1B5E20', fontweight='bold')

# Matrix multiply
ax.annotate('', xy=(3.2, 5.3), xytext=(2.3, 5.3),
    arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.5))
ax.annotate('', xy=(3.2, 4.7), xytext=(2.3, 4.1),
    arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.5))

ax.text(2.5, 5.0, '×', fontsize=16, ha='center', va='center', color='black')

# Scores / sqrt(d)
ax.add_patch(mpatches.FancyBboxPatch((3.2, 5.5), 3.0, 1.2,
    boxstyle='round,pad=0.1', fc='#FFF9C4', ec='#F9A825', lw=2))
ax.text(4.7, 6.1, 'Q @ K^T / √d', fontsize=10, ha='center', va='center',
        color='#E65100', fontweight='bold')
ax.text(4.7, 5.6, 'scores (n×m)', fontsize=9, ha='center', va='center', color='#616161')

# Softmax
ax.add_patch(mpatches.FancyBboxPatch((7.0, 5.5), 2.0, 1.2,
    boxstyle='round,pad=0.1', fc='#F8BBD0', ec='#AD1457', lw=2))
ax.text(8.0, 6.1, 'softmax', fontsize=10, ha='center', va='center',
        color='#880E4F', fontweight='bold')
ax.annotate('', xy=(7.0, 6.0), xytext=(6.2, 6.0),
    arrowprops=dict(arrowstyle='->', color='#AD1457', lw=1.5))

# Attention weights -> V
ax.add_patch(mpatches.FancyBboxPatch((7.0, 1.3), 2.0, 1.2,
    boxstyle='round,pad=0.1', fc='#E1BEE7', ec='#6A1B9A', lw=2))
ax.text(8.0, 1.9, 'weights\n(n×m)', fontsize=9, ha='center', va='center',
        color='#4A148C', fontweight='bold')
ax.annotate('', xy=(8.0, 2.5), xytext=(8.0, 5.5),
    arrowprops=dict(arrowstyle='->', color='#6A1B9A', lw=1.5))

# V multiply
ax.annotate('', xy=(9.0, 1.9), xytext=(2.3, 1.9),
    arrowprops=dict(arrowstyle='->', color='#E65100', lw=1.5, connectionstyle='arc3,rad=-0.1'))
ax.text(5.5, 1.2, 'V (m×v)', fontsize=9, ha='center', va='center', color='#BF360C')

ax.text(5.5, 2.5, '×', fontsize=16, ha='center', va='center', color='black')
ax.add_patch(mpatches.FancyBboxPatch((9.0, 1.3), 2.5, 1.2,
    boxstyle='round,pad=0.1', fc='#DCEDC8', ec='#33691E', lw=2.5))
ax.text(10.25, 1.9, 'Output\n(n×v)', fontsize=10, ha='center', va='center',
        color='#1B5E20', fontweight='bold')
ax.annotate('', xy=(9.0, 2.5), xytext=(9.0, 2.5),
    arrowprops=dict(arrowstyle='->', color='#33691E', lw=1.5))

ax.text(5.0, 0.4, 'Parameters: 0 extra  |  Complexity: O(n·m·d)', fontsize=9,
        ha='center', style='italic', color='#616161')

# --- Additive Attention ---
ax = axes[1]
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_title('Additive Attention (Eq. 11.3.7)', fontsize=12, fontweight='bold',
             color='#4A148C', pad=8)

# Q box
ax.add_patch(mpatches.FancyBboxPatch((0.5, 5.5), 1.8, 1.2,
    boxstyle='round,pad=0.1', fc='#BBDEFB', ec='#1565C0', lw=2))
ax.text(1.4, 6.1, 'Q\n(n×q)', fontsize=10, ha='center', va='center', color='#0D47A1', fontweight='bold')

# K box
ax.add_patch(mpatches.FancyBboxPatch((0.5, 3.5), 1.8, 1.2,
    boxstyle='round,pad=0.1', fc='#C8E6C9', ec='#2E7D32', lw=2))
ax.text(1.4, 4.1, 'K\n(m×k)', fontsize=10, ha='center', va='center', color='#1B5E20', fontweight='bold')

# W_q
ax.add_patch(mpatches.FancyBboxPatch((3.0, 5.5), 1.6, 1.2,
    boxstyle='round,pad=0.1', fc='#E3F2FD', ec='#1565C0', lw=1.5))
ax.text(3.8, 6.1, 'W_q\n(h×q)', fontsize=9, ha='center', va='center', color='#0D47A1')
ax.annotate('', xy=(3.0, 6.0), xytext=(2.3, 6.0),
    arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.5))

# W_k
ax.add_patch(mpatches.FancyBboxPatch((3.0, 3.5), 1.6, 1.2,
    boxstyle='round,pad=0.1', fc='#E8F5E9', ec='#2E7D32', lw=1.5))
ax.text(3.8, 4.1, 'W_k\n(h×k)', fontsize=9, ha='center', va='center', color='#1B5E20')
ax.annotate('', xy=(3.0, 4.0), xytext=(2.3, 4.0),
    arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.5))

# tanh + broadcast
ax.add_patch(mpatches.FancyBboxPatch((5.4, 3.5), 2.5, 3.2,
    boxstyle='round,pad=0.1', fc='#FFF9C4', ec='#F9A825', lw=2))
ax.text(6.65, 6.2, 'Broadcast +', fontsize=10, ha='center', va='center',
        color='#E65100', fontweight='bold')
ax.text(6.65, 5.7, 'tanh activation', fontsize=9, ha='center', va='center', color='#616161')

# Annotate broadcast
ax.annotate('', xy=(5.4, 5.7), xytext=(4.6, 5.7),
    arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.5))
ax.annotate('', xy=(5.4, 4.0), xytext=(4.6, 4.0),
    arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.5))

# w_v
ax.add_patch(mpatches.FancyBboxPatch((8.5, 4.5), 1.5, 1.2,
    boxstyle='round,pad=0.1', fc='#E1BEE7', ec='#6A1B9A', lw=2))
ax.text(9.25, 5.1, 'w_v^T\n(1×h)', fontsize=9, ha='center', va='center', color='#4A148C')
ax.add_patch(mpatches.FancyBboxPatch((8.5, 1.5), 1.5, 1.2,
    boxstyle='round,pad=0.1', fc='#F8BBD0', ec='#AD1457', lw=2))
ax.text(9.25, 2.1, 'softmax', fontsize=10, ha='center', va='center', color='#880E4F')

ax.annotate('', xy=(8.5, 5.0), xytext=(7.9, 5.0),
    arrowprops=dict(arrowstyle='->', color='#6A1B9A', lw=1.5))

# V
ax.add_patch(mpatches.FancyBboxPatch((0.5, 1.3), 1.8, 1.2,
    boxstyle='round,pad=0.1', fc='#FFCCBC', ec='#E65100', lw=2))
ax.text(1.4, 1.9, 'V\n(m×v)', fontsize=10, ha='center', va='center', color='#BF360C', fontweight='bold')

# Final output
ax.add_patch(mpatches.FancyBboxPatch((10.2, 3.0), 1.6, 1.5,
    boxstyle='round,pad=0.1', fc='#DCEDC8', ec='#33691E', lw=2.5))
ax.text(11.0, 3.75, 'Output\n(n×v)', fontsize=10, ha='center', va='center',
        color='#1B5E20', fontweight='bold')
ax.annotate('', xy=(10.2, 3.75), xytext=(10.0, 3.75),
    arrowprops=dict(arrowstyle='->', color='#33691E', lw=1.5))
ax.annotate('', xy=(10.2, 3.75), xytext=(2.3, 1.9),
    arrowprops=dict(arrowstyle='->', color='#E65100', lw=1.5, connectionstyle='arc3,rad=-0.05'))

ax.text(6.0, 0.4, 'Parameters: W_q, W_k, w_v  |  Complexity: O(n·m·h)', fontsize=9,
        ha='center', style='italic', color='#616161')

fig.suptitle('Fig. 11.3.4  Dot Product vs Additive Attention Architecture', fontsize=12,
             fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/home/sakana/Code/Obsidian-Vault/assets/attachments/d2l-buoi-52/dot-product-vs-additive.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: dot-product-vs-additive.png")

# ============================================================
# 5. BMM — Batch Matrix Multiplication illustration
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
plt.rcParams['font.size'] = 10

# Q: n × a × b
ax = axes[0]
Q_blocks = [
    [1, 0, 0, 0],  # Q_1: 1×4
    [0, 1, 0, 0],  # Q_2: 1×4
]
ax.imshow(Q_blocks, cmap='Blues', aspect='auto')
ax.set_title('Q ∈ R^(2×1×4)\n2 matrices, each 1×4', fontsize=10, fontweight='bold')
ax.set_xlabel('columns (4)')
ax.set_ylabel('batch (2)')
ax.set_xticks([0, 1, 2, 3])
ax.text(0, -0.15, 'Q_1=[1,0,0,0]', fontsize=8, ha='center', transform=ax.transAxes)
ax.text(0, -0.28, 'Q_2=[0,1,0,0]', fontsize=8, ha='center', transform=ax.transAxes)
for i in range(2):
    for j in range(4):
        ax.text(j, i, f'{Q_blocks[i][j]}', ha='center', va='center',
                fontsize=9, color='white' if Q_blocks[i][j] > 0.5 else 'black')

# K: n × 4 × 6
ax = axes[1]
K_blocks = [
    [[1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,1,0,0,0], [0,0,0,1,0,0]],  # K_1: 4×6
    [[1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,1,0,0,0], [0,0,0,1,0,0]],  # K_2: 4×6
]
K_flat = np.zeros((2, 4*6))
for b in range(2):
    for i in range(4):
        for j in range(6):
            K_flat[b, i*6 + j] = K_blocks[b][i][j]
ax.imshow(K_flat, cmap='Greens', aspect='auto')
ax.set_title('K ∈ R^(2×4×6)\n2 matrices, each 4×6', fontsize=10, fontweight='bold')
ax.set_xlabel('flattened (4×6=24)')
ax.set_ylabel('batch (2)')
# Mark block boundaries
ax.axvline(x=5.5, color='red', lw=1, ls='--')
ax.axvline(x=11.5, color='red', lw=1, ls='--')
ax.axvline(x=17.5, color='red', lw=1, ls='--')
for i in range(2):
    for j in range(24):
        ax.text(j, i, f'{K_flat[i,j]:.0f}', ha='center', va='center',
                fontsize=5, color='white' if K_flat[i,j] > 0.5 else 'black')

# Result: n × 1 × 6
ax = axes[2]
result = np.zeros((2, 6))
result[0] = [1, 0, 0, 0, 0, 0]
result[1] = [0, 1, 0, 0, 0, 0]
ax.imshow(result, cmap='Reds', aspect='auto')
ax.set_title('BMM(Q,K) ∈ R^(2×1×6)\nQ_1@K_1, Q_2@K_2', fontsize=10, fontweight='bold')
ax.set_xlabel('columns (6)')
ax.set_ylabel('batch (2)')
ax.set_xticks(range(6))
for i in range(2):
    for j in range(6):
        ax.text(j, i, f'{result[i,j]:.0f}', ha='center', va='center',
                fontsize=9, color='white' if result[i,j] > 0.5 else 'black')

fig.suptitle('Fig. 11.3.5  Batch Matrix Multiplication (BMM)', fontsize=12, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('/home/sakana/Code/Obsidian-Vault/assets/attachments/d2l-buoi-52/batch-matrix-multiplication.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: batch-matrix-multiplication.png")

# ============================================================
# 6. Dot Product Attention — complete forward pass shapes
# ============================================================
fig, ax = plt.subplots(figsize=(12, 5))
ax.axis('off')
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)

ax.text(6, 6.6, 'Fig. 11.3.6  DotProductAttention — Full Forward Pass Shapes', fontsize=12,
        fontweight='bold', ha='center')

# Q: (batch, n_queries, d)
ax.add_patch(mpatches.FancyBboxPatch((0.3, 4.5), 2.2, 1.2,
    boxstyle='round,pad=0.1', fc='#BBDEFB', ec='#1565C0', lw=2))
ax.text(1.4, 5.1, 'Q\n(batch, n, d)', fontsize=10, ha='center', va='center',
        fontweight='bold', color='#0D47A1')

# K: (batch, m_keys, d)
ax.add_patch(mpatches.FancyBboxPatch((0.3, 2.5), 2.2, 1.2,
    boxstyle='round,pad=0.1', fc='#C8E6C9', ec='#2E7D32', lw=2))
ax.text(1.4, 3.1, 'K\n(batch, m, d)', fontsize=10, ha='center', va='center',
        fontweight='bold', color='#1B5E20')

# V: (batch, m_keys, v)
ax.add_patch(mpatches.FancyBboxPatch((0.3, 0.5), 2.2, 1.2,
    boxstyle='round,pad=0.1', fc='#FFCCBC', ec='#E65100', lw=2))
ax.text(1.4, 1.1, 'V\n(batch, m, v)', fontsize=10, ha='center', va='center',
        fontweight='bold', color='#BF360C')

# Step 1: Q @ K^T
ax.add_patch(mpatches.FancyBboxPatch((3.5, 3.5), 2.8, 2.2,
    boxstyle='round,pad=0.1', fc='#FFF9C4', ec='#F9A825', lw=2))
ax.text(4.9, 5.0, 'scores = Q @ K^T / √d', fontsize=9.5, ha='center', va='center',
        color='#E65100', fontweight='bold')
ax.text(4.9, 4.3, 'Shape: (batch, n, m)', fontsize=9, ha='center', va='center',
        color='#616161', style='italic')

ax.annotate('', xy=(3.5, 4.6), xytext=(2.5, 4.6),
    arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.5))
ax.annotate('', xy=(3.5, 4.2), xytext=(2.5, 3.6),
    arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.5))
ax.text(2.8, 4.4, '×', fontsize=14, ha='center', va='center')

# Step 2: masked_softmax
ax.add_patch(mpatches.FancyBboxPatch((7.2, 3.5), 2.3, 2.2,
    boxstyle='round,pad=0.1', fc='#F8BBD0', ec='#AD1457', lw=2))
ax.text(8.35, 5.0, 'masked_softmax(scores)', fontsize=9, ha='center', va='center',
        color='#880E4F', fontweight='bold')
ax.text(8.35, 4.3, 'Shape: (batch, n, m)', fontsize=9, ha='center', va='center',
        color='#616161', style='italic')
ax.annotate('', xy=(7.2, 4.6), xytext=(6.3, 4.6),
    arrowprops=dict(arrowstyle='->', color='#F9A825', lw=1.5))

# Step 3: attention_weights @ V
ax.add_patch(mpatches.FancyBboxPatch((10.3, 3.5), 1.5, 2.2,
    boxstyle='round,pad=0.1', fc='#DCEDC8', ec='#33691E', lw=2.5))
ax.text(11.05, 5.0, 'out =\nattn @ V', fontsize=9.5, ha='center', va='center',
        color='#1B5E20', fontweight='bold')
ax.text(11.05, 4.1, '(b,n,v)', fontsize=9, ha='center', va='center',
        color='#616161', style='italic')

ax.annotate('', xy=(10.3, 4.6), xytext=(9.5, 4.6),
    arrowprops=dict(arrowstyle='->', color='#AD1457', lw=1.5))
ax.annotate('', xy=(11.05, 3.5), xytext=(11.05, 1.1),
    arrowprops=dict(arrowstyle='->', color='#E65100', lw=1.5))
ax.annotate('', xy=(10.3, 3.5), xytext=(10.3, 1.1),
    arrowprops=dict(arrowstyle='->', color='#33691E', lw=1.5))
ax.text(11.5, 2.0, '×', fontsize=14, ha='center', va='center')

plt.tight_layout()
plt.savefig('/home/sakana/Code/Obsidian-Vault/assets/attachments/d2l-buoi-52/dot-product-forward-shapes.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: dot-product-forward-shapes.png")

print("\nAll visualizations for Buổi 52 saved successfully.")
