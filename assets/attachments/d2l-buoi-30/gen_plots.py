import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ============================================================
# Figure 1: Receptive Field — 2×Conv 3×3 vs 1×Conv 5×5
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Left: 1×Conv 5×5 ---
ax = axes[0]
ax.set_xlim(-0.5, 6.5)
ax.set_ylim(-0.5, 6.5)
ax.set_aspect('equal')
ax.set_title('1×Conv 5×5\nReceptive Field = 5×5', fontsize=14, fontweight='bold', pad=15)

# Draw grid 5×5 — input pixels
for i in range(5):
    for j in range(5):
        rect = mpatches.FancyBboxPatch((i+0.05, j+0.05), 0.9, 0.9, 
                                        boxstyle="round,pad=0.05",
                                        facecolor='#ef476f', alpha=0.6, 
                                        edgecolor='#333', linewidth=1.5)
        ax.add_patch(rect)

# Output pixel
rect = mpatches.FancyBboxPatch((2.05, 6.05), 0.9, 0.9,
                                boxstyle="round,pad=0.05",
                                facecolor='#06D6A0', alpha=0.8,
                                edgecolor='#333', linewidth=2)
ax.add_patch(rect)
ax.text(2.5, 6.5, 'out', ha='center', va='center', fontsize=10, fontweight='bold')

# Arrow
ax.annotate('', xy=(2.5, 5.9), xytext=(2.5, 5.1),
            arrowprops=dict(arrowstyle='->', lw=2, color='#333'))

ax.text(2.5, 5.5, 'Conv 5×5', ha='center', va='center', fontsize=11, 
        fontweight='bold', color='#4A90D9',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#4A90D9', alpha=0.9))

# Labels
ax.text(2.5, -0.3, f'Params: 25c²', ha='center', fontsize=12, fontweight='bold', color='#ef476f')
ax.text(2.5, -0.7, f'ReLU: 1 lần', ha='center', fontsize=11, color='#666')
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)


# --- Right: 2×Conv 3×3 ---
ax = axes[1]
ax.set_xlim(-0.5, 6.5)
ax.set_ylim(-1.5, 9.5)
ax.set_aspect('equal')
ax.set_title('2×Conv 3×3\nReceptive Field = 5×5 (tương đương!)', fontsize=14, fontweight='bold', pad=15)

# Layer 0: Input pixels 5×5
for i in range(5):
    for j in range(5):
        rect = mpatches.FancyBboxPatch((i+0.05, j+0.05), 0.9, 0.9,
                                        boxstyle="round,pad=0.05",
                                        facecolor='#ef476f', alpha=0.4,
                                        edgecolor='#333', linewidth=1)
        ax.add_patch(rect)

# Highlight 3×3 center in input
for i in range(1, 4):
    for j in range(1, 4):
        rect = mpatches.FancyBboxPatch((i+0.05, j+0.05), 0.9, 0.9,
                                        boxstyle="round,pad=0.05",
                                        facecolor='#ef476f', alpha=0.7,
                                        edgecolor='#333', linewidth=1.5)
        ax.add_patch(rect)

# Arrow Conv1
ax.annotate('', xy=(2.5, 5.9), xytext=(2.5, 5.1),
            arrowprops=dict(arrowstyle='->', lw=2, color='#333'))
ax.text(2.5, 5.5, 'Conv 3×3 #1\n+ ReLU', ha='center', va='center', fontsize=10,
        fontweight='bold', color='#E8A838',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#E8A838', alpha=0.9))

# Layer 1: After first conv — 3×3 intermediate
for i in range(1, 4):
    for j in range(6, 9):
        rect = mpatches.FancyBboxPatch((i+0.05, j+0.05-6+6), 0.9, 0.9,
                                        boxstyle="round,pad=0.05",
                                        facecolor='#E8A838', alpha=0.5,
                                        edgecolor='#333', linewidth=1.5)
        ax.add_patch(rect)

# Arrow Conv2
ax.annotate('', xy=(2.5, 9.1), xytext=(2.5, 8.95+0.1),
            arrowprops=dict(arrowstyle='->', lw=2, color='#333'))

# Actually fix the drawing — layers at y=0-4 (input), y=6-8 (intermediate), y=10 (output)
# Let me redo this properly
ax.clear()
ax.set_xlim(-0.5, 6.5)
ax.set_ylim(-1.5, 12.5)
ax.set_aspect('equal')
ax.set_title('2×Conv 3×3\nReceptive Field = 5×5 (tương đương!)', fontsize=14, fontweight='bold', pad=15)

# Layer 0 (Input): 5×5 grid at y=0..4
for i in range(5):
    for j in range(5):
        alpha = 0.7 if (1 <= i <= 3 and 1 <= j <= 3) else 0.35
        rect = mpatches.FancyBboxPatch((i+0.05, j+0.05), 0.9, 0.9,
                                        boxstyle="round,pad=0.05",
                                        facecolor='#ef476f', alpha=alpha,
                                        edgecolor='#333', linewidth=1)
        ax.add_patch(rect)
ax.text(2.5, -0.5, 'Input (5×5)', ha='center', fontsize=11, color='#666')

# Arrow + Label Conv1
ax.text(2.5, 5.7, 'Conv 3×3 #1 + ReLU', ha='center', va='center', fontsize=10,
        fontweight='bold', color='#E8A838',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#E8A838', alpha=0.9))
ax.annotate('', xy=(2.5, 6.5), xytext=(2.5, 6.1),
            arrowprops=dict(arrowstyle='->', lw=2, color='#333'))

# Layer 1 (Intermediate): 3×3 grid at y=6.5..8.5
for i in range(1, 4):
    for j in range(3):
        rect = mpatches.FancyBboxPatch((i+0.05, j+6.5+0.05), 0.9, 0.9,
                                        boxstyle="round,pad=0.05",
                                        facecolor='#E8A838', alpha=0.5,
                                        edgecolor='#333', linewidth=1.5)
        ax.add_patch(rect)
ax.text(5, 7.5, 'Feature Map\n(3×3)', ha='center', fontsize=10, color='#666')

# Arrow + Label Conv2
ax.text(2.5, 10, 'Conv 3×3 #2 + ReLU', ha='center', va='center', fontsize=10,
        fontweight='bold', color='#4A90D9',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#4A90D9', alpha=0.9))
ax.annotate('', xy=(2.5, 10.8), xytext=(2.5, 10.4),
            arrowprops=dict(arrowstyle='->', lw=2, color='#333'))

# Layer 2 (Output): 1×1 at y=11
rect = mpatches.FancyBboxPatch((2.05, 11.05), 0.9, 0.9,
                                boxstyle="round,pad=0.05",
                                facecolor='#06D6A0', alpha=0.8,
                                edgecolor='#333', linewidth=2)
ax.add_patch(rect)
ax.text(2.5, 11.5, 'out', ha='center', va='center', fontsize=10, fontweight='bold')

# Params & ReLU
ax.text(2.5, -1.0, f'Params: 2×9c² = 18c² (ít hơn 28%)', ha='center', fontsize=12, fontweight='bold', color='#06D6A0')
ax.text(2.5, -1.4, f'ReLU: 2 lần (tăng tính phi tuyến)', ha='center', fontsize=11, color='#666')

ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.savefig('/home/sakana/Code/Obsidian-Vault/assets/attachments/d2l-buoi-30/receptive_field_comparison.png', 
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ============================================================
# Figure 2: VGG-11 Architecture
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(16, 8))
ax.set_xlim(0, 18)
ax.set_ylim(0, 8)
ax.set_aspect('equal')

blocks = [
    {'name': 'Block 1\n1×Conv3 64', 'in': '224²', 'out': '112²×64', 'color': '#4A90D9', 'x': 1},
    {'name': 'Block 2\n1×Conv3 128', 'in': '', 'out': '56²×128', 'color': '#4A90D9', 'x': 3.5},
    {'name': 'Block 3\n2×Conv3 256', 'in': '', 'out': '28²×256', 'color': '#E8A838', 'x': 6},
    {'name': 'Block 4\n2×Conv3 512', 'in': '', 'out': '14²×512', 'color': '#E8A838', 'x': 8.5},
    {'name': 'Block 5\n2×Conv3 512', 'in': '', 'out': '7²×512', 'color': '#E8A838', 'x': 11},
]

fc_layers = [
    {'name': 'FC\n4096', 'color': '#ef476f', 'x': 13.5},
    {'name': 'FC\n4096', 'color': '#ef476f', 'x': 15},
    {'name': 'FC\n10', 'color': '#06D6A0', 'x': 16.5},
]

# Draw VGG Blocks
y_center = 4
block_height = 3
block_width = 1.8

# Title
ax.text(9, 7.5, 'Kiến trúc VGG-11: 5 VGG Blocks + 3 FC Layers', 
        ha='center', fontsize=16, fontweight='bold', color='#333')

# Input
ax.text(0.2, y_center, '📷\nInput\n1×224×224', ha='center', va='center', fontsize=9, color='#333')

for i, b in enumerate(blocks):
    h = block_height - i*0.3
    rect = mpatches.FancyBboxPatch((b['x']-block_width/2, y_center-h/2), block_width, h,
                                    boxstyle="round,pad=0.1",
                                    facecolor=b['color'], alpha=0.7,
                                    edgecolor='#333', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(b['x'], y_center+0.2, b['name'], ha='center', va='center', fontsize=9, 
            fontweight='bold', color='white')
    ax.text(b['x'], y_center-h/2-0.3, b['out'], ha='center', fontsize=8, color='#666')
    
    if i > 0:
        ax.annotate('', xy=(b['x']-block_width/2-0.1, y_center), 
                    xytext=(blocks[i-1]['x']+block_width/2+0.1, y_center),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='#333'))

# Arrow from input to Block 1
ax.annotate('', xy=(blocks[0]['x']-block_width/2-0.1, y_center), 
            xytext=(0.6, y_center),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='#333'))

# Flatten arrow
ax.annotate('', xy=(fc_layers[0]['x']-block_width/2-0.1, y_center),
            xytext=(blocks[-1]['x']+block_width/2+0.1, y_center),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='#333'))
ax.text(12.2, y_center+0.4, 'Flatten\n25088', ha='center', fontsize=8, color='#999')

for i, fc in enumerate(fc_layers):
    h = 2.0
    rect = mpatches.FancyBboxPatch((fc['x']-0.5, y_center-h/2), 1.0, h,
                                    boxstyle="round,pad=0.1",
                                    facecolor=fc['color'], alpha=0.7,
                                    edgecolor='#333', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(fc['x'], y_center, fc['name'], ha='center', va='center', fontsize=9,
            fontweight='bold', color='white')
    
    if i > 0:
        ax.annotate('', xy=(fc['x']-0.5-0.05, y_center),
                    xytext=(fc_layers[i-1]['x']+0.5+0.05, y_center),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='#333'))

# Patterns annotation
ax.text(6, 1.2, '📐 Spatial: 224 → 112 → 56 → 28 → 14 → 7  (mỗi block ÷2)', 
        ha='center', fontsize=11, fontweight='bold', color='#4A90D9')
ax.text(6, 0.6, '📊 Channels: 64 → 128 → 256 → 512 → 512  (mỗi block ×2)', 
        ha='center', fontsize=11, fontweight='bold', color='#E8A838')

ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.savefig('/home/sakana/Code/Obsidian-Vault/assets/attachments/d2l-buoi-30/vgg11_architecture.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ============================================================
# Figure 3: VGG Block — Detailed view
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(8, 10))
ax.set_xlim(0, 8)
ax.set_ylim(0, 12)
ax.set_aspect('equal')

ax.set_title('VGG Block (num_convs=2, channels=128)', fontsize=14, fontweight='bold', pad=20)

layers = [
    {'name': 'Input\n(C, H, W)', 'color': '#4A90D9', 'y': 10.5, 'h': 1.2},
    {'name': 'Conv 3×3, pad=1\n+ ReLU', 'color': '#E8A838', 'y': 8.5, 'h': 1.2},
    {'name': 'Conv 3×3, pad=1\n+ ReLU', 'color': '#E8A838', 'y': 6.5, 'h': 1.2},
    {'name': 'MaxPool 2×2\nstride=2', 'color': '#ef476f', 'y': 4.5, 'h': 1.2},
    {'name': 'Output\n(128, H/2, W/2)', 'color': '#06D6A0', 'y': 2.5, 'h': 1.2},
]

annotations = [
    '',  # Input
    'Giữ nguyên H×W\n(nhờ padding=1)',
    'Giữ nguyên H×W\n(nhờ padding=1)',
    'Giảm H×W còn 1/2',
    '',  # Output
]

for i, l in enumerate(layers):
    w = 3.5
    rect = mpatches.FancyBboxPatch((4-w/2, l['y']-l['h']/2), w, l['h'],
                                    boxstyle="round,pad=0.1",
                                    facecolor=l['color'], alpha=0.7,
                                    edgecolor='#333', linewidth=2)
    ax.add_patch(rect)
    ax.text(4, l['y'], l['name'], ha='center', va='center', fontsize=11,
            fontweight='bold', color='white')
    
    # Annotation on the right
    if annotations[i]:
        ax.text(6.2, l['y'], annotations[i], ha='left', va='center', fontsize=9, color='#666',
                fontstyle='italic')
    
    # Arrow
    if i < len(layers) - 1:
        ax.annotate('', xy=(4, layers[i+1]['y']+layers[i+1]['h']/2+0.1),
                    xytext=(4, l['y']-l['h']/2-0.1),
                    arrowprops=dict(arrowstyle='->', lw=2, color='#333'))

# Border for the block
border = mpatches.FancyBboxPatch((1.5, 3.7), 5, 7.2,
                                  boxstyle="round,pad=0.2",
                                  facecolor='none',
                                  edgecolor='#999', linewidth=2, linestyle='--')
ax.add_patch(border)
ax.text(6.8, 10.3, 'VGG Block', fontsize=12, fontweight='bold', color='#999', rotation=-90)

# Formula
ax.text(4, 1.5, r'Công thức output: $\frac{in + 2(1) - 3}{1} + 1 = in$ → giữ nguyên size', 
        ha='center', fontsize=10, color='#333',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0', edgecolor='#ccc'))
ax.text(4, 0.7, r'MaxPool: $\frac{in - 2}{2} + 1 = \frac{in}{2}$ → giảm 2×', 
        ha='center', fontsize=10, color='#333',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0', edgecolor='#ccc'))

ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.savefig('/home/sakana/Code/Obsidian-Vault/assets/attachments/d2l-buoi-30/vgg_block_detail.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ All 3 figures generated successfully!")
