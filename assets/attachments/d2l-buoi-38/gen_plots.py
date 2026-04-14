import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

out = "/home/sakana/Code/Obsidian-Vault/assets/attachments/d2l-buoi-38"

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.facecolor': 'white',
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

# ===== 1. Text Processing Pipeline =====
fig, ax = plt.subplots(1, 1, figsize=(14, 5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 5)
ax.axis('off')
ax.set_title('Pipeline: Tu van ban tho den chuoi so (Text → Indices)', fontsize=16, fontweight='bold', pad=15)

# Boxes
boxes = [
    (0.5, 2.0, 2.8, 1.5, '#3498DB', 'Buoc 1:\nDoc van ban\n(Raw Text)'),
    (3.8, 2.0, 2.8, 1.5, '#E67E22', 'Buoc 2:\nTien xu ly\n(Preprocessing)'),
    (7.1, 2.0, 2.8, 1.5, '#2ECC71', 'Buoc 3:\nTokenization\n(Tach token)'),
    (10.4, 2.0, 2.8, 1.5, '#9B59B6', 'Buoc 4:\nVocab → Indices\n(So hoa)'),
]

for x, y, w, h, color, text in boxes:
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                     facecolor=color, edgecolor='white', linewidth=2, alpha=0.9)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=11, 
            color='white', fontweight='bold', linespacing=1.5)

# Arrows
for i in range(3):
    x_start = boxes[i][0] + boxes[i][2]
    x_end = boxes[i+1][0]
    y_mid = boxes[i][1] + boxes[i][3] / 2
    ax.annotate('', xy=(x_end, y_mid), xytext=(x_start, y_mid),
                arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=2.5))

# Examples below
examples = [
    (1.9, 1.3, '"The Time Machine,\nby H. G. Wells"', '#2C3E50'),
    (5.2, 1.3, '"the time machine\nby h g wells"', '#2C3E50'),
    (8.5, 1.3, "['t','h','e',' ',\n't','i','m','e']", '#2C3E50'),
    (11.8, 1.3, '[21, 9, 6, 0,\n21, 10, 14, 6]', '#2C3E50'),
]
for x, y, text, color in examples:
    ax.text(x, y, text, ha='center', va='top', fontsize=9, color=color,
            style='italic', family='monospace', linespacing=1.4)

plt.savefig(os.path.join(out, 'text_processing_pipeline.png'))
plt.close()

# ===== 2. Tokenization comparison =====
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Ba phuong phap Tokenization', fontsize=16, fontweight='bold', y=1.02)

text_input = "the time machine"

# Word-level
ax = axes[0]
ax.set_title('Word-level', fontsize=13, color='#3498DB', fontweight='bold')
words = text_input.split()
for i, w in enumerate(words):
    rect = mpatches.FancyBboxPatch((0.1, 2.5 - i*0.9), 0.8, 0.6, boxstyle="round,pad=0.05", 
                                     facecolor='#3498DB', alpha=0.8, edgecolor='white')
    ax.add_patch(rect)
    ax.text(0.5, 2.8 - i*0.9, w, ha='center', va='center', fontsize=12, color='white', fontweight='bold')
ax.text(0.5, 0.3, f'Vocab ~ 4,580 tu\n(toan bo corpus)', ha='center', fontsize=9, color='#7f8c8d')
ax.set_xlim(0, 1); ax.set_ylim(0, 3.5); ax.axis('off')

# Char-level  
ax = axes[1]
ax.set_title('Character-level', fontsize=13, color='#E67E22', fontweight='bold')
chars = list(text_input)
cols = 6
for i, c in enumerate(chars):
    row = i // cols
    col = i % cols
    color = '#E67E22' if c != ' ' else '#BDC3C7'
    rect = mpatches.FancyBboxPatch((0.05 + col*0.15, 2.5 - row*0.7), 0.12, 0.5, 
                                     boxstyle="round,pad=0.02", facecolor=color, alpha=0.8, edgecolor='white')
    ax.add_patch(rect)
    label = c if c != ' ' else '_'
    ax.text(0.11 + col*0.15, 2.75 - row*0.7, label, ha='center', va='center', fontsize=10, color='white', fontweight='bold')
ax.text(0.5, 0.3, f'Vocab = 28 ky tu\n(a-z + space + unk)', ha='center', fontsize=9, color='#7f8c8d')
ax.set_xlim(0, 1); ax.set_ylim(0, 3.5); ax.axis('off')

# Subword
ax = axes[2]
ax.set_title('Subword (BPE)', fontsize=13, color='#2ECC71', fontweight='bold')
subwords = ['the', '_time', '_mach', 'ine']
for i, sw in enumerate(subwords):
    rect = mpatches.FancyBboxPatch((0.05, 2.5 - i*0.7), 0.9, 0.5, boxstyle="round,pad=0.05",
                                     facecolor='#2ECC71', alpha=0.8, edgecolor='white')
    ax.add_patch(rect)
    ax.text(0.5, 2.75 - i*0.7, sw, ha='center', va='center', fontsize=12, color='white', fontweight='bold')
ax.text(0.5, 0.3, f'Vocab ~ 30,000\n(balance)', ha='center', fontsize=9, color='#7f8c8d')
ax.set_xlim(0, 1); ax.set_ylim(0, 3.5); ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(out, 'tokenization_comparison.png'))
plt.close()

# ===== 3. Zipf's Law =====
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Luat Zipf: Tan suat tu trong ngon ngu tu nhien", fontsize=16, fontweight='bold')

# Simulate Zipf data
np.random.seed(42)
vocab_size = 500
ranks = np.arange(1, vocab_size + 1)
alpha = 1.0
unigram_freqs = 2261 / ranks**alpha + np.random.uniform(0, 5, vocab_size)
bigram_freqs = 309 / ranks**(alpha*0.95) + np.random.uniform(0, 2, vocab_size)
trigram_freqs = 59 / ranks**(alpha*0.90) + np.random.uniform(0, 1, vocab_size)

ax = axes[0]
ax.loglog(ranks, unigram_freqs, 'o-', color='#3498DB', markersize=2, label='Unigram', alpha=0.8)
ax.loglog(ranks[:300], bigram_freqs[:300], 's-', color='#E67E22', markersize=2, label='Bigram', alpha=0.8)
ax.loglog(ranks[:200], trigram_freqs[:200], '^-', color='#E74C3C', markersize=2, label='Trigram', alpha=0.8)
ax.set_xlabel('Rank (thu hang)', fontsize=12)
ax.set_ylabel('Tan suat n(x)', fontsize=12)
ax.set_title('Log-log plot: Tan suat vs Rank', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

ax = axes[1]
top_words = ['the', 'i', 'and', 'of', 'a', 'to', 'was', 'in', 'that', 'my']
top_freqs = [2261, 1267, 1245, 1155, 816, 695, 552, 541, 443, 440]
bars = ax.barh(range(len(top_words)-1, -1, -1), top_freqs, color='#3498DB', alpha=0.8, edgecolor='white')
ax.set_yticks(range(len(top_words)-1, -1, -1))
ax.set_yticklabels(top_words, fontsize=11)
ax.set_xlabel('Tan suat', fontsize=12)
ax.set_title('Top 10 tu pho bien nhat\n(The Time Machine)', fontsize=13)
for i, (freq, bar) in enumerate(zip(top_freqs, bars)):
    ax.text(freq + 30, len(top_words)-1-i, str(freq), va='center', fontsize=10, color='#2C3E50')
ax.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(out, 'zipf_law.png'))
plt.close()

# ===== 4. Language Model Concepts =====
fig, ax = plt.subplots(1, 1, figsize=(14, 6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis('off')
ax.set_title('Language Model: Uoc luong xac suat chuoi\nP(x1, x2, ..., xT) = ?', fontsize=16, fontweight='bold', pad=15)

# Chain rule decomposition visual
words_example = ['I', 'love', 'deep', 'learning']
probs = ['P(I)', 'P(love|I)', 'P(deep|I,love)', 'P(learning|I,love,deep)']
colors = ['#3498DB', '#E67E22', '#2ECC71', '#9B59B6']

for i, (word, prob, color) in enumerate(zip(words_example, probs, colors)):
    x = 1.5 + i * 3
    # Word box
    rect = mpatches.FancyBboxPatch((x-0.6, 4.5), 1.2, 0.8, boxstyle="round,pad=0.1",
                                     facecolor=color, edgecolor='white', linewidth=2, alpha=0.9)
    ax.add_patch(rect)
    ax.text(x, 4.9, word, ha='center', va='center', fontsize=14, color='white', fontweight='bold')
    
    # Probability below
    ax.text(x, 3.5, prob, ha='center', va='center', fontsize=10, color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.1, edgecolor=color))
    
    # Multiply sign
    if i < 3:
        ax.text(x + 1.5, 3.5, 'x', ha='center', va='center', fontsize=16, color='#2C3E50', fontweight='bold')

# Bottom: perplexity explanation
ax.text(7, 1.8, 'Perplexity = exp(Cross-Entropy trung binh)', ha='center', fontsize=14, 
        fontweight='bold', color='#E74C3C',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#E74C3C', alpha=0.1, edgecolor='#E74C3C'))

# Cases
cases = [
    (2.5, 0.7, 'PP = 1\n(hoan hao)', '#2ECC71'),
    (7, 0.7, 'PP = |V|\n(uniform/random)', '#E67E22'),
    (11.5, 0.7, 'PP = inf\n(P = 0 cho 1 tu)', '#E74C3C'),
]
for x, y, text, color in cases:
    ax.text(x, y, text, ha='center', va='center', fontsize=11, color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.15, edgecolor=color))

plt.savefig(os.path.join(out, 'language_model_concepts.png'))
plt.close()

# ===== 5. Sequence Partitioning =====
fig, ax = plt.subplots(1, 1, figsize=(14, 5))
ax.set_xlim(0, 16)
ax.set_ylim(0, 6)
ax.axis('off')
ax.set_title('Phan chia chuoi thanh Input-Target pairs (n=5, d=2)', fontsize=15, fontweight='bold', pad=15)

# Full corpus
corpus = list(range(22))
# Show tokens
for i, tok in enumerate(corpus[:17]):
    x = 0.5 + i * 0.9
    color = '#BDC3C7' if i < 2 else '#3498DB'  # d=2 discarded
    alpha_val = 0.3 if i < 2 else 0.8
    rect = mpatches.FancyBboxPatch((x-0.35, 4.8), 0.7, 0.6, boxstyle="round,pad=0.05",
                                     facecolor=color, alpha=alpha_val, edgecolor='white')
    ax.add_patch(rect)
    ax.text(x, 5.1, f'x{tok}', ha='center', va='center', fontsize=8, color='white', fontweight='bold')

ax.text(1.4, 4.3, 'Bo di\n(d=2)', ha='center', fontsize=8, color='#E74C3C', fontstyle='italic')

# Input-Target pairs
pair_colors = ['#3498DB', '#E67E22', '#2ECC71', '#9B59B6', '#E74C3C']
for pair_idx in range(5):
    start = 2 + pair_idx
    y_input = 3.2
    y_target = 1.8
    color = pair_colors[pair_idx]
    
    # Input
    for j in range(5):
        x = 2 + pair_idx * 3 + j * 0.5
        if x > 15: break
    
    # Simplified - just labels
    x_label = 1 + pair_idx * 3
    input_text = f'Input {pair_idx+1}: x{2+pair_idx}..x{2+pair_idx+4}'
    target_text = f'Target {pair_idx+1}: x{3+pair_idx}..x{3+pair_idx+4}'
    
    ax.text(x_label + 1, y_input, input_text, ha='center', va='center', fontsize=9, color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.15, edgecolor=color))
    ax.text(x_label + 1, y_target, target_text, ha='center', va='center', fontsize=9, color=color,
            bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.08, edgecolor=color, linestyle='--'))

ax.text(8, 0.8, 'Target = Input dich sang phai 1 buoc (shifted by 1)', ha='center', fontsize=12,
        color='#2C3E50', fontweight='bold', fontstyle='italic')

plt.savefig(os.path.join(out, 'sequence_partitioning.png'))
plt.close()

print("All plots generated successfully!")
