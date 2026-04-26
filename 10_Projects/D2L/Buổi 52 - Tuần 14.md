---
session: "D2L Tuần 14, Buổi 52 — 11.3 Attention Scoring Functions"
aliases: ["Buổi 52"]
tags: [d2l, deep-learning, attention, dot-product, additive-attention, masked-softmax, bmm, nlp]
status: growth
source: "D2L Chapter 11.3 — Attention Scoring Functions"
created: 2026-04-25
related:
  - "[[Buổi 50 - Tuần 14]]"
  - "[[Buổi 51 - Tuần 14]]"
  - "[[Buổi 53 - Tuần 14]]"
---

# Buổi 52 — 11.3 Attention Scoring Functions

> [!NOTE] Mục tiêu buổi học
> - Hiểu derivation từ Gaussian kernel → scaled dot product attention
> - Nắm lý do tại sao cần chia $\sqrt{d}$ (variance control)
> - Thực hiện được masked softmax cho variable-length sequences
> - Hiểu BMM (Batch Matrix Multiplication) cho efficient minibatch computation
> - Implement DotProductAttention và AdditiveAttention từ scratch
> - Phân biệt khi nào dùng dot product vs additive attention

---

## Active Recall — Ôn lại Buổi 51

### Câu hỏi truy hồi

1. Nadaraya-Watson estimator là gì? Công thức (11.2.2).
2. 4 kernel functions: Gaussian, Boxcar, Epanechikov, Constant — viết công thức.
3. Tại sao Gaussian, Boxcar, Epanechikov cho kết quả regression gần giống nhau?
4. Bias-variance trade-off của Gaussian width $\sigma$.
5. Đơn giản hóa Gaussian kernel khi $\|\mathbf{x}\|=1$ (unit sphere). Implication cho attention?

### Tự trả lời

1. $f(\mathbf{q}) = \sum_i \mathbf{v}_i \cdot \frac{\alpha(\mathbf{q}, \mathbf{k}_i)}{\sum_j \alpha(\mathbf{q}, \mathbf{k}_j)}$. Non-parametric — không có learnable parameters.
2. Gaussian: $\exp(-\|q-k\|^2/2)$, Boxcar: $\mathbb{1}(\|q-k\| \leq 1)$, Epanechikov: $\max(0, 1-\|q-k\|)$, Constant: $1$.
3. Vì chúng đều "ưu tiên điểm gần hơn điểm xa". Sự khác biệt về functional form ít quan trọng bằng việc có weighting scheme hay không. Constant thất bại vì không có weighting.
4. Narrow $\sigma$ → low bias, high variance (noise-sensitive). Wide $\sigma$ → high bias, low variance (stable). Optimal ở giữa.
5. $\|x-x_i\|^2 = 2 - 2x^\top x_i$. Gaussian becomes $\exp(-\sigma^2(1 - x^\top x_i)) \propto \exp(\sigma^2 \cdot x^\top x_i)$. → **Essentially scaled dot-product attention!**

---

# PHẦN I — TỪ GAUSSIAN KERNEL ĐẾN DOT PRODUCT

## 1.1 Derivation: Gaussian → Dot Product (D2L Eq. 11.3.1)

[!NOTE] ELI5
> Gaussian kernel đo khoảng cách giữa query và key. Nhưng tính khoảng cách bình phương $\|\mathbf{q} - \mathbf{k}_i\|^2$ tốn $O(d)$ mỗi cặp. Dot product chỉ tốn 1 phép nhân-scaling. Attention hiệu quả hơn khi dùng dot product.

**Định nghĩa kỹ thuật (D2L 11.3.1):**

Khai triển Gaussian kernel theo squared distance:

$$a(\mathbf{q}, \mathbf{k}_i) = -\frac{1}{2} \|\mathbf{q} - \mathbf{k}_i\|^2$$

Khai triển:
$$= -\frac{1}{2}(\|\mathbf{q}\|^2 - 2\mathbf{q}^\top\mathbf{k}_i + \|\mathbf{k}_i\|^2)$$
$$= \mathbf{q}^\top\mathbf{k}_i - \frac{1}{2}\|\mathbf{k}_i\|^2 - \frac{1}{2}\|\mathbf{q}\|^2$$

**3 bước để đơn giản hóa:**

| Bước | Giải thích | Term biến mất |
|-------|-----------|--------------|
| **1. Normalization** | Softmax normalize đảm bảo $\sum_i \alpha_i = 1$ | Term $-\frac{1}{2}\|\mathbf{q}\|^2$ giống nhau cho mọi pair → cancel out |
| **2. Layer Norm** | Khi $\mathbf{k}_i$ được sinh từ layer norm → $\|\mathbf{k}_i\| \approx \text{constant}$ | Term $-\frac{1}{2}\|\mathbf{k}_i\|^2$ trở nên constant → cancel out |
| **3. Result** | Còn lại $\mathbf{q}^\top\mathbf{k}_i$ | Chỉ cần dot product! |

**Kết quả:** $a(\mathbf{q}, \mathbf{k}_i) = \mathbf{q}^\top\mathbf{k}_i$

---

## 1.2 Scaled Dot Product — Tại sao cần chia $\sqrt{d}$? (D2L Eq. 11.3.2)

[!NOTE] ELI5
> Dot product của 2 vectors ngẫu nhiên có "phương sai tăng theo chiều dài". Với $d=512$, dot product có thể lên tới vài trăm — softmax sẽ bão hòa (1 vector chiếm gần như 100% weight). Chia $\sqrt{d}$ giữ phương sai = 1, softmax hoạt động cân bằng.

**Định nghĩa kỹ thuật (D2L 11.3.2):**

Giả sử $\mathbf{q}, \mathbf{k}_i \in \mathbb{R}^d$ với elements i.i.d. $\sim \mathcal{N}(0, 1)$:

- $\mathbb{E}[\mathbf{q}^\top\mathbf{k}_i] = 0$ (zero mean) ✅
- $\text{Var}(\mathbf{q}^\top\mathbf{k}_i) = d$ (variance grows with $d$) ❌

**Vấn đề:** Khi $d$ lớn (e.g., $d=512$), dot product có thể có giá trị lớn $\sim O(\sqrt{d})$. Softmax của các giá trị lớn và chênh lệch sẽ bão hòa → 1 token chiếm ~100% weight, gradient vanish.

**Giải pháp:** Chia cho $\sqrt{d}$:

$$\boxed{a(\mathbf{q}, \mathbf{k}_i) = \frac{\mathbf{q}^\top \mathbf{k}_i}{\sqrt{d}}}$$

Với scaling này: $\text{Var}\!\left(\frac{\mathbf{q}^\top\mathbf{k}_i}{\sqrt{d}}\right) = \frac{d}{d} = 1$. ✅

**Final formula — Scaled Dot Product Attention (D2L 11.3.3):**

$$\alpha(\mathbf{q}, \mathbf{k}_i) = \text{softmax}\!\left(\frac{\mathbf{q}^\top \mathbf{k}_i}{\sqrt{d}}\right) = \frac{\exp(\mathbf{q}^\top \mathbf{k}_i / \sqrt{d})}{\sum_{j=1}^m \exp(\mathbf{q}^\top \mathbf{k}_j / \sqrt{d})}$$

![[assets/attachments/d2l-buoi-52/scaled-dot-product.png]]

> [!CRITICAL]- Tại sao Vaswani et al. (2017) dùng $\sqrt{d_k}$?
> Vì với standard initialization (mean=0, var=1), dot product variance = $d$. Scaling by $\sqrt{d}$ đưa variance về 1 → softmax scores ở "vùng an toàn" (không quá sharp, không quá flat) → gradient flow tốt.
>
> Đây là lý do $\sqrt{d_k}$ xuất hiện trong **attention scaling** của Transformer — không phải magic number mà là variance normalization.

> [!WARNING]- Dấu hiệu nhồi nhét
> Nếu bạn nhớ "chia $\sqrt{d}$" mà không hiểu tại sao → bạn đang nhồi nhét. Hãy tự hỏi: nếu $d$ rất nhỏ (e.g., $d=2$) thì sao? Dot product variance = 2 → chia $\sqrt{2}$ → variance = 1. Nếu $d=1$ → variance = 1 → không cần chia? Đúng!

---

# PHẦN II — CONVENIENCE FUNCTIONS: MASKED SOFTMAX VÀ BMM

## 2.1 Masked Softmax — Xử lý Variable-Length Sequences (D2L 11.3.2.1)

[!NOTE] ELI5
> Khi batch các câu có độ dài khác nhau, ta pad thêm tokens "<blank>". Masked softmax đảm bảo model không "chú ý" vào những tokens pad này — bằng cách set attention score = -1e6 → softmax weight ≈ 0.

**Định nghĩa kỹ thuật:**

Trong NLP, sequences có độ dài khác nhau (batch 3 câu):

```
Dive  into  Deep    Learning
Learn to    code    <blank>
Hello world <blank> <blank>
```

Các tokens `<blank>` (padding) không mang ý nghĩa. Ta cần giới hạn:
$$\sum_{i=1}^n \alpha(\mathbf{q}, \mathbf{k}_i)\mathbf{v}_i \to \sum_{i=1}^l \alpha(\mathbf{q}, \mathbf{k}_i)\mathbf{v}_i \quad (l \leq n)$$

**Cách implement (D2L):**

```python
def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis.
    
    X: 3D tensor (batch_size, n_queries, n_keys)
    valid_lens: tensor chứa độ dài hợp lệ của mỗi sequence
    """
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            # valid_lens = [2, 3]: batch 0 có 2 tokens valid, batch 1 có 3
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            # valid_lens 2D: [batch, n_queries] → mỗi query có độ dài riêng
            valid_lens = valid_lens.reshape(-1)
        
        # Thay thế masked elements bằng -1e6
        # exp(-1e6) ≈ 0 → gradient ≈ 0
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)
```

```python
def _sequence_mask(X, valid_len, value=0):
    """Tạo mask: True cho positions hợp lệ, False cho padded positions."""
    maxlen = X.size(1)
    mask = torch.arange(maxlen, dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X
```

**Cơ chế hoạt động:**

1. Tạo boolean mask: `True` cho positions valid, `False` cho padded
2. Set attention scores tại masked positions = `-1e6`
3. Softmax: $\exp(-\text{1e6}) \approx 0$ → attention weight $\approx 0$
4. Gradient tại masked positions: $\approx 0$ → không affect training

**Tại sao -1e6?** $\exp(-10^6) \approx 0$ với floating point precision. Gradient cũng $\approx 0$.

**valid_lens formats:**

| Format | Ví dụ | Shape | Ý nghĩa |
|--------|-------|-------|---------|
| 1D | `[2, 3]` | `(batch_size,)` | Mỗi example có cùng valid length cho mọi query |
| 2D | `[[1, 3], [2, 4]]` | `(batch_size, n_queries)` | Mỗi query có độ dài riêng |

![[assets/attachments/d2l-buoi-52/masked-softmax.png]]

---

## 2.2 Batch Matrix Multiplication (BMM) — Efficient Computation (D2L 11.3.2.2)

[!NOTE] ELI5
> Thay vì loop qua từng query một, BMM cho phép nhân **tất cả queries với tất cả keys cùng lúc** trong một ma trận. GPU parallelize hiệu quả — nhanh hơn loop rất nhiều.

**Định nghĩa kỹ thuật (D2L 11.3.4-11.3.5):**

$$\mathbf{Q} = [\mathbf{Q}_1, \mathbf{Q}_2, \ldots, \mathbf{Q}_n] \in \mathbb{R}^{n \times a \times b}$$
$$\mathbf{K} = [\mathbf{K}_1, \mathbf{K}_2, \ldots, \mathbf{K}_n] \in \mathbb{R}^{n \times b \times c}$$

$$\text{BMM}(\mathbf{Q}, \mathbf{K}) = [\mathbf{Q}_1\mathbf{K}_1, \mathbf{Q}_2\mathbf{K}_2, \ldots, \mathbf{Q}_n\mathbf{K}_n] \in \mathbb{R}^{n \times a \times c}$$

```python
Q = torch.ones((2, 3, 4))   # 2 matrices: each 3x4
K = torch.ones((2, 4, 6))   # 2 matrices: each 4x6
d2l.check_shape(torch.bmm(Q, K), (2, 3, 6))  # 2 matrices: each 3x6
```

**So sánh:**

| Method | Shape | Operation |
|--------|-------|-----------|
| Loop | `for i in range(n): out[i] = Q[i] @ K[i]` | $n$ matrix multiplications riêng lẻ |
| BMM | `torch.bmm(Q, K)` | 1 batch operation — GPU parallelizes |

![[assets/attachments/d2l-buoi-52/batch-matrix-multiplication.png]]

---

# PHẦN III — SCALED DOT PRODUCT ATTENTION (D2L 11.3.3)

## 3.1 Full Forward Pass — Shape Analysis

[!NOTE] ELI5
> Dot product attention nhận batch of queries, keys, values → tính scores → softmax → weighted sum → output. Tất cả trong 3 bước matrix operations.

**Định nghĩa kỹ thuật (D2L 11.3.6):**

Cho $\mathbf{Q} \in \mathbb{R}^{n \times d}$ (queries), $\mathbf{K} \in \mathbb{R}^{m \times d}$ (keys), $\mathbf{V} \in \mathbb{R}^{m \times v}$ (values):

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}\right)\mathbf{V} \in \mathbb{R}^{n \times v}$$

**Bước 1:** $\mathbf{S} = \frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}$ — scores matrix $(n \times m)$
**Bước 2:** $\mathbf{A} = \text{masked\_softmax}(\mathbf{S})$ — attention weights $(n \times m)$, rows sum to 1
**Bước 3:** $\mathbf{O} = \mathbf{A}\mathbf{V}$ — output $(n \times v)$

![[assets/attachments/d2l-buoi-52/d2l-fig-11-3-1.png]]

**Shape diagram:**

![[assets/attachments/d2l-buoi-52/dot-product-forward-shapes.png]]

## 3.2 Implementation — DotProductAttention Class

```python
class DotProductAttention(nn.Module):
    """Scaled dot product attention."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        """
        queries:  (batch_size, n_queries, d)
        keys:     (batch_size, m_keys, d)
        values:   (batch_size, m_keys, v)
        valid_lens: (batch_size,) hoặc (batch_size, n_queries)
        Returns:  (batch_size, n_queries, v)
        """
        d = queries.shape[-1]
        # BMM: Q @ K^T — scores (batch, n, m)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        # Masked softmax — attention weights
        self.attention_weights = masked_softmax(scores, valid_lens)
        # BMM: attention_weights @ V — output (batch, n, v)
        return torch.bmm(self.dropout(self.attention_weights), values)
```

**Dropout** được dùng để regularize attention weights — ngẫu nhiên "tắt" một số attention connections trong training.

```python
# Ví dụ với shapes
queries = torch.randn(2, 1, 2)    # batch=2, 1 query mỗi batch, d=2
keys    = torch.randn(2, 10, 2)   # batch=2, 10 keys, d=2
values  = torch.randn(2, 10, 4)   # batch=2, 10 values, v=4
valid_lens = torch.tensor([2, 6]) # batch 0: 2 keys valid, batch 1: 6 keys valid

attention = DotProductAttention(dropout=0.5)
output = attention(queries, keys, values, valid_lens)
# output shape: (2, 1, 4) — đúng như kỳ vọng
```

---

# PHẦN IV — ADDITIVE ATTENTION (D2L 11.3.4)

## 4.1 Khi nào cần Additive Attention?

[!NOTE] ELI5
> Dot product yêu cầu query và key cùng chiều. Additive attention cho phép query và key có **chiều khác nhau** — bằng cách project chúng vào cùng không gian trước khi so sánh.

**Định nghĩa kỹ thuật (D2L 11.3.7):**

Cho $\mathbf{q} \in \mathbb{R}^q$ và $\mathbf{k} \in \mathbb{R}^k$ (có thể $q \neq k$):

$$a(\mathbf{q}, \mathbf{k}) = \mathbf{w}_v^\top \tanh(\mathbf{W}_q\mathbf{q} + \mathbf{W}_k\mathbf{k}) \in \mathbb{R}$$

Trong đó $\mathbf{W}_q \in \mathbb{R}^{h \times q}$, $\mathbf{W}_k \in \mathbb{R}^{h \times k}$, $\mathbf{w}_v \in \mathbb{R}^h$ là learnable parameters.

**Interpretations:**

| Interpretation | Giải thích |
|---------------|-----------|
| **MLP view** | Query và Key được concatenate → đi qua MLP 1 hidden layer với $\tanh$ activation |
| **Additive view** | Project q và k riêng vào cùng không gian $h$, cộng lại, activate |
| **Bahdanau (2014)** | Đây chính là scoring function trong Bahdanau attention cho NMT |

**Khi nào dùng?**

| Scoring Function | Khi nào |
|-----------------|---------|
| **Dot Product** | $q = k$ (cùng dimension) — Transformer, self-attention |
| **Additive** | $q \neq k$ (khác dimension) — Bahdanau attention, cross-modal |

## 4.2 Implementation — AdditiveAttention Class

```python
class AdditiveAttention(nn.Module):
    """Additive attention (Bahdanau-style)."""
    def __init__(self, num_hiddens, dropout):
        super().__init__()
        # Project queries và keys vào cùng không gian h
        self.W_k = nn.LazyLinear(num_hiddens, bias=False)  # k → h
        self.W_q = nn.LazyLinear(num_hiddens, bias=False)  # q → h
        self.w_v = nn.LazyLinear(1, bias=False)            # h → 1 (scalar score)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        """
        queries: (batch_size, n_queries, q)
        keys:    (batch_size, m_keys, k)
        values:  (batch_size, m_keys, v)
        """
        queries, keys = self.W_q(queries), self.W_k(keys)
        # Feature expansion: shape (batch, n, 1, h) và (batch, 1, m, h)
        # Broadcast: cộng → (batch, n, m, h)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # w_v: h → 1, reshape → (batch, n, m)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # BMM: (batch, n, m) @ (batch, m, v) → (batch, n, v)
        return torch.bmm(self.dropout(self.attention_weights), values)
```

**Broadcasting trick:**

```python
# queries: (batch, n, h) → unsqueeze(2) → (batch, n, 1, h)
# keys:    (batch, m, h) → unsqueeze(1) → (batch, 1, m, h)
# features = queries.unsqueeze(2) + keys.unsqueeze(1)
# Result: (batch, n, m, h) — tất cả n×m pairs cùng lúc
```

---

# PHẦN V — SO SÁNH DOT PRODUCT VS ADDITIVE

## 5.1 Architecture Comparison

![[assets/attachments/d2l-buoi-52/dot-product-vs-additive.png]]

**So sánh chi tiết:**

| Khía cạnh | Dot Product Attention | Additive Attention |
|-----------|---------------------|-------------------|
| **Scoring** | $\mathbf{q}^\top\mathbf{k} / \sqrt{d}$ | $\mathbf{w}_v^\top \tanh(\mathbf{W}_q\mathbf{q} + \mathbf{W}_k\mathbf{k})$ |
| **Parameters** | 0 extra | $W_q \in \mathbb{R}^{h \times q}$, $W_k \in \mathbb{R}^{h \times k}$, $w_v \in \mathbb{R}^h$ |
| **Dimensionality** | Yêu cầu $q = k$ | Cho phép $q \neq k$ |
| **Complexity** | $O(n \cdot m \cdot d)$ | $O(n \cdot m \cdot h)$ |
| **MLP size** | Không có | 1 hidden layer, $h$ units |
| **Used in** | Transformer, self-attention | Bahdanau NMT |
| **Introduced** | Vaswani et al. (2017) | Bahdanau et al. (2014) |

> [!KEY]- D2L Note
> "In practice, the dot product attention is the mainstay of modern Transformer architectures. When queries and keys are vectors of different lengths, we can use the additive attention scoring function instead."
>
> Additive attention ra đời trước (2014, Bahdanau), dùng trong early Seq2Seq+attention models. Dot product attention ra đời sau (2017, Vaswani), phổ biến trong Transformer vì:
> 1. **Không có parameters** — simpler
> 2. **$O(1)$ extra parameters** thay vì MLP overhead
> 3. **Highly optimized** trên GPU (matrix multiplication rất nhanh)

## 5.2 Computational Complexity

**Dot Product Attention:**

| Operation | Complexity |
|-----------|-----------|
| $\mathbf{Q}\mathbf{K}^\top$ | $O(n \cdot m \cdot d)$ |
| Softmax | $O(n \cdot m)$ |
| $\mathbf{A}\mathbf{V}$ | $O(n \cdot m \cdot v)$ |
| **Total** | $O(n \cdot m \cdot (d + v))$ |

**Additive Attention:**

| Operation | Complexity |
|-----------|-----------|
| $\mathbf{W}_q\mathbf{Q}$ | $O(n \cdot q \cdot h)$ |
| $\mathbf{W}_k\mathbf{K}$ | $O(m \cdot k \cdot h)$ |
| $\tanh$ + $\mathbf{w}_v$ | $O(n \cdot m \cdot h)$ |
| **Total** | $O(n \cdot m \cdot h + (n \cdot q + m \cdot k) \cdot h)$ |

**Memory bandwidth:** BMM operations tốn bandwidth nhiều hơn computation → NVIDIA Transformer Engine tập trung vào optimizing attention.

---

# PHẦN VI — CODE: COMPLETE IMPLEMENTATION

## 6.1 Full Module với Masked Softmax

```python
import math
import torch
from torch import nn
from torch.nn import functional as F

# ============================================================
# masked_softmax — xử lý variable-length sequences
# ============================================================
def masked_softmax(X, valid_lens):
    """Softmax với masking cho padded tokens.
    
    X: 3D tensor (batch_size, n_queries, n_keys)
    valid_lens: (batch_size,) hoặc (batch_size, n_queries)
    """
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1)
        mask = torch.arange(maxlen, dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X

    if valid_lens is None:
        return F.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # Set masked positions = -1e6 → exp → ~0
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return F.softmax(X.reshape(shape), dim=-1)


# ============================================================
# DotProductAttention — scaled dot product (Transformer-style)
# ============================================================
class DotProductAttention(nn.Module):
    """Scaled dot product attention với dropout và masking."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        """
        Shapes:
          queries: (batch, n_queries, d)
          keys:    (batch, m_keys, d)
          values:  (batch, m_keys, v)
          valid_lens: (batch,) hoặc (batch, n_queries)
        Returns:
          (batch, n_queries, v)
        """
        d = queries.shape[-1]
        # BMM: Q @ K^T / sqrt(d)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


# ============================================================
# AdditiveAttention — Bahdanau-style với MLP scoring
# ============================================================
class AdditiveAttention(nn.Module):
    """Additive attention cho q != k dimensions."""
    def __init__(self, num_hiddens, dropout):
        super().__init__()
        self.W_k = nn.LazyLinear(num_hiddens, bias=False)  # k → h
        self.W_q = nn.LazyLinear(num_hiddens, bias=False)  # q → h
        self.w_v = nn.LazyLinear(1, bias=False)            # h → 1
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # Broadcast: (batch, n, 1, h) + (batch, 1, m, h) → (batch, n, m, h)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        scores = self.w_v(features).squeeze(-1)  # (batch, n, m)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
```

---

# PHẦN VII — SUMMARY VÀ LIÊN KẾT

## 7.1 Tóm tắt buổi

| Khái niệm | Hiểu | Cần ôn |
|-----------|------|--------|
| Gaussian → Dot Product derivation | ✅ | |
| Tại sao chia $\sqrt{d}$ (variance = 1) | ✅ | |
| Masked softmax (valid_lens, -1e6 trick) | ✅ | |
| BMM — batch matrix multiplication | ✅ | |
| DotProductAttention forward pass (shapes) | ✅ | |
| AdditiveAttention forward pass (MLP scoring) | ✅ | |
| Dot product vs additive: khi nào dùng | ✅ | |
| Complexity comparison | ✅ | |

## 7.2 Liên kết với các buổi tiếp theo

| Buổi | Chủ đề | Liên kết |
|------|--------|---------|
| **Buổi 53** (11.4) | Bahdanau Attention | Additive attention được dùng trong encoder-decoder attention |
| **Buổi 54** (11.5) | Multi-Head Attention | DotProductAttention × $h$ heads song song |
| **Buổi 55** (11.6) | Self-Attention & Positional Encoding | Q = K = V trong cùng sequence |

## 7.3 Bảng thuật ngữ

| Thuật ngữ | Định nghĩa |
|-----------|-----------|
| Attention scoring function $a(\mathbf{q}, \mathbf{k}_i)$ | Hàm tính "độ phù hợp" giữa query và key, trước softmax |
| Scaled dot product | $a(\mathbf{q}, \mathbf{k}_i) = \mathbf{q}^\top\mathbf{k}_i / \sqrt{d}$ |
| Variance normalization | Scaling by $\sqrt{d}$ giữ $\text{Var}(a) = 1$ khi $q, k_i \sim \mathcal{N}(0,1)$ |
| Masked softmax | Softmax với -1e6 cho padded positions → attention weight ≈ 0 |
| valid_lens | Tensor chỉ định độ dài hợp lệ của mỗi sequence trong batch |
| BMM (Batch Matrix Multiplication) | $\text{BMM}(\mathbf{Q}, \mathbf{K})$ — nhân batches of matrices element-wise |
| Additive attention | $a(\mathbf{q}, \mathbf{k}) = \mathbf{w}_v^\top \tanh(\mathbf{W}_q\mathbf{q} + \mathbf{W}_k\mathbf{k})$ |
| DotProductAttention | Attention với scaled dot product scoring, không extra parameters |
| Broadcasting in attention | $\text{unsqueeze}(q) + \text{unsqueeze}(k)$ → $(n, m, h)$ pairs cùng lúc |

---

## Bài tập D2L 11.3.6

1. **Distance-based attention**: Modify `DotProductAttention` để tính distance-based scores thay vì dot product. Gợi ý: chỉ cần squared norms $\|\mathbf{k}_i\|^2$.

2. **Different dimensions**: Modify dot product attention cho phép queries và keys có chiều khác nhau. Gợi ý: dùng matrix $\mathbf{M}$ để project giữa spaces.

3. **Complexity analysis**: Phân tích computational cost và memory bandwidth theo $d, v, n, m$. Khi nào trở thành bottleneck? Gợi ý: FlashAttention (Dao et al., 2022) giải quyết memory bottleneck.

---

## Active Recall — Câu hỏi về Buổi 52

1. **Cho $q, k_i \in \mathbb{R}^{d=64}$ với i.i.d. $\sim \mathcal{N}(0,1)$. Tính Var(dot product) với và không có scaling.** → Không scaling: $\text{Var} = d = 64$. Có scaling: $\text{Var} = 64/64 = 1$.

2. **Tại sao masked softmax dùng -1e6 thay vì -inf?** → $\exp(-\infty) = 0$ về mặt toán, nhưng -inf gây NaN trong một số implementation. -1e6 là safe approximation: $\exp(-10^6) \approx 0$ với float32.

3. **Cho AdditiveAttention với num_hiddens=8, query dim=20, key dim=2. Tính số parameters.** → $W_q$: $20 \times 8 = 160$, $W_k$: $2 \times 8 = 16$, $w_v$: $8 \times 1 = 8$. Total = 184 params.

4. **Tại sao dot product attention được ưu tiên hơn additive attention trong Transformer?** → (a) Không có parameters extra → simpler; (b) Chỉ dùng matrix multiplication → GPU-optimized; (c) Complexity $O(n \cdot m \cdot d)$ thấp hơn MLP-based attention.

5. **BMM vs loop: khi nào BMM nhanh hơn?** → BMM nhanh hơn khi $n$ lớn (nhiều matrices trong batch) — GPU parallelizes batch operations. Với $n=1$, overhead của BMM có thể lớn hơn loop đơn giản.

6. **Trong DotProductAttention, attention_weights có shape gì?** → $(batch, n\_queries, m\_keys)$ — mỗi query có $m$ attention weights (1 cho mỗi key). Sau softmax, mỗi hàng sum = 1.

---

## TODO

- [ ] Experiment: so sánh attention weights với và không có $\sqrt{d}$ scaling
- [ ] Implement distance-based attention như bài tập D2L
- [ ] Tìm hiểu FlashAttention (Dao et al., 2022) — giải quyết memory bottleneck
- [ ] Đọc paper gốc Vaswani (2017) — "Attention Is All You Need"

---

## Liên kết

- [[Buổi 50 - Tuần 14|QKV — Queries, Keys, and Values]]
- [[Buổi 51 - Tuần 14|Attention Pooling by Similarity]]
- [[Buổi 53 - Tuần 14|Bahdanau Attention]]
- [[Buổi 54 - Tuần 14|Multi-Head Attention]]
- [[Softmax Function]]
- [[Attention Mechanism]] *(concept note — cần tạo)*
- [[Transformer Architecture]] *(concept note đã có)*
