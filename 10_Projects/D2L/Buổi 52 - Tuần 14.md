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
>
> - [ ] Hiểu derivation từ Gaussian kernel sang scaled dot product
> - [ ] Nắm lý do tại sao cần chia $\sqrt{d}$ (kiểm soát phương sai)
> - [ ] Thực hiện được masked softmax cho variable-length sequences
> - [ ] Hiểu BMM (Batch Matrix Multiplication) cho tính toán hiệu quả với minibatch
> - [ ] Implement DotProductAttention và AdditiveAttention từ scratch
> - [ ] Phân biệt khi nào dùng dot product vs additive attention

---

## Active Recall — Ôn lại Buổi 51

### Câu hỏi truy hồi (không nhìn tài liệu)

1. Nadaraya-Watson estimator là gì? Công thức (11.2.2).
2. Bốn kernel functions: Gaussian, Boxcar, Epanechikov, Constant — viết công thức.
3. Tại sao Gaussian, Boxcar, Epanechikov cho kết quả regression gần giống nhau?
4. Bias-variance trade-off của Gaussian width $\sigma$.
5. Đơn giản hóa Gaussian kernel khi $\|\mathbf{x}\|=1$ (unit sphere). Implication cho attention?

### Trả lời nhanh

- **Q1:** $f(\mathbf{q}) = \sum_i \mathbf{v}_i \cdot \frac{\alpha(\mathbf{q}, \mathbf{k}_i)}{\sum_j \alpha(\mathbf{q}, \mathbf{k}_j)}$. **Claim** → **Reasoning** → Phi tham số — không có learnable parameters. **Evidence** → Eq. 11.2.2 trong D2L.

- **Q2:** Gaussian: $\exp(-\|q-k\|^2/2)$, Boxcar: $\mathbb{1}(\|q-k\| \leq 1)$, Epanechikov: $\max(0, 1-\|q-k\|)$, Constant: $1$. **Claim** → **Reasoning** → Mọi kernel đều đo "khoảng cách" giữa query và key. **Evidence** → D2L Eq. 11.2.1.

- **Q3:** Vì chúng đều "ưu tiên điểm gần hơn điểm xa". **Claim** → **Reasoning** → Sự khác biệt về functional form ít quan trọng bằng việc có weighting scheme hay không. Constant kernel thất bại vì không có weighting. **Evidence** → D2L Section 11.2.2.

- **Q4:** Kernel hẹp $\sigma$ → bias thấp, variance cao (nhạy với noise). Kernel rộng $\sigma$ → bias cao, variance thấp (ổn định). **Claim** → **Reasoning** → Optimal $\sigma$ nằm đâu đó ở giữa. **Evidence** → D2L Section 11.2.3.

- **Q5:** $\|x-x_i\|^2 = 2 - 2x^\top x_i$. Gaussian becomes $\exp(-\sigma^2(1 - x^\top x_i)) \propto \exp(\sigma^2 \cdot x^\top x_i)$. **Claim** → **Reasoning** → Trong softmax, constant factor cancel out → attention score essentially reduces to **scaled dot-product**. **Evidence** → Buổi 51 Active Recall Q3.

### Liên kết cần ôn lại

- [[Buổi 51 - Tuần 14|Nadaraya-Watson Kernel Regression]]
- [[Buổi 50 - Tuần 14|QKV — Queries, Keys, and Values]]

---

# PHẦN I — TỪ GAUSSIAN KERNEL SANG DOT PRODUCT

## 1.1 Derivation: Gaussian sang Dot Product (D2L Eq. 11.3.1)

> [!NOTE] ELI5
>
> Gaussian kernel đo khoảng cách giữa query và key bằng cách tính $\|\mathbf{q} - \mathbf{k}_i\|^2$ — tốn $O(d)$ phép tính. Nhưng sau khi đơn giản hóa (bỏ các term không đổi), chỉ còn lại $\mathbf{q}^\top\mathbf{k}_i$ — chỉ tốn **một phép nhân ma trận**. Attention hiệu đại vì **nhanh hơn** và **có thể học được**.

**Định nghĩa kỹ thuật:**

- **Đây là gì?** Sự biến đổi toán học cho thấy Gaussian kernel có thể đơn giản hóa thành dot product giữa query và key.
- **Input/Output là gì?** Input: Gaussian kernel $-\frac{1}{2}\|\mathbf{q} - \mathbf{k}_i\|^2$. Output: $\mathbf{q}^\top\mathbf{k}_i$ (sau khi loại bỏ các term không đổi).
- **Giải quyết vấn đề gì?** Giảm độ phức tạp tính toán từ $O(d)$ còn $O(1)$ cho mỗi cặp (query, key).
- **Thay thế/gợi ý giải pháp nào trước đây?** Gaussian kernel (NW) cần tính squared distance cho mỗi pair.

### Derivation chi tiết

Khai triển Gaussian kernel theo squared distance:

$$a(\mathbf{q}, \mathbf{k}_i) = -\frac{1}{2} \|\mathbf{q} - \mathbf{k}_i\|^2$$

Khai triển:

$$= -\frac{1}{2}(\|\mathbf{q}\|^2 - 2\mathbf{q}^\top\mathbf{k}_i + \|\mathbf{k}_i\|^2)$$

$$= \mathbf{q}^\top\mathbf{k}_i - \frac{1}{2}\|\mathbf{k}_i\|^2 - \frac{1}{2}\|\mathbf{q}\|^2$$

**Từ điển ký hiệu:**

| Ký hiệu | Định nghĩa |
|---------|-------------|
| $\mathbf{q}$ | Query vector $\in \mathbb{R}^d$ |
| $\mathbf{k}_i$ | Key vector thứ $i$ $\in \mathbb{R}^d$ |
| $\|\mathbf{q}\|^2$ | squared norm của $\mathbf{q}$ |
| $a(\mathbf{q}, \mathbf{k}_i)$ | Attention score (trước softmax) |

### Ba bước để đơn giản hóa

| Bước | Giải thích | Term biến mất |
|-------|-----------|---------------|
| **1. Normalization** | Softmax normalize đảm bảo $\sum_i \alpha_i = 1$ | Term $-\frac{1}{2}\|\mathbf{q}\|^2$ giống nhau cho mọi pair → cancel out |
| **2. Layer Norm** | Khi $\mathbf{k}_i$ được sinh từ layer norm → $\|\mathbf{k}_i\| \approx \text{constant}$ | Term $-\frac{1}{2}\|\mathbf{k}_i\|^2$ trở nên constant → cancel out |
| **3. Result** | Còn lại $\mathbf{q}^\top\mathbf{k}_i$ | Chỉ cần dot product! |

**Kết quả:** $a(\mathbf{q}, \mathbf{k}_i) = \mathbf{q}^\top\mathbf{k}_i$

> [!CHECKLIST]-
> Reader tự kiểm tra:
>
> - [ ] Tôi khai triển được $\|\mathbf{q} - \mathbf{k}_i\|^2$ thành ba terms
> - [ ] Tôi giải thích được tại sao hai terms cancel out trong softmax
> - [ ] Tôi hiểu kết quả: dot product thay vì squared distance

---

## 1.2 Scaled Dot Product — Tại sao cần chia $\sqrt{d}$? (D2L Eq. 11.3.2)

> [!NOTE] ELI5
>
> Dot product của hai vectors ngẫu nhiên có "độ lớn" tăng theo chiều dài. Với $d=512$, dot product có thể lên tới vài trăm. Softmax của các số lớn và chênh lệch sẽ **bão hòa** — một token chiếm gần như 100% weight, các token khác gần như bị lãng quên. Chia $\sqrt{d}$ giữ "độ lớn" = 1, softmax hoạt động cân bằng.

**Định nghĩa kỹ thuật:**

- **Đây là gì?** Scaled dot product là dot product được chia cho $\sqrt{d}$ để kiểm soát phương sai.
- **Input/Output là gì?** Input: $\mathbf{q}, \mathbf{k}_i \in \mathbb{R}^d$. Output: scalar score $\mathbf{q}^\top\mathbf{k}_i / \sqrt{d}$.
- **Giải quyết vấn đề gì?** Ngăn softmax bão hòa (saturation) khi $d$ lớn.
- **Thay thế/gợi ý giải pháp nào trước đây?** Trước đây dùng raw dot product → gradient vanishing khi $d$ lớn.

### Phân tích phương sai

Giả sử $\mathbf{q}, \mathbf{k}_i \in \mathbb{R}^d$ với elements i.i.d. $\sim \mathcal{N}(0, 1)$:

| Thống kê | Giá trị | Đánh giá |
|----------|---------|-----------|
| $\mathbb{E}[\mathbf{q}^\top\mathbf{k}_i]$ | $0$ | ✅ Zero mean |
| $\text{Var}(\mathbf{q}^\top\mathbf{k}_i)$ | $d$ | ❌ Tăng theo $d$ |

**Từ điển ký hiệu:**

| Ký hiệu | Định nghĩa |
|---------|-------------|
| $\mathbb{E}[\cdot]$ | Expected value |
| $\text{Var}(\cdot)$ | Variance |
| i.i.d. $\sim \mathcal{N}(0, 1)$ | Independent and identically distributed, standard normal |

**Vấn đề:** Khi $d$ lớn (ví dụ: $d=512$), dot product có thể có giá trị lớn $\sim O(\sqrt{d})$. Softmax của các giá trị lớn và chênh lệch sẽ bão hòa → một token chiếm ~100% weight, gradient biến mất.

### Giải pháp

**Giải pháp:** Chia cho $\sqrt{d}$:

$$\boxed{a(\mathbf{q}, \mathbf{k}_i) = \frac{\mathbf{q}^\top \mathbf{k}_i}{\sqrt{d}}}$$

Với scaling này: $\text{Var}\!\left(\frac{\mathbf{q}^\top\mathbf{k}_i}{\sqrt{d}}\right) = \frac{d}{d} = 1$. ✅

> [!CRITICAL]- Tại sao Vaswani et al. (2017) dùng $\sqrt{d_k}$?
>
> Vì với standard initialization (mean=0, var=1), dot product variance = $d$. Scaling by $\sqrt{d}$ đưa variance về 1 → softmax scores ở "vùng an toàn" (không quá sharp, không quá flat) → gradient flow tốt.
>
> Đây là lý do $\sqrt{d_k}$ xuất hiện trong **attention scaling** của Transformer — không phải magic number mà là **variance normalization**.

> [!WARNING]- Dấu hiệu nhồi nhét
>
> Nếu bạn nhớ "chia $\sqrt{d}$" mà không hiểu tại sao → bạn đang nhồi nhét. Hãy tự hỏi: nếu $d$ rất nhỏ (ví dụ: $d=2$) thì sao? Dot product variance = 2 → chia $\sqrt{2}$ → variance = 1. Nếu $d=1$ → variance = 1 → không cần chia? **Đúng!**

### Final formula — Scaled Dot Product Attention (D2L 11.3.3)

$$\alpha(\mathbf{q}, \mathbf{k}_i) = \text{softmax}\!\left(\frac{\mathbf{q}^\top \mathbf{k}_i}{\sqrt{d}}\right) = \frac{\exp(\mathbf{q}^\top \mathbf{k}_i / \sqrt{d})}{\sum_{j=1}^m \exp(\mathbf{q}^\top \mathbf{k}_j / \sqrt{d})}$$

**Từ điển ký hiệu:**

| Ký hiệu | Định nghĩa |
|---------|-------------|
| $\alpha(\mathbf{q}, \mathbf{k}_i)$ | Trọng số chú ý (attention weights) (sau softmax) |
| $\sqrt{d}$ | Scaling factor (căn bậc hai của dimension) |
| $m$ | Số lượng keys |

> [!CHECKLIST]-
> Reader tự kiểm tra:
>
> - [ ] Tôi tính được Var(dot product) = $d$ với standard initialization
> - [ ] Tôi giải thích được: chia $\sqrt{d}$ đưa variance về 1
> - [ ] Tôi hiểu: đây là variance normalization, không phải magic number

---

# PHẦN II — CONVENIENCE FUNCTIONS: MASKED SOFTMAX VÀ BMM

## 2.1 Masked Softmax — Xử lý Variable-Length Sequences (D2L 11.3.2.1)

> [!NOTE] ELI5
>
> Khi batch các câu có độ dài khác nhau, ta pad thêm tokens "trắng" vào cuối. Masked softmax đảm bảo model không "chú ý" vào những tokens trắng này — bằng cách nói với softmax "đừng quan tâm đến mấy tokens đó".

**Định nghĩa kỹ thuật:**

- **Đây là gì?** Masked softmax là softmax được điều chỉnh để bỏ qua các vị trí padding.
- **Input/Output là gì?** Input: scores matrix với valid_lens chỉ định độ dài hợp lệ. Output: trọng số chú ý (attention weights) với các vị trí padding có weight ≈ 0.
- **Giải quyết vấn đề gì?** Xử lý sequences có độ dài khác nhau trong batch mà không bị ảnh hưởng bởi padding.
- **Thay thế/gợi ý giải pháp nào trước đây?** Trước đây dùng manual masking với separate mask array.

### Ví dụ thực tế

Trong NLP, sequences có độ dài khác nhau (batch ba câu):

```
Dive  into  Deep    Learning
Learn to    code    <blank>
Hello world <blank> <blank>
```

Các tokens `<blank>` (padding) không mang ý nghĩa. Ta cần giới hạn:

$$\sum_{i=1}^n \alpha(\mathbf{q}, \mathbf{k}_i)\mathbf{v}_i \to \sum_{i=1}^l \alpha(\mathbf{q}, \mathbf{k}_i)\mathbf{v}_i \quad (l \leq n)$$

**Từ điển ký hiệu:**

| Ký hiệu | Định nghĩa |
|---------|-------------|
| $n$ | Tổng số keys (bao gồm padding) |
| $l$ | Số keys thực sự (không padding) |
| valid_lens | Tensor chỉ định độ dài hợp lệ |

### Cách implement (D2L)

```python
def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis.

    Args:
        X: 3D tensor (batch_size, n_queries, n_keys)
        valid_lens: tensor chứa độ dài hợp lệ của mỗi sequence

    Returns:
        softmax(X) với các vị trí masked có weight ≈ 0
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
3. Softmax: $\exp(-\text{1e6}) \approx 0$ → trọng số chú ý (attention weights) $\approx 0$
4. Gradient tại masked positions: $\approx 0$ → không ảnh hưởng training

> [!IMPORTANT]- Tại sao -1e6?
>
> $\exp(-10^6) \approx 0$ với floating point precision. Gradient cũng $\approx 0$. Giá trị -inf có thể gây NaN trong một số implementation, nên -1e6 là **safe approximation**.

### valid_lens formats

| Format | Ví dụ | Kích thước (shape) | Ý nghĩa |
|--------|-------|---------------------|---------|
| 1D | `[2, 3]` | `(batch_size,)` | Mỗi example có cùng valid length cho mọi query |
| 2D | `[[1, 3], [2, 4]]` | `(batch_size, n_queries)` | Mỗi query có độ dài riêng |

> [!CHECKLIST]-
> Reader tự kiểm tra:
>
> - [ ] Tôi giải thích được: -1e6 → exp → ~0 → weight ~0
> - [ ] Tôi biết: 1D valid_lens = cùng length, 2D = mỗi query khác nhau
> - [ ] Tôi hiểu: masked softmax không ảnh hưởng gradient flow

---

## 2.2 Batch Matrix Multiplication (BMM) — Tính toán Hiệu quả (D2L 11.3.2.2)

> [!NOTE] ELI5
>
> Thay vì đi từng cửa hàng một để hỏi giá (loop), BMM cho phép bạn **hỏi tất cả cửa hàng cùng lúc** trong một cuộc gọi. GPU làm điều này cực nhanh — nhanh hơn loop rất nhiều.

**Định nghĩa kỹ thuật:**

- **Đây là gì?** BMM là phép nhân ma trận trên batch of matrices cùng lúc.
- **Input/Output là gì?** Input: hai tensors 3D $\mathbf{Q} \in \mathbb{R}^{n \times a \times b}$, $\mathbf{K} \in \mathbb{R}^{n \times b \times c}$. Output: $\mathbf{Q}\mathbf{K}^\top \in \mathbb{R}^{n \times a \times c}$.
- **Giải quyết vấn đề gì?** Tăng tốc độ tính toán attention bằng cách tận dụng GPU parallelism.
- **Thay thế/gợi ý giải pháp nào trước đây?** Trước đây dùng Python loop qua từng query.

### Định nghĩa toán học

$$\mathbf{Q} = [\mathbf{Q}_1, \mathbf{Q}_2, \ldots, \mathbf{Q}_n] \in \mathbb{R}^{n \times a \times b}$$

$$\mathbf{K} = [\mathbf{K}_1, \mathbf{K}_2, \ldots, \mathbf{K}_n] \in \mathbb{R}^{n \times b \times c}$$

$$\text{BMM}(\mathbf{Q}, \mathbf{K}) = [\mathbf{Q}_1\mathbf{K}_1, \mathbf{Q}_2\mathbf{K}_2, \ldots, \mathbf{Q}_n\mathbf{K}_n] \in \mathbb{R}^{n \times a \times c}$$

**Từ điển ký hiệu:**

| Ký hiệu | Định nghĩa |
|---------|-------------|
| $n$ | Batch size |
| $a$ | Số queries |
| $b$ | Query/key dimension |
| $c$ | Value dimension |

### Ví dụ code

```python
Q = torch.ones((2, 3, 4))   # 2 matrices: each 3x4
K = torch.ones((2, 4, 6))   # 2 matrices: each 4x6
d2l.check_shape(torch.bmm(Q, K), (2, 3, 6))  # 2 matrices: each 3x6
```

### So sánh BMM vs Loop

| Method | Kích thước (shape) | Operation |
|--------|-------|-----------|
| Loop | `for i in range(n): out[i] = Q[i] @ K[i]` | $n$ matrix multiplications riêng lẻ |
| BMM | `torch.bmm(Q, K)` | 1 batch operation — GPU parallelizes |

> [!CHECKLIST]-
> Reader tự kiểm tra:
>
> - [ ] Tôi hiểu: BMM nhân nhiều ma trận cùng lúc
> - [ ] Tôi biết: kích thước (shape) input là 3D, output là 3D
> - [ ] Tôi giải thích được: tại sao BMM nhanh hơn loop

---

# PHẦN III — SCALED DOT PRODUCT ATTENTION (D2L 11.3.3)

## 3.1 Full Forward Pass — Shape Analysis

> [!NOTE] ELI5
>
> Dot product attention nhận batch of queries, keys, values → tính scores → softmax → tổng có trọng số (weighted sum) → output. Giống như bạn có đội ngũ tư vấn, hỏi ý kiến tất cả, rồi lấy trung bình có trọng số theo mức độ tin tưởng.

**Định nghĩa kỹ thuật:**

- **Đây là gì?** Scaled dot product attention là attention mechanism phổ biến nhất trong Transformer.
- **Input/Output là gì?** Input: $\mathbf{Q} \in \mathbb{R}^{n \times d}$, $\mathbf{K} \in \mathbb{R}^{m \times d}$, $\mathbf{V} \in \mathbb{R}^{m \times v}$. Output: $\mathbb{R}^{n \times v}$.
- **Giải quyết vấn đề gì?** Tính attention output một cách hiệu quả và khả vi.

**Từ điển ký hiệu:**

| Ký hiệu | Kích thước (shape) | Định nghĩa |
|---------|-------|-------------|
| $\mathbf{Q}$ | $(n, d)$ | Queries matrix |
| $\mathbf{K}$ | $(m, d)$ | Keys matrix |
| $\mathbf{V}$ | $(m, v)$ | Values matrix |
| $n$ | scalar | Số queries |
| $m$ | scalar | Số keys |
| $d$ | scalar | Query/key dimension |
| $v$ | scalar | Value dimension |

### Ba bước trong forward pass

Cho $\mathbf{Q} \in \mathbb{R}^{n \times d}$ (queries), $\mathbf{K} \in \mathbb{R}^{m \times d}$ (keys), $\mathbf{V} \in \mathbb{R}^{m \times v}$ (values):

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}\right)\mathbf{V} \in \mathbb{R}^{n \times v}$$

| Bước | Operation | Kích thước (shape) |
|-------|-----------|---------------------|
| **1** | $\mathbf{S} = \frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}$ | $(n \times m)$ — scores matrix |
| **2** | $\mathbf{A} = \text{masked\_softmax}(\mathbf{S})$ | $(n \times m)$ — trọng số chú ý (attention weights), các hàng sum = 1 |
| **3** | $\mathbf{O} = \mathbf{A}\mathbf{V}$ | $(n \times v)$ — output |

## 3.2 Implementation — Lớp DotProductAttention

```python
class DotProductAttention(nn.Module):
    """Scaled dot product attention."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        """
        Args:
            queries:  (batch_size, n_queries, d)
            keys:     (batch_size, m_keys, d)
            values:   (batch_size, m_keys, v)
            valid_lens: (batch_size,) hoặc (batch_size, n_queries)

        Returns:  (batch_size, n_queries, v)
        """
        d = queries.shape[-1]
        # BMM: Q @ K^T — scores (batch, n, m)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        # Masked softmax — trọng số chú ý (attention weights)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # BMM: trọng số chú ý (attention_weights) @ V — output (batch, n, v)
        return torch.bmm(self.dropout(self.attention_weights), values)
```

**Dropout** được dùng để regularize trọng số chú ý (attention weights) — ngẫu nhiên "tắt" một số attention connections trong training.

```python
# Ví dụ với kích thước (shapes)
queries = torch.randn(2, 1, 2)    # batch=2, 1 query mỗi batch, d=2
keys    = torch.randn(2, 10, 2)   # batch=2, 10 keys, d=2
values  = torch.randn(2, 10, 4)   # batch=2, 10 values, v=4
valid_lens = torch.tensor([2, 6]) # batch 0: 2 keys valid, batch 1: 6 keys valid

attention = DotProductAttention(dropout=0.5)
output = attention(queries, keys, values, valid_lens)
# output kích thước (shape): (2, 1, 4) — đúng như kỳ vọng
```

> [!CHECKLIST]-
> Reader tự kiểm tra:
>
> - [ ] Tôi đếm được 3 BMM operations: Q@K^T, softmax, A@V
> - [ ] Tôi biết: trọng số chú ý (attention_weights) có kích thước (shape) (batch, n_queries, m_keys)
> - [ ] Tôi giải thích được: tại sao dùng dropout trong attention

---

# PHẦN IV — ADDITIVE ATTENTION (D2L 11.3.4)

## 4.1 Khi nào cần Additive Attention?

> [!NOTE] ELI5
>
> Dot product yêu cầu query và key cùng chiều — giống như so sánh hai thanh kiếm cùng độ dài. Additive attention cho phép query và key có **chiều khác nhau** — giống như so sánh một thanh kiếm với một cây cung. Ta cần project chúng vào cùng không gian trước khi so sánh.

**Định nghĩa kỹ thuật:**

- **Đây là gì?** Additive attention là attention mechanism sử dụng MLP (multi-layer perceptron) để tính similarity score.
- **Input/Output là gì?** Input: $\mathbf{q} \in \mathbb{R}^q$, $\mathbf{k} \in \mathbb{R}^k$ (có thể $q \neq k$). Output: scalar score.
- **Giải quyết vấn đề gì?** Xử lý trường hợp query và key có chiều khác nhau.
- **Thay thế/gợi ý giải pháp nào trước đây?** Trước đây chỉ dùng dot product với $q = k$.

### Công thức

Cho $\mathbf{q} \in \mathbb{R}^q$ và $\mathbf{k} \in \mathbb{R}^k$ (có thể $q \neq k$):

$$a(\mathbf{q}, \mathbf{k}) = \mathbf{w}_v^\top \tanh(\mathbf{W}_q\mathbf{q} + \mathbf{W}_k\mathbf{k}) \in \mathbb{R}$$

**Từ điển ký hiệu:**

| Ký hiệu | Kích thước (shape) | Định nghĩa |
|---------|-------|-------------|
| $\mathbf{q}$ | $(q,)$ | Query vector |
| $\mathbf{k}$ | $(k,)$ | Key vector |
| $\mathbf{W}_q$ | $(h, q)$ | Project query vào không gian $h$ |
| $\mathbf{W}_k$ | $(h, k)$ | Project key vào không gian $h$ |
| $\mathbf{w}_v$ | $(h,)$ | Scalar projection |
| $h$ | scalar | Hidden dimension (số units trong MLP) |

### Interpretations

| Interpretation | Giải thích |
|---------------|------------|
| **MLP view** | Query và Key được concatenate → đi qua MLP 1 hidden layer với $\tanh$ activation |
| **Additive view** | Project q và k riêng vào cùng không gian $h$, cộng lại, activate |
| **Bahdanau (2014)** | Đây chính là scoring function trong Bahdanau attention cho NMT |

### Khi nào dùng?

| Scoring Function | Khi nào |
|-----------------|---------|
| **Dot Product** | $q = k$ (cùng dimension) — Transformer, self-attention |
| **Additive** | $q \neq k$ (khác dimension) — Bahdanau attention, cross-modal |

## 4.2 Implementation — Lớp AdditiveAttention

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
        Args:
            queries: (batch_size, n_queries, q)
            keys:    (batch_size, m_keys, k)
            values:  (batch_size, m_keys, v)
        """
        queries, keys = self.W_q(queries), self.W_k(keys)
        # Feature expansion: kích thước (shape) (batch, n, 1, h) và (batch, 1, m, h)
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

> [!CHECKLIST]-
> Reader tự kiểm tra:
>
> - [ ] Tôi đếm được 3 learnable matrices: $W_q, W_k, w_v$
> - [ ] Tôi giải thích được: broadcasting để tính tất cả pairs cùng lúc
> - [ ] Tôi biết: khi nào dùng additive (q ≠ k) vs dot product (q = k)

---

# PHẦN V — SO SÁNH DOT PRODUCT VS ADDITIVE

## 5.1 Architecture Comparison

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
>
> "Trong thực tế, dot product attention là mainstay của các kiến trúc Transformer hiện đại. Khi queries và keys là vectors có độ dài khác nhau, chúng ta có thể dùng additive attention scoring function thay thế."
>
> Additive attention ra đời trước (2014, Bahdanau), dùng trong early Seq2Seq+attention models. Dot product attention ra đời sau (2017, Vaswani), phổ biến trong Transformer vì:
> 1. **Không có parameters extra** — simpler
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

> [!CHECKLIST]-
> Reader tự kiểm tra:
>
> - [ ] Tôi so sánh được: dot product 0 params vs additive 3 params
> - [ ] Tôi biết: khi nào dùng cái nào (q=k vs q≠k)
> - [ ] Tôi hiểu: tại sao Transformer dùng dot product

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

    Args:
        X: 3D tensor (batch_size, n_queries, n_keys)
        valid_lens: (batch_size,) hoặc (batch_size, n_queries)

    Returns:
        softmax(X) với các vị trí padded có weight ≈ 0
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
        Args:
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

# PHẦN VII — TÓM TẮT VÀ LIÊN KẾT

## 7.1 Tóm tắt buổi

| Khái niệm | Hiểu | Cần ôn |
|-----------|------|--------|
| Gaussian sang Dot Product derivation | ✅ | |
| Tại sao chia $\sqrt{d}$ (phương sai = 1) | ✅ | |
| Masked softmax (valid_lens, -1e6 trick) | ✅ | |
| BMM — batch matrix multiplication | ✅ | |
| DotProductAttention forward pass (kích thước/shapes) | ✅ | |
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
| Masked softmax | Softmax với -1e6 cho padded positions → trọng số chú ý (attention weights) ≈ 0 |
| valid_lens | Tensor chỉ định độ dài hợp lệ của mỗi sequence trong batch |
| BMM (Batch Matrix Multiplication) | $\text{BMM}(\mathbf{Q}, \mathbf{K})$ — nhân batches of matrices element-wise |
| Additive attention | $a(\mathbf{q}, \mathbf{k}) = \mathbf{w}_v^\top \tanh(\mathbf{W}_q\mathbf{q} + \mathbf{W}_k\mathbf{k})$ |
| DotProductAttention | Attention với scaled dot product scoring, không extra parameters |
| Broadcasting in attention | $\text{unsqueeze}(q) + \text{unsqueeze}(k)$ → $(n, m, h)$ pairs cùng lúc |
| Trọng số chú ý (attention weights) | Trọng số $\alpha_i$ trong tổng có trọng số (weighted sum), quyết định mức độ "chú ý" vào mỗi value |

---

## Bài tập D2L 11.3.6

1. **Distance-based attention**: Modify `DotProductAttention` để tính distance-based scores thay vì dot product. Gợi ý: chỉ cần squared norms $\|\mathbf{k}_i\|^2$.

2. **Different dimensions**: Modify dot product attention cho phép queries và keys có chiều khác nhau. Gợi ý: dùng ma trận $\mathbf{M}$ để project giữa spaces.

3. **Complexity analysis**: Phân tích computational cost và memory bandwidth theo $d, v, n, m$. Khi nào trở thành bottleneck? Gợi ý: FlashAttention (Dao et al., 2022) giải quyết memory bottleneck.

---

## Active Recall — Câu hỏi về Buổi 52

1. **Cho $q, k_i \in \mathbb{R}^{d=64}$ với i.i.d. $\sim \mathcal{N}(0,1)$. Tính Var(dot product) với và không có scaling.**

→ **Claim**: Không scaling: $\text{Var} = 64$. Có scaling: $\text{Var} = 1$.
→ **Reasoning**: Với standard initialization, $\text{Var}(\mathbf{q}^\top\mathbf{k}_i) = d = 64$. Chia $\sqrt{d} = 8$ → $\text{Var} = 64/64 = 1$.
→ **Evidence**: Variance normalization property.

2. **Tại sao masked softmax dùng -1e6 thay vì -inf?**

→ **Claim**: -1e6 là safe approximation, tránh NaN.
→ **Reasoning**: $\exp(-\infty) = 0$ về mặt toán, nhưng -inf gây NaN trong một số implementation. -1e6 là safe approximation: $\exp(-10^6) \approx 0$ với float32.
→ **Evidence**: Implementation detail trong PyTorch.

3. **Cho AdditiveAttention với num_hiddens=8, query dim=20, key dim=2. Tính số parameters.**

→ **Claim**: Total = 184 params.
→ **Reasoning**: $W_q$: $20 \times 8 = 160$, $W_k$: $2 \times 8 = 16$, $w_v$: $8 \times 1 = 8$. Total = 184 params.
→ **Evidence**: Đếm parameters trong implementation.

4. **Tại sao dot product attention được ưu tiên hơn additive attention trong Transformer?**

→ **Claim**: Dot product simpler, faster, GPU-optimized.
→ **Reasoning**: (a) Không có parameters extra → simpler; (b) Chỉ dùng matrix multiplication → GPU-optimized; (c) Complexity $O(n \cdot m \cdot d)$ thấp hơn MLP-based attention.
→ **Evidence**: D2L Section 11.3.4.

5. **BMM vs loop: khi nào BMM nhanh hơn?**

→ **Claim**: BMM nhanh hơn khi $n$ lớn.
→ **Reasoning**: BMM nhanh hơn khi $n$ lớn (nhiều matrices trong batch) — GPU parallelizes batch operations. Với $n=1$, overhead của BMM có thể lớn hơn loop đơn giản.
→ **Evidence**: GPU architecture.

6. **Trong DotProductAttention, trọng số chú ý (attention_weights) có kích thước (shape) gì?**

→ **Claim**: $(batch, n\_queries, m\_keys)$.
→ **Reasoning**: Mỗi query có $m$ trọng số chú ý (attention weights) (một cho mỗi key). Sau softmax, mỗi hàng sum = 1.
→ **Evidence**: Forward pass implementation.

---

## TODO

- [ ] Experiment: so sánh trọng số chú ý (attention weights) với và không có $\sqrt{d}$ scaling
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
