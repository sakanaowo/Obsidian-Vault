---
session: "D2L Tuần 14, Buổi 54 — 11.5 Multi-Head Attention"
aliases: ["Buổi 54"]
tags: [d2l, deep-learning, attention, multi-head, transformer, nlp]
status: growth
source: "D2L Chapter 11.5 — Multi-Head Attention"
created: 2026-05-02
related:
  - "[[Buổi 52 - Tuần 14]]"
  - "[[Buổi 53 - Tuần 14]]"
  - "[[Buổi 55 - Tuần 14]]"
---

# Buổi 54 — 11.5 Multi-Head Attention

> [!NOTE] Mục tiêu buổi học
>
> - [ ] Hiểu tại sao cần nhiều heads — học các "quantities" khác nhau
> - [ ] Nắm công thức multi-head attention (D2L Eq. 11.5.1, 11.5.2)
> - [ ] Hiểu cách implement với transpose_qkv và transpose_output
> - [ ] Phân biệt được với single-head attention và Bahdanau attention
> - [ ] Implement MultiHeadAttention từ scratch

---

## Bảng thuật ngữ — ĐỌC TRƯỚC KHI TIẾP TỤC

| Thuật ngữ | Tiếng Việt | Giải thích bằng tiếng Việt |
|-----------|------------|------------------------------|
| Multi-head attention | Chú ý đa đầu | Chạy nhiều attention heads song song, mỗi head học một "khía cạnh" khác nhau |
| Head | Đầu chú ý | Một attention mechanism độc lập, học một projection riêng |
| $h$ | Số heads | Thường = 8 trong Transformer (d2l use 4) |
| $p_q = p_k = p_v$ | Chiều mỗi head | Thường = $d_{model} / h$ |
| $W_i^{(q)}, W_i^{(k)}, W_i^{(v)}$ | Ma trận projection của head $i$ | Học cách biến đổi Q, K, V riêng cho mỗi head |
| $W_o$ | Ma trận output | Trộn outputs từ $h$ heads thành output cuối cùng |
| Concatenation | Nối ghép | Ghép $h$ vectors có chiều $p_v$ thành vector có chiều $h \cdot p_v$ |

---

## Active Recall — Ôn lại Buổi 53

### Câu hỏi truy hồi (không nhìn tài liệu)

1. Trong Bahdanau attention, Query, Key, Value đến từ đâu?
2. Bahdanau attention giải quyết vấn đề gì của Seq2Seq cổ điển?
3. Công thức attention scores trong Bahdanau là gì? Tại sao dùng MLP thay vì dot product?
4. Context vector $\mathbf{c}_t$ được tính như thế nào?
5. Sự khác biệt giữa encoder-decoder attention và self-attention là gì?

### Trả lời chi tiết

---

**Câu 1: Trong Bahdanau attention, Query, Key, Value đến từ đâu?**

**Đáp án:**

| Thành phần | Đến từ                                  | Giải thích                                       |
| ---------- | --------------------------------------- | ------------------------------------------------ |
| **Query**  | Decoder hidden state $\mathbf{s}_{t-1}$ | Decoder hỏi: "Từ nào trong câu nguồn liên quan?" |
| **Key**    | Encoder hidden states $\mathbf{h}_i$    | Encoder trả lời: "Tôi ở vị trí nào"              |
| **Value**  | Encoder hidden states $\mathbf{h}_i$    | Encoder cung cấp: "Nội dung thực sự tôi mang"    |

---

**Câu 2: Bahdanau attention giải quyết vấn đề gì của Seq2Seq cổ điển?**

**Vấn đề của Seq2Seq cổ điển:**
- Encoder nén **toàn bộ** câu nguồn vào **một** context vector $\mathbf{c}$
- Context vector có kích thước cố định (ví dụ: 256 chiều)
- Câu nguồn có thể dài 50+ từ
- **Nút thắt cổ chai (bottleneck)**: Thông tin bị mất khi nén

**Giải pháp của Bahdanau:**
- Thay vì một context vector duy nhất, mỗi bước decode có **context vector riêng**
- Decoder được "nhìn" vào **tất cả** encoder hidden states
- Mỗi từ đích có context vector phù hợp với nhu cầu của nó

**Ví dụ:**
```
Câu nguồn: "The cat sat on the mat"
Câu đích: "Con mèo ngồi trên thảm"

Khi sinh từ "mèo":
  - Context vector chú ý nhiều vào "cat" (chủ ngữ)
  
Khi sinh từ "ngồi":
  - Context vector chú ý nhiều vào "sat" (vị ngữ)
```

---

**Câu 3: Công thức attention scores trong Bahdanau là gì? Tại sao dùng MLP thay vì dot product?**

**Công thức:**
$$a(\mathbf{s}_{t-1}, \mathbf{h}_i) = \mathbf{v}^\top \tanh(\mathbf{W}_s \mathbf{s}_{t-1} + \mathbf{W}_h \mathbf{h}_i)$$

**Giải thích từng phần:**

| Phần | Giải thích |
|------|------------|
| $\mathbf{s}_{t-1}$ | Decoder hidden state (Query) |
| $\mathbf{h}_i$ | Encoder hidden state (Key) |
| $\mathbf{W}_s$ | Ma trận biến đổi decoder state |
| $\mathbf{W}_h$ | Ma trận biến đổi encoder state |
| $\tanh$ | Activation, giữ giá trị trong [-1, 1] |
| $\mathbf{v}$ | Vector tổng hợp thành 1 số |

**Tại sao dùng MLP thay vì dot product?**

| Lý do               | Giải thích                                               |
| ------------------- | -------------------------------------------------------- |
| **Chiều khác nhau** | Decoder và encoder có thể có hidden dimensions khác nhau |
| **Tính linh hoạt**  | MLP có thể học mối quan hệ phức tạp hơn                  |
| **Non-linearity**   | $\tanh$ cho phép mô hình quan hệ phi tuyến tính          |

---

**Câu 4: Context vector $\mathbf{c}_t$ được tính như thế nào?**

**3 bước để tính context vector:**

```
Bước 1: Tính attention scores
  a_i = v^T × tanh(W_s × s_{t-1} + W_h × h_i)

Bước 2: Softmax thành attention weights
  α_i = softmax(a_i) = exp(a_i) / Σ exp(a_j)

Bước 3: Weighted sum
  c_t = Σ α_i × h_i
```

**Ví dụ cụ thể:**
```
Giả sử 3 attention scores: [8.0, 0.5, 3.0]

Sau softmax (với temperature = 1):
  α = [0.95, 0.01, 0.04]
  
  → 95% weight cho position 1
  → 4% weight cho position 3
  → 1% weight cho position 2

Context vector:
  c_t = 0.95 × h_1 + 0.01 × h_2 + 0.04 × h_3
       ≈ h_1 (lấy thông tin chủ yếu từ position 1)
```

---

**Câu 5: Sự khác biệt giữa encoder-decoder attention và self-attention là gì?**

| Khía cạnh | Encoder-Decoder (Bahdanau) | Self-Attention |
|-----------|---------------------------|---------------|
| **Query đến từ** | Decoder | Cùng sequence |
| **Key/Value đến từ** | Encoder | Cùng sequence |
| **Mục đích** | Align input với output | Capture dependencies trong cùng sequence |
| **Ví dụ** | "Từ nào trong câu nguồn?" | "Từ nào trong câu này?" |
| **Dùng trong** | Seq2Seq + Attention | Transformer, BERT |

---

### Liên kết cần ôn lại

- [[Buổi 53 - Tuần 14|Bahdanau Attention]]
- [[Buổi 52 - Tuần 14|Attention Scoring Functions]]

---

# PHẦN I — TẠI SAO CẦN NHIỀU HEADS?

## 1.1 Hạn chế của single-head attention

### Bảng thuật ngữ phần này

| Thuật ngữ | Giải thích |
|-----------|------------|
| Single-head | Chỉ một attention head duy nhất |
| Limitation | Hạn chế, giới hạn |
| Capture | Nắm bắt, thu thập |

### Vấn đề với single-head attention

**Giả sử bạn chỉ có một đầu chú ý:**

```
Bạn đang dịch: "The bank of the river"

Single-head attention chỉ có thể học MỘT loại dependency:
  → "bank" phụ thuộc vào "river" (nghĩa: bờ sông)
  HOẶC
  → "bank" phụ thuộc vào "money" (nghĩa: ngân hàng)
  NHƯNG KHÔNG THỂ cùng lúc!

Một head không thể nắm bắt cả hai loại thông tin khác nhau.
```

### Tại sao một head không đủ?

**Phân tích:**
- Mỗi head có một bộ ma trận projection $W_Q, W_K, W_V$ riêng
- Các heads khác nhau học các **representation subspaces** khác nhau
- Một head có thể tập trung vào **syntax** (ngữ pháp)
- Head khác tập trung vào **semantics** (nghĩa)
- Head khác nữa tập trung vào **coreference** (đại từ tham chiếu)

---

## 1.2 Giải pháp: Multi-head attention

### Bảng thuật ngữ phần này

| Thuật ngữ | Giải thích |
|-----------|------------|
| Multi-head | Nhiều heads |
| Parallel | Song song, cùng lúc |
| Subspace | Không gian con (một phần của không gian lớn) |

### Giải thích bằng ví dụ cụ thể

```
┌─────────────────────────────────────────────────────────────┐
│                 MULTI-HEAD ATTENTION                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  INPUT: Câu "The bank of the river"                         │
│                                                             │
│  ┌─────────┐                                                │
│  │  HEAD 1 │ → Học: "bank" ↔ "river" (nghĩa: bờ sông)       │
│  │ Syntax  │    Weight cao: bank → river                    │
│  └────┬────┘                                                │
│       │                                                     │
│  ┌────┴────┐                                                │
│  │  HEAD 2 │ → Học: "bank" ↔ "money" (nghĩa: ngân hàng)     │
│  │ Semantic│    Weight cao: bank → financial                │
│  └────┬────┘                                                │
│       │                                                     │
│  ┌────┴────┐                                                │
│  │  HEAD 3 │ → Học: "the" ↔ "bank" (mạo từ)                 │
│  │ Articles │    Weight cao: the → bank                     │
│  └────┬────┘                                                │
│       │                                                     │
│  ┌────┴────┐                                                │
│  │  HEAD 4 │ → Học: position dependencies                   │
│  │ Position│    Weight cao: nearby tokens                   │
│  └────┬────┘                                                │
│       │                                                     │
│       ↓                                                     │
│  ┌─────────────────────────────────┐                        │
│  │      CONCATENATION + LINEAR     │                        │
│  │  [h1; h2; h3; h4] @ W_o         │                        │
│  └─────────────────────────────────┘                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Ý tưởng chính:**
- Mỗi head học một "khía cạnh" khác nhau
- Cuối cùng, concatenate tất cả heads và biến đổi tuyến tính
- Model tự quyết định head nào quan trọng cho task nào

---

# PHẦN II — CÔNG THỨC TOÁN HỌC

## 2.1 Công thức từng Head (D2L Eq. 11.5.1)

### Bảng ký hiệu — ĐỌC TRƯỚC KHI XEM CÔNG THỨC

| Ký hiệu         | Kích thước         | Giải thích bằng tiếng Việt                             |
| --------------- | ------------------ | ------------------------------------------------------ |
| $\mathbf{q}$    | $(d_q,)$           | Query vector                                           |
| $\mathbf{k}$    | $(d_k,)$           | Key vector                                             |
| $\mathbf{v}$    | $(d_v,)$           | Value vector                                           |
| $h$             | scalar             | Số heads                                               |
| $p_q, p_k, p_v$ | scalar             | Chiều của mỗi head                                     |
| $W_i^{(q)}$     | $(p_q \times d_q)$ | Ma trận projection cho Query của head $i$              |
| $W_i^{(k)}$     | $(p_k \times d_k)$ | Ma trận projection cho Key của head $i$                |
| $W_i^{(v)}$     | $(p_v \times d_v)$ | Ma trận projection cho Value của head $i$              |
| $f$             | -                  | Attention pooling function (ví dụ: scaled dot product) |

### Công thức cho một head

$$\mathbf{h}_i = f\left(\mathbf{W}_i^{(q)}\mathbf{q}, \mathbf{W}_i^{(k)}\mathbf{k}, \mathbf{W}_i^{(v)}\mathbf{v}\right) \in \mathbb{R}^{p_v}$$

**Giải thích từng bước:**
Bước 1: Project query, key, value vào không gian của head i
  $q' = W_i^{(q)} × q$      $(d_q → p_q)$
  $k' = W_i^{(k)} × k$      $(d_k → p_k)$
  $v' = W_i^{(v)} × v$      $(d_v → p_v)$

Bước 2: Tính attention với các vectors đã project
  $h_i = f(q', k', v')$

Bước 3: Kết quả có chiều $p_v$


---

## 2.2 Công thức Output cuối cùng (D2L Eq. 11.5.2)

### Bảng ký hiệu

| Ký hiệu | Kích thước | Giải thích |
|---------|-------------|-------------|
| $\mathbf{W}_o$ | $(p_o \times h p_v)$ | Ma trận output |
| $[\mathbf{h}_1; \mathbf{h}_2; \ldots; \mathbf{h}_h]$ | $(h p_v,)$ | Concatenation của $h$ heads |
| $\mathbf{o}$ | $(p_o,)$ | Output cuối cùng |

### Công thức

$$\mathbf{o} = \mathbf{W}_o \begin{bmatrix} \mathbf{h}_1 \\ \vdots \\ \mathbf{h}_h \end{bmatrix} \in \mathbb{R}^{p_{o}} $$

**Giải thích:**
Input: h vectors, mỗi vector chiều p_v
       $[h_1; h_2; ...; h_h]$ có chiều $h × p_v$

Output: Nhân với $W_o$
        $o = W_o × [h_1; ...; h_h]$
        o có chiều $p_o$

Thông thường: $p_q = p_k = p_v = p_o / h = d_{model} / h$

---

## 2.3 Tại sao chia chiều cho h?

### Bảng thuật ngữ

| Thuật ngữ | Giải thích |
|-----------|------------|
| Computational cost | Chi phí tính toán |
| Parameterization | Số lượng tham số |

### Phân tích chi phí

**So sánh: Multi-head vs Single-head**

| Khía cạnh | Single-head ($d$) | Multi-head ($h$ heads, chiều $d/h$) |
|------------|-------------------|-------------------------------------|
| **Chiều mỗi head** | $d$ | $d/h$ |
| **Tổng chiều** | $d$ | $h \times (d/h) = d$ |
| **Ma trận Q, K, V** | $3 \times d \times d$ | $h \times 3 \times (d/h) \times (d/h) = 3 \times d^2 / h$ |
| **Output projection** | $d \times d$ | $d \times d$ |

**Kết luận:**
- Multi-head giữ nguyên **tổng chiều** output
- Nhưng giảm chiều **mỗi head** xuống $d/h$
- Điều này giúp **giảm computation** mà vẫn có nhiều representation subspaces

---

# PHẦN III — IMPLEMENTATION

## 3.1 Bảng thuật ngữ cho code

| Tên trong code | Giải thích |
|----------------|------------|
| `num_hiddens` | Chiều output cuối cùng $p_o$ |
| `num_heads` | Số heads $h$ |
| `transpose_qkv` | Transpose để tính attention song song |
| `transpose_output` | Reverse transpose_qkv |
| `DotProductAttention` | Attention layer để dùng trong mỗi head |

## 3.2 Code với comment giải thích từng dòng

```python
class MultiHeadAttention(nn.Module):
    """Multi-head attention: chạy nhiều attention heads song song.
    
    Mỗi head học một representation subspace khác nhau.
    Cuối cùng, concatenate tất cả heads và biến đổi tuyến tính.
    
    Args:
        num_hiddens: Chiều output cuối cùng (thường = d_model)
        num_heads: Số heads (thường = 8)
        dropout: Tỷ lệ dropout
        bias: Có dùng bias trong linear layers không
    """
    def __init__(self, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        
        # ============================================================
        # DotProductAttention: attention mechanism cho mỗi head
        # ============================================================
        # Mỗi head sử dụng scaled dot-product attention
        self.attention = d2l.DotProductAttention(dropout)
        
        # ============================================================
        # 4 Linear layers để project Q, K, V và output
        # ============================================================
        
        # W_q: biến đổi queries
        # Input: (batch, seq_len, d_model)
        # Output: (batch, seq_len, num_hiddens)
        self.W_q = nn.LazyLinear(num_hiddens, bias=bias)
        
        # W_k: biến đổi keys
        self.W_k = nn.LazyLinear(num_hiddens, bias=bias)
        
        # W_v: biến đổi values
        self.W_v = nn.LazyLinear(num_hiddens, bias=bias)
        
        # W_o: biến đổi output cuối cùng (sau khi concatenate heads)
        self.W_o = nn.LazyLinear(num_hiddens, bias=bias)
    
    def forward(self, queries, keys, values, valid_lens):
        """Forward pass của multi-head attention.
        
        Args:
            queries: (batch_size, n_queries, d_model)
            keys: (batch_size, m_keys, d_model)
            values: (batch_size, m_keys, d_model)
            valid_lens: (batch_size,) hoặc (batch_size, n_queries)
        
        Returns:
            Output: (batch_size, n_queries, num_hiddens)
        """
        # ============================================================
        # Bước 1: Project Q, K, V
        # ============================================================
        
        # queries: (batch, n, d_model) → (batch, n, num_hiddens)
        queries = self.transpose_qkv(self.W_q(queries))
        
        # keys: (batch, m, d_model) → (batch, m, num_hiddens)
        keys = self.transpose_qkv(self.W_k(keys))
        
        # values: (batch, m, d_model) → (batch, m, num_hiddens)
        values = self.transpose_qkv(self.W_v(values))
        
        # ============================================================
        # Bước 2: Xử lý valid_lens cho nhiều heads
        # ============================================================
        # valid_lens ban đầu: (batch,) 
        # Cần repeat cho mỗi head
        
        if valid_lens is not None:
            # valid_lens = [2, 3], num_heads = 4
            # Sau repeat: [2, 2, 2, 2, 3, 3, 3, 3]
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)
        
        # ============================================================
        # Bước 3: Tính attention
        # ============================================================
        # Output shape: (batch * num_heads, n_queries, num_hiddens / num_heads)
        output = self.attention(queries, keys, values, valid_lens)
        
        # ============================================================
        # Bước 4: Reverse transpose và concatenate heads
        # ============================================================
        # output_concat: (batch, n_queries, num_hiddens)
        output_concat = self.transpose_output(output)
        
        # ============================================================
        # Bước 5: Final linear transformation
        # ============================================================
        # W_o: (batch, n_queries, num_hiddens) → (batch, n_queries, num_hiddens)
        return self.W_o(output_concat)
```

---

## 3.3 Hàm Transpose — Chi tiết

### transpose_qkv

```python
def transpose_qkv(self, X):
    """Transpose để tính attention song song cho nhiều heads.
    
    Shape transformation:
    Input:  (batch_size, seq_len, num_hiddens)
           ↓ reshape: (batch_size, seq_len, num_heads, num_hiddens/num_heads)
           ↓ permute: (batch_size, num_heads, seq_len, num_hiddens/num_heads)
           ↓ reshape: (batch_size * num_heads, seq_len, num_hiddens/num_heads)
    Output: (batch_size * num_heads, seq_len, num_hiddens/num_heads)
    
    Tại sao cần reshape và permute?
    - Muốn tính attention cho từng head độc lập
    - Nhưng vẫn dùng batched matrix multiplication để tính nhanh
    """
    # X.shape[0] = batch_size
    # X.shape[1] = seq_len
    # X.shape[2] = num_hiddens
    
    # Thêm chiều num_heads
    # Shape: (batch, seq_len, num_hiddens) → (batch, seq_len, num_heads, num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
    
    # Permute để đưa num_heads lên trước
    # Shape: (batch, seq_len, num_heads, num_hiddens/num_heads) 
    #       → (batch, num_heads, seq_len, num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)
    
    # Merge batch và num_heads
    # Shape: (batch, num_heads, seq_len, num_hiddens/num_heads)
    #       → (batch * num_heads, seq_len, num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])
```

### transpose_output

```python
def transpose_output(self, X):
    """Reverse của transpose_qkv.
    
    Shape transformation:
    Input:  (batch_size * num_heads, seq_len, num_hiddens/num_heads)
           ↓ reshape: (batch, num_heads, seq_len, num_hiddens/num_heads)
           ↓ permute: (batch, seq_len, num_heads, num_hiddens/num_heads)
           ↓ reshape: (batch, seq_len, num_hiddens)
    Output: (batch_size, seq_len, num_hiddens)
    """
    # Tách batch và num_heads
    # Shape: (batch*heads, seq, dim) → (batch, heads, seq, dim)
    X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
    
    # Đưa num_heads về sau
    # Shape: (batch, heads, seq, dim) → (batch, seq, heads, dim)
    X = X.permute(0, 2, 1, 3)
    
    # Merge lại thành num_hiddens
    # Shape: (batch, seq, heads, dim) → (batch, seq, heads*dim)
    return X.reshape(X.shape[0], X.shape[1], -1)
```

---

## 3.4 Ví dụ với shapes cụ thể

```python
# Ví dụ
num_hiddens = 100  # d_model
num_heads = 5
batch_size = 2
num_queries = 4
num_kvpairs = 6

# Tạo attention layer
attention = MultiHeadAttention(num_hiddens, num_heads, dropout=0.5)

# Input
X = torch.ones((batch_size, num_queries, num_hiddens))
Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
valid_lens = torch.tensor([3, 2])

# Forward pass
output = attention(X, Y, Y, valid_lens)

# Kiểm tra shape
print(output.shape)  # torch.Size([2, 4, 100])
# Batch 2, 4 queries, mỗi query có 100 chiều

# Giải thích:
# - Batch 2: 2 câu trong batch
# - 4 queries: 4 vị trí cần tính attention
# - 100 chiều: num_hiddens = 100
```

---

# PHẦN IV — SO SÁNH

## 4.1 Bảng so sánh các loại Attention

| Khía cạnh | Single-head | Multi-head ($h$ heads) | Bahdanau |
|-----------|-------------|----------------------|----------|
| **Số heads** | 1 | $h$ | 1 |
| **Chiều mỗi head** | $d$ | $d/h$ | Khác nhau có thể |
| **Tổng chiều** | $d$ | $d$ | Decoder/Encoder dims |
| **Ma trận Q, K, V** | 1 bộ | $h$ bộ | Additive MLP |
| **Output** | Vector $d$ | Concatenate → $d$ | Vector $d$ |
| **Captures** | 1 loại dependency | $h$ loại dependencies | 1 loại alignment |
| **Dùng trong** | - | Transformer | Seq2Seq |

## 4.2 Multi-head vs Bahdanau attention

| Khía cạnh | Multi-head | Bahdanau |
|-----------|-----------|----------|
| **Query đến từ** | Decoder hoặc cùng sequence | Decoder hidden state |
| **Key/Value đến từ** | Encoder hoặc cùng sequence | Encoder hidden states |
| **Scoring function** | Scaled dot product | Additive MLP |
| **Tính song song** | Tất cả heads cùng lúc | Tuần tự |
| **Năm ra đời** | 2017 (Transformer) | 2014 |
| **Học được** | Q, K, V projections | W_s, W_h, v |

---

# PHẦN V — TÓM TẮT

## 5.1 Ghi nhớ chính

```
┌─────────────────────────────────────────────────────────────┐
│              MULTI-HEAD ATTENTION — TÓM TẮT                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  VẤN ĐỀ:                                                   │
│  - Single-head chỉ học được MỘT loại dependency              │
│  - Cần nhiều heads để capture các khía cạnh khác nhau       │
│                                                             │
│  GIẢI PHÁP:                                                │
│  - Chạy h attention heads SONG SONG                         │
│  - Mỗi head có W_Q, W_K, W_V RIÊNG                        │
│  - Concatenate outputs → Final linear                       │
│                                                             │
│  CÔNG THỨC:                                                │
│  h_i = f(W_i^{(q)}q, W_i^{(k)}k, W_i^{(v)}v)            │
│  o = W_o [h_1; h_2; ...; h_h]                            │
│                                                             │
│  IMPLEMENTATION:                                           │
│  - transpose_qkv: reshape để tính song song                 │
│  - transpose_output: reverse reshape                        │
│  - valid_lens: repeat cho mỗi head                       │
│                                                             │
│  TẠI SAO CHIA CHIỀU?                                      │
│  - Giữ tổng chiều = d                                      │
│  - Giảm computation = d²/h thay vì d²                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 5.2 Bảng thuật ngữ cuối buổi

| Thuật ngữ | Tiếng Việt | Ghi nhớ |
|-----------|------------|---------|
| Multi-head attention | Chú ý đa đầu | $h$ heads song song |
| Head | Đầu chú ý | Mỗi head = 1 attention mechanism |
| Subspace | Không gian con | Mỗi head học 1 subspace |
| transpose_qkv | Transpose cho QKV | Reshape để tính song song |
| Concatenation | Nối ghép | Ghép $h$ vectors thành 1 |

---

## Bài tập

1. **Visualize attention weights**: Vẽ attention weights của nhiều heads cho cùng một câu. Nhận xét sự khác biệt giữa các heads.

2. **Pruning heads**: Giả sử muốn cắt bớt heads không quan trọng để tăng tốc. Làm sao đo lường "tầm quan trọng" của mỗi head?

3. **Different d_model/h**: Thay đổi số heads và chiều mỗi head. Quan sát ảnh hưởng đến kết quả.

---

## TODO

- [ ] Visualize attention weights của nhiều heads
- [ ] Experiment với số heads khác nhau
- [ ] Đọc paper Vaswani (2017) "Attention Is All You Need"

---

## Liên kết

- [[Buổi 52 - Tuần 14|Attention Scoring Functions]]
- [[Buổi 53 - Tuần 14|Bahdanau Attention]]
- [[Buổi 55 - Tuần 14|Self-Attention & Positional Encoding]]
- [[Transformer Architecture]] *(concept note đã có)*
