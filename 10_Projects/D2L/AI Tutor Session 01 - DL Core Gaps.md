---
title: "AI Tutor Session 01 - DL Core Gaps"
tags: [ai-tutor, deep-learning, backpropagation, batchnorm, attention, study-note]
created: 2026-04-20
session: "AI Tutor — Buổi 1: Backpropagation + BatchNorm + Attention"
tutor_topics: [backpropagation, batch-normalization, scaled-dot-product-attention]
student_diagnosis: "Undergraduate"
tutor_progress:
  0.1_backpropagation: completed
  0.2_batchnorm: pending
  0.3_attention: pending
source:
  - Based on diagnostic test 2026-04-20
  - Obsidian vault AI Tutor skill
related:
  - "[[Batch Normalization]]"
  - "[[Self-Attention]]"
  - "[[Buổi 33 - Tuần 9]]"
---

# AI Tutor — Buổi 1: Lấp lỗ hổng DL Core

**Chẩn đoán:** Undergraduate (DL Core: 2/4)
**Mục tiêu buổi:** Lấp 3 lỗ hổng: Backprop hidden layer, BatchNorm formula, Scaled Attention

---

## 0.1 — Backpropagation: Công thức đầy đủ

> [!NOTE] ELI5
> Backpropagation giống như một "cuộc thi sửa sai" ngược từ cuối về đầu. Sau mỗi lần dự đoán sai, thay vì chỉ nói "sai rồi", ta sẽ đi ngược lại từng bước, hỏi: "Lỗi này do ai lỗi nhiều nhất?" — Layer cuối chịu trách nhiệm lớn nhất, rồi lan dần về trước. Mỗi layer tính xem mình cần "điều chỉnh" bao nhiêu để lần sau làm tốt hơn.

- **Đây là gì?** Thuật toán tính gradient của loss function theo từng weight trong mạng, bằng cách áp dụng chain rule từ output ngược về input.
- **Input/Output:** Forward pass output → Tính loss → Backward pass trả về $\frac{\partial L}{\partial W}$ cho mọi weight.
- **Tại sao cần?** Để cập nhật weights theo hướng giảm loss — không có gradient thì không có learning.

### 0.1.1 — Chain Rule cơ bản

Với network 2 lớp:

```
x → h = W₁·x + b₁ → a = ReLU(h) → y = W₂·a + b₂ → L = Loss(y, ŷ)
```

**Output layer gradient** (∂L/∂W₂):

$$\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial W_2}$$

Với MSE loss: $L = \frac{1}{2}(y - \hat{y})^2$

$$\frac{\partial L}{\partial y} = y - \hat{y}$$

→ Vậy: $\frac{\partial L}{\partial W_2} = (y - \hat{y}) \cdot a^T$

### 0.1.2 — Hidden layer gradient (phần thiếu)

**Điểm khác biệt quan trọng:** ∂L/∂W₁ phải đi qua **thêm** activation và W₂:

$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial a} \cdot \frac{\partial a}{\partial h} \cdot \frac{\partial h}{\partial W_1}$$

Viết tắt (delta notation):

$$\delta_L = \frac{\partial L}{\partial y} \quad \text{(output error)}$$

$$\delta_{h} = (W_2^T \cdot \delta_L) \odot \sigma'(h) \quad \text{(hidden error)}$$

$$\frac{\partial L}{\partial W_1} = \delta_h \cdot x^T$$

> [!WARNING] Điểm dễ nhầm
> Đạo hàm activation ReLU: $\sigma'(h) = 1$ nếu $h > 0$, $= 0$ nếu $h \leq 0$.
> Nếu neuron "chết" (dead neuron) → gradient = 0 → không học được.

### 0.1.3 — Full backward pass algorithm

```python
# Giả định: 1 sample, loss = MSE
y_hat = model(x)           # Forward pass
loss = 0.5 * (y_hat - y)**2

# Backward pass
grad_y = y_hat - y         # ∂L/∂y = (ŷ - y)

# Output layer
grad_W2 = torch.outer(a, grad_y)  # ∂L/∂W2 = a^T · δ
grad_b2 = grad_y                     # ∂L/∂b2 = δ

# Hidden layer
grad_a = W2.T @ grad_y              # δ · W2^T
grad_h = grad_a * (h > 0).float()   # ⊙ ReLU'(h)
grad_W1 = torch.outer(x, grad_h)     # ∂L/∂W1 = x^T · δ_h
grad_b1 = grad_h                     # ∂L/∂b1 = δ_h
```

### 0.1.4 — Sự khác biệt key giữa ∂L/∂W₂ và ∂L/∂W₁

| | Output layer (W₂) | Hidden layer (W₁) |
|---|---|---|
| Chain rule length | Ngắn (2 terms) | Dài (4+ terms) |
| Phụ thuộc | Chỉ cần output error | Cần "propagate" qua activation + W₂ |
| Tên gọi | Local gradient | Propagated gradient |
| Vanishing tendency | Ít | Nhiều hơn (nhân nhiều đạo hàm) |

> [!TIP] Để nhớ
> Layer càng xa output → chain rule càng dài → gradient càng có nguy cơ vanish/explode.
> Đây là lý do mạng sâu khó train → cần ReLU, residual connections, batch norm...

---

## 0.2 — Batch Normalization: Công thức + Tại sao theo batch

> [!NOTE] ELI5
> Tưởng tượng một đội bóng mà mỗi cầu thủ có thể lực rất khác nhau: người chạy nhanh quá, người lại yếu quá. Nếu huấn luyện viên cứ để họ thi đấu với nhau ngay → cả đội sẽ hỗn loạn. **Batch Normalization** giống như việc "chuẩn hóa thể lực" mỗi tuần: đo thể lực cả đội (batch), điều chỉnh về mức trung bình, rồi mới cho thi đấu. Nhờ đó, cả đội hoạt động ổn định hơn.

- **Đây là gì?** Kỹ thuật chuẩn hóa activation trong mỗi layer bằng cách đưa mean và variance về (0, 1) theo batch, rồi scale/shift bằng learnable parameters γ, β.
- **Input/Output:** Input $(N, C, H, W)$ → Output cùng shape, nhưng normalized.
- **Tại sao cần?** Giải quyết **Internal Covariate Shift** — phân phối activation thay đổi qua mỗi layer khi training.

### 0.2.1 — Công thức đầy đủ (3 bước)

**Bước 1 — Tính batch statistics:**

$$\mu_B = \frac{1}{m} \sum_{i=1}^{m} x_i, \quad \sigma^2_B = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2$$

**Bước 2 — Standardize:**

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma^2_B + \epsilon}}$$

**Bước 3 — Scale & Shift (learnable):**

$$y_i = \gamma \hat{x}_i + \beta$$

### 0.2.2 — Tại sao normalize theo batch, không phải feature?

**Normalize theo batch** = mỗi feature (channel) tính mean/variance **chung** từ tất cả samples trong batch.

**Normalize theo feature** (LayerNorm) = mỗi sample tính mean/variance riêng.

| Cách normalize | Tính mean/variance theo | Ưu điểm | Nhược điểm |
|---|---|---|---|
| **BatchNorm** | Batch (dim=0) | Dùng được batch statistics để regularize | Phụ thuộc batch size, không tốt cho online/inference |
| **LayerNorm** | Feature (dim=1) | Ổn định, không phụ thuộc batch | Không có batch regularization |

**Lý do chọn BatchNorm:**
1. **Batch statistics mang tính regularize** — mỗi batch ngẫu nhiên → đưa nhiễu vào → giống dropout
2. **Tính toán hiệu quả** — vectorized operation trên batch
3. **Transformer dùng LayerNorm** vì lý do khác: sequence length thay đổi, batch size nhỏ → batch statistics không ổn định

### 0.2.3 — Training vs Inference

| Mode | Mean/Variance | Code |
|---|---|---|
| **Training** | Batch stats $\mu_B, \sigma^2_B$ | Dùng trực tiếp |
| **Inference** | Running stats (EMA) | `model.eval()`, dùng `torch.no_grad()` |

**Running statistics update:**

$$\mu_{run} \leftarrow (1 - momentum) \cdot \mu_{run} + momentum \cdot \mu_B$$

```python
# PyTorch BatchNorm tự động cập nhật running stats
# momentum mặc định = 0.1
bn = nn.BatchNorm2d(num_features)
bn.eval()  # Inference mode

# Khi eval, BatchNorm dùng:
# - running_mean, running_var (đã update từ training)
# - NOT batch statistics
output = bn(input)  # Nếu chưa eval(), dùng batch stats → kết quả không ổn định
```

> [!WARNING] Lỗi phổ biến
> Quên `model.eval()` → inference dùng batch stats → kết quả thay đổi mỗi lần chạy.

### 0.2.4 — PyTorch Implementation

```python
class BatchNorm2d(nn.Module):
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))  # γ
        self.beta = nn.Parameter(torch.zeros(num_features))  # β
        self.momentum = momentum
        self.eps = eps
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        if self.training:
            # Bước 1: Tính batch statistics (theo H, W)
            mu = x.mean(dim=(0, 2, 3), keepdim=True)
            var = x.var(dim=(0, 2, 3), keepdim=True)
            # Bước 2: Standardize
            x_norm = (x - mu) / torch.sqrt(var + self.eps)
            # Cập nhật running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu.squeeze()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
        else:
            # Inference: dùng running stats
            mu = self.running_mean.view(1, -1, 1, 1)
            var = self.running_var.view(1, -1, 1, 1)
            x_norm = (x - mu) / torch.sqrt(var + self.eps)
        # Bước 3: Scale & Shift
        return self.gamma.view(1, -1, 1, 1) * x_norm + self.beta.view(1, -1, 1, 1)
```

---

## 0.3 — Scaled Dot-Product Attention

> [!NOTE] ELI5
> Attention giống như khi bạn đọc một câu hỏi (Query), tìm từng từ trong bài có liên quan (Key), rồi lấy nội dung của từ đó (Value) để trả lời. Nhưng nếu từ đó quá "phổ biến" (điểm tương đồng cao với mọi từ khác), bạn cần chia cho căn bậc 2 độ dài vector để "công bằng" hơn — không để từ nào chiếm ưu thế quá.

- **Đây là gì?** Cơ chế tính attention weights bằng dot product giữa Query và Key, chia cho √d_k để scale, rồi dùng softmax để được phân phối xác suất, cuối cùng nhân với Value.
- **Input:** Q, K, V (cùng shape $(N, d_k)$, $(N, d_k)$, $(N, d_v)$)
- **Output:** Attention output $(N, d_v)$
- **Tại sao cần?** Cho phép mô hình "chú ý" đến các phần khác nhau của input một cách có trọng số — không phải tất cả đều quan trọng như nhau.

### 0.3.1 — Công thức đầy đủ

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**Giải thích từng bước:**

1. **$QK^T$** — Tính similarity giữa mỗi query và mỗi key. Kết quả: ma trận $(N, N)$ với $A_{ij}$ = "query $i$ quan tâm đến key $j$ bao nhiêu?"

2. **$\div \sqrt{d_k}$** — Scale để tránh gradient vanish (chi tiết bên dưới)

3. **softmax** — Chuyển thành phân phối xác suất (tổng = 1 theo hàng)

4. **$\times V$** — Nhân attention weights với values để lấy weighted sum

### 0.3.2 — Tại sao chia cho √d_k?

**Vấn đề:** Khi $d_k$ lớn, dot product $q \cdot k = \sum_i q_i k_i$ có thể có giá trị rất lớn (magnitude tăng theo $\sqrt{d_k}$).

**Hệ quả:** Khi input vào softmax quá lớn → softmax trở nên "sharp" (nearly one-hot) → gradient rất nhỏ → **vanishing gradient**.

**Giải pháp:** Chia cho $\sqrt{d_k}$ để variance của dot product giữ ổn định:

$$\text{Var}(q \cdot k) = \text{Var}\left(\sum_i q_i k_i\right) = d_k \cdot \text{Var}(q_i) \cdot \text{Var}(k_i)$$

Nếu $q_i, k_i$ có variance = 1 → $\text{Var}(q \cdot k) = d_k$

→ Chia cho $\sqrt{d_k}$ → $\text{Var}(q \cdot k / \sqrt{d_k}) = 1$ ✓

> [!IMPORTANT] Ý nghĩa
> $\sqrt{d_k}$ giữ cho softmax input ở **quy mô phù hợp** → gradient ổn định → training hiệu quả.

### 0.3.3 — PyTorch Implementation

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Args:
        Q: (batch, n_heads, seq_len, d_k)
        K: (batch, n_heads, seq_len, d_k)
        V: (batch, n_heads, seq_len, d_v)
    Returns:
        output: (batch, n_heads, seq_len, d_v)
        attention_weights: (batch, n_heads, seq_len, seq_len)
    """
    d_k = Q.size(-1)

    # Bước 1: Tính dot product QK^T
    scores = torch.matmul(Q, K.transpose(-2, -1))  # (B, H, L, L)

    # Bước 2: Scale
    scores = scores / math.sqrt(d_k)

    # Bước 3: Mask (optional - cho decoder attention)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Bước 4: Softmax
    attention_weights = F.softmax(scores, dim=-1)

    # Bước 5: Nhân với V
    output = torch.matmul(attention_weights, V)  # (B, H, L, d_v)

    return output, attention_weights
```

### 0.3.4 — Tại sao dùng Q, K, V thay vì chỉ dùng X?

**Thiết kế 3 vectors (Q, K, V) cho phép:**
1. **Query** — "tôi đang tìm gì" (từ vị trí hiện tại)
2. **Key** — "tôi có gì để match" (biểu diễn của mỗi vị trí)
3. **Value** — "nội dung thực sự cần lấy" (có thể khác Key)

**Ví dụ:** Khi hỏi "Con mèo ngồi trên ghế":
- Query "Con mèo" có thể match với Key "Con mèo" (cao) và "Ghế" (thấp)
- Nhưng Value của "Ghế" có thể chứa thông tin về "màu sắc" — không liên quan
- → Q,K,V cho phép tách biệt "matching" và "content retrieval"

> [!TIP] Để nhớ
> Q = "tôi muốn tìm gì"
> K = "tôi có gì để so sánh"
> V = "tôi thực sự muốn lấy thông tin gì"

---

## Tóm tắt Buổi 1

### Key Takeaways

1. **Backpropagation:** Chain rule nhân dồi gradient từ output → input. Hidden layer gradient dài hơn output layer → nguy cơ vanish cao hơn.

2. **BatchNorm:** 3 bước (standardize → scale γ → shift β). Training dùng batch stats, inference dùng running stats (EMA). Normalize theo batch vì batch statistics có tác dụng regularize.

3. **Scaled Attention:** $\text{softmax}(QK^T/\sqrt{d_k})V$. Chia $\sqrt{d_k}$ để giữ variance ổn định, tránh softmax "sharp" → vanishing gradient.

### Liên hệ

- Backpropagation → kết nối với [[Loss Functions]] (MSE gradient = y - ŷ)
- BatchNorm → pre-requisite cho [[Buổi 33 - Tuần 9]] và [[Buổi 34 - Tuần 9]] (ResNet)
- Attention → nền tảng của [[Self-Attention]] và Transformer Architecture

---

## Tiếp theo

| Bước | Lệnh | Nội dung |
|------|-------|----------|
| 1 | `/tutor-test` | Test Backpropagation |
| 2 | `/tutor-start` 2 | BatchNorm (buổi 2) |
| 3 | `/tutor-start` 3 | Scaled Attention (buổi 3) |

> [!NOTE] Gợi ý
> - Muốn hiểu sâu hơn về BatchNorm vs LayerNorm? → Hỏi về normalization comparison
> - Muốn biết attention được dùng trong LLM như thế nào? → Hỏi về Transformer
> - Muốn thực hành code? → `/tutor-code` Backpropagation

---

**Tutor Session:** 01
**Ngày:** 2026-04-20
**Kết thúc buổi:** 3 bài học hoàn thành
