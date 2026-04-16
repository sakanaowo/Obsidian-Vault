---
tags:
  - concept
  - deep-learning
  - optimization
  - rnn
  - training
aliases:
  - Gradient Norm Clipping
  - Grad Clip
date: 2026-04-15
status: seedling
---

# Gradient Clipping

> [!NOTE] ELI5
> Bạn đang đổ dốc trượt tuyết. Bình thường bạn kiểm soát được tốc độ. Nhưng gặp chỗ cực dốc, bạn lao quá nhanh và sẽ bay ra khỏi đường. **Gradient Clipping** là giới hạn tốc độ tối đa — nếu tốc độ (gradient) vượt ngưỡng, tự động giảm xuống mức an toàn, nhưng **giữ nguyên hướng di chuyển**.

**Gradient Clipping** là kỹ thuật **ổn định training** bằng cách giới hạn norm của gradient vector trước khi cập nhật trọng số. Khi gradient norm vượt ngưỡng $\theta$, gradient bị thu nhỏ tỉ lệ sao cho norm đúng bằng $\theta$, giữ nguyên hướng. Kỹ thuật này **bắt buộc** khi training RNN do hiện tượng exploding gradient từ chuỗi nhân $W_{hh}$ qua nhiều time steps.

## Công thức

$$\mathbf{g} \leftarrow \min\!\left(1, \frac{\theta}{\|\mathbf{g}\|}\right) \cdot \mathbf{g}$$

- $\mathbf{g}$: gradient vector (concatenate gradient tất cả parameters)
- $\|\mathbf{g}\| = \sqrt{\sum_i g_i^2}$: L2 norm toàn cục
- $\theta$: ngưỡng clipping (hyperparameter), thường $\theta = 1$ hoặc $\theta = 5$

**Logic:**

- $\|\mathbf{g}\| \leq \theta$ → giữ nguyên (hệ số nhân = 1)
- $\|\mathbf{g}\| > \theta$ → thu nhỏ: $\mathbf{g}_{\text{new}} = \theta \cdot \frac{\mathbf{g}}{\|\mathbf{g}\|}$ (norm = $\theta$, hướng giữ nguyên)

## Tại sao cần — Exploding Gradient trong RNN

Khi backprop qua $T$ time steps:

$$\frac{\partial H_T}{\partial H_0} = \prod_{t=1}^{T} W_{hh}^T \cdot \text{diag}(\phi')$$

- $\|W_{hh}\| > 1$: gradient tăng **exponentially** → nan/inf
- $\|W_{hh}\| < 1$: gradient giảm exponentially → **vanishing** (clipping không giúp)

## Implementation

```python
# From scratch
def clip_gradients(model, theta):
    params = [p for p in model.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params))
    if norm > theta:
        for p in params:
            p.grad[:] *= theta / norm

# PyTorch built-in
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=theta)
```

## So sánh với các phương pháp khác

| Phương pháp            | Giải quyết               | Nhược điểm                             |
| ---------------------- | ------------------------ | -------------------------------------- |
| **Gradient Clipping**  | Exploding gradient       | Không giải quyết vanishing             |
| **Giảm learning rate** | Giảm bước cập nhật       | Chậm training khi gradient bình thường |
| **LSTM/GRU**           | Vanishing gradient       | Nhiều params hơn vanilla RNN           |
| **Gradient penalty**   | Regularize gradient norm | Tốn compute (tính Hessian)             |

## Liên kết

- [[Recurrent Neural Network]] — kiến trúc chính cần gradient clipping
- [[Buổi 40 - Tuần 11]] — implementation chi tiết
- [[Backpropagation Through Time]] — cơ chế gây exploding/vanishing gradient

---

> [!TODO]
>
> - Thêm phân tích gradient clipping cho Transformer (gradient norm behavior khác RNN)
> - Thêm adaptive clipping methods (e.g., gradient centralization)
> - So sánh max-norm clipping vs value clipping
