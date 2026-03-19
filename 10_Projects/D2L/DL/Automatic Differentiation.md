---
title: "Automatic Differentiation"
aliases: [autograd, autodiff, "tự động vi phân", backpropagation-mechanism]
tags: [concept, deep-learning, d2l, math, fundamentals, pytorch]
created: 2026-03-07
session: "D2L Tuần 1, Buổi 5 — Automatic Differentiation (Autograd)"
source: "D2L Chapter Preliminaries - sec_autograd"
related:
  - "[[Calculus for Deep Learning]]"
  - "[[Tensor Operations]]"
  - "[[Backpropagation]]"
---

# Automatic Differentiation (Autograd)

> [!NOTE] ELI5
> Khi bạn muốn nướng bánh hoàn hảo, bạn ghi lại từng bước làm (cho bột → nhào → nướng). Nếu bánh bị cháy, bạn đọc ngược lại để tìm bước nào sai. **Autograd** làm y chang: khi tính toán, nó **ghi lại mọi bước** vào một "cuốn sổ" (computational graph). Khi cần gradient, nó đọc ngược sổ từ output về input, áp dụng chain rule ở mỗi bước. Không cần viết đạo hàm tay nữa!

## 1. Vấn đề & Giải pháp

### Tại sao không tính gradient thủ công?

Neural net với hàng triệu parameters — tính đạo hàm tay thủ công từng layer là:

- **Tedious** — cực kỳ mất thời gian
- **Error-prone** — dễ nhầm ký hiệu, quên chain rule
- **Infeasible** — với dynamic architecture (RNN, transformers) không thể làm trước

### Autograd giải quyết thế nào?

1. **Forward pass:** Mỗi phép tính được ghi vào **computational graph** — cây tính toán theo dõi từng value phụ thuộc vào value nào
2. **Backward pass (`backward()`):** Đi ngược graph, áp dụng **chain rule** tự động ở mỗi node
3. **Kết quả:** Gradient của mọi leaf tensor có `requires_grad=True`

---

## 2. Computational Graph

> [!NOTE] ELI5
> Computational graph là bản đồ "ai sinh ra ai". Ví dụ: $y = 2x^\top x$. Bản đồ ghi: $x$ → nhân dot với $x$ → nhân 2 → $y$. Khi backward, đi ngược bản đồ này, áp dụng chain rule từng bước.

```lua
x → dot(x, x) → * 2 → y
    [node 1]    [node 2]
```

**Backward:**

- $\frac{\partial y}{\partial \text{node2}} = 1$
- $\frac{\partial \text{node2}}{\partial \text{node1}} = 2$
- $\frac{\partial \text{node1}}{\partial x} = 2x$ (vì $\frac{\partial}{\partial x} x^\top x = 2x$)
- By chain rule: $\frac{\partial y}{\partial x} = 1 \cdot 2 \cdot 2x = 4x$ ✓

---

## 3. PyTorch Workflow — 4 Bước

```python
import torch

# Bước 1: Tạo tensor, đánh dấu requires_grad=True
x = torch.arange(4.0, requires_grad=True)
# Hoặc: x = torch.arange(4.0); x.requires_grad_(True)

# Bước 2: Tính toán (PyTorch tự build computational graph)
y = 2 * torch.dot(x, x)   # y = 2 * x^T * x = scalar

# Bước 3: Gọi backward()
y.backward()

# Bước 4: Đọc gradient từ .grad
print(x.grad)  # tensor([0., 4., 8., 12.]) = 4x ✓

# Xác minh
print(x.grad == 4 * x)  # tensor([True, True, True, True])
```

> [!WARNING] PyTorch không tự reset gradient buffer!
> Khác với MXNet/TensorFlow, PyTorch **cộng dồn gradient** vào `.grad` mỗi lần `backward()`. Trước mỗi lần tính mới, phải gọi `x.grad.zero_()`:
>
> ```python
> x.grad.zero_()  # Reset về 0
> y = x.sum()
> y.backward()
> print(x.grad)  # tensor([1., 1., 1., 1.])
> ```
>
> **Tại sao?** Thiết kế này hữu ích khi optimize tổng nhiều objective functions.

---

## 4. Backward cho Non-Scalar Output

Thông thường, loss là một **scalar**. Nhưng nếu output `y` là vector?

**Jacobian:** Derivative của vector-valued function là một **matrix**:
$$\mathbf{J} = \frac{\partial \mathbf{y}}{\partial \mathbf{x}} \in \mathbb{R}^{m \times n}$$

Trong DL, ta thường cần **sum gradient** qua batch — không cần full Jacobian. PyTorch API:

```python
x.grad.zero_()
y = x * x   # y là vector [x0², x1², x2², x3²]

# PyTorch yêu cầu cung cấp "gradient" argument cho non-scalar backward
# Về bản chất: tính v^T * J thay vì J đầy đủ
y.backward(gradient=torch.ones(len(y)))  # v = [1,1,1,1]
# Equivalent: y.sum().backward()

print(x.grad)  # tensor([0., 2., 4., 6.]) = 2x (gradient của sum(x*x))
```

> [!NOTE] Tại sao tham số tên là `gradient` thay vì `v`?
> Về mặt toán học, argument này là vector $\mathbf{v}$ để tính $\mathbf{v}^\top \mathbf{J}$. Tên "gradient" gây nhầm lẫn nhưng có lý do lịch sử. Khi $\mathbf{v} = \mathbf{1}$, kết quả = gradient của $\sum y_i$ theo $x$.

---

## 5. Detaching Computation

Đôi khi ta muốn **tách một phần** khỏi computational graph — không muốn gradient chạy qua đó.

**Ví dụ:** Tính $z = x \cdot y$ với $y = x^2$, nhưng chỉ muốn gradient theo "direct effect" của $x$.

```python
x.grad.zero_()
y = x * x           # y = x²
u = y.detach()      # u = same value as y, nhưng bị cắt khỏi graph

z = u * x           # z = u * x (u treated as constant)
z.sum().backward()

print(x.grad)       # = u = x², KHÔNG phải 3x² như khi không detach
print(x.grad == u)  # True
```

**So sánh:**

| Cách         | `z = x * y * (...) = x * x * x`     | Gradient               |
| ------------ | ----------------------------------- | ---------------------- |
| Không detach | $z = x^3$                           | $\nabla_x z = 3x^2$    |
| Detach $y$   | $z = u \cdot x$, $u = \text{const}$ | $\nabla_x z = u = x^2$ |

**Ứng dụng thực tế:**

- **Target network** trong reinforcement learning (DDPG, DQN)
- **Stop-gradient** trong self-supervised learning (SimSiam, BYOL)
- Khi muốn fake "constants" trong optimization

---

## 6. Gradient qua Python Control Flow

Đây là sức mạnh cực kỳ quan trọng: autograd **hoạt động ngay cả với if/while/loops**!

```python
def f(a):
    b = a * 2
    while b.norm() < 1000:   # Python while!
        b = b * 2
    if b.sum() > 0:           # Python if!
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()

# Gradient chính xác tuy hàm có if/while
print(a.grad == d / a)  # True (vì f tuyến tính theo a, piecewise)
```

**Tại sao hoạt động được?** Computational graph được xây dựng **dynamically** khi run từng dòng code cụ thể với input cụ thể. Mỗi lần gọi `f(a)` với input khác → graph khác! Đây gọi là **dynamic computation graph** (PyTorch) vs static graph (TF1, Theano).

> [!NOTE] Dynamic vs Static Graph
> | | **Dynamic (PyTorch, JAX)** | **Static (TF1, Theano)** |
> |---|---|---|
> | Graph xây dựng | Khi chạy code (eager) | Trước khi chạy (compile) |
> | Debug | Dễ — dùng print/pdb | Khó |
> | Control flow | Tự nhiên (Python if/while) | Cần ops đặc biệt (tf.cond) |
> | Performance | Tốt (XLA compile optional) | Nhanh hơn khi đã compile |

---

## 7. Tóm tắt 4 Quy tắc Vàng (từ D2L)

1. **Attach** gradient: `requires_grad=True` hoặc `x.requires_grad_(True)`
2. **Record** computation: chỉ cần tính toán như bình thường (PyTorch tự ghi)
3. **Execute backprop**: `loss.backward()`
4. **Access** gradient: `x.grad`

---

## 8. Bảng so sánh Framework

| Feature               | PyTorch              | TensorFlow 2          | JAX              |
| --------------------- | -------------------- | --------------------- | ---------------- |
| Kích hoạt tracking    | `requires_grad=True` | `tf.Variable(x)`      | automatic        |
| Record scope          | tự động              | `tf.GradientTape()`   | tự động          |
| Backward              | `y.backward()`       | `tape.gradient(y, x)` | `grad(f)(x)`     |
| Reset gradient        | `x.grad.zero_()`     | automatic             | N/A (functional) |
| Gradient accumulation | **có (mặc định)**    | không                 | N/A              |

---

## 9. Kết nối với Backpropagation

Autograd là **cài đặt kỹ thuật** của backpropagation:

- **Backprop (lý thuyết):** Áp dụng chain rule ngược qua layers
- **Autograd (thực hành):** Tự động áp dụng chain rule trên computational graph

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y_n} \cdot \frac{\partial y_n}{\partial y_{n-1}} \cdots \frac{\partial y_1}{\partial x} = \prod_{i} \mathbf{J}_i$$

Mỗi `backward()` call = một lần traverse toàn bộ cây này từ output về input.

---

## Exercises (từ D2L)

1. Tại sao second derivative ($f''$) đắt hơn nhiều so với first derivative?
2. Sau `backward()`, gọi lại ngay lần nữa — điều gì xảy ra? (Hint: PyTorch xóa graph sau backward mặc định)
3. Trong ví dụ control flow, nếu `a` là vector thay vì scalar, điều gì xảy ra?
4. Vẽ $f(x) = \sin(x)$ và $f'(x)$ **chỉ dùng autograd**, không hardcode $\cos(x)$.

---

> [!TODO] Mở rộng
>
> - [[Backpropagation]] — phân tích toán học chi tiết qua multi-layer network
> - Tìm hiểu: **Higher-order derivatives** — `torch.autograd.grad` với `create_graph=True`
> - Tìm hiểu: Forward-mode vs Reverse-mode autodiff (khi nào dùng cái nào?)
> - Tìm hiểu: **JAX functional approach** — `grad`, `jit`, `vmap` stack
