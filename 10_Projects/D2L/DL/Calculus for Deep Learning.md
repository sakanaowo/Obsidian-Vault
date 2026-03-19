---
title: "Calculus for Deep Learning"
aliases: [calculus, "giải tích", derivatives, gradients, "chain rule"]
tags: [concept, deep-learning, d2l, math, fundamentals]
created: 2026-03-07
session: "D2L Tuần 1, Buổi 4 — Calculus"
source: "D2L Chapter Preliminaries - sec_calculus"
related:
  - "[[Linear Algebra for Deep Learning]]"
  - "[[Automatic Differentiation]]"
  - "[[Backpropagation]]"
---

# Calculus for Deep Learning

> [!NOTE] ELI5
> Tưởng tượng bạn đang đứng trên một ngọn đồi lởm chởm và muốn xuống thấp nhất (minimize loss). **Giải tích** cho bạn biết: chỗ bạn đứng đang **dốc theo hướng nào** và **dốc bao nhiêu**. Thông tin đó chính là **gradient** — cái mà mọi thuật toán training đều cần. Không có giải tích, không có backpropagation, không có Deep Learning.

## 1. Tại sao Calculus quan trọng trong DL?

Deep learning = tìm parameters $\theta$ để minimize loss $L(\theta)$.

**Cách duy nhất hiệu quả để tìm minimum:** đi theo hướng ngược gradient.

$$\theta \leftarrow \theta - \eta \cdot \nabla_\theta L(\theta)$$

Để tính $\nabla_\theta L(\theta)$, ta cần:

1. **Derivative** — đạo hàm của hàm 1 biến
2. **Partial derivative** — đạo hàm riêng của hàm nhiều biến
3. **Gradient** — vector gom tất cả partial derivatives
4. **Chain rule** — cách tính gradient qua nhiều lớp hàm lồng nhau (= backpropagation)

---

## 2. Derivative — Đạo hàm

> [!NOTE] ELI5
> Đạo hàm $f'(x)$ = **tốc độ thay đổi tức thời** của $f$ tại $x$. Nếu bạn tăng $x$ thêm một chút $h$, hàm $f$ thay đổi khoảng $f'(x) \cdot h$. Trực quan hơn: $f'(x)$ chính là **độ dốc** (slope) của đường tiếp tuyến với đồ thị $f$ tại điểm $x$.

### Định nghĩa chính thức

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

### Ký hiệu tương đương

$$f'(x) = y' = \frac{dy}{dx} = \frac{df}{dx} = \frac{d}{dx}f(x) = Df(x)$$

### Các đạo hàm cơ bản

| Hàm           | Đạo hàm        |
| ------------- | -------------- |
| $C$ (hằng số) | $0$            |
| $x^n$         | $nx^{n-1}$     |
| $e^x$         | $e^x$          |
| $\ln x$       | $x^{-1} = 1/x$ |
| $\sin x$      | $\cos x$       |
| $\cos x$      | $-\sin x$      |

### Quy tắc đạo hàm

| Quy tắc               | Công thức                                                              |
| --------------------- | ---------------------------------------------------------------------- |
| **Constant multiple** | $\frac{d}{dx}[Cf(x)] = C \cdot f'(x)$                                  |
| **Sum rule**          | $\frac{d}{dx}[f(x) + g(x)] = f'(x) + g'(x)$                            |
| **Product rule**      | $\frac{d}{dx}[f(x)g(x)] = f(x)g'(x) + g(x)f'(x)$                       |
| **Quotient rule**     | $\frac{d}{dx}\frac{f(x)}{g(x)} = \frac{g(x)f'(x) - f(x)g'(x)}{g(x)^2}$ |

### Ví dụ minh họa

$$\frac{d}{dx}[3x^2 - 4x] = 3 \cdot 2x - 4 \cdot 1 = 6x - 4$$

Tại $x = 1$: $f'(1) = 6(1) - 4 = 2$ → độ dốc tại $x=1$ là 2.

```python
# Kiểm tra numerically
def f(x):
    return 3 * x**2 - 4 * x

for h in 10.0**np.arange(-1, -6, -1):
    print(f'h={h:.5f}, numerical limit={(f(1+h)-f(1))/h:.5f}')
# Kết quả hội tụ về 2.0 khi h → 0
```

---

## 3. Partial Derivatives & Gradient

> [!NOTE] ELI5
> Hàm DL thường có **hàng triệu biến** (parameters). Partial derivative $\frac{\partial f}{\partial x_i}$ hỏi: "Nếu chỉ thay đổi **một mình** $x_i$, $f$ thay đổi như thế nào?" Rồi ta gom tất cả partial derivatives thành 1 vector → đó là **gradient** $\nabla f$. Gradient chỉ hướng tăng nhanh nhất của $f$, nên để giảm $f$ ta đi **ngược chiều** gradient.

### Partial Derivative

Cho $y = f(x_1, x_2, \ldots, x_n)$:

$$\frac{\partial y}{\partial x_i} = \lim_{h \to 0} \frac{f(\ldots, x_i + h, \ldots) - f(\ldots, x_i, \ldots)}{h}$$

**Cách tính:** giữ tất cả biến khác cố định, đạo hàm theo $x_i$ như hàm 1 biến.

**Ký hiệu:** $\frac{\partial y}{\partial x_i} = \partial_{x_i} f = f_{x_i} = \partial_i f$ — tất cả giống nhau.

### Gradient

$$\nabla_{\mathbf{x}} f(\mathbf{x}) = \begin{bmatrix} \partial_{x_1} f \\ \partial_{x_2} f \\ \vdots \\ \partial_{x_n} f \end{bmatrix} \in \mathbb{R}^n$$

**Ý nghĩa hình học:**

- $\nabla f(\mathbf{x})$ trỏ **hướng tăng nhanh nhất** của $f$ tại $\mathbf{x}$
- $-\nabla f(\mathbf{x})$ trỏ **hướng giảm nhanh nhất** → đây là hướng gradient descent đi

### Gradient của các biểu thức ma trận quan trọng

| Biểu thức                                      | Gradient theo $\mathbf{x}$                 |
| ---------------------------------------------- | ------------------------------------------ |
| $\mathbf{A}\mathbf{x}$                         | $\mathbf{A}^\top$                          |
| $\mathbf{x}^\top\mathbf{A}$                    | $\mathbf{A}$                               |
| $\mathbf{x}^\top\mathbf{A}\mathbf{x}$          | $(\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$ |
| $\|\mathbf{x}\|^2 = \mathbf{x}^\top\mathbf{x}$ | $2\mathbf{x}$                              |
| $\|\mathbf{X}\|_F^2$                           | $2\mathbf{X}$                              |

> [!TIP] Tại sao các công thức này quan trọng?
>
> - $\nabla_{\mathbf{x}} \|\mathbf{x}\|^2 = 2\mathbf{x}$ → gradient của MSE loss chứa term này!
> - $\nabla_{\mathbf{W}} \|\mathbf{W}\|_F^2 = 2\mathbf{W}$ → gradient của L2 regularization!

---

## 4. Chain Rule — Quy tắc dây chuyền

> [!NOTE] ELI5
> Neural network là hàm lồng nhau nhiều tầng: `loss(softmax(linear(relu(linear(x)))))`. **Chain rule** cho phép tính gradient của cái này theo từng layer, từ ngoài vào trong, bằng cách **nhân các gradient lại với nhau**. Đây chính là trái tim của **backpropagation**.

### Chain rule — Hàm 1 biến

Nếu $y = f(g(x))$, với $u = g(x)$:

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

**Ví dụ:** $y = (3x^2 - 4x)^5$

- $u = 3x^2 - 4x$, $y = u^5$
- $\frac{dy}{du} = 5u^4$, $\frac{du}{dx} = 6x - 4$
- $\frac{dy}{dx} = 5(3x^2 - 4x)^4 \cdot (6x - 4)$

### Chain rule — Hàm nhiều biến

Nếu $y = f(\mathbf{u})$ và $\mathbf{u} = g(\mathbf{x})$ (vector-valued):

$$\frac{\partial y}{\partial x_i} = \sum_{j=1}^m \frac{\partial y}{\partial u_j} \cdot \frac{\partial u_j}{\partial x_i}$$

Dạng ma trận:

$$\nabla_{\mathbf{x}} y = \mathbf{A} \cdot \nabla_{\mathbf{u}} y$$

Trong đó $\mathbf{A}_{ij} = \frac{\partial u_j}{\partial x_i}$ là **Jacobian matrix**.

### Liên hệ với Backpropagation

```
Forward pass:  x → u₁ → u₂ → ... → y = L (loss)
                ↑      ↑      ↑
Backward:   ∂L/∂x ← ∂L/∂u₁ ← ∂L/∂u₂ ← ... ← ∂L/∂y = 1
```

Chain rule cho phép truyền gradient từ output ngược về input, **nhân qua từng layer**. Đây chính là lý do linear algebra quan trọng: mỗi bước backward = một **vector-matrix product**.

---

## 5. Kết nối với Optimization

### Gradient Descent cơ bản

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta L(\theta_t)$$

- $\theta$: parameters (weights)
- $\eta$: learning rate (tốc độ học)
- $\nabla_\theta L$: gradient của loss theo parameters

### Ý nghĩa của các điểm đặc biệt

| Điều kiện                    | Ý nghĩa                                         |
| ---------------------------- | ----------------------------------------------- |
| $f'(x) = 0$                  | Critical point (có thể là min/max/saddle point) |
| $f'(x) > 0$                  | $f$ đang tăng → đi về bên trái để giảm          |
| $f'(x) < 0$                  | $f$ đang giảm → đi về bên phải để giảm          |
| $f''(x) > 0$ tại $f'(x) = 0$ | Local minimum                                   |
| $f''(x) < 0$ tại $f'(x) = 0$ | Local maximum                                   |

> [!WARNING] Saddle Points trong DL
> Với hàm nhiều chiều, $\nabla f = \mathbf{0}$ không nhất thiết là minimum! Có thể là **saddle point** (minimum theo 1 hướng, maximum theo hướng khác). Đây là vấn đề thực tế trong training DNN với hàng triệu parameters.

---

## 6. Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 3 * x**2 - 4 * x

x = np.arange(0, 3, 0.1)
tangent_at_1 = 2 * x - 3  # y = f(1) + f'(1)*(x-1) = -1 + 2*(x-1) = 2x - 3

plt.plot(x, f(x), label='f(x) = 3x²-4x')
plt.plot(x, tangent_at_1, '--', label='Tangent at x=1 (slope=2)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid()
```

---

## 7. Tổng kết

| Khái niệm               | Định nghĩa                                   | Ứng dụng DL                                |
| ----------------------- | -------------------------------------------- | ------------------------------------------ |
| **Derivative $f'(x)$**  | Tốc độ thay đổi tức thời, độ dốc tiếp tuyến  | Làm thế nào để thay đổi param để giảm loss |
| **Partial derivative**  | Đạo hàm khi giữ các biến khác cố định        | Gradient của mỗi weight riêng lẻ           |
| **Gradient $\nabla f$** | Vector tất cả partial derivatives            | Hướng đi trong gradient descent            |
| **Chain rule**          | $\frac{dy}{dx} = \frac{dy}{du}\frac{du}{dx}$ | Backpropagation qua nhiều layers           |

---

## Exercises (từ D2L)

1. Chứng minh từ định nghĩa: $\frac{d}{dx} x^n = nx^{n-1}$ (hint: khai triển $(x+h)^n$)
2. Tính đạo hàm $f(x) = x^x$ (hint: viết $x^x = e^{x \ln x}$, dùng chain rule)
3. Tìm gradient của $f(\mathbf{x}) = 3x_1^2 + 5e^{x_2}$
4. Gradient của $f(\mathbf{x}) = \|\mathbf{x}\|_2$ là gì? Điều gì xảy ra tại $\mathbf{x} = \mathbf{0}$?
5. $f'(x) = 0$ có nghĩa là gì? Cho ví dụ cụ thể.

---

> [!TODO] Mở rộng
>
> - [[Automatic Differentiation]] — cách máy tính tự tính gradient (Buổi 5)
> - [[Backpropagation]] — áp dụng chain rule qua computation graph
> - Tìm hiểu: Second-order methods (Newton's method, L-BFGS)
> - Tìm hiểu: Jacobian và Hessian matrix
