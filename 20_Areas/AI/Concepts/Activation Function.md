---
title: "Activation Function"
aliases: [hàm kích hoạt, activation, nonlinearity, phi tuyến]
tags: [concept, deep-learning, neural-network, activation]
created: 2026-03-24
---

# Activation Function

> [!NOTE] ELI5
> Nếu mạng neuron chỉ có phép **nhân + cộng** (tuyến tính), thì xếp bao nhiêu tầng cũng vô nghĩa — kết quả luôn rút gọn thành 1 tầng duy nhất. **Activation function** là "nút bẻ cong" — nó thêm **phi tuyến** (nonlinearity), giúp mạng có thể học được những quy luật phức tạp mà đường thẳng không thể biểu diễn.

## 1. Tại sao cần Activation Function?

Nếu không có activation function:

$$\mathbf{H} = \mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)}, \quad \mathbf{O} = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}$$

Có thể rút gọn thành:

$$\mathbf{O} = \mathbf{X}\underbrace{\mathbf{W}^{(1)}\mathbf{W}^{(2)}}_{\mathbf{W}} + \underbrace{\mathbf{b}^{(1)}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}}_{\mathbf{b}}$$

→ **Equivalent với 1 layer!** Thêm layer mà không thêm nonlinearity = vô nghĩa.

Với activation $\sigma$:
$$\mathbf{H} = \sigma(\mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)})$$
→ **Không thể rút gọn** → mạng thật sự "sâu".

## 2. ReLU (Rectified Linear Unit)

$$\text{ReLU}(x) = \max(0, x)$$

| Tính chất | Giá trị |
| --- | --- |
| Range | $[0, +\infty)$ |
| Đạo hàm | $0$ nếu $x < 0$, $1$ nếu $x > 0$ |
| Ưu điểm | Đơn giản, gradient không vanish (khi $x>0$) |
| Nhược điểm | "Dead neurons" (gradient = 0 vĩnh viễn khi $x < 0$) |

**Phổ biến nhất** trong deep learning hiện đại. Sử dụng mặc định cho hidden layers.

### Biến thể
- **Leaky ReLU**: $\max(0.01x, x)$ — cho gradient nhỏ khi $x < 0$
- **pReLU**: $\max(\alpha x, x)$ — $\alpha$ là tham số học được
- **GELU**: $x \cdot \Phi(x)$ — dùng trong Transformer (GPT, BERT)

## 3. Sigmoid

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

| Tính chất | Giá trị |
| --- | --- |
| Range | $(0, 1)$ |
| Đạo hàm | $\sigma(x)(1 - \sigma(x))$, max = 0.25 tại $x=0$ |
| Ưu điểm | Output giải thích được như xác suất |
| Nhược điểm | Gradient vanish khi $|x|$ lớn (max chỉ 0.25) |

Dùng ở **output layer** cho binary classification. **Ít dùng** cho hidden layers (gradient vanish).

## 4. Tanh (Hyperbolic Tangent)

$$\tanh(x) = \frac{1 - e^{-2x}}{1 + e^{-2x}}$$

| Tính chất | Giá trị |
| --- | --- |
| Range | $(-1, 1)$ |
| Đạo hàm | $1 - \tanh^2(x)$, max = 1 tại $x=0$ |
| Ưu điểm | Zero-centered (output trung bình = 0) |
| Nhược điểm | Gradient vanish khi $|x|$ lớn |

Quan hệ: $\tanh(x) = 2\sigma(2x) - 1$. Dùng trong **LSTM/GRU** (RNN).

## 5. Bảng so sánh

| | ReLU | Sigmoid | Tanh |
| --- | --- | --- | --- |
| Range | $[0, \infty)$ | $(0, 1)$ | $(-1, 1)$ |
| Gradient max | 1 | 0.25 | 1 |
| Vanishing? | Không (khi $x>0$) | Có | Có |
| Zero-centered? | Không | Không | Có |
| Dùng ở | Hidden layers | Output (binary) | LSTM/GRU |

## TODO

- [ ] Thêm GELU, Swish, Mish
- [ ] Code visualization tất cả activation functions
