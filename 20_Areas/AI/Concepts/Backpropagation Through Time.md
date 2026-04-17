---
tags:
  - ai
  - deep-learning
  - rnn
  - optimization
  - gradient
aliases:
  - BPTT
  - Backprop Through Time
date: 2026-04-19
---

# Backpropagation Through Time (BPTT)

> [!NOTE] ELI5
> Hãy tưởng tượng một dây chuyền lắp ráp dài 100 người. Nếu sản phẩm cuối bị lỗi, quản lý phải truy ngược lại từng người để tìm ai gây ra lỗi. Nhưng qua mỗi người, "manh mối" (gradient) bị méo đi — qua 100 người thì hoặc biến mất hoàn toàn (vanishing) hoặc bị phóng đại phi lý (exploding). BPTT là quy trình "truy lỗi ngược" này trong RNN.

**Backpropagation Through Time (BPTT)** là thuật toán tính gradient cho Recurrent Neural Networks (RNN). Về bản chất, nó là backpropagation thông thường áp dụng lên computational graph của RNN **đã unrolled theo thời gian** — biến mạng hồi quy thành một mạng feedforward rất sâu ($T$ layers tương ứng $T$ time steps).

**Tại sao BPTT đặc biệt?** Trong feedforward networks, mỗi layer có trọng số riêng. Trong RNN, **cùng một ma trận** $W_{hh}$ được nhân lại $T$ lần. Gradient do đó chứa **lũy thừa** $(W_{hh})^k$, gây ra vấn đề vanishing/exploding gradient.

## Công thức cốt lõi

Với mô hình RNN đơn giản $h_t = W_{hx} x_t + W_{hh} h_{t-1}$, $o_t = W_{qh} h_t$:

$$\frac{\partial L}{\partial h_t} = \sum_{i=t}^{T} \left(W_{hh}^T\right)^{T-i} \cdot W_{qh}^T \cdot \frac{\partial L}{\partial o_i}$$

Lũy thừa $(W_{hh}^T)^k$ là nguyên nhân chính:

- $|\lambda_{\max}| < 1$ → gradient vanish (mất thông tin xa)
- $|\lambda_{\max}| > 1$ → gradient explode (training diverge)

## Các chiến lược thực tế

| Chiến lược         | Mô tả                                  | Ưu/Nhược                                              |
| ------------------ | -------------------------------------- | ----------------------------------------------------- |
| **Full BPTT**      | Truyền ngược toàn bộ $T$ steps         | Chính xác nhưng $O(T)$ bộ nhớ, vanishing/exploding    |
| **Truncated BPTT** | Chỉ truyền ngược $\tau$ steps gần nhất | **Mặc định** — biased nhưng ổn định, $O(\tau)$ bộ nhớ |
| **Randomized**     | Truncate ngẫu nhiên                    | Unbiased nhưng variance cao                           |

Trong PyTorch, truncated BPTT được thực hiện qua `state.detach_()`.

## Liên kết

- [[Recurrent Neural Network]] — kiến trúc mà BPTT áp dụng
- [[Gradient Clipping]] — xử lý exploding gradient (không xử lý vanishing)
- [[Gradient Descent]] — thuật toán tối ưu sử dụng gradient từ BPTT

---

> [!TODO] Mở rộng
>
> - Thêm so sánh BPTT vs RTRL (Real-Time Recurrent Learning)
> - Phân tích chi tiết hơn về Jacobian spectrum trong thực tế
> - Kết nối với gate mechanism của LSTM/GRU
