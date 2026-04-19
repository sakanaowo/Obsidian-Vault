---
tags:
  - concept
  - deep-learning
  - rnn
  - lstm
  - gating
  - sequence-model
aliases:
  - LSTM
  - Long Short-Term Memory
  - Bộ nhớ Ngắn-Dài hạn
date: 2026-04-20
status: mature
---

# Long Short-Term Memory (LSTM)

> [!NOTE] Giải thích đơn giản
> LSTM giống một người đọc sách **có sổ tay ghi chú**: ở mỗi trang, họ quyết định (1) xóa ghi chú cũ nào, (2) ghi thêm gì mới, (3) khi nào mở sổ cho người khác xem. Nhờ vậy, họ có thể nhớ chi tiết từ trang 10 khi đọc đến trang 490.

## Định nghĩa kỹ thuật

**LSTM** là kiến trúc mạng hồi quy (RNN) thay thế mỗi nút ẩn thường bằng một **ô nhớ có cổng** (gated memory cell). Mỗi ô nhớ duy trì hai trạng thái song song:

- **Trạng thái ô nhớ** $C_t$ — bộ nhớ nội bộ, được bảo vệ bởi cổng
- **Trạng thái ẩn** $H_t$ — output cho tầng tiếp theo

Ba cổng sigmoid kiểm soát dòng thông tin: cổng quên ($F_t$), cổng đầu vào ($I_t$), cổng đầu ra ($O_t$).

**Giải quyết vấn đề:** [[Vanishing and Exploding Gradients|Gradient biến mất]] trong RNN thường, nhờ cập nhật $C_t$ bằng phép cộng tuyến tính thay vì phép nhân ma trận.

**Đề xuất bởi:** Hochreiter & Schmidhuber (1997).

## Công thức cốt lõi

$$I_t = \sigma(X_t W_{xi} + H_{t-1} W_{hi} + b_i)$$
$$F_t = \sigma(X_t W_{xf} + H_{t-1} W_{hf} + b_f)$$
$$O_t = \sigma(X_t W_{xo} + H_{t-1} W_{ho} + b_o)$$
$$\tilde{C}_t = \tanh(X_t W_{xc} + H_{t-1} W_{hc} + b_c)$$
$$C_t = F_t \odot C_{t-1} + I_t \odot \tilde{C}_t$$
$$H_t = O_t \odot \tanh(C_t)$$

## Đặc điểm

| Khía cạnh | Chi tiết |
|---|---|
| Tham số | $4(dh + h^2 + h)$ — gấp 4 lần RNN thường |
| Đầu vào | $X_t \in \mathbb{R}^{n \times d}$, $(H_{t-1}, C_{t-1})$ |
| Đầu ra | $H_t \in \mathbb{R}^{n \times h}$, $C_t \in \mathbb{R}^{n \times h}$ |
| Hàm kích hoạt | Sigmoid (cổng) + Tanh (ứng viên + output) |
| Gradient flow | Ổn định qua đường $C_t$ khi $F_t \approx 1$ |

## Liên kết

- **Tiền thân:** [[Recurrent Neural Network]]
- **Đơn giản hơn:** [[GRU]] (gộp cổng quên + đầu vào, bỏ $C_t$ riêng)
- **Kế thừa:** [[Transformer Architecture]] (thay tuần tự bằng song song)
- **Kỹ thuật liên quan:** [[Gradient Clipping]], [[Backpropagation Through Time]]
- **Hàm kích hoạt:** [[Sigmoid Function]], [[Activation Function]]

## Nguồn tham khảo

- [[Buổi 43 - Tuần 12]] — Phân tích chi tiết LSTM từ D2L 10.1
- D2L Section 10.1: [d2l.ai/chapter_recurrent-modern/lstm.html](https://d2l.ai/chapter_recurrent-modern/lstm.html)
