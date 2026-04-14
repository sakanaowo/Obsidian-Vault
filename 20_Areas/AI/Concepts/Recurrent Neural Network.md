---
tags:
  - concept
  - deep-learning
  - rnn
  - sequence-modeling
aliases:
  - RNN
  - Vanilla RNN
  - Elman Network
status: seedling
date: 2026-04-14
---

# Recurrent Neural Network

> [!NOTE] ELI5
> Hãy tưởng tượng bạn đang nghe một câu chuyện. Mỗi khi nghe thêm 1 câu mới, bạn không quên hết câu trước mà **cập nhật** hiểu biết của mình: trộn câu vừa nghe với trí nhớ cũ → tạo ra trí nhớ mới. **RNN** hoạt động y hệt: nó đọc dữ liệu theo thứ tự, mỗi bước trộn input mới với "bộ nhớ" cũ, nhờ đó "hiểu" ngữ cảnh.

## Định nghĩa kỹ thuật

**Recurrent Neural Network (RNN)** là kiến trúc mạng nơ-ron dùng **recurrent computation** cho hidden state: tại mỗi time step $t$, hidden state $H_t$ được tính từ input hiện tại $X_t$ **kết hợp** hidden state trước $H_{t-1}$.

- **Input:** Chuỗi $(X_1, X_2, \ldots, X_T)$, mỗi $X_t \in \mathbb{R}^{n \times d}$
- **Output:** Chuỗi $(O_1, O_2, \ldots, O_T)$, mỗi $O_t \in \mathbb{R}^{n \times q}$
- **Giải quyết:** Sequence modeling — mô hình hóa phụ thuộc thứ tự trong dữ liệu chuỗi mà N-gram và MLP không xử lý được do bùng nổ tham số hoặc thiếu bộ nhớ

## Công thức cốt lõi

**Hidden state update:**
$$H_t = \phi(X_t W_{xh} + H_{t-1} W_{hh} + b_h)$$

**Output:**
$$O_t = H_t W_{hq} + b_q$$

| Parameter | Shape    | Ý nghĩa                                |
| --------- | -------- | -------------------------------------- |
| $W_{xh}$  | $(d, h)$ | Input → hidden                         |
| $W_{hh}$  | $(h, h)$ | Hidden → hidden (recurrent connection) |
| $b_h$     | $(1, h)$ | Bias hidden                            |
| $W_{hq}$  | $(h, q)$ | Hidden → output                        |
| $b_q$     | $(1, q)$ | Bias output                            |
| $\phi$    | —        | Activation (thường **tanh**)           |

## Đặc tính quan trọng

1. **Weight Sharing:** Cùng bộ tham số cho mọi time step → params không phụ thuộc vào $T$
2. **Latent Variable Model:** $H_t$ nén toàn bộ lịch sử $(X_1, \ldots, X_t)$
3. **Sequential Computation:** Phải tính $H_1 \to H_2 \to \ldots$ tuần tự

## Hạn chế

- **Vanishing/Exploding Gradient** khi $T$ lớn → giải quyết bởi [[LSTM]], [[GRU]]
- **Không song song hóa** được → giải quyết bởi [[Transformer]]
- **Effective memory** hữu hạn trong thực tế (~50–200 steps)

## Liên kết

- Học từ: [[Buổi 39 - Tuần 11]]
- Liên quan: [[Autoregressive Model]], [[N-gram Language Model]]
- Phát triển: [[LSTM]], [[GRU]], [[Bidirectional RNN]]
- Nguồn: [D2L 9.4](https://d2l.ai/chapter_recurrent-neural-networks/rnn.html)

---

> [!TODO]
>
> - Thêm phần so sánh Vanilla RNN vs LSTM vs GRU khi học Chapter 10
> - Thêm code implementation chi tiết từ Buổi 40
> - Thêm hình minh họa gradient flow
