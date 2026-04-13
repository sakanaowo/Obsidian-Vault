---
tags:
  - ai
  - deep-learning
  - sequence-modeling
  - autoregressive
  - language-model
aliases:
  - AR Model
  - Autoregressive
  - Self-Regressive Model
date: 2026-04-13
status: seedling
---

# Autoregressive Model

> [!NOTE] ELI5
> Hãy tưởng tượng bạn đang viết tiếp một câu chuyện. Mỗi từ bạn viết phụ thuộc vào **tất cả các từ bạn đã viết trước đó**. "Auto" = "tự"; "Regressive" = "hồi quy" — bạn dùng chính **output cũ của mình** làm input để sinh output mới. Google auto-complete, ChatGPT, và dự báo thời tiết đều hoạt động theo nguyên tắc này.

## Định nghĩa kỹ thuật

**Autoregressive (AR) model** là model ước lượng **xác suất có điều kiện** của phần tử tiếp theo dựa trên toàn bộ lịch sử các phần tử trước:

$$P(x_t \mid x_{t-1}, x_{t-2}, \ldots, x_1)$$

- **Input:** Chuỗi các giá trị quá khứ $x_1, \ldots, x_{t-1}$
- **Output:** Phân phối xác suất trên giá trị tiếp theo $x_t$
- **Giải quyết:** Bài toán dự đoán tuần tự — sử dụng **chain rule** để phân rã xác suất đồng thời:

$$P(x_1, \ldots, x_T) = P(x_1) \cdot \prod_{t=2}^{T} P(x_t \mid x_{t-1}, \ldots, x_1)$$

## Cơ chế chi tiết

### Hai chiến lược chính

| Chiến lược                | Công thức                                       | Ưu điểm                         | Nhược điểm                    | Mô hình đại diện   |
| ------------------------- | ----------------------------------------------- | ------------------------------- | ----------------------------- | ------------------ |
| **Fixed Window** ($\tau$) | $\hat{x}_t = f(x_{t-\tau}, \ldots, x_{t-1})$    | Đơn giản, input cố định         | Mất context xa hơn $\tau$     | N-gram, MLP        |
| **Latent Autoregressive** | $h_t = g(h_{t-1}, x_{t-1}); \hat{x}_t = f(h_t)$ | Nhớ toàn bộ lịch sử (lý thuyết) | Khó train, vanishing gradient | **RNN**, LSTM, GRU |

### Ứng dụng thực tế

- **GPT** (Generative Pre-trained Transformer): dự đoán token tiếp theo → sinh văn bản
- **WaveNet**: dự đoán audio sample tiếp theo → sinh giọng nói
- **Time series forecasting**: dự đoán giá cổ phiếu, nhiệt độ, lưu lượng mạng

### Hạn chế chính

- **Error accumulation**: khi generate multi-step, sai số tích lũy theo hàm mũ
- **Exposure bias**: training dùng ground truth (teacher forcing) nhưng inference dùng own predictions
- **Unidirectional**: chỉ nhìn context trái → phải (so với BERT nhìn cả hai hướng)

## Liên kết

- [[Markov Chain]] — AR model với Markov assumption
- [[N-gram Language Model]] — AR model cổ điển dùng fixed window
- Source: [[Buổi 37 - Tuần 10]] §3

---

> [!TODO]
>
> - [ ] Mở rộng phần so sánh AR vs non-AR (BERT, Diffusion)
> - [ ] Thêm code implementation cơ bản
> - [ ] Liên kết với GPT architecture khi học Transformer
