---
title: "Sigmoid Function"
aliases: [logistic function, hàm sigmoid]
tags: [concept, math, activation-function, deep-learning, sigmoid]
created: 2026-03-19
related:
  - "[[Softmax Function]]"
  - "[[Cross-Entropy Loss]]"
  - "[[Maximum Likelihood Estimation]]"
---

# Sigmoid Function

> [!NOTE] ELI5
> Sigmoid là một hàm biến một số bất kỳ thành một giá trị trong khoảng 0 đến 1.
> Bạn có thể xem nó như một "nút xoay" mức độ tin tưởng: số lớn thì gần 1, số nhỏ thì gần 0.
> Vì vậy sigmoid rất hợp khi ta muốn diễn giải output thành "xác suất" cho bài toán 2 lớp.

## 1) Bản chất (First Principles): Tại sao định nghĩa như vậy?

**Claim:** Sigmoid được chọn vì nó ánh xạ từ trục thực $(-\infty, +\infty)$ về khoảng mở $(0,1)$, phù hợp với diễn giải xác suất.

**Reasoning:** Trong binary classification, mô hình tuyến tính tạo ra điểm thô (logit) có thể rất âm hoặc rất dương. Xác suất thì không thể nhỏ hơn 0 hay lớn hơn 1, nên cần một phép biến đổi vừa liên tục, vừa đơn điệu, vừa bị chặn trên dưới.

**Evidence:** Sigmoid thỏa tất cả điều kiện đó, và giữ được thứ tự của logit: logit lớn hơn sẽ cho xác suất lớn hơn.

## 2) Ví dụ đời thường cụ thể

Bạn có hệ thống lọc email spam. Mô hình tính điểm thô $z$ cho mỗi email:

- $z = -4$ -> email rất không giống spam
- $z = 0$ -> không chắc
- $z = 4$ -> email rất giống spam

Qua sigmoid:

- $\sigma(-4) \approx 0.018$
- $\sigma(0) = 0.5$
- $\sigma(4) \approx 0.982$

Ý nghĩa: cùng một mô hình tuyến tính, sau sigmoid ta có xác suất để đặt ngưỡng quyết định (ví dụ 0.5, 0.7, ...).

## 3) Công thức và giải thích từng ký hiệu

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

- $z$: logit (điểm thô), thường là $z = \mathbf{w}^T\mathbf{x} + b$
- $e$: hằng số Euler
- $\sigma(z)$: output trong $(0,1)$

### Tính chất quan trọng

1. **Range:** $0 < \sigma(z) < 1$
2. **Monotonic:** $z_1 > z_2 \Rightarrow \sigma(z_1) > \sigma(z_2)$
3. **Đối xứng:** $\sigma(-z) = 1 - \sigma(z)$
4. **Đạo hàm đẹp:**

$$
\sigma'(z) = \sigma(z)(1-\sigma(z))
$$

Đạo hàm lớn nhất tại $z=0$ (bằng $0.25$), nhỏ dần khi $|z|$ lớn. Đây là lý do có hiện tượng **saturation** (gradient gần 0 ở hai đầu).

## 4) Ứng dụng thực tế trong Deep Learning

### 4.1 Binary classification

Xác suất lớp dương:

$$
p(y=1\mid x)=\sigma(z)
$$

Kết hợp với Binary Cross-Entropy (BCE):

$$
\mathcal{L}_{\text{BCE}} = -\left[y\log \hat{y} + (1-y)\log(1-\hat{y})\right]
$$

### 4.2 Logistic Regression và MLE

- Bernoulli likelihood + MLE dẫn tới BCE loss.
- Vì thế Logistic Regression là mô hình xác suất rõ ràng, không chỉ là "hàm kích hoạt".

### 4.3 Gate trong RNN/LSTM/GRU

Sigmoid được dùng để tạo "cổng" (gate) trong LSTM/GRU, vì output 0..1 phù hợp vai trò "đóng/mở" dòng thông tin.

## 5) So sánh nhanh: Sigmoid vs Softmax

| | Sigmoid | Softmax |
| --- | --- | --- |
| Bài toán chính | Binary classification | Multi-class single-label |
| Đầu ra | 1 xác suất | Vector xác suất tổng = 1 |
| Quan hệ | Tương đương softmax 2 lớp | Tổng quát hóa tương tự sigmoid |
| Lưu ý | Dùng cho multi-label (mỗi nhãn độc lập) | Dùng khi các lớp loại trừ lẫn nhau |

## 6) Các điểm dễ nhầm trong thực hành

1. Không nên gọi `sigmoid` rồi đưa vào `BCEWithLogitsLoss` (vì hàm này đã gồm sigmoid bên trong).
2. Sigmoid ở lớp ẩn có thể gây vanishing gradient; thường ưu tiên ReLU/GELU trong hidden layers.
3. Ngưỡng 0.5 không luôn tối ưu khi dữ liệu lệch lớp; cần tune threshold theo metric business.

## 7) Reader checklist

- [ ] Mình hiểu vì sao sigmoid map được về (0,1).
- [ ] Mình tính được $\sigma(z)$ cho vài giá trị cơ bản (âm, 0, dương).
- [ ] Mình giải thích được vì sao sigmoid + BCE hợp với binary classification.
- [ ] Mình biết khi nào nên dùng sigmoid, khi nào nên dùng softmax.

## Liên kết

- [[Softmax Function]]
- [[Cross-Entropy Loss]]
- [[Maximum Likelihood Estimation]]

## TODO

- [ ] Thêm một ví dụ threshold tuning (precision-recall trade-off) trên dữ liệu lệch lớp.
- [ ] Thêm section "calibration" và Platt scaling/temperature scaling cho output sigmoid.
