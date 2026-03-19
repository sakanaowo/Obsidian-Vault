---
title: "Softmax Function"
aliases: [softmax, hàm softmax, normalized exponential, softargmax]
tags: [concept, machine-learning, activation-function, classification]
created: 2026-03-19
---

# Softmax Function

> [!NOTE] ELI5
> Bạn có 3 điểm thi thô (chưa chuẩn hóa): Toán 8, Lý 5, Hóa 2. Softmax biến chúng thành **phần trăm**: Toán ~88%, Lý ~10%, Hóa ~2%. Tổng luôn bằng 100%. Điểm cao nhất chiếm tỉ lệ lớn nhất, nhưng các điểm khác vẫn có "cơ hội" nhỏ — không bị gạt bỏ hoàn toàn.

## 1. Bản chất — Tại sao cần Softmax?

Sau khi mô hình tuyến tính tính ra các **logits** (điểm thô) $o_1, o_2, \ldots, o_q$, ta gặp hai vấn đề:

1. Logits **có thể âm** → không phải xác suất hợp lệ
2. Logits **không cộng lại bằng 1** → không thể hiểu là phân phối xác suất

Softmax giải quyết cả hai bằng cách: **(1)** dùng $e^x$ để biến mọi giá trị thành dương, **(2)** chia cho tổng để chuẩn hóa.

## 2. Công thức

$$\hat{y}_i = \text{softmax}(\mathbf{o})_i = \frac{\exp(o_i)}{\sum_{j=1}^q \exp(o_j)}$$

| Ký hiệu | Nghĩa |
| --- | --- |
| $o_i$ | Logit (điểm thô) cho class $i$ |
| $\exp(o_i)$ | $e^{o_i}$ — luôn dương |
| $\sum_j \exp(o_j)$ | Tổng tất cả $e^{o_j}$ — dùng để chuẩn hóa |
| $\hat{y}_i$ | Xác suất ước lượng cho class $i$ |

### Tính chất

- $\hat{y}_i \in (0, 1)$ cho mọi $i$
- $\sum_{i=1}^q \hat{y}_i = 1$
- **Giữ nguyên thứ tự**: $o_i > o_j \Rightarrow \hat{y}_i > \hat{y}_j$
- Vì giữ thứ tự: $\arg\max_j \hat{y}_j = \arg\max_j o_j$ (không cần tính softmax để biết class thắng)

## 3. Ví dụ số cụ thể

Cho $\mathbf{o} = (2.0, 1.0, 0.1)$:

| Bước | $o_1 = 2.0$ | $o_2 = 1.0$ | $o_3 = 0.1$ |
| --- | --- | --- | --- |
| $\exp(o_i)$ | 7.39 | 2.72 | 1.11 |
| $\hat{y}_i$ | $\frac{7.39}{11.22}$ = 0.659 | $\frac{2.72}{11.22}$ = 0.242 | $\frac{1.11}{11.22}$ = 0.099 |
| Phần trăm | **65.9%** | 24.2% | 9.9% |

## 4. Tại sao dùng $e^x$? (Lịch sử)

Ý tưởng softmax bắt nguồn từ **vật lý thống kê** (Boltzmann, 1868). Trong nhiệt động lực học, xác suất một trạng thái năng lượng $E$ tỉ lệ với $\exp(-E / kT)$, trong đó $T$ là **nhiệt độ**.

Trong ML, khi nói "tăng/giảm temperature" của softmax, ta nhân logits với $\frac{1}{T}$:

$$\hat{y}_i = \frac{\exp(o_i / T)}{\sum_j \exp(o_j / T)}$$

- **$T$ nhỏ** → softmax trở nên "sắc" (sharp) → gần argmax
- **$T$ lớn** → softmax trở nên "phẳng" (flat) → gần uniform distribution
- Kỹ thuật này được dùng trong **knowledge distillation** và **LLM text generation**

## 5. Numerical Stability

> [!WARNING] Vấn đề tràn số
> Nếu $o_i$ rất lớn (ví dụ 1000), $\exp(1000)$ sẽ **overflow** (vượt giới hạn float). Framework giải quyết bằng cách trừ đi max trước: $\text{softmax}(o_i - \max_j o_j)$ — kết quả toán học không đổi nhưng tránh overflow.

## 6. So sánh Softmax vs Sigmoid

| | Sigmoid | Softmax |
| --- | --- | --- |
| **Số classes** | 2 (binary) | $q$ (multi-class) |
| **Công thức** | $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\frac{e^{o_i}}{\sum_j e^{o_j}}$ |
| **Output** | 1 số trong $(0,1)$ | Vector trong $(0,1)^q$, tổng = 1 |
| **Quan hệ** | Sigmoid = Softmax với 2 classes | Softmax = generalized sigmoid |

Thực tế, với 2 classes: $\text{softmax}(o_1, o_2)_1 = \sigma(o_1 - o_2)$.

## 7. Ứng dụng trong DL

- **Output layer** của mọi classification model
- **Attention mechanisms**: tính attention weights trong [[Transformer Architecture|Transformers]]
- **RL**: policy networks dùng softmax để chọn action
- **LLM**: sampling token tiếp theo

## TODO

- [ ] Thêm code PyTorch implementation
- [ ] So sánh softmax vs sparsemax
- [ ] Liên kết với Gumbel-Softmax trick
