---
title: "Batch Normalization"
aliases: ["BatchNorm", "BN", "Batch Norm", "chuẩn hóa theo batch"]
tags:
  [concept, deep-learning, normalization, training-stability, regularization]
created: 2026-04-11
---

# Batch Normalization (Chuẩn hóa theo Batch)

> [!NOTE] ELI5
> Tưởng tượng bạn đang nấu ăn và mỗi nguyên liệu có độ mặn rất khác nhau — muối thì rất mặn, đường thì không mặn chút nào. **Batch Normalization** giống như việc cân chỉnh lại mọi nguyên liệu về cùng một thang đo trước khi cho vào nồi. Nhờ đó, "lò nấu" (mạng neural) không bị rối loạn vì chênh lệch quá lớn giữa các thành phần, và học được nhanh hơn, ổn định hơn.

**Batch Normalization (BN)** là kỹ thuật chuẩn hóa các activation bên trong mạng neural trong quá trình training. Với mỗi mini-batch, BN tính mean và variance từ batch đó, dùng chúng để chuẩn hóa activations về phân phối chuẩn $\mathcal{N}(0, 1)$, sau đó rescale và shift bằng 2 learnable parameters $\gamma$ (scale) và $\beta$ (shift). Mục đích: ổn định quá trình training, cho phép dùng learning rate lớn hơn, và giảm nhạy cảm với weight initialization.

## Cơ chế toán học

Với một mini-batch $\mathcal{B} = \{x_1, x_2, \ldots, x_m\}$:

**Bước 1 — Tính batch statistics:**
$$\mu_\mathcal{B} = \frac{1}{m} \sum_{i=1}^{m} x_i, \quad \sigma^2_\mathcal{B} = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_\mathcal{B})^2$$

**Bước 2 — Standardize:**
$$\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma^2_\mathcal{B} + \epsilon}}$$

**Bước 3 — Scale & Shift (learnable):**
$$y_i = \gamma \hat{x}_i + \beta$$

Trong đó:

- $\epsilon$ — hằng số nhỏ (thường $10^{-5}$) tránh chia cho 0
- $\gamma, \beta$ — learnable parameters, cho phép mô hình "học lại" phân phối tối ưu
- $\gamma$ khởi tạo = 1, $\beta$ khởi tạo = 0

## Training vs Inference

| Mode          | Mean/Var dùng                                                | Lý do                                    |
| ------------- | ------------------------------------------------------------ | ---------------------------------------- |
| **Training**  | Batch statistics ($\mu_\mathcal{B}$, $\sigma^2_\mathcal{B}$) | Tính từ mini-batch hiện tại              |
| **Inference** | Running statistics (EMA)                                     | Batch size = 1 không có batch statistics |

**Running statistics** được cập nhật trong training bằng Exponential Moving Average:
$$\mu_{run} \leftarrow (1 - \alpha) \mu_{run} + \alpha \mu_\mathcal{B}$$

> [!WARNING] Lỗi phổ biến
> Quên gọi `model.eval()` trước inference → mô hình vẫn dùng batch statistics → kết quả không ổn định khi test.

## BN cho FC vs Convolutional Layers

| Layer type | Normalize theo chiều          | Lý do                                            |
| ---------- | ----------------------------- | ------------------------------------------------ |
| FC         | dim=0 (batch)                 | Mỗi feature là 1 scalar                          |
| Conv       | dim=(0, 2, 3) (batch + H + W) | Cùng channel = cùng feature map → normalize cùng |

Với Conv: mỗi channel có 1 cặp $(\gamma_c, \beta_c)$ riêng → số params BN = $2 \times C$ (rất ít).

## Hiệu ứng Regularization

BN dùng **batch statistics** (nhiễu từ mini-batch ngẫu nhiên) → đưa nhiễu vào mạng → hoạt động như một dạng regularization tương tự [[Dropout]]. Tuy nhiên, đây là **side effect**, không phải mục đích chính.

## Liên kết

- Đã học chi tiết ở [[Buổi 33 - Tuần 9]]
- Được sử dụng trong [[Buổi 34 - Tuần 9]] (ResNet — pre-activation: BN → ReLU → Conv)
- Được sử dụng trong [[Buổi 35 - Tuần 9]] (DenseNet — conv block)
- Liên quan: [[Dropout]], [[Activation Function]]
- Source: [d2l.ai — 8.5 Batch Normalization](https://d2l.ai/chapter_convolutional-modern/batch-norm.html)

---

> [!TODO]
>
> - So sánh chi tiết BN vs Layer Normalization vs Group Normalization vs Instance Normalization
> - Phân tích tại sao BN giúp "landscape smoothing" (Li et al., 2018)
> - Tranh cãi về Internal Covariate Shift — BN thực sự giải quyết vấn đề gì?
> - Xu hướng: tại sao Transformer dùng Layer Norm thay BN
