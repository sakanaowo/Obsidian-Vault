---
type: concept
title: Mean Squared Error
aliases:
  - MSE
  - L2 Loss
  - Squared Error Loss
tags:
  - ai
  - machine-learning
  - loss-functions
---

**Mean Squared Error (MSE)** là một trong những hàm loss cơ bản nhất trong machine learning, đo **trung bình bình phương sai số** giữa giá trị dự đoán và giá trị thực:

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

Trong đó:
- $y_i$: giá trị thực (ground truth)
- $\hat{y}_i$: giá trị dự đoán
- $n$: số lượng samples

**Đặc điểm của MSE**

1. **Nhạy với outliers**: Vì lấy bình phương, các sai số lớn bị phạt nặng hơn nhiều so với sai số nhỏ. Ví dụ, sai số 10 đóng góp 100 vào loss, trong khi sai số 1 chỉ đóng góp 1.

2. **Luôn không âm**: MSE ≥ 0, và MSE = 0 khi và chỉ khi dự đoán hoàn hảo.

3. **Liên hệ với Gaussian likelihood**: Tối thiểu MSE tương đương với tối đa hóa log-likelihood của Gaussian distribution với variance cố định.

**MSE trong Masked Autoencoders**

[[Masked Autoencoders (MAE)]] sử dụng MSE để đo sai số giữa pixel tái tạo và pixel gốc:

$$\mathcal{L}_{\text{MAE}} = \frac{1}{|M|} \sum_{i \in M} \| x_i - \hat{x}_i \|^2$$

Điểm đặc biệt: MAE **chỉ tính loss trên các patch bị mask** ($M$), không tính trên patch nhìn thấy. Paper giải thích: tính loss trên toàn bộ pixel làm giảm accuracy ~0.5%. Trực giác: loss trên patch nhìn thấy là "nhiễu" — encoder đã thấy chúng rồi, reconstruction trivial.

**So sánh với các loss khác**

| Loss | Công thức | Đặc điểm |
|------|-----------|---------|
| MSE (L2) | $(y - \hat{y})^2$ | Nhạy outliers, smooth gradient |
| MAE (L1) | $\|y - \hat{y}\|$ | Robust hơn với outliers |
| Huber | Kết hợp L1 + L2 | Smooth cho small errors, robust cho large |
| Cross-Entropy | $-\log \hat{p}$ | Dùng cho classification, discrete outputs |

**Normalized pixel reconstruction**

Paper MAE cũng khảo sát biến thể **per-patch normalization**: chuẩn hóa pixel trong mỗi patch về mean=0, std=1 trước khi tính MSE. Kết quả:

| Target | Fine-tuning | Linear Probing |
|--------|-------------|----------------|
| Pixel (w/o norm) | 84.9% | 73.5% |
| Pixel (w/ norm) | **85.4%** | **73.9%** |

Normalization tăng contrast cục bộ và giữ high-frequency components, giúp cải thiện cả fine-tuning và linear probing.
