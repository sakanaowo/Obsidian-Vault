---
title: "Maximum Likelihood Estimation"
aliases: [MLE, ước lượng hợp lý tối đa]
tags: [concept, statistics, machine-learning]
created: 2026-03-16
---

# Maximum Likelihood Estimation

> [!NOTE] ELI5
> Hãy xem mô hình như một "máy tạo dữ liệu". **MLE** chọn bộ tham số làm cho dữ liệu bạn quan sát được trở nên "có khả năng xuất hiện cao nhất" theo mô hình đó. Nói đơn giản: chọn tham số khiến dữ liệu thực trông hợp lý nhất.

Với dữ liệu độc lập $(x_i, y_i)$, MLE tối ưu:

$$
\hat{\theta}_{\text{MLE}}=\arg\max_\theta\prod_{i=1}^n p(y_i\mid x_i;\theta)
$$

Thường đổi sang tối thiểu hóa negative log-likelihood để tính ổn định hơn:

$$
\hat{\theta}_{\text{MLE}}=\arg\min_\theta\left[-\sum_{i=1}^n\log p(y_i\mid x_i;\theta)\right]
$$

Trong linear regression với nhiễu Gaussian cộng, tối ưu này tương đương với tối thiểu hóa MSE.

## TODO

- [ ] Thêm ví dụ so sánh MLE cho Gaussian vs Bernoulli (liên hệ MSE vs cross-entropy).
- [ ] Thêm phần bias-variance và tính nhất quán của estimator.
- [ ] Liên kết với MAP estimation và Bayesian inference.
