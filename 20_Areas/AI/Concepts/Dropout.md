---
title: Dropout
tags: [concept, regularization, deep-learning]
created: 2026-04-02
aliases: ["Dropout Regularization", "dropout"]
---

# Dropout

> [!NOTE] ELI5
> Dropout giống việc **tắt ngẫu nhiên** một số neurons trong mạng khi training. Mỗi lần train, mạng phải **học cách hoạt động ngay cả khi thiếu vài neurons** — giống học nhóm mà mỗi buổi vắng vài bạn, ép mỗi bạn phải tự giỏi thay vì dựa vào 1 "bạn giỏi nhất". Kết quả: mạng **robust hơn**, ít overfitting.

## Cơ chế

- **Training**: Với xác suất $p$ (thường 0.5), mỗi hidden unit bị "tắt" (set = 0). Kết quả rescale bằng $\frac{1}{1-p}$ để giữ expectation.
- **Inference**: Tất cả neurons hoạt động bình thường — không dropout.

$$h' = \begin{cases} 0 & \text{với xác suất } p \\ \frac{h}{1-p} & \text{với xác suất } 1-p \end{cases}$$

## Liên kết

- Đã học chi tiết ở [[Buổi 22 - Tuần 6]]
- Sử dụng trong [[Buổi 29 - Tuần 8]] (AlexNet)
- Source: [d2l.ai — 5.6 Dropout](https://d2l.ai/chapter_multilayer-perceptrons/dropout.html)

---

> [!TODO]
> - Mở rộng phần phân tích: tại sao Dropout hoạt động (ensemble interpretation, Bayesian interpretation)
> - So sánh Dropout vs Weight Decay vs Batch Norm
> - Thêm code ví dụ implement Dropout from scratch
