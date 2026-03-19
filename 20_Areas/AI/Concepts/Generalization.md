---
title: "Generalization"
aliases: [khả năng tổng quát hóa, out-of-sample performance, khả năng khái quát]
tags: [concept, machine-learning, evaluation, generalization]
created: 2026-03-16
updated: 2026-03-18
---

# Generalization

> [!NOTE] ELI5
> Học thuộc lòng bài tập cũ chưa chắc làm được bài mới. Trong ML cũng vậy: mô hình tốt không phải chỉ đúng trên dữ liệu đã thấy, mà còn phải đúng trên **dữ liệu mới chưa từng gặp**. Khả năng đó gọi là generalization.

## 1. Bản chất — Tại sao generalization là vấn đề cốt lõi?

Machine learning (và rộng hơn là khoa học) luôn phải đối mặt với câu hỏi: **khi nào ta được phép khái quát từ những quan sát cụ thể sang quy luật chung?**

- Ta không muốn dự đoán giá cổ phiếu **hôm qua** — mà là **ngày mai**
- Ta không cần chẩn đoán lại bệnh nhân **đã biết** — mà là bệnh nhân **mới**

**Generalization** chính là khả năng đạt lỗi thấp trên phân phối dữ liệu ngoài tập huấn luyện.

## 2. Các khái niệm liên quan

### [[Training Error vs Generalization Error]]

- **Training error** ($R_{\text{emp}}$): trung bình lỗi trên training set — tính được chính xác
- **Generalization error** ($R$): kỳ vọng lỗi trên toàn bộ phân phối — **không thể tính chính xác**, chỉ ước lượng qua validation/test set

### [[Overfitting and Underfitting]]

- **Overfitting**: mô hình quá phức tạp → training error thấp nhưng generalization error cao
- **Underfitting**: mô hình quá đơn giản → cả hai đều cao

### [[Bias-Variance Tradeoff]]

$$\text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

- Mô hình đơn giản = high bias, low variance
- Mô hình phức tạp = low bias, high variance
- Cần tìm **sweet spot** để minimize tổng lỗi

## 3. Các yếu tố ảnh hưởng

### Model Complexity (Độ phức tạp mô hình)

Mô hình phức tạp hơn (nhiều tham số, phạm vi tham số rộng) có khả năng fit training data tốt hơn, nhưng dễ overfit hơn. Triết gia Karl Popper: *một lý thuyết giải thích được mọi thứ thì thực ra không giải thích được gì cả.*

### Dataset Size (Kích thước dữ liệu)

- Data ít → dễ overfit
- Data nhiều → generalization error giảm
- Quy tắc: model complexity không nên tăng nhanh hơn lượng data

### Regularization

Kỹ thuật hạn chế sự phức tạp của mô hình:
- **L2 (Weight Decay)**: phạt trọng số lớn → giữ mô hình "mượt"
- **L1 (Lasso)**: đẩy trọng số về 0 → feature selection
- **Dropout**, **Early stopping**, **Data augmentation**

## 4. Đánh giá thực tế

1. Tách data thành **train/validation/test**
2. Train trên training set
3. Chọn mô hình tốt nhất dựa trên **validation error**
4. Đánh giá cuối cùng trên **test set** (chỉ 1 lần)
5. Khi data ít: dùng [[Cross-Validation|K-fold cross-validation]]

## 5. Lưu ý quan trọng trong Deep Learning

> [!IMPORTANT]
> Trong DL, mô hình tốt nhất thường vẫn có gap giữa training error và validation error. Gap ≠ xấu — điều quan trọng là **generalization error có đủ thấp** cho bài toán thực tế hay không.

Các mạng neural sâu có khả năng fit arbitrary labels (memorize hoàn toàn), nhưng trong thực tế vẫn generalize tốt. Đây là một hiện tượng mà lý thuyết truyền thống chưa giải thích trọn vẹn.

## TODO

- [ ] Bổ sung PAC learning/Rademacher complexity ở mức nhập môn
- [ ] Liên kết với double descent trong mô hình lớn
- [ ] Thêm distribution shift / domain adaptation
