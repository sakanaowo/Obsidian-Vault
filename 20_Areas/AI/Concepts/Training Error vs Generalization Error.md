---
title: "Training Error vs Generalization Error"
aliases: [training loss, generalization loss, empirical risk, population risk, lỗi huấn luyện, lỗi tổng quát hóa]
tags: [concept, machine-learning, evaluation, generalization]
created: 2026-03-18
---

# Training Error vs Generalization Error

> [!NOTE] ELI5
> Bạn làm thử 10 bài tập ở nhà và được 10/10. Nhưng khi vào phòng thi, bạn chỉ được 7/10 vì đề thi khác bài tập. **Training Error** là điểm bài tập ở nhà. **Generalization Error** là điểm thi thật. Cái chúng ta thật sự quan tâm là điểm thi thật.

## 1. Bản chất — Tại sao phải phân biệt?

Trong machine learning, mô hình được huấn luyện trên một **tập dữ liệu cố định** (training set). Nhưng mục tiêu thực sự không phải là "đúng trên dữ liệu cũ" mà là **"đúng trên dữ liệu mới chưa từng thấy"**.

Nếu chỉ đo lỗi trên training set, ta đang tự chấm điểm cho chính mình — kết quả sẽ **lạc quan giả tạo**. Mô hình có thể đã "học vẹt" (memorize) training data thay vì học được quy luật tổng quát.

## 2. Định nghĩa kỹ thuật

### Training Error (Lỗi huấn luyện) — $R_{\text{emp}}$

$$R_{\text{emp}}[\mathbf{X}, \mathbf{y}, f] = \frac{1}{n} \sum_{i=1}^n l(\mathbf{x}^{(i)}, y^{(i)}, f(\mathbf{x}^{(i)}))$$

| Ký hiệu | Nghĩa |
| --- | --- |
| $n$ | Số lượng mẫu trong training set |
| $\mathbf{x}^{(i)}$ | Đặc trưng (features) của mẫu thứ $i$ |
| $y^{(i)}$ | Nhãn thật (ground truth) của mẫu thứ $i$ |
| $f(\mathbf{x}^{(i)})$ | Dự đoán của mô hình cho mẫu thứ $i$ |
| $l(\cdot)$ | Hàm loss (ví dụ: MSE, Cross-entropy) |

Đây là **trung bình cộng** lỗi trên toàn bộ training set — tính được chính xác.

### Generalization Error (Lỗi tổng quát hóa) — $R$

$$R[p, f] = E_{(\mathbf{x}, y) \sim P} [l(\mathbf{x}, y, f(\mathbf{x}))] = \int \int l(\mathbf{x}, y, f(\mathbf{x})) \, p(\mathbf{x}, y) \, d\mathbf{x} \, dy$$

Đây là **kỳ vọng** (expectation) của lỗi trên **toàn bộ phân phối dữ liệu** — bao gồm cả dữ liệu chưa từng thấy. **Không thể tính chính xác** vì ta không biết $p(\mathbf{x}, y)$ thật sự.

## 3. IID Assumption

Để việc đánh giá có ý nghĩa, ta cần giả định **IID (Independent and Identically Distributed)**:

- Training data và test data đều được lấy **độc lập** từ **cùng một phân phối** $P(X, Y)$.
- Nếu phân phối thay đổi giữa train và test (distribution shift), mọi kết luận về generalization đều mất hiệu lực.

> [!WARNING] Khi nào IID bị vi phạm?
> - Dữ liệu theo thời gian (stock prices: train trên 2023, test trên 2025)
> - Dữ liệu y tế từ bệnh viện khác nhau
> - Dữ liệu ảnh train trong phòng lab, test ngoài đời thật

## 4. Generalization Gap

$$\text{Generalization Gap} = R - R_{\text{emp}}$$

- **Gap nhỏ** → mô hình generalize tốt
- **Gap lớn** → [[Overfitting and Underfitting|overfitting]] (train tốt nhưng test kém)
- **Cả hai đều cao** → [[Overfitting and Underfitting|underfitting]] (mô hình quá đơn giản)

## 5. Ứng dụng trong DL

- **Validation set** dùng để **ước lượng** generalization error trong quá trình train
- **Test set** dùng **một lần duy nhất** để đánh giá cuối cùng
- Kỹ thuật giảm gap: [[Generalization|regularization]], early stopping, data augmentation, dropout

## TODO

- [ ] Thêm ví dụ số cụ thể với Linear Regression
- [ ] Liên kết với distribution shift / domain adaptation
