---
title: "Overfitting and Underfitting"
aliases: [overfitting, underfitting, quá khớp, kém khớp, học vẹt, học chưa đủ]
tags: [concept, machine-learning, generalization, training]
created: 2026-03-18
---

# Overfitting and Underfitting

> [!NOTE] ELI5
> **Underfitting** giống như bạn chỉ đọc lướt sách giáo khoa 5 phút rồi đi thi — chưa học đủ nên cái gì cũng sai. **Overfitting** giống như bạn học thuộc lòng từng dấu chấm phẩy trong sách nhưng gặp đề mới là bó tay — vì bạn nhớ chi tiết thay vì hiểu bản chất.

## 1. Bản chất — Tại sao xảy ra?

Mọi mô hình ML đều phải cân bằng giữa hai thái cực:

- **Quá đơn giản** → không nắm được quy luật trong dữ liệu → **underfitting**
- **Quá phức tạp** → nắm luôn cả nhiễu ngẫu nhiên → **overfitting**

Đây chính là biểu hiện thực tế của [[Bias-Variance Tradeoff]].

## 2. Underfitting (Kém khớp)

### Dấu hiệu

| Chỉ số | Giá trị |
| --- | --- |
| Training error | **Cao** |
| Validation error | **Cao** |
| Gap (val - train) | **Nhỏ** |

### Nguyên nhân

1. Mô hình quá đơn giản (ví dụ: dùng đường thẳng fit dữ liệu hình sin)
2. Features không đủ thông tin
3. Train quá ít epochs
4. Regularization quá mạnh

### Ví dụ cụ thể

Dữ liệu thật tuân theo $y = x^2 + \text{noise}$. Nếu dùng mô hình tuyến tính $\hat{y} = wx + b$ (bậc 1), đường thẳng không bao giờ uốn cong được → lỗi lớn trên cả train lẫn test.

### Cách khắc phục

- Tăng độ phức tạp mô hình (thêm layers, thêm features)
- Giảm regularization
- Train lâu hơn
- Thêm features phù hợp

## 3. Overfitting (Quá khớp)

### Dấu hiệu

| Chỉ số | Giá trị |
| --- | --- |
| Training error | **Rất thấp** (gần 0) |
| Validation error | **Cao** |
| Gap (val - train) | **Rất lớn** |

### Nguyên nhân

1. Mô hình quá phức tạp so với lượng dữ liệu
2. Train quá nhiều epochs (mô hình bắt đầu "nhớ" training data)
3. Dữ liệu train quá ít hoặc thiếu đa dạng
4. Không có regularization

### Ví dụ cụ thể

Cùng dữ liệu $y = x^2 + \text{noise}$. Nếu dùng polynomial bậc 15 với chỉ 10 điểm dữ liệu, đường cong sẽ đi qua **mọi điểm** (kể cả nhiễu), tạo ra đường ngoằn ngoèo kỳ lạ. Training error ≈ 0 nhưng dự đoán trên điểm mới sẽ sai lệch rất nhiều.

### Cách khắc phục

- Thêm dữ liệu (More Data)
- Regularization: L1 (Lasso), L2 (Ridge/Weight Decay), Dropout
- Early stopping (dừng train khi validation error bắt đầu tăng)
- Data augmentation
- Giảm độ phức tạp mô hình
- [[Cross-Validation]]

## 4. So sánh tổng hợp

| | Underfitting | Good Fit | Overfitting |
| --- | --- | --- | --- |
| Train Error | Cao | Thấp | Rất thấp |
| Val/Test Error | Cao | Thấp | Cao |
| Gap | Nhỏ | Nhỏ | Lớn |
| Bias | Cao | Vừa | Thấp |
| Variance | Thấp | Vừa | Cao |
| Mô hình | Quá đơn giản | Vừa đủ | Quá phức tạp |

## 5. Ứng dụng thực tế trong DL

> [!IMPORTANT] Lưu ý quan trọng từ D2L
> Trong deep learning, overfitting **không phải lúc nào cũng xấu**. Các mô hình tốt nhất thường có training error thấp hơn nhiều so với validation error. Điều quan trọng là **generalization error** có đủ thấp hay không, chứ không phải gap có bằng 0 hay không.

- **Early stopping**: theo dõi validation loss, dừng khi nó bắt đầu tăng
- **Dropout**: tắt ngẫu nhiên neurons khi train → ép mô hình không phụ thuộc vào bất kỳ neuron đơn lẻ nào
- **Weight decay** (L2 regularization): phạt trọng số lớn, giữ mô hình "đơn giản"
- **Data augmentation**: tạo thêm dữ liệu giả từ dữ liệu thật

## TODO

- [ ] Thêm code visualization overfitting với polynomial
- [ ] Liên kết với double descent phenomenon
