---
title: "Cross-Validation"
aliases: [K-fold cross-validation, cross validation, kiểm chứng chéo, CV]
tags: [concept, machine-learning, evaluation, model-selection]
created: 2026-03-18
---

# Cross-Validation

> [!NOTE] ELI5
> Bạn có 50 bài tập để ôn thi. Thay vì luôn dùng 40 bài để học và 10 bài để kiểm tra, bạn **chia thành 5 phần**, mỗi lần dùng 1 phần khác nhau làm bài kiểm tra. Làm 5 lần, lấy trung bình điểm → đánh giá chính xác hơn nhiều so với chỉ kiểm tra 1 lần.

## 1. Bản chất — Tại sao cần Cross-Validation?

Khi data ít, nếu tách ra một validation set cố định, ta sẽ:

1. **Mất dữ liệu training** quý giá
2. **Kết quả đánh giá** phụ thuộc vào việc "may mắn" chọn được validation set đại diện hay không

Cross-Validation giải quyết cả hai vấn đề bằng cách **xoay vòng** phần nào là validation.

## 2. K-Fold Cross-Validation

### Quy trình

1. Chia toàn bộ training data thành $K$ phần bằng nhau (gọi là "folds")
2. Lặp $K$ lần:
   - Lần thứ $i$: dùng fold $i$ làm validation, $K-1$ folds còn lại làm training
   - Train mô hình, đo lỗi trên fold $i$ → được $\text{Score}_i$
3. Kết quả cuối: trung bình $K$ scores

$$\text{CV Score} = \frac{1}{K} \sum_{i=1}^{K} \text{Score}_i$$

### Giá trị K phổ biến

| K | Tên gọi | Ưu điểm | Nhược điểm |
| --- | --- | --- | --- |
| 5 | 5-fold CV | Cân bằng giữa tính toán và độ tin cậy | Phổ biến nhất |
| 10 | 10-fold CV | Đánh giá ổn định hơn | Tốn gấp đôi thời gian so với 5-fold |
| $n$ | Leave-One-Out (LOO) | Dùng tối đa data để train | Cực kỳ tốn tính toán |

## 3. Ưu điểm

- **Tận dụng tối đa data**: mỗi mẫu đều được dùng cả để train lẫn validate
- **Ước lượng ổn định**: trung bình nhiều lần giảm phương sai
- **Phát hiện overfitting**: nếu scores dao động mạnh → mô hình không ổn định

## 4. Nhược điểm

- **Tốn tính toán**: phải train mô hình $K$ lần
- **Bias nhẹ**: mỗi lần chỉ train trên $\frac{K-1}{K}$ data (ít hơn full training set)
- **Không phù hợp** data có thứ tự thời gian (cần Time Series CV thay thế)

## 5. Khi nào dùng?

- ✅ Data ít (< vài nghìn mẫu)
- ✅ Cần so sánh nhiều mô hình khác nhau
- ✅ Muốn ước lượng generalization error đáng tin cậy
- ❌ Data rất lớn (tách validation set cố định đã đủ)
- ❌ Training rất tốn thời gian (deep learning lớn)

## 6. Ứng dụng trong DL

Trong deep learning, K-fold CV ít được dùng vì:
- Training mỗi model mất hàng giờ/ngày
- Dataset thường đủ lớn để split cố định

Tuy nhiên, K-fold vẫn hữu ích khi:
- Fine-tune trên dataset nhỏ
- Medical imaging (data ít, mỗi mẫu quý)
- Kaggle competitions (squeeze thêm performance)

## TODO

- [ ] Thêm code example K-fold với sklearn
- [ ] Liên kết với Stratified K-Fold
- [ ] Thêm Time Series Cross-Validation
