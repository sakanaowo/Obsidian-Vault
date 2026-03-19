---
title: "One-Hot Encoding"
aliases: [one-hot, one hot vector, mã hóa one-hot, biểu diễn one-hot]
tags: [concept, machine-learning, data-representation, classification]
created: 2026-03-19
---

# One-Hot Encoding

> [!NOTE] ELI5
> Bạn có 3 con vật: Mèo, Gà, Chó. Nếu đánh số Mèo=1, Gà=2, Chó=3 thì máy tính sẽ **hiểu nhầm** rằng Gà "lớn hơn" Mèo và Chó "lớn hơn" Gà. Thay vào đó, ta dùng dãy chỉ có 0 và 1: Mèo = [1,0,0], Gà = [0,1,0], Chó = [0,0,1]. Mỗi vị trí đại diện cho 1 loại, loại nào đang xét thì bật 1, còn lại tắt 0.

## 1. Bản chất — Tại sao không dùng số thứ tự?

Khi nhãn phân loại **không có thứ tự tự nhiên** (ví dụ: mèo, chó, gà — không con nào "lớn hơn" con nào), việc gán số thứ tự (1, 2, 3) sẽ tạo ra **quan hệ giả** mà dữ liệu không hề có. Mô hình tuyến tính sẽ lợi dụng quan hệ giả này → sai lệch kết quả.

**One-hot encoding** giải quyết vấn đề bằng cách biểu diễn mỗi nhãn thành **vector có đúng 1 phần tử bằng 1**, phần còn lại bằng 0.

## 2. Công thức

Với $q$ nhóm (classes), nhãn của class $j$ là vector $\mathbf{y} \in \{0, 1\}^q$ thỏa mãn:

$$y_i = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}$$

Ví dụ 3 classes:

| Class | One-hot vector |
| --- | --- |
| Cat | $(1, 0, 0)$ |
| Chicken | $(0, 1, 0)$ |
| Dog | $(0, 0, 1)$ |

## 3. Mối liên hệ với Softmax

Output của [[Softmax Function]] là vector xác suất $\hat{\mathbf{y}} = (\hat{y}_1, \hat{y}_2, \hat{y}_3)$ — cùng kích thước với one-hot vector. [[Cross-Entropy Loss]] so sánh hai vector này để tính lỗi.

Khi nhãn là one-hot $(0, 0, 1, 0)$, cross-entropy loss đơn giản hóa thành $-\log \hat{y}_j$ (chỉ còn 1 term vì các phần tử khác bằng 0).

## 4. Ứng dụng

- **Classification**: biểu diễn nhãn trong softmax regression, neural networks
- **NLP**: biểu diễn từ (word) trong word-level models (trước khi có embeddings)
- **Data preprocessing**: chuyển đổi categorical features (xem [[Data Preprocessing with Pandas]])

## TODO

- [ ] So sánh với label encoding, ordinal encoding
- [ ] Thêm ví dụ multi-label (nhiều 1 trong vector)
