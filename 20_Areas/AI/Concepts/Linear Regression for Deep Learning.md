---
title: "Linear Regression for Deep Learning"
aliases: [linear regression, hồi quy tuyến tính, least squares]
tags: [concept, deep-learning, machine-learning, optimization, statistics]
created: 2026-03-16
session: "D2L Tuần 3, Buổi 10 — Linear Regression"
source:
  - "30_Resources/Books/Dive Into Deep Learning/d2l-en/chapter_linear-regression/linear-regression.md"
  - "30_Resources/Books/Dive Into Deep Learning/d2l-en/chapter_linear-regression/linear-regression-scratch.md"
related:
  - "[[Maximum Likelihood Estimation]]"
  - "[[Gradient Descent]]"
  - "[[Generalization]]"
---

# Linear Regression for Deep Learning

> [!NOTE] ELI5
> Bạn có một công tắc dự đoán: mỗi thông tin đầu vào (diện tích, tuổi nhà, ...) được nhân với một "mức quan trọng", rồi cộng lại để ra kết quả. Học mô hình tức là chỉnh các mức quan trọng đó sao cho dự đoán gần đúng nhất. Sai số dùng bình phương để phạt lỗi lớn nặng hơn. Đây là mô hình nền tảng để hiểu cơ chế huấn luyện mạng sâu.

## 1. Bản chất (First Principles)

**Linear Regression** tìm một ánh xạ affine từ đặc trưng sang giá trị số:

$$
\hat{y}=\mathbf{w}^\top\mathbf{x}+b
$$

Mục tiêu không phải chỉ "fit đường thẳng", mà là ước lượng **kỳ vọng có điều kiện** $E[Y\mid X=\mathbf{x}]$ dưới giả định quan hệ gần tuyến tính. A đúng vì B, được thể hiện qua C:

- A: Mô hình tuyến tính có ý nghĩa thống kê.
- B: Khi giả định nhiễu cộng zero-mean, trung bình của $Y$ quanh mặt phẳng tuyến tính là thứ cần học.
- C: D2L diễn giải trực tiếp $E[Y\mid X=\mathbf{x}]$ là weighted sum của feature.

## 2. Cơ chế tối ưu

Ta tối thiểu hóa squared loss:

$$
L(\mathbf{w},b)=\frac{1}{n}\sum_{i=1}^n\frac{1}{2}(\hat{y}^{(i)}-y^{(i)})^2
$$

Gradient theo tham số cho phép cập nhật lặp:

$$
\theta\leftarrow\theta-\eta\nabla_\theta L
$$

với $\theta=(\mathbf{w},b)$. Trong thực tế ta dùng minibatch để cân bằng giữa tốc độ tính toán và độ ổn định gradient.

## 3. Nghiệm đóng vs nghiệm lặp

### 3.1 Nghiệm đóng (Normal Equation)

$$
\mathbf{w}^*=(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}
$$

Điều kiện: $\mathbf{X}^\top\mathbf{X}$ khả nghịch (cột độc lập tuyến tính).

### 3.2 Tối ưu lặp (Minibatch SGD)

$$
(\mathbf{w},b)\leftarrow(\mathbf{w},b)-\frac{\eta}{|\mathcal{B}|}\sum_{i\in\mathcal{B}_t}\partial_{(\mathbf{w},b)}l^{(i)}
$$

Trong DL, cách lặp là chuẩn vì mô hình thường không có nghiệm đóng khả thi.

## 4. Góc nhìn xác suất: MSE = MLE (Gaussian noise)

Nếu:

$$
y=\mathbf{w}^\top\mathbf{x}+b+\epsilon,\quad \epsilon\sim\mathcal{N}(0,\sigma^2)
$$

thì tối đa hóa likelihood tương đương tối thiểu hóa tổng bình phương sai số (khác nhau bởi hằng số và hệ số tỉ lệ khi $\sigma$ cố định).

Ý nghĩa: lựa chọn MSE không chỉ vì tiện đạo hàm, mà còn có nền tảng xác suất rõ ràng.

## 5. Ví dụ thực tế ngắn

Bài toán định giá nhà:

- Feature: diện tích, số phòng, tuổi nhà.
- Label: giá bán.
- Dự đoán: $\hat{y}=w_1x_1+w_2x_2+w_3x_3+b$.

Nếu dữ liệu có outlier lớn, squared loss sẽ phạt mạnh các điểm này, có thể làm mô hình nhạy quá mức. Khi đó có thể cân nhắc loss robust (ví dụ Huber), như bài tập mở rộng trong D2L.

## 6. Liên hệ với mạng nơ-ron

Linear regression có thể nhìn như mạng nơ-ron 1 tầng fully-connected, không hidden layer, đầu ra scalar. Đây là "phiên bản tối giản" của hầu hết pipeline huấn luyện trong deep learning: model, loss, optimizer, training loop.

## 7. Khi nào nên và không nên dùng

Nên dùng khi:

- Cần baseline mạnh, dễ giải thích.
- Quan hệ gần tuyến tính hoặc đã feature engineering tốt.

Không nên dùng đơn lẻ khi:

- Quan hệ phi tuyến mạnh.
- Dữ liệu có cấu trúc phức tạp (ảnh, chuỗi, ngôn ngữ) chưa qua biến đổi đặc trưng phù hợp.

## TODO

- [ ] Bổ sung mục so sánh sâu giữa MSE, MAE, Huber dưới các mô hình nhiễu khác nhau.
- [ ] Thêm ví dụ định lượng về multicollinearity và ảnh hưởng đến $(X^TX)^{-1}$.
- [ ] Liên kết sang note thực nghiệm với `linear-regression-concise.md`.
