---
title: "Buổi 10 - Tuần 3: Linear Regression (D2L)"
tags: [d2l, linear-regression, deep-learning, study-note]
created: 2026-03-16
session: "D2L Tuần 3, Buổi 10 — Linear Regression"
source:
  - "30_Resources/Books/Dive Into Deep Learning/d2l-en/chapter_linear-regression/linear-regression.md"
  - "30_Resources/Books/Dive Into Deep Learning/d2l-en/chapter_linear-regression/linear-regression-scratch.md"
related:
  - "[[Linear Regression for Deep Learning]]"
  - "[[Maximum Likelihood Estimation]]"
  - "[[Gradient Descent]]"
---

# Buổi 10 - Tuần 3: Linear Regression

> [!NOTE] ELI5
> Hãy tưởng tượng em cần đoán giá nhà nhưng chưa biết công thức. Em lấy từng thông tin như diện tích, số phòng, tuổi nhà, cho mỗi cái một "điểm" rồi cộng lại để ra giá dự đoán. Nếu đoán sai, em sửa điểm đó một chút. Lặp lại nhiều lần, em sẽ đoán ngày càng đúng hơn. Linear Regression chính là cách học "cho điểm" như vậy.

## 1. Cách đọc file này (quan trọng)

Nếu bạn đang thấy "không hiểu gì", hãy đọc theo đúng thứ tự sau:

1. Đọc phần 2 để nắm từ vựng (dịch chữ + nghĩa kỹ thuật).
2. Đọc phần 3 và 4 để biết bài toán là gì, mô hình làm gì.
3. Đọc phần 5 để hiểu "thế nào là sai" (loss).
4. Đọc phần 6 để hiểu cách sửa sai (SGD) và cách giải trực tiếp (Normal Equation).
5. Đọc phần 7 và 8 để thấy ví dụ số cụ thể, không còn mơ hồ.
6. Đọc phần 9 để nối với code PyTorch.

## 2. Mục tiêu buổi học

Buổi này bám nội dung d2l.ai nhưng trình bày lại theo hướng dễ hiểu:

1. Hiểu bài toán hồi quy là gì.
2. Hiểu từng khái niệm xuất hiện trong chương.
3. Nắm công thức cốt lõi, biết rõ từng ký hiệu nghĩa là gì.
4. Biết vì sao MSE, SGD và Normal Equation cùng xuất hiện.
5. Biết cách dịch từ tiếng Anh sang nghĩa kỹ thuật để không bị "học vẹt thuật ngữ".

## 3. Từ điển khái niệm trong buổi này

> [!NOTE] ELI5
> Bạn hãy xem bảng này như "từ điển mini". Cùng một từ tiếng Anh, có thể có nghĩa dịch chữ và nghĩa kỹ thuật khác nhau. Nếu không tách hai lớp nghĩa này, rất dễ hiểu lầm.

| Thuật ngữ                  | Dịch nghĩa từ ngữ                                       | Giải nghĩa trong ngữ cảnh bài                                    | Ví dụ nhanh                                |
| -------------------------- | ------------------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------ |
| **Regression**             | Hồi quy (quay về/ước lượng theo xu hướng)               | Dạng bài toán dự đoán **một số thực liên tục**                   | Dự đoán giá nhà: 2.35 tỷ                   |
| **Feature**                | Đặc trưng (đặc điểm đo được)                            | Biến đầu vào mà mô hình dùng để dự đoán                          | diện tích, số phòng                        |
| **Label / Target**         | Nhãn / Mục tiêu                                         | Giá trị đúng mô hình phải học để dự đoán                         | giá bán thật của căn nhà                   |
| **Model**                  | Mô hình (mẫu/công thức)                                 | Hàm ánh xạ từ feature sang dự đoán                               | $\hat y = w^Tx+b$                          |
| **Weight ($w$)**           | Trọng số (mức nặng nhẹ)                                 | Hệ số cho biết feature ảnh hưởng mạnh hay yếu                    | $w_{area}$ lớn -> diện tích ảnh hưởng mạnh |
| **Bias ($b$)**             | Độ lệch / Hệ số chệch                                   | Hằng số bù để dịch dự đoán lên/xuống toàn cục                    | mọi dự đoán cộng thêm 0.2 tỷ               |
| **Prediction ($\hat y$)**  | Giá trị dự đoán (ước lượng)                             | Kết quả mô hình trả ra trước khi so với nhãn thật                | mô hình đoán 2.1 tỷ                        |
| **Loss**                   | Hao hụt / Mất mát                                       | Thước đo độ sai giữa dự đoán và giá trị thật                     | dự đoán sai càng nhiều, loss càng cao      |
| **MSE**                    | Mean Squared Error (trung bình bình phương sai số)      | Loss chuẩn cho hồi quy: phạt mạnh sai số lớn                     | sai 10 -> bình phương thành 100            |
| **Gradient**               | Đạo hàm dốc / véc-tơ độ dốc                             | Chỉ hướng loss tăng nhanh nhất theo tham số                      | đi ngược gradient để giảm loss             |
| **Learning Rate ($\eta$)** | Tốc độ học / bước học                                   | Độ lớn mỗi bước cập nhật tham số                                 | quá lớn dễ dao động, quá nhỏ học chậm      |
| **Minibatch**              | Lô nhỏ dữ liệu                                          | Nhóm mẫu nhỏ dùng cho một lần cập nhật                           | 64 mẫu/lần update                          |
| **SGD**                    | Stochastic Gradient Descent (hạ dốc ngẫu nhiên)         | Cập nhật tham số dựa trên gradient của lô dữ liệu nhỏ            | lặp: forward -> loss -> backward -> step   |
| **Normal Equation**        | Phương trình chuẩn                                      | Công thức nghiệm đóng của linear regression                      | $w^*=(X^TX)^{-1}X^Ty$                      |
| **MLE**                    | Maximum Likelihood Estimation (ước lượng hợp lý tối đa) | Chọn tham số làm dữ liệu quan sát "có khả năng sinh ra cao nhất" | với noise Gaussian thì tương đương MSE     |

> [!NOTE] Cách đọc thuật ngữ
> Với mỗi mục, hãy nhớ 3 tầng: (1) tên gốc tiếng Anh, (2) dịch nghĩa từ ngữ, (3) nghĩa kỹ thuật khi dùng trong mô hình. Đây là cách tránh nhầm giữa "dịch chữ" và "nghĩa toán-học".

## 4. Bài toán hồi quy đang giải

> [!NOTE] ELI5
> Ta có một bảng dữ liệu cũ: mỗi dòng là một căn nhà, kèm đặc điểm và giá bán thật. Mục tiêu là học từ bảng cũ để sau này gặp nhà mới thì đoán được giá.

Trong D2L, hồi quy dự đoán biến liên tục: giá, nhu cầu, thời gian, mức tiêu thụ.

- **Feature**: đầu vào $\mathbf{x}$.
- **Label/Target**: đầu ra $y$.
- **Dataset**: các cặp $(\mathbf{x}^{(i)}, y^{(i)})$.

Claim: Linear Regression phù hợp khi xu hướng chính gần tuyến tính.
Reasoning: ta giả định trung bình của $Y$ khi biết $X$ có thể viết thành tổ hợp tuyến tính.
Evidence: công thức trung tâm là affine map của feature.

Ví dụ dữ liệu (rất nhỏ):

| Diện tích (m2) | Tuổi nhà (năm) | Giá thật (tỷ) |
| --- | --- | --- |
| 50 | 1 | 2.3 |
| 60 | 3 | 2.6 |
| 70 | 8 | 2.9 |

Ở đây:

- Feature là 2 cột đầu: diện tích, tuổi nhà.
- Label là cột cuối: giá thật.

## 5. Công thức mô hình (và ý nghĩa từng ký hiệu)

> [!NOTE] ELI5
> Bạn cho mỗi feature một “điểm trọng số”, cộng lại, rồi cộng thêm bias để tinh chỉnh toàn cục.

### 4.1 Công thức

Cho $\mathbf{x}\in\mathbb{R}^d$, $\mathbf{w}\in\mathbb{R}^d$:

$$
\hat{y}=\mathbf{w}^\top\mathbf{x}+b
$$

Với toàn bộ dữ liệu (design matrix $\mathbf{X}\in\mathbb{R}^{n\times d}$):

$$
\hat{\mathbf{y}}=\mathbf{X}\mathbf{w}+b
$$

- $\mathbf{w}$: mức ảnh hưởng của từng feature.
- $b$: intercept/offset.
- $\hat{y}$: dự đoán.

Giải thích từng ký hiệu theo ngôn ngữ đời thường:

- $\mathbf{x}$: "thông tin đầu vào" của một căn nhà.
- $\mathbf{w}$: "bảng hệ số chấm điểm" tương ứng với từng loại thông tin.
- $b$: "nút chỉnh toàn cục" để kéo toàn bộ dự đoán lên hoặc xuống.
- $\hat y$: giá mô hình đoán ra.

### 4.2 Vì sao cần bias?

Nếu bỏ $b$, ta chỉ mô tả các hàm đi qua gốc. Trong thực tế, nhiều quan hệ không đi qua gốc nên cần thành phần tịnh tiến $b$ để tăng năng lực biểu diễn.

Ví dụ cụ thể:

- Nếu một căn hộ cực nhỏ vẫn có giá nền tối thiểu do vị trí trung tâm, giá đó chính là phần mà $b$ đang "gánh".

## 6. Loss: tại sao dùng MSE?

> [!NOTE] ELI5
> Sai nhỏ thì phạt nhỏ, sai lớn thì phạt lớn hơn rất nhanh vì có bình phương.

### 5.1 Công thức

Với 1 mẫu:

$$
l^{(i)}=\frac{1}{2}(\hat y^{(i)}-y^{(i)})^2
$$

Với toàn bộ dữ liệu:

$$
L(\mathbf w,b)=\frac{1}{n}\sum_{i=1}^{n}\frac{1}{2}(\hat y^{(i)}-y^{(i)})^2
$$

### 5.2 Ý nghĩa trực giác

- Trơn và dễ tối ưu bằng gradient.
- Phạt mạnh outlier (vừa là điểm mạnh, vừa là điểm yếu).
- Khớp đẹp với giả định Gaussian noise (phần 8).

### 5.3 Ví dụ cụ thể

Trong định giá nhà:

- Dự đoán sai 50 triệu là đáng lo hơn rất nhiều so với sai 5 triệu.
- Bình phương biến sai số lớn thành chi phí rất lớn, khiến mô hình ưu tiên giảm những lỗi nguy hiểm.

So sánh nhanh MSE vs MAE bằng số:

- Sai số 2: MSE phạt $2^2=4$, MAE phạt $|2|=2$.
- Sai số 10: MSE phạt $10^2=100$, MAE phạt $|10|=10$.

Kết luận: MSE "ghét" lỗi lớn hơn MAE rất nhiều.

## 7. Hai cách học tham số: Nghiệm đóng vs học lặp

> [!NOTE] ELI5
> Cách 1: giải trực tiếp ra đáp án (Normal Equation). Cách 2: sửa dần từng bước (SGD). Linear regression có thể dùng cả hai.

### 6.1 Normal Equation (giải trực tiếp)

$$
\mathbf{w}^*=(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}
$$

Điều kiện quan trọng: $\mathbf{X}^\top\mathbf{X}$ phải khả nghịch.

Giải nghĩa từng thành phần:

- $\mathbf{X}$: ma trận dữ liệu, mỗi hàng là 1 mẫu, mỗi cột là 1 feature.
- $\mathbf{X}^\top\mathbf{X}$: ma trận tương quan giữa các feature.
- $(\cdot)^{-1}$: phép nghịch đảo ma trận (chỉ tồn tại khi ma trận không suy biến).

Khi dễ lỗi:

- Feature gần trùng nhau (đa cộng tuyến) -> ma trận gần suy biến -> tính nghịch đảo không ổn định.

### 6.2 Minibatch SGD (học dần)

$$
(\mathbf w,b)\leftarrow(\mathbf w,b)-\frac{\eta}{|\mathcal B|}\sum_{i\in\mathcal B_t}\partial_{(\mathbf w,b)}l^{(i)}
$$

- $\eta$: learning rate.
- $\mathcal B_t$: minibatch tại bước $t$.
- $|\mathcal B|$: kích thước minibatch.

Giải thích bằng 4 bước cố định mỗi vòng lặp:

1. Lấy một minibatch dữ liệu.
2. Tính dự đoán và loss.
3. Tính gradient qua backward.
4. Cập nhật tham số theo công thức trên.

### 6.3 Khi nào dùng cách nào?

- Dataset nhỏ, feature ít, ma trận ổn định: Normal Equation tiện.
- Dataset lớn hoặc mô hình sâu: SGD là tiêu chuẩn.
- Trong deep learning thực tế: hầu như luôn dùng optimizer dạng SGD/Adam.

## 8. Ví dụ mini bằng số (từng bước cụ thể)

> [!NOTE] ELI5
> Mình sẽ lấy 1 căn nhà, đoán thử, tính sai số, rồi xem tham số nên tăng hay giảm. Chỉ một vòng như vậy bạn sẽ thấy SGD hoạt động ra sao.

Giả sử dự đoán giá nhà theo 2 feature:

$$
\hat y = w_1\cdot \text{area}+w_2\cdot \text{age}+b
$$

Giả sử tạm thời chuẩn hóa feature để dễ tính:

- area = 0.8
- age = 0.2
- giá thật $y=2.0$

Giả sử tham số hiện tại:

- $w_1=1.0$
- $w_2=-0.5$
- $b=0.3$

Bước 1. Dự đoán:

$$
\hat y = 1.0\cdot 0.8 + (-0.5)\cdot 0.2 + 0.3 = 1.0
$$

Bước 2. Sai số:

$$
e = \hat y - y = 1.0 - 2.0 = -1.0
$$

Bước 3. Loss của mẫu:

$$
l=\frac{1}{2}e^2 = 0.5
$$

Bước 4. Hướng cập nhật trực giác:

- Sai số âm nghĩa là mô hình đoán thấp hơn thực tế.
- Cần tăng đầu ra dự đoán.
- Vì area dương và khá lớn, gradient sẽ gợi ý tăng $w_1$.
- Bias cũng có xu hướng tăng để đẩy toàn bộ dự đoán lên.

Đây là trực giác quan trọng nhất của training.

## 9. Liên hệ MSE và MLE (Gaussian noise)

> [!NOTE] ELI5
> Nếu bạn tin rằng sai số đo đạc quanh giá thật có dạng chuông Gaussian, thì tối ưu xác suất dữ liệu xảy ra sẽ dẫn đúng về MSE.

Giả sử mô hình sinh dữ liệu:

$$
y=\mathbf w^\top\mathbf x+b+\epsilon,\quad \epsilon\sim\mathcal N(0,\sigma^2)
$$

Âm log-likelihood sẽ có hạng chính là bình phương sai số, nên:

- Tối đa hóa likelihood
- Tương đương tối thiểu hóa MSE (khi $\sigma$ cố định)

Đây là lý do thống kê đứng sau loss MSE.

Nói ngắn gọn theo cấu trúc Claim/Reasoning/Evidence:

- Claim: "Dùng MSE không phải chọn bừa".
- Reasoning: Nếu nhiễu là Gaussian, log-likelihood sinh ra hạng bình phương sai số.
- Evidence: Tối đa hóa likelihood tương đương tối thiểu hóa tổng bình phương sai số.

## 10. Code skeleton dễ nhớ

1. Khởi tạo `w`, `b`.
2. Viết hàm `net(X) = X @ w + b`.
3. Viết hàm loss MSE.
4. `backward()` để lấy gradient.
5. Cập nhật tham số theo SGD.

```python
import torch

w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

def net(X):
  return X @ w + b

def loss(y_hat, y):
  return ((y_hat - y.reshape(y_hat.shape)) ** 2 / 2).mean()

def sgd(params, lr):
  with torch.no_grad():
    for p in params:
      p -= lr * p.grad
      p.grad.zero_()
```

Giải nghĩa nhanh từng hàm:

- `net(X)`: nhận dữ liệu đầu vào, trả về dự đoán.
- `loss(y_hat, y)`: đo độ sai giữa dự đoán và nhãn thật.
- `sgd(...)`: sửa tham số đi một bước nhỏ theo hướng giảm loss.

## 11. Điểm dễ nhầm

1. **Linear** và **affine** khác nhau ở bias $b$.
2. MSE tốt khi giả định noise phù hợp; không phải luôn tốt nhất cho mọi dữ liệu.
3. Train loss thấp không đồng nghĩa mô hình sẽ tốt trên dữ liệu mới.
4. Quên reset gradient trong PyTorch sẽ làm cập nhật sai.

## 12. Bài tập tự kiểm tra (kèm gợi ý tự chấm)

1. Giải thích bằng lời: vì sao cần bias?
2. Viết lại update rule SGD và nêu ý nghĩa từng ký hiệu.
3. Tại sao MSE phạt outlier mạnh hơn MAE?
4. Khi nào Normal Equation dễ gặp vấn đề số học?

Gợi ý tự chấm:

- Nếu bạn trả lời được bằng câu chữ đời thường trước, rồi mới viết công thức sau, nghĩa là bạn đã hiểu.

## 13. Kết luận ngắn

Linear Regression là mô hình đơn giản nhất nhưng chứa đủ xương sống của deep learning:

1. mô hình tham số,
2. loss,
3. gradient,
4. tối ưu,
5. đánh giá và tổng quát hóa.

Buổi sau học triển khai đầy đủ từ scratch theo d2l linear-regression-scratch.

## 14. Checklist "đã hiểu thật chưa?"

Nếu còn mơ hồ, kiểm tra 6 câu sau:

1. Bạn có phân biệt được feature và label không?
2. Bạn có giải thích được vì sao phải có bias không?
3. Bạn có nói được MSE phạt lỗi lớn mạnh hơn MAE như thế nào không?
4. Bạn có mô tả được 4 bước của một vòng SGD không?
5. Bạn có biết khi nào nên ưu tiên Normal Equation, khi nào nên dùng SGD không?
6. Bạn có diễn giải được câu "MSE gắn với giả định Gaussian noise" bằng lời thường không?

Nếu chưa qua checklist, hãy đọc lại phần 3 -> 8 trước khi qua buổi tiếp theo.
