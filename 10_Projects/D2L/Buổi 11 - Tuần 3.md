---
title: "Buổi 11 - Tuần 3: Linear Regression from Scratch (D2L)"
tags: [d2l, linear-regression, scratch, deep-learning, study-note]
created: 2026-03-16
session: "D2L Tuần 3, Buổi 11 — Linear Regression from Scratch"
source:
  - "30_Resources/Books/Dive Into Deep Learning/d2l-en/chapter_linear-regression/linear-regression-scratch.md"
  - "30_Resources/Books/Dive Into Deep Learning/d2l-en/chapter_linear-regression/synthetic-regression-data.md"
related:
  - "[[Buổi 10 - Tuần 3]]"
  - "[[Linear Regression for Deep Learning]]"
  - "[[Gradient Descent]]"
  - "[[Maximum Likelihood Estimation]]"
---

# Buổi 11 - Tuần 3: Linear Regression from Scratch

> [!NOTE] ELI5
> Buổi này giống như bạn học lái xe số sàn để hiểu máy vận hành. Sau đó bạn đi xe số tự động sẽ tự tin hơn. Ở đây cũng vậy: tự code từ đầu để hiểu bản chất training.

## 1. Nếu bạn chỉ nhớ 3 ý

1. Mô hình dự đoán bằng công thức: cộng các "điểm" từ feature.
2. Nếu đoán sai thì tính loss để biết sai bao nhiêu.
3. Dùng SGD để sửa tham số một chút sau mỗi batch.

## 2. Mục tiêu buổi này

1. Hiểu rõ từ đầu tới cuối pipeline train.
2. Tự viết được model, loss, optimizer, training loop.
3. Biết đọc output để xác nhận code chạy đúng.

## 3. Bài toán hôm nay đang làm gì?

> [!NOTE] ELI5
> Ta có nhiều căn nhà cũ với giá thật. Ta học từ dữ liệu đó để sau này gặp nhà mới thì đoán giá.

Dữ liệu mỗi dòng gồm:

- Feature: ví dụ diện tích, tuổi nhà.
- Label: giá thật.

Mục tiêu: học tham số để hàm dự đoán gần giá thật nhất.

## 4. Vì sao dùng dữ liệu synthetic trước?

> [!NOTE] ELI5
> Vì ta tự tạo đề có đáp án sẵn, nên biết chắc chương trình có học đúng không.

Ta tạo dữ liệu theo công thức:

$$
y = Xw + b + \epsilon
$$

Trong đó:

- $w, b$ là "đáp án thật" (ground truth) ta đặt trước.
- $\epsilon$ là nhiễu nhỏ để dữ liệu giống thực tế hơn.

A đúng vì B, được thể hiện qua C:

- **Claim**: Dữ liệu synthetic rất tốt để kiểm tra implementation.
- **Reasoning**: Ta biết trước tham số đúng nên dễ đối chiếu.
- **Evidence**: Sau train, nếu tham số học được gần ground truth thì code đúng.

Ví dụ số cụ thể:

$$
w=[2,-3.4]^\top,\quad b=4.2
$$

Với $x=[1,2]$:

$$
y_{clean}=2\cdot1 + (-3.4)\cdot2 + 4.2 = -0.6
$$

Nếu cộng nhiễu $\epsilon=0.01$ thì $y=-0.59$.

## 5. Model: công thức dự đoán

> [!NOTE] ELI5
> Model giống máy tính điểm: mỗi feature nhân với một trọng số, rồi cộng lại.

$$
\hat y = Xw + b
$$

Giải nghĩa ký hiệu:

- $X$: dữ liệu đầu vào của một batch.
- $w$: trọng số cần học.
- $b$: hằng số bù (bias).
- $\hat y$: giá mô hình dự đoán.

Lưu ý quan trọng:

- Shape phải khớp, nếu không loss sẽ sai.
- `b` thường được broadcast cho toàn batch.

## 6. Loss: đo sai bao nhiêu

> [!NOTE] ELI5
> Loss là thước đo sai số. Sai nhiều thì loss lớn.

Với một mẫu:

$$
l = \frac{1}{2}(\hat y - y)^2
$$

Với một batch:

$$
L = \frac{1}{B}\sum_{i=1}^{B} l_i
$$

Vì sao bình phương?

1. Không âm.
2. Trơn, dễ tối ưu bằng gradient.
3. Phạt mạnh lỗi lớn.

Ví dụ:

- Sai 1 -> phạt 1.
- Sai 10 -> phạt 100.

Nghĩa là mô hình sẽ ưu tiên giảm lỗi lớn trước.

## 7. SGD: cách sửa tham số

> [!NOTE] ELI5
> Sau khi biết mình sai bao nhiêu, ta sửa tham số một chút theo hướng làm giảm sai.

Công thức chung:

$$
θ \leftarrow θ - η ∇_θ L
$$

- $\theta$: tham số (ở đây là $w,b$).
- $\eta$: learning rate.
- $\nabla_\theta L$: gradient.

Giải thích 4 tầng cho gradient:

1. **Bản chất**: gradient chỉ hướng loss tăng nhanh nhất.
2. **Ví dụ đời thường**: đi ngược dốc để xuống núi nhanh.
3. **Công thức**:
$$
\nabla_\theta L = \left[\frac{\partial L}{\partial \theta_1}, \frac{\partial L}{\partial \theta_2}, ...\right]
$$
4. **Ứng dụng DL**: mọi mô hình sâu đều cập nhật theo nguyên lý này.

## 8. Training loop: vòng lặp quan trọng nhất

> [!NOTE] ELI5
> Mỗi vòng gồm 4 việc: đoán -> tính lỗi -> tính gradient -> sửa tham số.

Cho mỗi minibatch:

1. `y_hat = net(Xb)`
2. `l = loss(y_hat, yb)`
3. `l.backward()`
4. `sgd([w,b], lr)`

Lặp đủ nhiều vòng, loss sẽ giảm.

## 9. Code PyTorch tối giản (đọc theo khối)

```python
import torch

# 1) Tao du lieu tong hop
true_w = torch.tensor([2.0, -3.4])
true_b = 4.2
n = 1000
X = torch.randn(n, 2)
noise = 0.01 * torch.randn(n, 1)
y = X @ true_w.reshape(-1, 1) + true_b + noise

# 2) Khoi tao tham so can hoc
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 3) Dinh nghia model + loss + sgd

def net(X):
    return X @ w + b


def loss(y_hat, y):
    return ((y_hat - y) ** 2 / 2).mean()


def sgd(params, lr):
    with torch.no_grad():
        for p in params:
            p -= lr * p.grad
            p.grad.zero_()

# 4) Training loop
def data_iter(batch_size, X, y):
    idx = torch.randperm(X.shape[0])
    for i in range(0, X.shape[0], batch_size):
        batch_idx = idx[i:i + batch_size]
        yield X[batch_idx], y[batch_idx]

lr = 0.03
batch_size = 32
epochs = 3

for epoch in range(epochs):
    for Xb, yb in data_iter(batch_size, X, y):
        l = loss(net(Xb), yb)
        l.backward()
        sgd([w, b], lr)
    with torch.no_grad():
        train_l = loss(net(X), y)
        print(f"epoch {epoch+1}, loss {train_l.item():.6f}")

print("w that:", true_w)
print("w hoc duoc:", w.reshape(-1).detach())
print("b that:", true_b)
print("b hoc duoc:", b.item())
```

Cách đọc output đúng:

1. Loss giảm qua từng epoch.
2. `w hoc duoc` gần `w that`.
3. `b hoc duoc` gần `b that`.

Nếu 3 điều này đúng, code của bạn nhiều khả năng đã chuẩn.

## 10. Từ điển thuật ngữ buổi 11

| Thuật ngữ | Dịch nghĩa từ ngữ | Nghĩa kỹ thuật trong bài | Dấu hiệu bạn đã hiểu |
| --- | --- | --- | --- |
| Ground truth | Sự thật gốc | Tham số/nhãn đúng dùng làm chuẩn so sánh | Bạn giải thích được vì sao synthetic data cần ground truth |
| From scratch | Từ đầu, bằng tay | Tự viết các thành phần cốt lõi, không phụ thuộc API high-level | Bạn tự viết được net/loss/sgd |
| Epoch | Lượt quét dữ liệu | Một lần đi hết tập train | Bạn phân biệt được epoch và batch |
| Batch/Minibatch | Lô dữ liệu | Nhóm mẫu xử lý cùng lúc để cập nhật tham số | Bạn biết batch size ảnh hưởng tốc độ và nhiễu gradient |
| Autograd | Tự động vi phân | Cơ chế framework tính gradient tự động | Bạn biết cần `backward()` và `zero_grad()` |

## 11. Những lỗi thực tế cần tránh

1. Sai shape giữa `y_hat` và `y` làm loss sai.
2. Không shuffle dữ liệu train -> học kém ổn định.
3. Chỉ nhìn train loss mà không kiểm tra khả năng tổng quát hóa.
4. Kỳ vọng tham số học đúng y hệt ground truth dù có nhiễu.

## 12. Bài tự kiểm tra (siêu ngắn)

1. Tại sao buổi này phải dùng dữ liệu synthetic thay vì dữ liệu thật ngay từ đầu?
2. Nếu quên `zero_grad()`, chuyện gì xảy ra với gradient?
3. Viết lại bằng lời công thức SGD update.
4. Nói rõ khác nhau giữa buổi 10 (lý thuyết) và buổi 11 (from scratch).

## 13. Kết luận

Buổi 11 là bước chuyển từ "biết công thức" sang "tự vận hành cả pipeline học".

Khi nắm được buổi này, bạn đã hiểu khung nền của rất nhiều mô hình deep learning:

1. Định nghĩa mô hình.
2. Chọn loss.
3. Tính gradient.
4. Cập nhật tham số.
5. Lặp lại cho tới khi học tốt.

Buổi tiếp theo (Buổi 12) dùng API concise để bạn thấy rõ: framework chỉ là lớp bọc tiện lợi của đúng các bước trên.
