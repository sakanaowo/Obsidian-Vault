---
title: "Buổi 12 - Tuần 3: Linear Regression Concise (D2L)"
tags: [d2l, linear-regression, concise, pytorch, deep-learning, study-note]
created: 2026-03-17
session: "D2L Tuần 3, Buổi 12 — Linear Regression Concise"
source:
  - "30_Resources/Books/Dive Into Deep Learning/d2l-en/chapter_linear-regression/linear-regression-concise.md"
related:
  - "[[Buổi 10 - Tuần 3]]"
  - "[[Buổi 11 - Tuần 3]]"
  - "[[Linear Regression for Deep Learning]]"
  - "[[Gradient Descent]]"
---

# Buổi 12 - Tuần 3: Linear Regression Concise

> [!NOTE] ELI5
> Buổi 11 bạn tự lắp máy bằng tay. Buổi 12 bạn dùng máy có sẵn của framework. Kết quả toán học không đổi, chỉ khác là code gọn hơn, ít lỗi vặt hơn, và chạy nhanh hơn.

## 1. Mục tiêu buổi học

1. Hiểu "concise" nghĩa là gì trong D2L.
2. Biết map từng phần của buổi 11 sang API high-level.
3. Biết vì sao kết quả vẫn là linear regression cũ dù code ngắn hơn.
4. Biết các điểm dễ nhầm khi dùng API có sẵn.

## 2. "Concise" thật ra thay thế cái gì?

> [!NOTE] ELI5
> Không phải framework làm phép màu. Nó chỉ đóng gói các phần bạn đã tự viết ở buổi 11.

### 2.1 Bảng đối chiếu Scratch vs Concise

| Thành phần | Buổi 11 (scratch) | Buổi 12 (concise) | Ý nghĩa |
| --- | --- | --- | --- |
| Model | Tự viết `net(X)=X@w+b` | `nn.LazyLinear(1)` | Vẫn là hàm affine tuyến tính |
| Loss | Tự viết MSE | `nn.MSELoss()` | Cùng đo sai số bình phương |
| Optimizer | Tự viết `sgd(...)` | `torch.optim.SGD(...)` | Cùng cập nhật theo gradient |
| Training loop | Tự lặp từng batch | Dùng loop gọn/`Trainer` | Cùng quy trình forward-backward-step |

### 2.2 Claim - Reasoning - Evidence

- **Claim**: Concise không đổi bản chất mô hình.
- **Reasoning**: Layer/loss/optimizer built-in chỉ là bản triển khai chuẩn của cùng công thức toán.
- **Evidence**: Tham số học được vẫn tiệm cận ground truth giống buổi 11.

## 3. Model high-level: `nn.LazyLinear(1)`

> [!NOTE] ELI5
> Bạn nói với PyTorch: "Tôi cần 1 đầu ra". Còn số đầu vào bao nhiêu, PyTorch tự suy ra lần đầu bạn đưa dữ liệu vào.

### 3.1 Bản chất

Với hồi quy tuyến tính 1 đầu ra, layer fully-connected có dạng:

$$
\hat y = XW + b
$$

`nn.LazyLinear(1)` đại diện đúng công thức trên, trong đó:

- `out_features = 1`
- `in_features` được suy ra tự động khi chạy lần đầu.

### 3.2 Vì sao phải khởi tạo tham số?

D2L gợi ý khởi tạo:

- Weight từ Gaussian nhỏ (std 0.01).
- Bias bằng 0.

Lý do:

1. Tránh khởi tạo quá lớn gây đầu ra/gradient bất ổn.
2. Giữ điểm bắt đầu trung tính, dễ học.

## 4. Loss built-in: `nn.MSELoss()`

> [!NOTE] ELI5
> MSELoss là bộ chấm điểm sai có sẵn. Bạn không cần tự viết lại công thức nữa.

### 4.1 Bản chất

`nn.MSELoss()` trả về trung bình:

$$
\text{MSE} = \frac{1}{B}\sum_{i=1}^{B}(\hat y_i - y_i)^2
$$

Lưu ý quan trọng: bản này không có hệ số $\frac{1}{2}$ như một số sách viết tay.

### 4.2 Điều này có làm đổi nghiệm tối ưu không?

- Không đổi nghiệm tối ưu.
- Chỉ đổi tỉ lệ gradient (hệ số hằng), có thể bù bằng learning rate.

A đúng vì B, được thể hiện qua C:

- **Claim**: Bỏ/giữ $\frac{1}{2}$ không làm đổi nghiệm tốt nhất.
- **Reasoning**: Nhân loss với hằng số dương không đổi vị trí điểm cực tiểu.
- **Evidence**: Cả hai phiên bản đều hội tụ về tham số gần ground truth khi cấu hình hợp lý.

## 5. Optimizer built-in: `torch.optim.SGD`

> [!NOTE] ELI5
> Trước đây bạn tự viết hàm sửa tham số. Giờ framework làm giúp cùng một công việc.

Công thức cốt lõi vẫn là:

$$
\theta \leftarrow \theta - \eta \nabla_\theta L
$$

Trong code concise:

1. `optimizer.zero_grad()`
2. `loss.backward()`
3. `optimizer.step()`

Đây chính là 3 thao tác bạn từng làm tay ở buổi 11.

## 6. Quy trình train concise (không bỏ qua trực giác)

> [!NOTE] ELI5
> Dù dùng API cao cấp, bạn vẫn phải hiểu mỗi batch đang diễn ra gì: đoán, chấm lỗi, sửa.

Pipeline cho mỗi minibatch:

1. Forward: `y_hat = net(Xb)`.
2. Loss: `l = mse(y_hat, yb)`.
3. Backward: `l.backward()`.
4. Update: `optimizer.step()`.
5. Reset gradient: `optimizer.zero_grad()` cho batch sau.

## 7. Code PyTorch concise tối giản

```python
import torch
from torch import nn

# 1) Tao du lieu tong hop
true_w = torch.tensor([2.0, -3.4])
true_b = 4.2
n = 1000
X = torch.randn(n, 2)
noise = 0.01 * torch.randn(n, 1)
y = X @ true_w.reshape(-1, 1) + true_b + noise

# 2) DataLoader
batch_size = 32
dataset = torch.utils.data.TensorDataset(X, y)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 3) Model concise
net = nn.LazyLinear(1)
net.weight.data.normal_(0, 0.01)
net.bias.data.fill_(0)

# 4) Loss + Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.03)

# 5) Train
epochs = 3
for epoch in range(epochs):
    for Xb, yb in loader:
        optimizer.zero_grad()
        l = criterion(net(Xb), yb)
        l.backward()
        optimizer.step()
    with torch.no_grad():
        train_l = criterion(net(X), y)
        print(f"epoch {epoch+1}, loss {train_l.item():.6f}")

# 6) So sanh tham so hoc duoc voi ground truth
w_hat = net.weight.data.reshape(-1)
b_hat = net.bias.data.item()
print("w that:", true_w)
print("w hoc duoc:", w_hat)
print("b that:", true_b)
print("b hoc duoc:", b_hat)
```

## 8. Đọc kết quả đúng cách

> [!NOTE] ELI5
> Loss giảm là tín hiệu tốt, nhưng chưa đủ. Bạn cần kiểm tra luôn tham số học được có gần tham số thật không.

Checklist:

1. Loss giảm theo epoch?
2. `w_hat` gần `true_w`?
3. `b_hat` gần `true_b`?
4. Nếu chưa gần: kiểm tra learning rate, epoch, batch size, hoặc lỗi shape.

## 9. Từ điển thuật ngữ Buổi 12

| Thuật ngữ | Dịch nghĩa từ ngữ | Nghĩa trong ngữ cảnh buổi này | Ví dụ |
| --- | --- | --- | --- |
| Concise API | API ngắn gọn | API cấp cao đã đóng gói logic phổ biến | `nn.Linear`, `nn.MSELoss`, `optim.SGD` |
| Fully Connected Layer | Lớp kết nối đầy đủ | Mỗi đầu vào nối đến đầu ra qua ma trận trọng số | `nn.LazyLinear(1)` |
| Lazy Initialization | Khởi tạo lười | Chưa chốt kích thước đầu vào cho tới lần forward đầu tiên | `LazyLinear` tự suy ra `in_features` |
| DataLoader | Bộ nạp dữ liệu | Chia dữ liệu thành batch, hỗ trợ shuffle | `DataLoader(dataset, batch_size=32)` |
| Trainer (D2L) | Bộ huấn luyện | Khung gọi fit theo epoch/batch | `trainer.fit(model, data)` |

## 10. So sánh thực chiến: nên dùng scratch hay concise?

### 10.1 Khi nên dùng scratch

1. Học nền tảng hoặc debug sâu.
2. Nghiên cứu thành phần mới chưa có sẵn.
3. Muốn kiểm soát từng phép toán.

### 10.2 Khi nên dùng concise

1. Bài toán chuẩn, thành phần đã có trong framework.
2. Cần tốc độ phát triển nhanh.
3. Muốn giảm bug vặt trong khâu huấn luyện.

Kết luận cân bằng:

- Học để hiểu bằng scratch.
- Làm thực tế ưu tiên concise.
- Không hiểu scratch thì dễ dùng concise sai.

## 11. Điểm dễ nhầm

1. Tưởng dùng concise thì không cần hiểu gradient.
2. Quên `zero_grad()` nên gradient cộng dồn.
3. Quên kiểm tra shape của `y` và `y_hat`.
4. Hiểu nhầm rằng loss công thức khác nhau nghĩa là mô hình khác nhau.

## 12. Bài tự kiểm tra

1. Tại sao `nn.MSELoss()` không có $\frac{1}{2}$ nhưng vẫn học đúng?
2. `LazyLinear(1)` tiện hơn `Linear(in_features, 1)` ở điểm nào?
3. Viết lại 3 dòng cốt lõi của tối ưu: zero_grad, backward, step.
4. Nêu 2 tình huống nên dùng scratch thay vì concise.

## 13. Kết luận

Buổi 12 giúp bạn chuyển từ "tự xây pipeline" sang "dùng framework đúng cách". Kiến thức toán không đổi, chỉ thay đổi mức trừu tượng của code.

Buổi 13 sẽ đi vào **generalization**: vì sao train tốt chưa chắc test tốt, và cách đọc hiện tượng overfitting/underfitting.
