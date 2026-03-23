---
title: "Buổi 15 - Tuần 4: Image Classification Dataset — Fashion-MNIST (D2L)"
tags: [d2l, fashion-mnist, dataset, dataloader, classification, study-note]
created: 2026-03-22
session: "D2L Tuần 4, Buổi 15 — Image Classification Dataset"
source:
  - "30_Resources/Books/Dive Into Deep Learning/d2l-en/chapter_linear-classification/image-classification-dataset.md"
related:
  - "[[Buổi 14 - Tuần 4]]"
  - "[[Fashion-MNIST Dataset]]"
  - "[[DataLoader (PyTorch)]]"
  - "[[Softmax Function]]"
  - "[[One-Hot Encoding]]"
---

# Buổi 15 — Fashion-MNIST: Bộ dữ liệu đầu tiên cho Classification

> [!NOTE] ELI5
> Buổi 14 bạn đã biết **lý thuyết** phân loại ([[Softmax Function|softmax]], [[Cross-Entropy Loss|cross entropy]]). Buổi 15 bạn chạm vào **dữ liệu thật**: 70,000 ảnh quần áo nhỏ xíu (28×28 pixel, trắng đen), chia thành 10 loại. Đây là "bộ đề thi chuẩn" mà gần như mọi model classification đều được thử trên đó.
>
> Bạn cũng sẽ học cách **nạp dữ liệu** vào model bằng DataLoader — "máy phát bài tập tự động" chia data thành từng batch nhỏ.

---

## 🎯 Mục tiêu buổi học

1. Hiểu **Fashion-MNIST** là gì, tại sao dùng thay MNIST gốc
2. Biết **10 categories** và cấu trúc dữ liệu
3. Hiểu **tensor shape convention** cho ảnh: $(c, h, w)$
4. Biết cách dùng **DataLoader** để nạp dữ liệu theo batch
5. Viết được **code load + visualize** Fashion-MNIST

---

## Phần 1: Tại sao Fashion-MNIST?

> [!NOTE] ELI5
> **MNIST** (chữ số viết tay) giống bài kiểm tra 15 phút — quá dễ, ai cũng đạt 95+%. **Fashion-MNIST** giống bài thi thật — đủ khó để phân biệt mô hình mạnh và yếu, nhưng vẫn đủ nhỏ để train nhanh trên laptop.

### 1.1 Lịch sử ngắn gọn

| Năm | Sự kiện |
| --- | --- |
| **1998** | MNIST ra đời — benchmark hàng đầu thập niên 2000 |
| ~2012 | MNIST trở nên **quá dễ** — mọi model đạt >95% |
| **2017** | Fashion-MNIST ra đời — thay thế MNIST, cùng format |
| Hiện tại | ImageNet là benchmark lớn, Fashion-MNIST dùng cho learning/prototype |

### 1.2 So sánh MNIST vs Fashion-MNIST

| | MNIST | Fashion-MNIST |
| --- | --- | --- |
| Nội dung | Chữ số 0-9 | 10 loại quần áo |
| Ảnh | 28×28, grayscale | 28×28, grayscale |
| Train/Test | 60,000 / 10,000 | 60,000 / 10,000 |
| Accuracy dễ đạt | >95% (quá dễ) | ~85-90% (khó hơn) |
| Dùng cho | Sanity check | Benchmark + learning |

> [!TIP] Tại sao cùng format?
> Fashion-MNIST được thiết kế **drop-in replacement** — cùng kích thước, cùng cấu trúc train/test. Bạn chỉ cần đổi 1 dòng code từ `MNIST` sang `FashionMNIST` là xong.

---

## Phần 2: 10 Categories của Fashion-MNIST

![[assets/attachments/D2L/Buổi 15/fashion_mnist_samples.png]]

| Label | Tên tiếng Anh | Tiếng Việt | Số ảnh train | Số ảnh test |
| --- | --- | --- | --- | --- |
| 0 | T-shirt/top | Áo thun | 6,000 | 1,000 |
| 1 | Trouser | Quần dài | 6,000 | 1,000 |
| 2 | Pullover | Áo len chui đầu | 6,000 | 1,000 |
| 3 | Dress | Váy/đầm | 6,000 | 1,000 |
| 4 | Coat | Áo khoác | 6,000 | 1,000 |
| 5 | Sandal | Dép sandal | 6,000 | 1,000 |
| 6 | Shirt | Áo sơ mi | 6,000 | 1,000 |
| 7 | Sneaker | Giày thể thao | 6,000 | 1,000 |
| 8 | Bag | Túi xách | 6,000 | 1,000 |
| 9 | Ankle boot | Giày boot cổ ngắn | 6,000 | 1,000 |

**Tổng**: 60,000 train + 10,000 test = **70,000 ảnh**.

> [!WARNING] Cặp dễ nhầm
> **T-shirt (0)** vs **Shirt (6)** và **Pullover (2)** vs **Coat (4)** là các cặp hay bị nhầm. Đây chính là thách thức chính của Fashion-MNIST.

> Xem thêm: [[Fashion-MNIST Dataset]]

---

## Phần 3: Tensor Shape — Ảnh được lưu thế nào?

> [!NOTE] ELI5
> Một ảnh là một **lưới số** — mỗi ô là 1 pixel (giá trị 0-255: đen → trắng). Ảnh 28×28 pixel = 784 con số. Khi train, ta gom 64 ảnh lại thành 1 batch.

### 3.1 Convention: $c \times h \times w$

| Dimension | Ý nghĩa | Fashion-MNIST |
| --- | --- | --- |
| $c$ (channels) | Kênh màu | 1 (grayscale) |
| $h$ (height) | Chiều cao | 28 pixels |
| $w$ (width) | Chiều rộng | 28 pixels |

**1 ảnh**: shape = $(1, 28, 28)$ — 1 channel, 28 hàng, 28 cột.

### 3.2 Minibatch shape

Khi gom $n$ ảnh thành 1 batch:

$$\text{Batch shape} = (n, c, h, w) = (64, 1, 28, 28)$$

| Dimension | Ý nghĩa | Giá trị |
| --- | --- | --- |
| Position 0 | Batch size | 64 |
| Position 1 | Channels | 1 |
| Position 2 | Height | 28 |
| Position 3 | Width | 28 |

### 3.3 Pixel values

| Trước transform | Sau `ToTensor()` |
| --- | --- |
| Integer 0-255 | Float 0.0-1.0 |
| dtype: uint8 | dtype: float32 |

`ToTensor()` tự động chia cho 255 → chuẩn hóa pixels về khoảng $[0, 1]$.

> [!TIP] Tại sao chuẩn hóa?
> Giá trị 0-255 quá lớn → gradient lớn → training không ổn định. Đưa về 0-1 giúp mô hình hội tụ dễ hơn.

---

## Phần 4: DataLoader — Máy phát dữ liệu tự động

> [!NOTE] ELI5
> Bạn có 60,000 bài tập. Không thể đọc hết 1 lần. DataLoader **chia thành từng bộ 64 bài**, **xáo trộn** thứ tự mỗi lần ôn, và **đưa từng bộ** cho bạn xử lý. Hết 1 vòng (epoch) = đã đọc hết 60,000 bài.

### 4.1 Pipeline

![[assets/attachments/D2L/Buổi 15/dataloader_pipeline.png]]

| Bước | Công việc | Tại sao? |
| --- | --- | --- |
| **1. Dataset** | Lưu trữ ảnh + label | Nguồn dữ liệu |
| **2. Transform** | Resize, ToTensor | Chuẩn hóa format |
| **3. Shuffle** | Xáo trộn thứ tự | Tránh model nhớ thứ tự |
| **4. Batch** | Gom 64 mẫu/batch | SGD cần minibatch |
| **5. Iterate** | Lặp qua từng batch | Feed vào model |

### 4.2 Code PyTorch load Fashion-MNIST

```python
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

# 1) Transform: chuyển ảnh sang tensor
trans = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()   # 0-255 → 0.0-1.0
])

# 2) Download + load dataset
train_data = torchvision.datasets.FashionMNIST(
    root='./data', train=True, transform=trans, download=True)
test_data = torchvision.datasets.FashionMNIST(
    root='./data', train=False, transform=trans, download=True)

print(f"Train: {len(train_data)} images")  # 60,000
print(f"Test:  {len(test_data)} images")   # 10,000

# 3) DataLoader
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=64, shuffle=False)

# 4) Đọc 1 batch
X, y = next(iter(train_loader))
print(f"X shape: {X.shape}")  # (64, 1, 28, 28)
print(f"y shape: {y.shape}")  # (64,)
print(f"X dtype: {X.dtype}")  # float32
print(f"y dtype: {y.dtype}")  # int64 (label)
```

### 4.3 Giải thích code từng dòng

| Code | Ý nghĩa |
| --- | --- |
| `transforms.ToTensor()` | Chuyển PIL Image → Tensor, chia 255 |
| `train=True` | Lấy training set (60k), `False` = test set (10k) |
| `download=True` | Tự download nếu chưa có |
| `batch_size=64` | Mỗi batch 64 ảnh |
| `shuffle=True` | Xáo trộn (chỉ cho train!) |
| `next(iter(loader))` | Lấy batch đầu tiên |

### 4.4 Mapping label → tên

```python
labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
          'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

# Ví dụ: y[0] = 7 → labels[7] = 'sneaker'
```

> Xem thêm: [[DataLoader (PyTorch)]]

---

## Phần 5: Visualize — Luôn nhìn dữ liệu trước khi train

> [!NOTE] ELI5
> Trước khi train model, **hãy luôn xem dữ liệu** bằng mắt. Con người rất giỏi phát hiện lỗi, bất thường. Nếu data có vấn đề mà bạn không biết → train ra model sai.

### 5.1 Code đơn giản để xem ảnh

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 8, figsize=(12, 4))
for i, ax in enumerate(axes.flat):
    img = train_data[i][0].squeeze()  # (1, 28, 28) → (28, 28)
    label = train_data[i][1]
    ax.imshow(img, cmap='gray')
    ax.set_title(labels[label], fontsize=9)
    ax.axis('off')
plt.tight_layout()
plt.show()
```

### 5.2 Checklist khi xem data

- [ ] Ảnh có đúng nội dung không? (không bị hỏng, không bị lệch)
- [ ] Label có khớp với ảnh không?
- [ ] Các class có cân bằng không? (Fashion-MNIST: 6,000/class → cân bằng ✅)
- [ ] Ảnh có đủ thông tin để phân biệt các class không?

---

## Phần 6: Tốc độ đọc data — Đừng để I/O thành bottleneck

> [!NOTE] ELI5
> Model train chậm có thể vì **chờ data** chứ không phải vì tính toán. DataLoader với `num_workers > 0` dùng nhiều CPU load data song song trong khi GPU đang train batch trước.

### 6.1 Benchmark thời gian load

```python
import time
tic = time.time()
for X, y in train_loader:
    pass
print(f"Thời gian load 1 epoch: {time.time() - tic:.2f} sec")
```

### 6.2 Tăng tốc

| Cách | Lệnh | Lý do |
| --- | --- | --- |
| Song song hóa | `num_workers=4` | Dùng 4 CPU đọc data parallel |
| Pin memory | `pin_memory=True` | Tăng tốc transfer CPU → GPU |
| Giảm batch cuối | `drop_last=True` | Batch cuối thường nhỏ, gây bottleneck |

---

## 📖 Từ điển thuật ngữ Buổi 15

| Thuật ngữ | Dịch nghĩa | Nghĩa trong buổi này | Ví dụ |
| --- | --- | --- | --- |
| **Fashion-MNIST** | — | Bộ 70k ảnh quần áo 28×28, 10 class | Dataset chuẩn thay MNIST |
| **MNIST** | — | Bộ ảnh chữ số viết tay (quá dễ) | Benchmark cũ, dùng sanity check |
| **Grayscale** | Ảnh xám | Ảnh 1 channel (0-255) | Khác RGB (3 channels) |
| **Tensor shape** | Kích thước tensor | $(c, h, w)$ hoặc $(n, c, h, w)$ | $(64, 1, 28, 28)$ |
| **Channel** | Kênh màu | Grayscale=1, RGB=3 | Fashion-MNIST: 1 channel |
| **DataLoader** | Bộ nạp dữ liệu | Chia data → batch, shuffle, iterate | `DataLoader(data, 64, True)` |
| **Batch size** | Kích thước batch | Số mẫu mỗi lần feed vào model | 64 |
| **Epoch** | Vòng lặp | 1 lần duyệt hết toàn bộ training data | 60000/64 ≈ 938 batches |
| **Transform** | Phép biến đổi | Resize, ToTensor, Normalize | `transforms.ToTensor()` |
| **Shuffle** | Xáo trộn | Ngẫu nhiên hóa thứ tự mỗi epoch | Chỉ dùng cho train set |
| **num_workers** | Số worker | Số CPU tải data song song | 0 = main thread only |

---

## ✅ Bài tự kiểm tra

1. Fashion-MNIST có **bao nhiêu ảnh** train và test? Mỗi class có bao nhiêu ảnh?
2. Shape của 1 batch 64 ảnh Fashion-MNIST là gì? Giải thích ý nghĩa từng dimension.
3. Tại sao `shuffle=True` cho train nhưng `shuffle=False` cho test?
4. `ToTensor()` làm gì với pixel values? Tại sao cần làm vậy?
5. Fashion-MNIST khó hơn MNIST ở điểm nào? Nêu 1 cặp categories dễ nhầm.

---

## 🔗 Liên kết

- **Buổi trước**: [[Buổi 14 - Tuần 4]] — Softmax Regression (lý thuyết)
- **Buổi sau**: [[Buổi 16 - Tuần 4]] — Softmax Regression from Scratch
- **Concept notes**: [[Fashion-MNIST Dataset]], [[DataLoader (PyTorch)]]

## 📝 Kết luận

Buổi 15 là buổi **thực hành đầu tiên** với dữ liệu phân loại thật. Fashion-MNIST sẽ theo bạn suốt nhiều buổi tiếp theo — từ softmax regression đơn giản đến CNN phức tạp. Hai kỹ năng quan trọng:

1. **Hiểu cấu trúc data**: shape $(n, c, h, w)$, pixel values $[0, 1]$, 10 classes cân bằng
2. **Dùng DataLoader**: chia batch, shuffle, iterate — pipeline chuẩn cho mọi bài toán DL

Buổi 16 sẽ **implement softmax regression from scratch** trên Fashion-MNIST — gắn lý thuyết buổi 14 vào data buổi 15.
