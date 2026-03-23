---
title: "DataLoader (PyTorch)"
aliases: [data loader, bộ nạp dữ liệu, DataLoader, data iterator]
tags: [concept, machine-learning, pytorch, data-pipeline, training]
created: 2026-03-22
---

# DataLoader (PyTorch)

> [!NOTE] ELI5
> Bạn có 60,000 bài tập nhưng không thể đọc hết 1 lần. DataLoader giúp bạn **chia thành từng bộ nhỏ** (batch), **xáo trộn thứ tự** mỗi lần ôn, và **đưa từng bộ vào model** để train. Nó giống cái máy phát bài tập tự động.

## 1. Bản chất — Tại sao cần DataLoader?

Training trên **toàn bộ dataset cùng lúc** gặp 2 vấn đề:

1. **RAM không đủ**: 60,000 ảnh × kích thước mỗi ảnh → tốn bộ nhớ
2. **SGD cần batch**: [[Gradient Descent|Stochastic Gradient Descent]] hoạt động trên **minibatch**, không phải toàn bộ data

DataLoader giải quyết bằng cách:
- Chia data thành **minibatches** cố định (ví dụ: 64 mẫu/batch)
- **Xáo trộn** (shuffle) mỗi epoch để tránh mô hình "nhớ thứ tự"
- **Tải song song** (num_workers) để tận dụng CPU trong khi GPU train

## 2. Pipeline

```
Dataset → Transform → Shuffle → Batch → Model
```

| Bước | Công việc | PyTorch API |
| --- | --- | --- |
| 1. Dataset | Lưu trữ dữ liệu + labels | `torchvision.datasets.FashionMNIST(...)` |
| 2. Transform | Resize, ToTensor, Normalize | `transforms.Compose([...])` |
| 3. Shuffle | Xáo trộn thứ tự (chỉ khi train) | `shuffle=True` |
| 4. Batch | Gom thành minibatch | `batch_size=64` |
| 5. Iterate | Lặp qua từng batch | `for X, y in loader:` |

## 3. Code mẫu (PyTorch)

```python
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

# 1. Transform: resize + chuyển sang tensor
trans = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()  # pixel 0-255 → 0.0-1.0
])

# 2. Dataset
train_data = torchvision.datasets.FashionMNIST(
    root='./data', train=True, transform=trans, download=True)

# 3. DataLoader
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# 4. Dùng trong training loop
for X, y in train_loader:
    # X.shape = (64, 1, 28, 28)
    # y.shape = (64,)
    pass
```

## 4. Các tham số quan trọng

| Tham số | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `batch_size` | 1 | Số mẫu mỗi batch |
| `shuffle` | False | Có xáo trộn không |
| `num_workers` | 0 | Số process tải data song song |
| `drop_last` | False | Bỏ batch cuối nếu thiếu mẫu |

> [!TIP] Khi nào shuffle?
> - **Training**: `shuffle=True` (luôn luôn)
> - **Validation/Test**: `shuffle=False` (kết quả cần reproducible)

## TODO

- [ ] Thêm custom Dataset class
- [ ] Pin memory cho GPU training
