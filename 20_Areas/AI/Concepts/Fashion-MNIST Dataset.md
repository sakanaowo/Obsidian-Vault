---
title: "Fashion-MNIST Dataset"
aliases: [Fashion MNIST, FashionMNIST, bộ dữ liệu Fashion-MNIST]
tags: [concept, machine-learning, dataset, computer-vision, classification]
created: 2026-03-22
---

# Fashion-MNIST Dataset

> [!NOTE] ELI5
> Fashion-MNIST giống bộ đề thi chuẩn cho bài toán nhận dạng ảnh. Thay vì nhận dạng chữ số viết tay (đã quá dễ), bộ này yêu cầu nhận dạng **10 loại quần áo**: áo thun, quần dài, giày, túi xách... Mỗi ảnh nhỏ 28×28 pixel, trắng đen. Ai cũng dùng bộ này để test model, nên kết quả có thể so sánh với nhau.

## 1. Tại sao không dùng MNIST gốc?

**MNIST** (chữ số 0-9 viết tay) ra đời năm 1998, từng là benchmark hàng đầu. Nhưng ngày nay, ngay cả mô hình đơn giản cũng đạt >95% accuracy → **không còn phân biệt** được mô hình mạnh vs yếu.

**Fashion-MNIST** (2017) được thiết kế để thay thế MNIST:

| | MNIST | Fashion-MNIST |
| --- | --- | --- |
| Nội dung | Chữ số 0-9 | 10 loại quần áo |
| Kích thước ảnh | 28×28 | 28×28 (giống!) |
| Channels | 1 (grayscale) | 1 (grayscale) |
| Train/Test | 60k/10k | 60k/10k (giống!) |
| Độ khó | Quá dễ | Khó hơn nhiều |

Thiết kế **cùng format** để dễ dàng swap vào code đã có sẵn cho MNIST.

## 2. 10 Categories

| Label | Tên | Mô tả |
| --- | --- | --- |
| 0 | T-shirt/top | Áo thun |
| 1 | Trouser | Quần dài |
| 2 | Pullover | Áo len chui đầu |
| 3 | Dress | Váy/đầm |
| 4 | Coat | Áo khoác |
| 5 | Sandal | Dép sandal |
| 6 | Shirt | Áo sơ mi |
| 7 | Sneaker | Giày thể thao |
| 8 | Bag | Túi xách |
| 9 | Ankle boot | Giày boot cổ ngắn |

Mỗi category có **6,000 train** + **1,000 test** = **7,000 ảnh**.

## 3. Tensor Shape Convention

Ảnh được lưu dạng tensor $c \times h \times w$:

| Dimension | Ý nghĩa | Giá trị |
| --- | --- | --- |
| $c$ | Channels (kênh màu) | 1 (grayscale) |
| $h$ | Height (chiều cao) | 28 pixels |
| $w$ | Width (chiều rộng) | 28 pixels |

Minibatch shape: $(n, c, h, w)$ = $(64, 1, 28, 28)$ với batch_size = 64.

## TODO

- [ ] Thêm so sánh kết quả accuracy của các model trên Fashion-MNIST
- [ ] Liên kết với CIFAR-10, ImageNet
