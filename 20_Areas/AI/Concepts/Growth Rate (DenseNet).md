---
title: "Growth Rate (DenseNet)"
aliases: ["Growth Rate", "tốc độ tăng trưởng channel", "k (DenseNet)"]
tags: [concept, deep-learning, architecture, densenet, hyperparameter]
created: 2026-04-11
---

# Growth Rate (DenseNet) — Tốc độ Tăng trưởng Channel

> [!NOTE] ELI5
> Growth rate giống như số người được thêm vào đội mỗi tháng. Nếu mỗi tháng đội thêm 32 người (growth rate = 32), thì sau 4 tháng đội có thêm 128 người. Trong DenseNet, mỗi conv layer "thêm" đúng $k$ channels mới vào tensor đang tích lũy — con số $k$ đó chính là growth rate.

**Growth rate** (ký hiệu $k$) là hyperparameter trong DenseNet định nghĩa số output channels mà **mỗi conv block** trong một Dense Block tạo ra. Vì DenseNet dùng concatenation (thay vì addition như ResNet), mỗi layer tích lũy thêm $k$ channels mới vào tensor. Sau $n$ layers trong một dense block với input $C_{in}$ channels:

$$C_{out} = C_{in} + n \times k$$

Growth rate nhỏ (thường $k = 12$–$32$) là đặc trưng của DenseNet, khác biệt hoàn toàn với các kiến trúc trước (ResNet dùng 64–512 channels/block).

## Lý do Growth Rate nhỏ lại hiệu quả

1. **Feature cũ không bị mất** — concatenation giữ nguyên tất cả feature maps từ layers trước
2. **Mỗi layer chỉ cần đóng góp thêm** $k$ "thông tin mới" thay vì học lại toàn bộ
3. **"Collective knowledge"** — layer $l$ có thể truy cập thông qua mọi $l$ feature sets trước đó

Kết quả: DenseNet với $k = 32$ thường ít tham số hơn ResNet ở cùng accuracy.

## Giá trị phổ biến

| Biến thể               | $k$   | Mục đích                |
| ---------------------- | ----- | ----------------------- |
| DenseNet-BC (CIFAR)    | 12    | Compact cho dataset nhỏ |
| DenseNet-121, 169, 201 | 32    | Chuẩn cho ImageNet      |
| DenseNet-264           | 32–48 | Cực sâu                 |

## Tác động lên Channel Count

Dense Block với $n$ layers, input $C_{in}$, growth rate $k$:

| Layer | Input             | Output (sau concat) |
| ----- | ----------------- | ------------------- |
| 1     | $C_{in}$          | $C_{in} + k$        |
| 2     | $C_{in} + k$      | $C_{in} + 2k$       |
| $n$   | $C_{in} + (n-1)k$ | $C_{in} + nk$       |

> [!TIP] Lưu ý
> Growth rate kiểm soát **tốc độ tăng channels TRONG** dense block. Để kiểm soát channels **giữa** các dense blocks, dùng **Transition Layer** (giảm 50% channels bằng Conv 1×1).

## Liên kết

- Đã học chi tiết ở [[Buổi 35 - Tuần 9]] (DenseNet — Dense Blocks)
- Liên quan: [[Residual Connection]] (trong ResNet channels tăng theo ×2 tại boundary, không tăng liên tục)
- Source: [d2l.ai — 8.7 DenseNet](https://d2l.ai/chapter_convolutional-modern/densenet.html)
- Paper gốc: Huang et al., 2017 — "Densely Connected Convolutional Networks"

---

> [!TODO]
>
> - Phân tích ảnh hưởng của $k$ đến memory consumption (trade-off với params)
> - Bottleneck DenseNet-B: thêm Conv 1×1 trước Conv 3×3 để giảm FLOPs khi $C_{in}$ lớn
> - Compression factor $\theta$ trong Transition Layer: cách DenseNet-BC kiểm soát kép
