---
title: "Residual Connection"
aliases:
  [
    "Residual Link",
    "Skip Connection",
    "shortcut connection",
    "kết nối tắt",
    "kết nối dư",
  ]
tags: [concept, deep-learning, architecture, resnet, gradient-flow]
created: 2026-04-11
---

# Residual Connection (Kết nối Dư)

> [!NOTE] ELI5
> Hãy tưởng tượng bạn đang viết bài luận. Thay vì viết lại hoàn toàn từ đầu, bạn lấy bản cũ rồi **chỉ viết thêm phần sửa đổi** — phần mới = phần cũ + chỉnh sửa. Residual connection hoạt động y hệt: thay vì layer học hàm $f(x)$ từ đầu, nó chỉ học "phần hiệu chỉnh" $g(x) = f(x) - x$, rồi cộng với input gốc: output $= x + g(x)$.

**Residual Connection** (hay skip connection) là kỹ thuật kiến trúc cho phép input $\mathbf{x}$ **bỏ qua** một hoặc nhiều layers và được cộng trực tiếp vào output của các layers đó. Công thức tổng quát:

$$\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$$

trong đó $\mathcal{F}(\mathbf{x})$ là phần biến đổi (thường là 2 conv layers + BN + ReLU), và $\mathbf{x}$ là "shortcut path" không qua biến đổi. Layer chỉ cần học **phần dư** (residual) $\mathcal{F}(\mathbf{x}) = \mathbf{y} - \mathbf{x}$, thay vì học toàn bộ mapping $\mathbf{y} = f(\mathbf{x})$.

## Tại sao cần Residual Connection?

### 1. Degradation Problem

Khi mạng ngày càng sâu (> ~20 layers), training accuracy **giảm** — không phải overfitting mà là tối ưu hóa thất bại. Nguyên nhân: gradient vanishing.

### 2. Gradient Flow qua Shortcut

Đạo hàm theo chain rule:
$$\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \frac{\partial \mathcal{F}}{\partial \mathbf{x}} + \mathbf{I}$$

Thành phần $\mathbf{I}$ (identity matrix) đảm bảo **gradient luôn chảy ngược** qua shortcut path, ngay cả khi $\frac{\partial \mathcal{F}}{\partial \mathbf{x}}$ rất nhỏ. Kết quả: có thể train mạng 100–1000+ layers.

### 3. Học Hàm Tầm Thường (Trivial Mapping)

Nếu layer không cần thay đổi gì, nó chỉ cần học $\mathcal{F}(\mathbf{x}) \approx 0$ (dễ hơn nhiều so với học identity mapping $f(\mathbf{x}) = \mathbf{x}$ từ đầu).

## Điều kiện để dùng Addition

Addition yêu cầu $\mathbf{x}$ và $\mathcal{F}(\mathbf{x})$ **cùng shape**. Khi channels thay đổi (e.g., 64 → 128), cần dùng **1×1 convolution** trên shortcut path để chiếu (project) lên cùng số channels:

$$\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{W}_s \mathbf{x}$$

## Residual Connection vs Dense Connection

|                  | Residual (ResNet)    | Dense (DenseNet)         |
| ---------------- | -------------------- | ------------------------ |
| Phép kết hợp     | **Addition** (+)     | **Concatenation** (cat)  |
| Nguồn skip       | Chỉ layer ngay trước | TẤT CẢ layers trước      |
| Channels         | Cố định              | Tăng tuyến tính (+k)     |
| Factor kiểm soát | Projection conv      | Growth rate + Transition |

## Liên kết

- Đã học chi tiết ở [[Buổi 34 - Tuần 9]] (ResNet architecture)
- Mở rộng sang dense connections trong [[Buổi 35 - Tuần 9]] (DenseNet)
- Liên quan: [[Skip Connection]], [[Batch Normalization]], [[Grouped Convolution]]
- Source: [d2l.ai — 8.6 Residual Networks](https://d2l.ai/chapter_convolutional-modern/resnet.html)

---

> [!TODO]
>
> - Pre-activation ResNet (ResNet v2): BN → ReLU → Conv vs Conv → BN → ReLU
> - Wide ResNet: tăng width thay depth
> - Resnext: grouped convolution kết hợp với residual
> - Liên hệ với Highway Networks (cùng thời kỳ)
