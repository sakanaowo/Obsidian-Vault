---
type: concept
title: Data Augmentation
aliases:
  - Image Augmentation
  - Augmentation
tags:
  - ai
  - machine-learning
  - computer-vision
  - regularization
---

**Data Augmentation** là kỹ thuật tăng cường dữ liệu bằng cách tạo ra các biến thể của training samples. Mục đích: tăng diversity của training data, giảm overfitting, và (trong self-supervised learning) tạo các "views" khác nhau của cùng một input.

**Các loại augmentation phổ biến trong Vision**

1. **Geometric transformations**:
   - **Random crop**: Cắt vùng ngẫu nhiên từ ảnh
   - **Horizontal flip**: Lật ngang
   - **Rotation**: Xoay ảnh
   - **Scale/Resize**: Thay đổi kích thước
   
2. **Color transformations**:
   - **Color jitter**: Thay đổi brightness, contrast, saturation, hue
   - **Grayscale**: Chuyển sang trắng đen
   - **Gaussian blur**: Làm mờ
   
3. **Advanced augmentations**:
   - **Mixup**: Trộn hai ảnh với nhau
   - **CutMix**: Cắt và paste vùng từ ảnh khác
   - **RandAugment**: Áp dụng ngẫu nhiên từ pool augmentations
   - **AutoAugment**: Học policy augmentation tối ưu

**Augmentation trong Contrastive Learning**

[[Contrastive Learning]] **phụ thuộc mạnh** vào augmentation để tạo positive pairs:
- SimCLR: random crop, color distortion, Gaussian blur
- MoCo: tương tự SimCLR
- BYOL: crop + color jitter (mạnh hơn)

Nghiên cứu cho thấy nếu chỉ dùng crop (không color jitter), accuracy giảm **13-28%** (BYOL, SimCLR).

**Augmentation trong MAE**

[[Masked Autoencoders (MAE)]] có tính chất đặc biệt về augmentation:

| Configuration | Fine-tuning | Linear Probing |
|--------------|-------------|----------------|
| crop + color jit | 84.3% | 71.9% |
| crop, random size | **84.9%** | **73.5%** |
| crop, fixed size | 84.7% | 73.1% |
| none (center-crop only) | 84.0% | 65.7% |

**Quan sát quan trọng**:
1. MAE hoạt động tốt với **augmentation tối giản** (chỉ crop)
2. Color jitter **làm giảm** performance (ngược với contrastive methods)
3. MAE **vẫn hoạt động** dù không có augmentation (center-crop only)

**Tại sao MAE không cần augmentation mạnh?**

Paper MAE giải thích: trong MAE, **random masking đóng vai trò augmentation**:
- Mỗi iteration, mask pattern khác nhau → training sample mới
- Bài toán đã khó nhờ masking → không cần augmentation để regularize

Điều này khác với contrastive learning:
- Nếu không có augmentation, hai views giống hệt → trivial solution
- Augmentation tạo invariance → mô hình học features bất biến với transforms

**Trade-off trong Augmentation**

| Augmentation | Pro | Con |
|--------------|-----|-----|
| Mạnh | Tăng diversity, giảm overfit | Có thể phá hủy semantic information |
| Nhẹ | Giữ nguyên semantics | Risk of overfitting |

MAE chọn hướng **augmentation nhẹ + masking nặng**: giữ nguyên semantics trong từng image, nhưng tạo diversity thông qua random masking.
