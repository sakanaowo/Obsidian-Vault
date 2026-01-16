---
type: concept
title: Contrastive Learning
aliases:
  - Contrastive Self-Supervised Learning
  - Contrastive Representation Learning
tags:
  - ai
  - self-supervised-learning
  - representation-learning
---

**Contrastive Learning** là một paradigm **self-supervised learning** trong đó mô hình học bằng cách **phân biệt** các mẫu tương tự (**positive pairs**) khỏi các mẫu không tương tự (**negative pairs**). Trực giác: đẩy các representation của positive pairs lại gần nhau trong không gian latent, đồng thời đẩy negative pairs ra xa.

Trong thị giác máy tính, positive pairs thường được tạo bằng **data augmentation**: hai "views" khác nhau của cùng một ảnh (crop, color jitter, blur, v.v.) là positive pair; hai ảnh khác nhau là negative pair. Các phương pháp contrastive nổi tiếng bao gồm **SimCLR** (Chen et al., 2020), **MoCo** (He et al., 2020), và **BYOL** (Grill et al., 2020).

**Cơ chế toán học**

Hàm loss contrastive điển hình (InfoNCE) có dạng:
$$
\mathcal{L} = -\log \frac{\exp(\text{sim}(z_i, z_j^+) / \tau)}{\sum_{k=1}^{N} \exp(\text{sim}(z_i, z_k) / \tau)}
$$

Trong đó:
- $z_i, z_j^+$: representations của positive pair
- $z_k$: representations của tất cả samples (positive + negatives)
- $\tau$: temperature parameter
- $\text{sim}$: hàm similarity (thường là cosine similarity)

**So sánh với MAE**

Contrastive learning và [[Masked Autoencoders (MAE)]] đại diện cho hai nhánh self-supervised learning khác nhau:

| Aspect | Contrastive Learning | MAE |
|--------|---------------------|-----|
| Paradigm | Discriminative (so sánh) | Generative (tái tạo) |
| Data augmentation | **Bắt buộc** (tạo views) | Tối thiểu (masking đóng vai trò augmentation) |
| Học gì | Invariance đối với augmentations | Cấu trúc tín hiệu (gestalt) |
| Negative samples | Cần (trừ BYOL) | Không cần |

**Điểm yếu của Contrastive Learning**

1. **Phụ thuộc mạnh vào augmentation**: Chất lượng representation phụ thuộc vào việc chọn augmentation phù hợp. Augmentations quá mạnh có thể phá hủy thông tin hữu ích; quá yếu có thể dẫn đến trivial solution.

2. **Cần nhiều negative samples**: SimCLR cần batch size rất lớn (4096+) để có đủ negatives. MoCo giải quyết bằng memory bank, nhưng vẫn phức tạp hơn MAE.

3. **Không học tái tạo**: Contrastive learning không yêu cầu mô hình hiểu chi tiết pixel-level, có thể bỏ qua thông tin hữu ích cho một số downstream tasks.

Paper MAE (He et al., 2022) so sánh trực tiếp: MAE hoạt động **không cần augmentation** (chỉ center-crop, no flipping), trong khi contrastive methods giảm 13-28% accuracy nếu chỉ dùng crop. Điều này cho thấy MAE học bằng cách "mô hình hóa cấu trúc tín hiệu" thay vì "học bất biến đối với augmentations".

**Các phương pháp tiêu biểu**
- **MoCo v1/v2/v3**: Momentum Contrast, sử dụng memory bank
- **SimCLR v1/v2**: Simple Contrastive Learning, cần large batch
- **BYOL**: Bootstrap Your Own Latent, không cần negative samples
- **SwAV**: Swapping Assignments between Views, kết hợp clustering
- **DINO**: Self-Distillation with no Labels, kết hợp knowledge distillation
