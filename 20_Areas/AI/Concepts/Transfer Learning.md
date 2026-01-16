---
type: concept
title: Transfer Learning
aliases:
  - Domain Adaptation
  - Knowledge Transfer
tags:
  - ai
  - machine-learning
  - deep-learning
---

**Transfer Learning** là paradigm trong machine learning khi một mô hình được huấn luyện trên một task/domain (**source**) và kiến thức học được được chuyển giao để giải quyết task/domain khác (**target**). Ý tưởng cốt lõi: features học được từ dữ liệu lớn, tổng quát có thể hữu ích cho nhiều downstream tasks.

**Tại sao Transfer Learning quan trọng?**

1. **Data efficiency**: Target task có thể có ít dữ liệu, nhưng pre-training trên dữ liệu lớn (như ImageNet, WebText) cung cấp "prior knowledge" giúp mô hình học nhanh hơn.

2. **Compute efficiency**: Thay vì train từ scratch, chỉ cần fine-tune một phần hoặc toàn bộ pre-trained model.

3. **Better generalization**: Pre-trained features thường robust hơn, generalize tốt hơn đặc biệt khi target data ít.

**Các dạng Transfer Learning trong Deep Learning**

1. **Feature extraction (frozen backbone)**: Giữ nguyên backbone, chỉ train classifier head mới.
   - Tương đương [[Linear Probing]] trong self-supervised learning

2. **Fine-tuning**: Cập nhật toàn bộ hoặc một phần backbone cho target task.
   - [[Fine-Tuning (Transfer Learning)]] là dạng phổ biến nhất

3. **Partial fine-tuning**: Freeze phần lớn backbone, chỉ tune một vài layers cuối.
   - Paper MAE so sánh điều này giữa MAE và MoCo v3

**Transfer Learning trong MAE**

[[Masked Autoencoders (MAE)]] chứng minh transfer learning mạnh mẽ:

| Task | Dataset | MAE ViT-L | Supervised ViT-L |
|------|---------|-----------|------------------|
| Detection | COCO | 53.3 APbox | 49.3 APbox |
| Segmentation | ADE20K | 53.6 mIoU | 49.9 mIoU |
| Classification | iNaturalist | 80.1% | - |

**Điểm đáng chú ý**:
- MAE outperforms supervised pre-training on all transfer tasks
- Gain lớn hơn khi model size tăng (ViT-L gains > ViT-B gains)
- MAE thể hiện **scaling behavior** tương tự self-supervised NLP

**Source-Target Mismatch**

Một thách thức trong transfer learning là **domain shift**: source và target có distribution khác nhau. Paper MAE demo điều này với Figure 3: MAE trained trên ImageNet được áp dụng trực tiếp lên COCO images — reconstructions vẫn hợp lý, cho thấy features đã học có tính tổng quát.

**Pre-training Paradigms trong Vision**

| Paradigm | Ví dụ | Đặc điểm |
|----------|-------|---------|
| Supervised | ImageNet classification | Cần labels, có thể có label bias |
| Contrastive | MoCo, SimCLR | Không cần labels, cần augmentation |
| Generative | MAE, BEiT | Không cần labels, học structure |

MAE cho thấy **generative self-supervised learning** có thể vượt qua supervised pre-training trong transfer learning, đặc biệt với model lớn.
