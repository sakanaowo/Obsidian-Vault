---
type: concept
title: ImageNet
aliases:
  - ImageNet-1K
  - ILSVRC
  - IN1K
tags:
  - ai
  - datasets
  - computer-vision
  - benchmarks
---

**ImageNet** là một trong những dataset và benchmark quan trọng nhất trong lịch sử computer vision. Được tạo bởi Fei-Fei Li và đội ngũ tại Stanford (2009), ImageNet đã đóng vai trò then chốt trong cuộc cách mạng deep learning, đặc biệt sau chiến thắng của AlexNet tại ILSVRC 2012.

**ImageNet-1K (ILSVRC)**

Phiên bản thường được sử dụng nhất là **ImageNet-1K** (hay ILSVRC - ImageNet Large Scale Visual Recognition Challenge):
- **1,000 classes**: 1,000 categories phân cấp theo WordNet hierarchy
- **~1.28 million training images**: Trung bình ~1,300 images per class
- **50,000 validation images**: 50 images per class
- **Image size**: Đa dạng, thường resize về 224×224 hoặc 256×256

**Vai trò trong nghiên cứu**

1. **Benchmark tiêu chuẩn**: ImageNet-1K top-1 accuracy là thước đo de facto cho image classification models.

2. **Pre-training dataset**: Các mô hình được pre-train trên ImageNet và transfer sang downstream tasks.

3. **Lịch sử**: 
   - 2012: AlexNet (58.9%) → Deep learning explosion
   - 2015: ResNet (96.4%) → Deeper is better
   - 2020: ViT (88.5% với JFT-300M) → Transformers cho vision
   - 2022: MAE ViT-H (87.8% chỉ với IN1K) → Self-supervised scaling

**ImageNet trong MAE**

[[Masked Autoencoders (MAE)]] sử dụng ImageNet-1K làm **pre-training và evaluation dataset**:

| Model | Pre-train Data | Top-1 Acc |
|-------|---------------|-----------|
| ViT-L supervised | IN1K | 82.6% |
| ViT-L MAE | IN1K | **85.9%** |
| ViT-H MAE | IN1K | **86.9%** |
| ViT-H MAE (448) | IN1K | **87.8%** |

**Điểm đáng chú ý**: MAE đạt state-of-the-art trong nhóm **chỉ dùng IN1K** (không dùng external data như JFT-300M hay DALL-E 250M). Điều này quan trọng vì:
- **Reproducibility**: IN1K publicly available, JFT-300M không
- **Fair comparison**: So sánh methods trên cùng dữ liệu

**ImageNet và External Data**

Một số methods sử dụng external data để boost performance:
- **JFT-300M** (Google): 300M images, not public
- **DALL-E data** (OpenAI): 250M images, used for BEiT tokenizer
- **Instagram-1B**: 1B images, used for semi-supervised learning

MAE chứng minh rằng với self-supervised learning được thiết kế tốt, **ImageNet-1K đủ để train ViT-Huge** mà không bị overfitting.

**Các dataset liên quan**

| Dataset | Size | Classes | Use case |
|---------|------|---------|----------|
| ImageNet-1K | 1.28M | 1,000 | Classification benchmark |
| ImageNet-21K | 14M | 21,841 | Pre-training (larger) |
| COCO | 118K | 80 | Object detection, segmentation |
| ADE20K | 20K | 150 | Semantic segmentation |
| iNaturalist | 675K | 5,089 | Fine-grained classification |
| Places | 1.8M | 365 | Scene classification |

MAE được đánh giá trên tất cả các dataset trên để chứng minh transfer learning mạnh mẽ.
