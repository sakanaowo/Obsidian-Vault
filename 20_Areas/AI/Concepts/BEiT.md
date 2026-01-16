---
type: concept
title: BEiT
aliases:
  - BERT Pre-Training of Image Transformers
tags:
  - ai
  - computer-vision
  - self-supervised-learning
  - transformers
---

**BEiT (BERT Pre-Training of Image Transformers)** là một phương pháp self-supervised learning cho hình ảnh do Bao et al. (Microsoft, 2021) đề xuất. BEiT áp dụng ý tưởng **masked prediction** từ [[BERT]] vào vision, nhưng với một twist: thay vì dự đoán pixel trực tiếp, BEiT dự đoán **discrete visual tokens** được tạo bởi một tokenizer (dVAE từ DALL-E).

**Cơ chế hoạt động**

1. **Tokenization**: Sử dụng **discrete VAE (dVAE)** được pre-train từ DALL-E (250M images) để chuyển ảnh thành một lưới visual tokens rời rạc. Mỗi patch 16×16 tương ứng với một token từ vocabulary ~8192.

2. **Masking**: Che một phần patches (thường ~40%) theo **block-wise sampling** — che các vùng liên tục thay vì random.

3. **Prediction**: Encoder (ViT) xử lý toàn bộ patches (bao gồm mask tokens), và dự đoán visual token ID cho các vị trí bị che bằng **cross-entropy loss**.

**So sánh BEiT vs MAE**

| Aspect | BEiT | [[Masked Autoencoders (MAE)]] |
|--------|------|-----|
| Reconstruction target | Discrete tokens (dVAE) | Continuous pixels |
| Tokenizer | Cần pre-train dVAE (250M images) | Không cần |
| Masking ratio | ~40% | 75% |
| Mask token in encoder | Có | **Không** |
| Masking strategy | Block-wise | Random |
| Loss | Cross-entropy | MSE |

**Nhược điểm của BEiT so với MAE**

Paper MAE (He et al., 2022) chỉ ra một số nhược điểm của BEiT:

1. **Cần tokenizer phức tạp**: dVAE cần pre-train riêng trên dataset lớn (250M images), thêm một stage training và không reproducible nếu không có dữ liệu DALL-E.

2. **Overhead compute**: dVAE encoder là CNN lớn chiếm ~40% FLOPs của ViT-L, thêm overhead đáng kể.

3. **Token không cần thiết**: Khi BEiT dự đoán pixel thay vì token, accuracy giảm 1.8%. Nhưng MAE cho thấy với thiết kế phù hợp (normalized pixels, không mask token trong encoder), pixel works as well or better.

4. **Mask token trong encoder**: BEiT đưa mask token vào encoder, tạo gap giữa pre-training và deployment (downstream không có mask token). MAE tránh điều này bằng cách chỉ đưa mask token vào decoder.

**Kết quả so sánh**

| Method | Pre-train Data | ViT-B | ViT-L |
|--------|---------------|-------|-------|
| BEiT | IN1K + DALLE (250M) | 83.2% | 85.2% |
| MAE | IN1K only | **83.6%** | **85.9%** |

MAE đạt accuracy cao hơn BEiT dù chỉ dùng ImageNet-1K (1.3M images) so với BEiT cần 250M images cho tokenizer. MAE cũng nhanh hơn **3.5× per epoch**.

**Bài học từ BEiT**

BEiT là bước đệm quan trọng cho thấy masked prediction có thể hoạt động trong vision. Tuy nhiên, MAE chứng minh rằng **complexity của tokenization là không cần thiết** nếu giải quyết đúng các thách thức khác (masking ratio, asymmetric architecture, no mask token in encoder).
