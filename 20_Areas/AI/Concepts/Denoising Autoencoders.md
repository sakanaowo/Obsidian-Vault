---
type: concept
title: Denoising Autoencoders
aliases:
  - DAE
  - Denoising AE
tags:
  - ai
  - autoencoders
  - representation-learning
  - self-supervised-learning
---

**Denoising Autoencoders (DAE)** là một biến thể của [[Autoencoders]] trong đó input bị **làm hỏng có chủ đích** (corrupted), và mô hình được huấn luyện để **tái tạo lại input gốc** (không bị hỏng). Ý tưởng được đề xuất bởi Vincent et al. (2008, 2010) và là một trong những phương pháp representation learning cổ điển.

**Cơ chế hoạt động**

1. **Corruption**: Thêm nhiễu vào input $x$ để tạo phiên bản bị hỏng $\tilde{x}$. Các loại corruption phổ biến:
   - **Additive Gaussian noise**: $\tilde{x} = x + \epsilon$, với $\epsilon \sim \mathcal{N}(0, \sigma^2)$
   - **Masking noise**: Đặt một phần input về 0 (hoặc giá trị trung bình)
   - **Salt-and-pepper noise**: Đặt ngẫu nhiên một số pixel về min/max
   
2. **Encoding**: Encoder $f$ ánh xạ input bị hỏng sang latent: $z = f(\tilde{x})$

3. **Decoding**: Decoder $g$ tái tạo input gốc: $\hat{x} = g(z)$

4. **Loss**: So sánh reconstruction với input **gốc** (không phải input bị hỏng):
   $$\mathcal{L} = \mathbb{E}[\| x - g(f(\tilde{x})) \|^2]$$

**Tại sao DAE học được representation tốt?**

Trực giác: để tái tạo được input gốc từ input bị hỏng, mô hình phải học **cấu trúc** của dữ liệu — không chỉ copy input. Corruption buộc mô hình phải "hiểu" distribution của dữ liệu để "điền vào" phần bị mất/bị nhiễu.

Từ góc nhìn toán học, DAE có thể được hiểu như **score matching**: học gradient của log-density $\nabla_x \log p(x)$. Điều này liên kết DAE với [[Diffusion Models]] — cũng học score function để sinh dữ liệu.

**MAE như một dạng DAE**

[[Masked Autoencoders (MAE)]] có thể được coi là **DAE hiện đại** với các cải tiến:

| Aspect | Classical DAE | MAE |
|--------|---------------|-----|
| Corruption type | Noise, masking nhẹ | Masking nặng (75% patches) |
| Architecture | Symmetric encoder-decoder | Asymmetric (encoder nhẹ hơn) |
| Loss | Trên toàn bộ output | Chỉ trên phần bị mask |
| Backbone | MLP, CNN | Transformer (ViT) |

Paper MAE (He et al., 2022) nhấn mạnh rằng MAE là "a form of denoising autoencoding, but different from the classical DAE in numerous ways". Sự khác biệt chính là:
1. **Masking ratio cực cao** (75% vs ~10-30% trong DAE cổ điển)
2. **Encoder không thấy mask token** → tránh distribution mismatch
3. **Decoder xử lý full set** → encoder có thể focus vào semantic features

**Lịch sử và ảnh hưởng**

DAE là một trong những phương pháp unsupervised representation learning đầu tiên cho thấy pre-training có thể cải thiện downstream performance. Trước deep learning "bùng nổ" (pre-2012), DAE và Restricted Boltzmann Machines (RBM) là các phương pháp chính để pre-train mạng sâu. Ngày nay, với sự phát triển của Transformers và self-supervised learning quy mô lớn, các biến thể hiện đại của DAE (như MAE) đang trở lại với hiệu quả vượt trội.
