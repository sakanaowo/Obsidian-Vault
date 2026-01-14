---
tags:
  - AI/Model
  - AI/Architecture
  - Concept
aliases:
  - Variational Autoencoder
created: 2026-01-04
---

### Định nghĩa

**VAE** (Variational Autoencoder) là một loại mạng neural nhân tạo dùng để học các biểu diễn dữ liệu hiệu quả (efficient data codings). Trong ngữ cảnh của Generative AI hiện đại, VAE thường được sử dụng làm thành phần nén và giải nén ảnh.

### Cấu trúc

VAE bao gồm hai phần chính:
1.  **Encoder:** Nhận đầu vào là ảnh gốc (Pixel Space), nén nó thành một biểu diễn ẩn ([[Latent Space]]) có kích thước nhỏ hơn rất nhiều nhưng vẫn giữ được các đặc trưng quan trọng.
2.  **Decoder:** Nhận đầu vào là Latent vector, tái tạo lại ảnh gốc (Pixel Space) từ biểu diễn nén đó.

### Vai trò trong Stable Diffusion

Trong **[[Stable Diffusion]]**, VAE không tham gia vào quá trình tạo sinh (diffusion) chính, mà đóng vai trò là "người vận chuyển":
*   **Bước 1 (Encoding):** Nếu dùng *Image-to-Image*, ảnh đầu vào sẽ được VAE Encoder nén xuống Latent Space.
*   **Bước 2 (Decoding):** Sau khi quá trình Diffusion (xảy ra trong Latent Space) hoàn tất và tạo ra một "Latent Image" sạch, VAE Decoder sẽ giải nén nó thành bức ảnh PNG/JPG cuối cùng mà mắt người có thể xem được.

> [!NOTE] Tại sao cần VAE?
> Nhờ VAE, Stable Diffusion có thể hoạt động trên không gian latent nhỏ (ví dụ 64x64) thay vì không gian pixel lớn (512x512), giúp giảm yêu cầu phần cứng từ Supercomputer xuống GPU cá nhân (Consumer GPU).
