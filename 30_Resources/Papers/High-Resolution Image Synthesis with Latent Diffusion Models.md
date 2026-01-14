---
tags:
  - Resources/Paper
  - AI/ImageGeneration
  - AI/DiffusionModels
aliases:
  - Stable Diffusion Paper
  - LDM Paper
authors:
  - Robin Rombach
  - Andreas Blattmann
  - Dominik Lorenz
  - Patrick Esser
  - Björn Ommer
year: 2022
link: https://arxiv.org/abs/2112.10752
---

# High-Resolution Image Synthesis with Latent Diffusion Models

## Tổng quan
Đây là bài báo khoa học nền tảng giới thiệu mô hình **Latent Diffusion Models (LDMs)**, cơ sở của **Stable Diffusion**.

## Đóng góp chính
Bài báo giải quyết vấn đề chi phí tính toán khổng lồ của các mô hình Diffusion hoạt động trên không gian pixel (Pixel Space).
1.  **Perceptual Compression:** Sử dụng một Autoencoder (VAE) để nén ảnh vào một không gian tiềm ẩn ([[Latent Space]]) có chiều thấp hơn nhưng vẫn bảo toàn ngữ nghĩa.
2.  **Latent Diffusion:** Quá trình khuếch tán (Diffusion) và khử nhiễu (Denoising) diễn ra hoàn toàn trong không gian latent này.
3.  **Cross-Attention Conditioning:** Tích hợp cơ chế Attention để điều khiển quá trình sinh ảnh bằng văn bản (Text Prompts), layout, hoặc các điều kiện khác.

## Kết quả
Phương pháp này cho phép huấn luyện các mô hình tạo ảnh chất lượng cao trên các GPU phổ thông (Consumer GPUs) thay vì cần siêu máy tính, mở ra kỷ nguyên "Dân chủ hóa AI Art".
