---
tags:
  - Resources/Paper
  - AI/ImageGeneration
  - AI/Google
aliases:
  - Imagen Paper
authors:
  - Chitwan Saharia
  - William Chan
  - et al. (Google Research)
year: 2022
link: https://arxiv.org/abs/2205.11487
---

# Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding

## Tổng quan
Bài báo giới thiệu **Imagen**, mô hình tạo ảnh của Google, cạnh tranh trực tiếp với DALL-E 2.

## Phát hiện quan trọng
Imagen không sử dụng CLIP hay Latent Diffusion. Thay vào đó, nó sử dụng một **Large Language Model (LLM)** cực lớn (như T5-XXL) để mã hóa văn bản.
*   **Key Insight:** Sức mạnh của bộ mã hóa văn bản (Text Encoder) quan trọng hơn việc tăng kích thước của mô hình tạo ảnh (Diffusion Model) để đạt được độ chính xác ngữ nghĩa (prompt adherence) và chất lượng hình ảnh.

Bài báo khẳng định xu hướng kết hợp sức mạnh hiểu ngôn ngữ của LLM vào các tác vụ thị giác.
