---
tags:
  - Resources/Paper
  - AI/ImageGeneration
  - AI/DALL-E
aliases:
  - DALL-E 2 Paper
authors:
  - Aditya Ramesh
  - Prafulla Dhariwal
  - Alex Nichol
  - Casey Chu
  - Mark Chen
year: 2022
link: https://arxiv.org/abs/2204.06125
---

# Hierarchical Text-Conditional Image Generation with CLIP Latents

## Tổng quan
Bài báo mô tả kiến trúc của **DALL-E 2**, mô hình tạo ảnh nổi tiếng của OpenAI.

## Kiến trúc (unCLIP)
Hệ thống gồm hai giai đoạn chính:
1.  **Prior:** Chuyển đổi caption văn bản thành CLIP image embedding.
2.  **Decoder:** Chuyển đổi CLIP image embedding thành hình ảnh thực tế (sử dụng Diffusion Model).

## Điểm nhấn
Bài báo chứng minh rằng việc sử dụng biểu diễn ảnh của CLIP (CLIP image latents) làm điều kiện đầu vào giúp tạo ra hình ảnh có độ đa dạng cao và bám sát ngữ nghĩa văn bản hơn so với các phương pháp trước đó. Nó cũng giới thiệu khả năng tạo biến thể (variations) của ảnh một cách mạnh mẽ.
