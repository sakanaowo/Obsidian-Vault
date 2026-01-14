---
tags:
  - Resources/Paper
  - AI/Foundations
  - AI/DiffusionModels
aliases:
  - DDPM Paper
authors:
  - Jonathan Ho
  - Ajay Jain
  - Pieter Abbeel
year: 2020
link: https://arxiv.org/abs/2006.11239
---

# Denoising Diffusion Probabilistic Models

## Tổng quan
Đây là bài báo hồi sinh sự quan tâm đến các mô hình khuếch tán (Diffusion Models), chứng minh rằng chúng có thể tạo ra hình ảnh chất lượng cao tương đương hoặc vượt trội hơn GANs.

## Nội dung cốt lõi
1.  **Forward Process:** Mô hình hóa quá trình thêm nhiễu Gaussian dần dần vào dữ liệu như một chuỗi Markov cố định.
2.  **Reverse Process:** Học một mạng neural để đảo ngược quá trình này (khử nhiễu).
3.  **Objective Function:** Đơn giản hóa hàm mục tiêu thành việc dự đoán nhiễu (noise prediction) tại mỗi bước thời gian $t$, sử dụng L2 loss (Mean Squared Error).

Bài báo đặt nền móng lý thuyết và thực nghiệm cho sự bùng nổ của Generative AI trong lĩnh vực hình ảnh sau này.
