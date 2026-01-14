---
tags:
  - Resources/Paper
  - AI/Security
  - AI/Ethics
aliases:
  - Diffusion Memorization Paper
authors:
  - Nicholas Carlini
  - et al.
year: 2023
link: https://arxiv.org/abs/2301.13188
---

# Extracting Training Data from Diffusion Models

## Tổng quan
Bài báo này nghiên cứu về vấn đề bảo mật và riêng tư của các mô hình khuếch tán (Diffusion Models) như Stable Diffusion và Imagen.

## Vấn đề
Các tác giả chứng minh rằng Diffusion Models có khả năng "ghi nhớ" (memorize) dữ liệu huấn luyện nhiều hơn so với các mô hình GAN trước đây.
*   Họ đã trích xuất thành công hàng nghìn hình ảnh từ dữ liệu huấn luyện (bao gồm ảnh cá nhân, ảnh có bản quyền) chỉ bằng cách query mô hình.

## Ý nghĩa
Nghiên cứu này dấy lên hồi chuông cảnh báo về rủi ro rò rỉ dữ liệu (Privacy Leakage) và bản quyền trong kỷ nguyên Generative AI, thúc đẩy các thảo luận về đạo đức và pháp lý.
