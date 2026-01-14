---
tags:
  - Resources/Paper
  - AI/Multimodal
  - AI/VisionLanguage
aliases:
  - CLIP Paper
authors:
  - Alec Radford
  - Jong Wook Kim
  - et al. (OpenAI)
year: 2021
link: https://arxiv.org/abs/2103.00020
---

# Learning Transferable Visual Models From Natural Language Supervision

## Tổng quan
Bài báo giới thiệu **CLIP (Contrastive Language-Image Pre-training)**, một mô hình học biểu diễn (representation learning) kết nối hình ảnh và văn bản.

## Cơ chế (Contrastive Learning)
Thay vì huấn luyện phân loại ảnh theo các nhãn cố định (như ImageNet), CLIP được huấn luyện trên 400 triệu cặp (ảnh, văn bản) thu thập từ internet.
*   Mô hình gồm 2 nhánh: Image Encoder và Text Encoder.
*   **Mục tiêu:** Tối ưu hóa sao cho vector embedding của ảnh và văn bản mô tả tương ứng có độ tương đồng cosine cao nhất (nằm gần nhau trong không gian vector), trong khi đẩy xa các cặp không khớp.

## Tầm quan trọng
CLIP là thành phần không thể thiếu trong các hệ thống Text-to-Image hiện đại (như DALL-E 2, Stable Diffusion). Nó đóng vai trò là "người phiên dịch", chuyển đổi prompt của người dùng thành tín hiệu mà mô hình tạo ảnh có thể hiểu được.
