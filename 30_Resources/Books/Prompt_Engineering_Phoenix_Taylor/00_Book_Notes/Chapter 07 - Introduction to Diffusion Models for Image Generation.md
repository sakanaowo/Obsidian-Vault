---
tags:
  - Resources/BookNote
  - AI/ImageGeneration
  - AI/DiffusionModels
  - AI/History
created: 2026-01-04
source: [[Prompt Engineering for Generative AI Future-Proof Inputs for Reliable Al Outputs (James Phoenix, Mike Taylor) (Z-Library).pdf]]
author: James Phoenix, Mike Taylor
---

# Chapter 07: Introduction to Diffusion Models for Image Generation

## 1. Sự trỗi dậy của Diffusion Models

Chương này mở đầu bằng sự chuyển dịch mô hình (paradigm shift) trong lĩnh vực tạo sinh hình ảnh. Trước năm 2020, **GANs (Generative Adversarial Networks)** thống trị nhưng gặp khó khăn trong việc huấn luyện và đa dạng hóa đầu ra. Sự xuất hiện của **[[Diffusion Models]]** (đặc biệt là DALL-E 2 năm 2022) đã thay đổi tất cả.

### Nguyên lý hoạt động (The Physics of Noise)
Diffusion Models lấy cảm hứng từ vật lý (nhiệt động lực học).
1.  **Forward Process (Phá hủy):** Tưởng tượng một giọt mực loang dần trong cốc nước cho đến khi nước đục ngầu. Trong ảnh, ta thêm nhiễu Gaussian (Gaussian noise) từ từ cho đến khi ảnh trở thành nhiễu trắng ngẫu nhiên.
2.  **Reverse Process (Tái tạo):** Dạy một mạng neural (AI) cách "đảo ngược" quá trình trên. Từ một đám nhiễu hỗn độn, AI dự đoán và loại bỏ từng lớp nhiễu để khôi phục lại "giọt mực" (bức ảnh) ban đầu.

![[fig_7-1_Diffusion_schematics_These_models_were_trained_on_.png]]
*Figure 7-1: Sơ đồ quá trình Forward và Reverse Diffusion.*

## 2. Giải phẫu một hệ thống tạo ảnh (Anatomy of Image Gen)

Để hiểu sâu hơn, sách phân tích kiến trúc của **[[Stable Diffusion]]** (một dạng Latent Diffusion Model - LDM). Nó không xử lý trực tiếp trên điểm ảnh (pixels) mà trên không gian tiềm ẩn ([[Latent Space]]).

![[fig_7-2_Encoding_and_decoding_process_These_vectors,_also_.png]]
*Figure 7-2: Quá trình Encoding và Decoding thông qua VAE.*

Hệ thống gồm 3 thành phần chính phối hợp nhịp nhàng:

1.  **[[VAE]] (Variational Autoencoder):** Bộ nén ảnh.
    *   *Encoder:* Nén ảnh to (Pixel) -> Vector nhỏ (Latent). Giúp giảm tải tính toán.
    *   *Decoder:* Giải nén Vector (Latent) -> Ảnh to (Pixel) để người dùng xem.
2.  **[[U-Net]]:** "Người thợ vẽ".
    *   Hoạt động trong Latent Space.
    *   Nhiệm vụ: Dự đoán nhiễu tại mỗi bước.
    *   Sử dụng cơ chế **Cross-Attention** để nhận chỉ đạo từ văn bản (Prompt).
3.  **[[CLIP]] (Text Encoder):** "Người phiên dịch".
    *   Chuyển đổi Prompt văn bản ("Con mèo lái xe") thành các vector embedding mà U-Net có thể hiểu được.

## 3. Không gian tiềm ẩn (Latent Space Walk)

Khái niệm **[[Latent Space]]** là chìa khóa để hiểu khả năng sáng tạo của AI. Mọi hình ảnh có thể có (possible images) đều nằm đâu đó trong không gian đa chiều này.

*   Khi bạn thay đổi prompt từ "chó" sang "mèo", bạn đang di chuyển vector điều hướng trong không gian latent.
*   Việc di chuyển mượt mà giữa hai điểm trong không gian này tạo ra hiệu ứng biến hình (morphing) giữa hai bức ảnh.

![[fig_7-3_A_random_walk_through_latent_space_Within_the_doma.png]]
*Figure 7-3: Minh họa việc di chuyển ngẫu nhiên trong Latent Space tạo ra các biến thể ảnh.*

## 4. Các "Ông lớn" trong ngành (Key Players)

Chương này so sánh chi tiết các mô hình hàng đầu:

### OpenAI DALL-E
*   **DALL-E 1 (2021):** Sử dụng dVAE, chưa phải Diffusion thuần túy.
*   **DALL-E 2 (2022):** Bước ngoặt sử dụng Diffusion. Tích hợp tính năng Inpainting (vẽ thêm vào ảnh) và Outpainting (mở rộng ảnh).
*   **Đặc điểm:** Dễ dùng, hiểu prompt tốt, nhưng là mã nguồn đóng (Closed Source).

![[fig_7-4_DALL-E_capabilities_The_DALL-E_model_was_not_open_.png]]
*Figure 7-4: Các khả năng của DALL-E (Variation, Inpainting).*

### Midjourney
*   **Định hướng:** Tập trung vào tính nghệ thuật (Artistic) và thẩm mỹ (Aesthetics) hơn là tả thực (Photorealism).
*   **Nền tảng:** Hoạt động chủ yếu qua Discord.
*   **Ưu điểm:** Chất lượng ánh sáng, bố cục cực tốt ngay cả với prompt ngắn.
*   **Nhược điểm:** Khó kiểm soát chi tiết, không có API chính thức (tại thời điểm viết sách).

![[fig_7-8_Midjourney’s_Discord_server,_July_2022_When_you_fi.png]]
*Figure 7-8: Giao diện tạo ảnh qua Discord của Midjourney.*

### Stable Diffusion (Stability AI)
*   **Triết lý:** Mã nguồn mở (Open Source). Dân chủ hóa AI.
*   **Sức mạnh:** Cộng đồng cực lớn. Có thể chạy offline trên GPU cá nhân.
*   **Tùy biến:** Hỗ trợ Fine-tuning ([[LoRA]], DreamBooth) và ControlNet để kiểm soát cấu trúc ảnh.

![[fig_7-11_AUTOMATIC1111’s_web_UI_for_Stable_Diffusion_Versio.png]]
*Figure 7-11: Giao diện AUTOMATIC1111 - công cụ phổ biến nhất để chạy Stable Diffusion.*

## 5. Đạo đức và Bản quyền

Sự bùng nổ của AI Art dẫn đến tranh cãi lớn:
*   **Dataset:** Các mô hình (như Stable Diffusion) được train trên tập dữ liệu khổng lồ **LAION-5B** (5 tỷ cặp ảnh-text từ internet), bao gồm cả ảnh có bản quyền.
*   **Style Mimicry:** AI có thể nhại lại phong cách của nghệ sĩ đang sống chỉ trong vài giây.
*   **Deepfakes:** Nguy cơ tạo ảnh giả mạo người nổi tiếng hoặc nội dung độc hại.

---
**Liên kết:** [[Chapter 06 - Autonomous Agents with Memory and Tools]] | [[Chapter 08 - Standard Practices for Image Generation]]

## 6. Tài liệu tham khảo & Đọc thêm (Seminal Papers)

Chương này được xây dựng dựa trên các nghiên cứu nền tảng sau:

1.  **Stable Diffusion (LDM):** [[High-Resolution Image Synthesis with Latent Diffusion Models]] - Bài báo giới thiệu kiến trúc Latent Diffusion giúp giảm chi phí tính toán.
2.  **CLIP:** [[Learning Transferable Visual Models From Natural Language Supervision]] - Mô hình kết nối văn bản và hình ảnh của OpenAI.
3.  **DALL-E 2:** [[Hierarchical Text-Conditional Image Generation with CLIP Latents]] - Kiến trúc unCLIP của OpenAI.
4.  **Diffusion Foundations:** [[Denoising Diffusion Probabilistic Models]] - Bài báo hồi sinh mô hình khuếch tán (DDPM).
5.  **Imagen:** [[Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding]] - Cách tiếp cận của Google sử dụng LLM lớn làm bộ mã hóa văn bản.
6.  **Security & Ethics:** [[Extracting Training Data from Diffusion Models]] - Nghiên cứu về rủi ro rò rỉ dữ liệu huấn luyện.
