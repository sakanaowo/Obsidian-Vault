---
tags:
  - AI/Model
  - AI/Architecture
  - Concept
aliases:
  - UNet Architecture
created: 2026-01-04
---

### Định nghĩa

**U-Net** là một kiến trúc mạng neural tích chập (CNN) có hình dạng giống chữ "U", ban đầu được thiết kế cho phân đoạn hình ảnh y tế (image segmentation). Tuy nhiên, trong kỷ nguyên Generative AI, nó đã trở thành "trái tim" của các mô hình **[[Diffusion Models]]**.

### Cấu trúc

U-Net gồm hai đường dẫn đối xứng:
1.  **Downsampling (Encoder):** Giảm kích thước không gian của ảnh (nén) để trích xuất các đặc trưng ngữ nghĩa mức cao (high-level features).
2.  **Upsampling (Decoder):** Tăng kích thước không gian trở lại để khôi phục chi tiết ảnh.
3.  **Skip Connections:** Các đường nối tắt giữa các lớp tương ứng của Encoder và Decoder, giúp bảo toàn thông tin chi tiết (low-level features) bị mất trong quá trình nén.

### Vai trò trong Stable Diffusion

Trong **[[Stable Diffusion]]**, nhiệm vụ của U-Net là **Dự đoán Nhiễu (Noise Prediction)**.
*   **Input:** Một Latent Image đang bị nhiễu + Thông tin thời gian (Timestep) + Thông tin điều kiện (Text Embedding từ [[CLIP]]).
*   **Process:** U-Net sử dụng các lớp **Cross-Attention** để "nhìn" vào Text Embedding và quyết định xem nên giữ lại hay bỏ đi phần nào của nhiễu để tạo ra hình ảnh khớp với mô tả.
*   **Output:** Một bản đồ nhiễu (Noise Map) dự đoán. Hệ thống sẽ lấy ảnh hiện tại trừ đi bản đồ này để có ảnh sạch hơn.

Đây là thành phần nặng nhất và tốn nhiều tài nguyên tính toán nhất trong pipeline tạo ảnh.
