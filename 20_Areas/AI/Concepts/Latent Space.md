---
tags:
  - AI/Concept
  - AI/Math
  - Data/Representation
aliases:
  - Latent Representation
  - Embedding Space
created: 2026-01-04
---

### Định nghĩa

**Latent Space** (Không gian tiềm ẩn) là một biểu diễn nén, trừu tượng của dữ liệu, nơi các đặc điểm quan trọng (features) được mã hóa dưới dạng các vector số. Trong không gian này, các điểm dữ liệu có tính chất giống nhau sẽ nằm gần nhau.

Hãy tưởng tượng Latent Space giống như một bản đồ tư duy của AI, nơi "Vua" - "Đàn ông" + "Phụ nữ" = "Nữ hoàng".

### Latent Space trong Image Generation

Đối với các mô hình tạo ảnh (như **[[Stable Diffusion]]**), việc tính toán trực tiếp trên từng điểm ảnh (Pixel Space) của một bức ảnh 1024x1024 là quá tốn kém (3 x 1024 x 1024 giá trị).

Giải pháp là **Latent Diffusion**:
1.  **Compression (VAE Encoder):** Dùng một mạng VAE để nén ảnh từ Pixel Space xuống Latent Space (ví dụ: giảm kích thước 64 lần, nhưng giữ lại các đặc trưng ngữ nghĩa quan trọng).
2.  **Diffusion Process:** Quá trình thêm nhiễu và khử nhiễu diễn ra hoàn toàn trong không gian Latent này (nhanh hơn và ít tốn VRAM hơn).
3.  **Decompression (VAE Decoder):** Sau khi sinh ra Latent vector sạch, dùng VAE Decoder để "giải nén" nó trở lại thành ảnh Pixel Space có độ phân giải cao.

```mermaid
graph LR
    Pixel[Pixel Space (Ảnh to)] -->|Encoder| Latent[Latent Space (Vector nén)]
    Latent -->|Diffusion Process| LatentProcessed[Latent đã xử lý]
    LatentProcessed -->|Decoder| PixelOut[Pixel Space (Ảnh kết quả)]
```

### Tại sao Latent Space quan trọng?

*   **Hiệu suất:** Giảm đáng kể chi phí tính toán.
*   **Semantic Manipulation:** Dễ dàng thực hiện các thao tác sửa đổi ngữ nghĩa (ví dụ: đổi màu tóc, thêm kính) bằng cách di chuyển vector trong Latent Space thay vì sửa từng pixel thủ công.
*   **Interpolation:** Có thể tạo ra các biến thể trung gian mượt mà giữa hai hình ảnh.