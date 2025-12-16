---
tags:
  - diffusion-models
  - generative-ai
  - image-generation
  - deep-learning
status: in_progress
created: 2025-12-10
source:
  - Prompt Engineering for Generative AI (James Phoenix, Mike Taylor)
  - RAG roadmap.pdf
---

# Diffusion Models

## Định nghĩa và Tổng quan

**Diffusion Models** là một loại mô hình tạo sinh (generative models) đã cho thấy những kết quả vượt trội trong việc tạo ra hình ảnh từ văn bản. Chúng được giới thiệu vào năm 2015 và trở nên phổ biến rộng rãi với sự ra đời của DALL-E 2, Stable Diffusion, và Midjourney. Các mô hình này đóng vai trò quan trọng trong lĩnh vực [[Image Generation]].

## Cơ chế Hoạt động

Các mô hình Diffusion hoạt động dựa trên một quy trình hai bước chính:

1.  **Quy trình làm nhiễu (Forward/Noising Process):**
    *   Mô hình từ từ thêm nhiễu ngẫu nhiên (random noise) vào một hình ảnh cho đến khi nó trở thành nhiễu hoàn toàn, không thể nhận dạng được.
    *   Quá trình này được lấy cảm hứng từ vật lý, mô phỏng cách các hạt khuếch tán (spread out) qua một môi trường.

2.  **Quy trình khử nhiễu (Reverse/Denoising Process):**
    *   Đây là bước tạo sinh chính. Mô hình được huấn luyện để dự đoán cách đảo ngược quy trình làm nhiễu, tức là loại bỏ nhiễu từng bước để khôi phục lại hình ảnh gốc.
    *   Các dự đoán khử nhiễu này được điều kiện hóa (conditioned) dựa trên mô tả văn bản (prompt) mà người dùng cung cấp. Nếu hình ảnh tạo ra không khớp với mô tả, trọng số của mạng nơ-ron sẽ được điều chỉnh để cải thiện độ chính xác.
    *   Khi được huấn luyện, mô hình có thể lấy nhiễu ngẫu nhiên và biến nó thành một hình ảnh khớp với mô tả trong prompt.

## Các Khái niệm Chính

*   **Latent Space (Không gian tiềm ẩn):** Là một không gian đa chiều trừu tượng nơi các hình ảnh được biểu diễn dưới dạng các [[Vector Representations]]. Các hình ảnh có nét tương đồng sẽ nằm gần nhau trong không gian này. Prompt được mã hóa thành vector trong không gian tiềm ẩn, và mô hình diffusion tạo ra hình ảnh phù hợp với vector đó.
*   **Embeddings:** Các vector số học đại diện cho văn bản hoặc hình ảnh, hoạt động như một "vị trí" hoặc "địa chỉ" trong không gian tiềm ẩn của mô hình.
*   **Denoising Strength:** Kiểm soát mức độ nhiễu được thêm vào hoặc loại bỏ, ảnh hưởng đến độ giống của hình ảnh tạo ra so với hình ảnh gốc (nếu có).

## Các Mô hình Diffusion Nổi bật

*   **DALL-E (OpenAI):** Nổi tiếng với khả năng tạo ra hình ảnh độc đáo và nghệ thuật. DALL-E 2 là một bước đột phá, và DALL-E 3 được tích hợp vào ChatGPT.
*   **Midjourney:** Phổ biến trong cộng đồng nghệ sĩ AI vì khả năng tạo ra hình ảnh chất lượng cao với phong cách thẩm mỹ đặc trưng, đặc biệt là trong lĩnh vực giả tưởng và siêu thực.
*   **Stable Diffusion (Stability AI):** Một mô hình mã nguồn mở, cho phép người dùng chạy cục bộ trên máy tính của họ (yêu cầu GPU). Stable Diffusion nổi bật về tính linh hoạt và khả năng tùy chỉnh, với một cộng đồng lớn đóng góp các tính năng mở rộng như [[ControlNet]] và [[DreamBooth Fine-Tuning]].
    *   **Stable Diffusion XL (SDXL):** Phiên bản nâng cấp với số lượng tham số lớn hơn, mang lại kết quả chất lượng cao hơn và đa dạng hơn về kích thước hình ảnh.

## Ứng dụng và Kỹ thuật

*   **Prompting with an Image (Img2Img):** Sử dụng một hình ảnh hiện có làm đầu vào cùng với prompt văn bản để hướng dẫn quá trình tạo hình ảnh, giúp kiểm soát tốt hơn phong cách và bố cục.
*   **Inpainting:** Chỉnh sửa các phần cụ thể của hình ảnh bằng cách xóa chúng và thêm prompt để tạo nội dung mới vào vùng đã xóa.
*   **Outpainting:** Mở rộng hình ảnh ra ngoài khung ban đầu, tạo thêm ngữ cảnh hoặc các yếu tố mới xung quanh hình ảnh hiện có.
*   **Negative Prompts:** Chỉ định các yếu tố hoặc khái niệm mà bạn *không* muốn xuất hiện trong hình ảnh tạo ra.
*   **Weighted Terms:** Gán trọng số khác nhau cho các từ trong prompt để kiểm soát mức độ ảnh hưởng của chúng đến hình ảnh cuối cùng.
*   **Prompt Rewriting / Meta Prompting:** Sử dụng một mô hình AI để viết lại hoặc cải thiện prompt ban đầu nhằm đạt được kết quả tốt hơn.

## Thách thức

*   **Quyền sở hữu trí tuệ:** Việc tạo ra hình ảnh theo phong cách nghệ sĩ hiện có đặt ra câu hỏi về quyền sở hữu và bản quyền.
*   **Hallucination:** Mặc dù diffusion models tạo hình ảnh, chúng vẫn có thể gặp khó khăn trong việc tạo ra các chi tiết chính xác hoặc liên tục (ví dụ: ngón tay, mắt, răng).
*   **Độ tin cậy:** Các mô hình diffusion không luôn tạo ra kết quả như mong đợi và yêu cầu thử nghiệm lặp đi lặp lại để đạt được kết quả mong muốn.
