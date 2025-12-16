---
tags:
  - latent-space
  - embeddings
  - generative-ai
  - machine-learning
status: in_progress
created: 2025-12-10
source:
  - Prompt Engineering for Generative AI (James Phoenix, Mike Taylor)
  - RAG roadmap.pdf
---

# Latent Space (Không gian tiềm ẩn)

## Định nghĩa

**Latent Space** (còn gọi là không gian tiềm ẩn, không gian đặc trưng hoặc không gian nhúng) là một không gian đa chiều trừu tượng, nơi các mô hình học máy biểu diễn dữ liệu thực tế (như hình ảnh, văn bản, âm thanh) dưới dạng các [[Vector Representations]]. Trong không gian này, dữ liệu không còn ở dạng ban đầu mà được mã hóa thành một tập hợp các số (vector) phản ánh các đặc điểm và mối quan hệ tiềm ẩn của chúng.

## Tầm quan trọng trong Generative AI

Latent Space đóng vai trò cực kỳ quan trọng trong các mô hình AI tạo sinh (Generative AI), đặc biệt là [[Diffusion Models]] và [[Large Language Models]] (LLMs):

*   **Biểu diễn súc tích:** Thay vì xử lý dữ liệu thô phức tạp (ví dụ: hàng triệu pixel của một hình ảnh), mô hình làm việc với các vector nhỏ gọn và có ý nghĩa hơn trong latent space.
*   **Nắm bắt mối quan hệ:** Trong latent space, các điểm dữ liệu có đặc điểm tương tự sẽ được nhóm lại gần nhau. Ví dụ, trong latent space của mô hình tạo hình ảnh, tất cả các hình ảnh về "chó" sẽ nằm gần nhau, và gần hơn với hình ảnh "mèo" hơn là "ô tô". Điều này cho phép mô hình học được các khái niệm và mối quan hệ ngữ nghĩa.
*   **Tạo sinh dữ liệu mới:**
    *   **Interpolation (Nội suy):** Bằng cách di chuyển theo một đường thẳng giữa hai điểm (hai vector) trong latent space, mô hình có thể tạo ra các biến thể mượt mà, liên tục giữa hai thực thể ban đầu. Ví dụ, chuyển đổi từ hình ảnh một con chó sang một bát trái cây.
    *   **Sampling (Lấy mẫu):** Các mô hình tạo sinh có thể lấy mẫu các vector ngẫu nhiên từ latent space và giải mã chúng thành dữ liệu mới, độc đáo nhưng vẫn có ý nghĩa (ví dụ: tạo ra các khuôn mặt người chưa từng tồn tại).
*   **Kiểm soát quá trình tạo sinh:** Bằng cách thao tác các vector trong latent space, người dùng có thể kiểm soát các thuộc tính của dữ liệu được tạo ra (ví dụ: thay đổi kiểu tóc, màu sắc của một người trong hình ảnh được tạo).

## Latent Space trong Diffusion Models

Trong Diffusion Models, latent space đặc biệt quan trọng:

*   **Quá trình Denoising:** Mô hình làm việc trực tiếp trong latent space để dần dần khử nhiễu từ một vector nhiễu ngẫu nhiên thành một biểu diễn có ý nghĩa của hình ảnh.
*   **Prompt Engineering:** Khi bạn cung cấp một prompt văn bản (ví dụ: "một con chó corgi đội mũ phi hành gia"), prompt này sẽ được mã hóa thành một vector trong latent space. Diffusion model sau đó sẽ tìm kiếm trong không gian này để tạo ra hình ảnh phù hợp nhất với vector đó.
*   **Điều hướng không gian tiềm ẩn:** Prompt engineering có thể được xem là một quá trình "điều hướng" trong latent space, tìm kiếm sự kết hợp phù hợp của các từ để tạo ra hình ảnh mong muốn.

## Các loại Embeddings và Latent Space

*   **Word Embeddings:** Các [[Embeddings]] của từ (ví dụ: word2vec, GloVe) là những ví dụ ban đầu về cách biểu diễn từ trong latent space.
*   **Contextual Embeddings:** Các mô hình hiện đại (như BERT, GPT) tạo ra các embeddings "ngữ cảnh", nghĩa là cùng một từ có thể có các vector khác nhau tùy thuộc vào ngữ cảnh xuất hiện trong câu. Điều này làm cho latent space trở nên phong phú và phức tạp hơn.

## Ví dụ trực quan

Mặc dù latent space thường có hàng trăm hoặc hàng nghìn chiều, việc trực quan hóa nó thường được thực hiện bằng cách chiếu xuống không gian 2D hoặc 3D đơn giản hơn để minh họa các khái niệm như khoảng cách giữa các từ, sự gần gũi của các nhóm dữ liệu tương tự.

Hiểu về Latent Space là chìa khóa để nắm bắt cách các mô hình AI tạo sinh hoạt động, cho phép chúng ta không chỉ tạo ra dữ liệu mới mà còn kiểm soát và thao tác các thuộc tính của dữ liệu đó một cách hiệu quả.
