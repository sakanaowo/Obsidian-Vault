---
tags:
  - AI/Concept
  - AI/NLP
  - Data/Representation
aliases:
  - Vector Embeddings
  - Text Embeddings
created: 2026-01-04
---

### Định nghĩa

**Embeddings** (Vector Embeddings) là kỹ thuật biểu diễn dữ liệu (văn bản, hình ảnh, âm thanh) dưới dạng các vector số thực trong không gian nhiều chiều. Mục tiêu của Embeddings là chuyển đổi ý nghĩa ngữ nghĩa (semantic meaning) của dữ liệu thành dạng số học mà máy tính có thể xử lý được.

Trong không gian vector này, các dữ liệu có ý nghĩa tương tự nhau sẽ nằm gần nhau (khoảng cách ngắn), và ngược lại.

### Cơ chế hoạt động

1.  **Input:** Một đoạn văn bản ("con mèo").
2.  **Model:** Một mô hình Embedding (như OpenAI `text-embedding-ada-002`, `SentenceTransformers`) xử lý văn bản.
3.  **Output:** Một danh sách các số thực (ví dụ: `[0.012, -0.931, 0.552, ...]`). Số chiều (dimensions) phụ thuộc vào model (ví dụ: 1536 chiều cho `ada-002`).

### Ý nghĩa của không gian vector

Hãy tưởng tượng một không gian 2 chiều đơn giản:
*   Trục X: Độ "dễ thương".
*   Trục Y: Kích thước.

Khi đó:
*   "Mèo con" -> (Cao, Nhỏ)
*   "Chó con" -> (Cao, Nhỏ) -> *Gần "Mèo con"*
*   "Sư tử" -> (Thấp, Lớn) -> *Xa "Mèo con"*

Trong thực tế, không gian embedding có hàng ngàn chiều, nắm bắt các sắc thái ngữ nghĩa phức tạp hơn nhiều (ngữ pháp, cảm xúc, chủ đề, mối quan hệ từ vựng...).

### Các loại Vector

*   **Dense Vectors (Vector dày):** Hầu hết các giá trị là khác 0. Các embedding hiện đại (BERT, OpenAI) là dense vectors. Chứa nhiều thông tin ngữ nghĩa.
*   **Sparse Vectors (Vector thưa):** Hầu hết các giá trị là 0. Ví dụ: One-hot encoding hoặc TF-IDF. Thường dựa trên sự xuất hiện của từ khóa chính xác (keyword matching).

### Ứng dụng

*   **Semantic Search:** Tìm kiếm dựa trên ý nghĩa thay vì từ khóa chính xác.
*   **Recommendation Systems:** Gợi ý sản phẩm tương tự.
*   **Clustering:** Phân nhóm dữ liệu.
*   **Anomaly Detection:** Phát hiện dữ liệu bất thường (nằm xa các cụm dữ liệu chuẩn).
