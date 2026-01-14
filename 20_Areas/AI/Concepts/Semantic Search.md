---
tags:
  - AI/Search
  - AI/NLP
  - Concept
aliases:
  - Vector Search
created: 2026-01-04
---

### Định nghĩa

**Semantic Search** (Tìm kiếm ngữ nghĩa) là phương pháp tìm kiếm thông tin dựa trên *ý định* và *ngữ cảnh* của truy vấn, thay vì chỉ so khớp từ khóa chính xác (lexical matching) như các công cụ tìm kiếm truyền thống.

Nó cho phép tìm thấy kết quả liên quan ngay cả khi các từ ngữ trong truy vấn và tài liệu không trùng khớp hoàn toàn, miễn là chúng có cùng ý nghĩa.

### So sánh với Lexical Search

| Đặc điểm | Lexical Search (Keyword) | Semantic Search (Vector) |
| :--- | :--- | :--- |
| **Cơ chế** | So khớp từ khóa chính xác (TF-IDF, BM25). | So sánh độ tương đồng vector ([[Embeddings]]). |
| **Ưu điểm** | Nhanh, chính xác với tên riêng, mã số. | Hiểu đồng nghĩa, ngữ cảnh, ngôn ngữ tự nhiên. |
| **Nhược điểm** | Thất bại nếu từ khóa khác nhau (Xe hơi vs Ô tô). | Tốn kém tài nguyên tính toán hơn. |
| **Ví dụ** | Query "Java" -> Tìm tài liệu có chữ "Java". | Query "Ngôn ngữ lập trình phổ biến" -> Tìm thấy "Python", "Java"... |

### Quy trình Semantic Search

1.  **Indexing:**
    *   Chia nhỏ tài liệu ([[Text Chunking]]).
    *   Tạo vector embedding cho từng chunk.
    *   Lưu vào [[Vector Databases]].
2.  **Querying:**
    *   Chuyển đổi câu hỏi của người dùng thành vector embedding (cùng model).
    *   Tính toán khoảng cách ([[Distance Metrics]]) giữa vector câu hỏi và các vector trong database.
    *   Trả về các chunk có khoảng cách gần nhất (Top-K nearest neighbors).

### Hybrid Search

Để đạt hiệu quả tốt nhất, các hệ thống hiện đại thường sử dụng **Hybrid Search**: kết hợp kết quả từ cả Keyword Search (để bắt chính xác từ khóa quan trọng) và Semantic Search (để hiểu ngữ cảnh), sau đó dùng thuật toán (như Reciprocal Rank Fusion) để xếp hạng lại.
