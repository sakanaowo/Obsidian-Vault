---
tags:
  - AI/Infrastructure
  - Data/Database
  - Concept
aliases:
  - Vector DB
  - Vector Store
created: 2026-01-04
---

### Định nghĩa

**Vector Database** là một loại cơ sở dữ liệu chuyên dụng để lưu trữ, quản lý và truy vấn các vector embedding ([[Embeddings]]). Khác với database truyền thống (SQL, NoSQL) tối ưu cho việc tìm kiếm chính xác (exact match), Vector Database được tối ưu cho **Similarity Search** (tìm kiếm tương đồng) trong không gian nhiều chiều.

Đây là thành phần "bộ nhớ dài hạn" (Long-term Memory) quan trọng cho các ứng dụng AI, đặc biệt là trong kiến trúc [[Retrieval Augmented Generation (RAG)]].

### Tại sao cần Vector Database?

Việc tính toán khoảng cách (như Cosine Similarity) giữa một vector truy vấn và hàng triệu vector trong database theo cách vét cạn (Brute-force) là quá chậm. Vector Database sử dụng các thuật toán **Indexing** đặc biệt để tăng tốc độ tìm kiếm này lên gấp nhiều lần mà vẫn giữ độ chính xác chấp nhận được (Approximate Nearest Neighbor - ANN).

### Các thuật toán Indexing phổ biến

*   **HNSW (Hierarchical Navigable Small World):** Phổ biến nhất hiện nay. Tạo ra một cấu trúc đồ thị phân cấp, cho phép "nhảy cóc" nhanh chóng đến vùng dữ liệu quan trọng. Tương tự như việc tìm đường đi ngắn nhất trên bản đồ.
*   **IVF (Inverted File Index):** Chia không gian vector thành các cụm (clusters) (như Voronoi cells). Khi tìm kiếm, chỉ cần tìm trong các cụm gần nhất thay vì toàn bộ không gian.
*   **PQ (Product Quantization):** Nén vector để giảm dung lượng bộ nhớ, chấp nhận giảm một chút độ chính xác để đổi lấy tốc độ và khả năng lưu trữ lớn hơn.

### Phân loại công cụ: Library vs Database

Chương 5 phân biệt rõ hai nhóm công cụ chính:

1.  **Vector Libraries (ví dụ: [[FAISS]]):**
    *   **Bản chất:** Là thư viện mã nguồn mở, chạy local (in-process).
    *   **Ưu điểm:** Cực nhanh, tối ưu hóa sâu, hỗ trợ GPU, miễn phí.
    *   **Nhược điểm:** Dữ liệu thường lưu trong RAM (dễ mất), không có tính năng quản lý database đầy đủ (như replication, sharding), khó mở rộng (scale) nếu không tự xây dựng hạ tầng bao quanh.

2.  **Vector Databases (ví dụ: [[Pinecone]], Weaviate, ChromaDB):**
    *   **Bản chất:** Là một hệ quản trị cơ sở dữ liệu đầy đủ (thường là Cloud Service).
    *   **Ưu điểm:** Cung cấp đầy đủ tính năng CRUD, quản lý metadata, scaling tự động, bảo mật, và độ bền dữ liệu (persistence). Dễ tích hợp vào ứng dụng production.
    *   **Nhược điểm:** Có thể tốn chi phí và độ trễ mạng (network latency).

### Quy trình hoạt động (CRUD)

1.  **Insert (Upsert):** Lưu vector cùng với ID và Metadata (ví dụ: `{"text": "Nội dung...", "source": "page_1"}`).
2.  **Query:** Gửi vector truy vấn -> Trả về danh sách vector tương đồng nhất (Top-K) + Metadata kèm theo.
3.  **Update/Delete:** Cập nhật hoặc xóa vector dựa trên ID. Lưu ý rằng việc update index vector phức tạp hơn nhiều so với B-Tree index truyền thống.

### Metadata Filtering (Self-Querying)

Một tính năng quan trọng là **Self-Querying**: Kết hợp tìm kiếm vector với lọc theo metadata.
*   **Ví dụ:** "Tìm các văn bản về 'AI' (Vector search) nhưng chỉ trong các tài liệu xuất bản năm '2023' (Metadata filter)".
*   Quá trình này có thể diễn ra trước (pre-filtering) hoặc sau (post-filtering) khi tìm kiếm vector, ảnh hưởng lớn đến hiệu năng và độ chính xác.