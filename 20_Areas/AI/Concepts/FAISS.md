---
tags:
  - AI/Library
  - Data/VectorSearch
  - Concept
aliases:
  - Facebook AI Similarity Search
created: 2026-01-04
---

### Định nghĩa

**FAISS** (Facebook AI Similarity Search) là một thư viện mã nguồn mở được phát triển bởi Facebook AI Research (Meta), chuyên dùng để tìm kiếm độ tương đồng (similarity search) và phân cụm (clustering) các vector mật độ cao (dense vectors) một cách cực kỳ hiệu quả.

Nó được thiết kế để xử lý các bộ dữ liệu vector có kích thước lớn (hàng tỷ vector) mà có thể không vừa với RAM, sử dụng các thuật toán indexing tối ưu.

### Đặc điểm chính

1.  **Hiệu năng cao:** Được viết bằng C++ với wrapper Python, tối ưu hóa sâu cho cả CPU và GPU.
2.  **Indexing đa dạng:** Hỗ trợ nhiều loại index như `IndexFlatL2` (Brute-force chính xác), `IndexIVFFlat` (Inverted File - nhanh hơn), `IndexHNSW` (Graph-based).
3.  **Local Execution:** Chạy trực tiếp trong tiến trình ứng dụng (in-process), không cần server riêng.

### Hạn chế (So với Vector DB)

*   **Không phải là Database:** FAISS chỉ là một index. Nó không quản lý việc lưu trữ dữ liệu gốc (metadata, text), không hỗ trợ giao dịch (transactions), và không đảm bảo độ bền dữ liệu (persistence) nếu không tự code thêm logic lưu file index ra đĩa.
*   **Khó Scale ngang:** Việc phân tán FAISS index ra nhiều máy chủ đòi hỏi kỹ thuật phức tạp hơn so với việc dùng một Managed Service như Pinecone.

### Ứng dụng

FAISS thường được dùng trong giai đoạn phát triển (prototyping) hoặc trong các hệ thống mà dữ liệu vector có thể load hết vào RAM, hoặc khi cần tích hợp sâu engine tìm kiếm vào ứng dụng local.
