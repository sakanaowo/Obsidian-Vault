---
tags:
  - vector-database
  - embeddings
  - rag
  - information-retrieval
status: in_progress
created: 2025-12-10
source:
  - Prompt Engineering for Generative AI (James Phoenix, Mike Taylor)
  - RAG roadmap.pdf
---

# Vector Databases (Cơ sở dữ liệu Vector)

## Định nghĩa

**Vector Databases** là một loại cơ sở dữ liệu chuyên biệt được thiết kế để lưu trữ, quản lý và truy vấn hiệu quả các [[Vector Representations]] (hay còn gọi là vector embeddings) của dữ liệu. Thay vì lưu trữ dữ liệu truyền thống theo bảng hoặc tài liệu, các cơ sở dữ liệu này tập trung vào các vector số, cho phép tìm kiếm dựa trên độ tương đồng ngữ nghĩa.

## Vai trò trong RAG và AI

Vector Databases là một thành phần cốt lõi của các hệ thống [[Retrieval Augmented Generation (RAG)]], cũng như nhiều ứng dụng AI khác.

*   **Lưu trữ Embeddings:** Chúng lưu trữ các vector embeddings đã được tạo từ văn bản, hình ảnh, âm thanh, hoặc bất kỳ loại dữ liệu nào mà [[Large Language Models]] (LLMs) có thể xử lý.
*   **Tìm kiếm độ tương đồng (Similarity Search):** Chức năng chính của Vector Databases là tìm kiếm nhanh chóng các vector tương tự với một vector truy vấn. Các thuật toán tìm kiếm hàng xóm gần nhất (Nearest Neighbor Search) được sử dụng để xác định các mục có ý nghĩa tương tự.
*   **Context cho LLM:** Trong RAG, khi người dùng đưa ra một câu hỏi, câu hỏi đó được chuyển đổi thành một vector truy vấn. Vector Database sau đó tìm kiếm các chunk tài liệu có vector tương tự nhất, và các chunk này được đưa trở lại làm ngữ cảnh cho LLM để tạo ra câu trả lời. Điều này giúp giảm thiểu [[Hallucination]] và cung cấp thông tin cập nhật.

## Cách hoạt động

1.  **Chuyển đổi thành Vector:** Dữ liệu phi cấu trúc hoặc có cấu trúc (văn bản, hình ảnh) được chuyển đổi thành các [[Vector Representations]] (embeddings) bằng cách sử dụng các mô hình embedding chuyên biệt.
2.  **Lập chỉ mục (Indexing):** Các vector này được lập chỉ mục trong cơ sở dữ liệu. Quá trình lập chỉ mục sử dụng các thuật toán như HNSW (Hierarchical Navigable Small World) để tổ chức các vector theo cách cho phép tìm kiếm hiệu quả theo độ tương đồng. HNSW tạo ra một cấu trúc nhiều lớp, giống như một "hệ thống đường cao tốc" cho các vector, giúp tăng tốc độ tìm kiếm mặc dù là tìm kiếm gần đúng (approximate).
3.  **Truy vấn (Querying):** Khi có một câu hỏi hoặc yêu cầu, nó cũng được chuyển đổi thành một vector truy vấn. Cơ sở dữ liệu sử dụng vector này để tìm kiếm các vector đã lưu trữ có khoảng cách ([[Distance Metrics]]) gần nhất.
4.  **Trả về kết quả:** Cơ sở dữ liệu trả về các vector (và siêu dữ liệu liên quan) giống nhất, sau đó có thể được sử dụng bởi LLM.

## Lợi ích của Vector Databases

*   **Giảm Hallucination:** Cung cấp ngữ cảnh đáng tin cậy cho LLM, ngăn chặn việc mô hình tạo ra thông tin bịa đặt.
*   **Tiếp cận kiến thức cập nhật:** Cho phép LLM truy cập thông tin bên ngoài đã được cập nhật mà không cần huấn luyện lại toàn bộ mô hình.
*   **Xử lý dữ liệu phi cấu trúc:** Giúp làm việc hiệu quả với các loại dữ liệu không dễ dàng phù hợp với cơ sở dữ liệu quan hệ truyền thống.
*   **Mở rộng quy mô:** Có thể xử lý hàng tỷ vector và thực hiện tìm kiếm độ tương đồng trong thời gian thực hoặc gần thời gian thực.
*   **Cải thiện trải nghiệm người dùng:** Cho phép các ứng dụng cung cấp thông tin liên quan và cá nhân hóa hơn.

## Các nhà cung cấp và Thư viện phổ biến

### 1. Hosted Vector Databases (Dịch vụ đám mây)

*   **Pinecone:** Một trong những nhà cung cấp hàng đầu, được quản lý hoàn toàn và dễ bắt đầu sử dụng.
*   **Weaviate:** Cung cấp tìm kiếm lai mạnh mẽ, mã nguồn mở.
*   **Chroma:** Mã nguồn mở, dễ sử dụng, thường được dùng cho các thử nghiệm.
*   **Milvus:** Có khả năng mở rộng cao, phổ biến trong các ứng dụng doanh nghiệp.

### 2. Open Source & Tích hợp

*   **FAISS (Facebook AI Similarity Search):** Một thư viện mã nguồn mở của Facebook AI để tìm kiếm độ tương đồng hiệu quả và phân cụm vector. Thường được sử dụng cục bộ.
*   **pgvector:** Một tiện ích mở rộng cho PostgreSQL, biến cơ sở dữ liệu quan hệ truyền thống thành một vector database.
*   **Elasticsearch, Redis:** Có thể tích hợp khả năng tìm kiếm vector.

## Các loại Metric khoảng cách (Distance Metrics)

Khi tìm kiếm vector tương đồng, các Vector Databases sử dụng các metric khoảng cách để đo lường "sự gần gũi" giữa các vector:

*   **Cosine Similarity (Độ tương đồng Cosine):** Đo góc giữa hai vector. Phổ biến nhất trong RAG, có giá trị từ -1 đến 1.
*   **Euclidean Distance (Khoảng cách Euclidean - L2):** Đo khoảng cách đường thẳng giữa hai vector.
*   **Dot Product (Tích vô hướng):** Liên quan đến Cosine Similarity nhưng cũng tính đến độ lớn của vector.

Vector Databases là một công nghệ then chốt để xây dựng các hệ thống AI mạnh mẽ và đáng tin cậy, đặc biệt là trong các ứng dụng cần truy xuất thông tin ngữ nghĩa và tạo sinh nội dung chất lượng cao.
