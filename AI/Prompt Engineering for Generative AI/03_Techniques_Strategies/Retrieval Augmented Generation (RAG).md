---
tags:
  - rag
  - retrieval-augmented-generation
  - llm
  - generative-ai
  - information-retrieval
status: in_progress
created: 2025-12-10
source:
  - Prompt Engineering for Generative AI (James Phoenix, Mike Taylor)
  - RAG roadmap.pdf
---

# Retrieval Augmented Generation (RAG)

## Định nghĩa

**Retrieval Augmented Generation (RAG)** là một kỹ thuật tiên tiến trong lĩnh vực [[Generative AI]] nhằm cải thiện độ chính xác, đáng tin cậy và liên quan của các phản hồi được tạo ra bởi [[Large Language Models]] (LLMs). RAG hoạt động bằng cách cho phép LLM truy xuất thông tin liên quan từ một kho tri thức bên ngoài (knowledge base) và sử dụng thông tin đó làm ngữ cảnh (context) trước khi tạo ra câu trả lời.

Kỹ thuật này giúp LLMs khắc phục các hạn chế cố hữu như tạo ra thông tin không chính xác hoặc bịa đặt (gọi là [[Hallucination]]) và bị giới hạn bởi [[Context Window]] của chúng.

## Quy trình Pipeline của RAG

Quy trình RAG thường bao gồm ba giai đoạn chính: **Ingest (Nạp dữ liệu), Retrieve (Truy xuất) và Generate (Tạo sinh)**.

### 1. Ingest (Nạp dữ liệu và Xử lý tài liệu)

Đây là giai đoạn chuẩn bị dữ liệu bên ngoài để có thể truy xuất.
*   **Document Processing & [[Chunking Text]]:**
    *   Tài liệu lớn được chia thành các đoạn nhỏ hơn (chunks) để phù hợp với giới hạn [[Context Window]] của LLM.
    *   Các chiến lược chunking bao gồm: chunking theo kích thước cố định (fixed-size) với chồng lấn (overlap) để giữ ngữ cảnh, chunking ngữ nghĩa (semantic chunking) dựa trên sự thay đổi ý nghĩa, chia theo ký tự đệ quy (recursive character splitting), và xử lý dựa trên cấu trúc (structure-aware parsing) cho HTML/PDF.
*   **[[Vector Embeddings]]:**
    *   Mỗi chunk văn bản được chuyển đổi thành các biểu diễn vector số học (embeddings).
    *   Các vector này được lưu trữ trong [[Vector Databases]] để có thể tìm kiếm theo độ tương đồng ngữ nghĩa.

### 2. Retrieve (Truy xuất thông tin)

Giai đoạn này tập trung vào việc tìm kiếm các chunk tài liệu phù hợp nhất với câu hỏi của người dùng.
*   **[[Vector Databases]]:** Đây là công cụ trung tâm để lưu trữ và lập chỉ mục các vector embeddings. Các cơ sở dữ liệu này cho phép tìm kiếm dựa trên độ tương đồng, khác với tìm kiếm dựa trên ID hoặc từ khóa truyền thống.
    *   **Các loại tìm kiếm:**
        *   **Lexical Search (Tìm kiếm từ vựng):** Tìm kiếm dựa trên từ khóa chính xác (ví dụ: BM25/TF-IDF). Thích hợp cho các mã sản phẩm, tên cụ thể.
        *   **Semantic Search (Tìm kiếm ngữ nghĩa):** Tìm kiếm dựa trên ý định/ý nghĩa của câu hỏi, ngay cả khi không có từ khóa trùng khớp chính xác.
        *   **Hybrid Search (Tìm kiếm lai):** Kết hợp cả lexical và semantic search để tận dụng ưu điểm của cả hai, thường sử dụng thuật toán như Reciprocal Rank Fusion (RRF) để kết hợp các kết quả xếp hạng.
*   **Metadata & Filtering:** Sử dụng siêu dữ liệu (metadata) của tài liệu (ví dụ: ngày xuất bản, tác giả, chủ đề) để lọc kết quả truy xuất, giúp tăng độ chính xác và liên quan.
*   **Reranking (Xếp hạng lại):** Sau khi truy xuất một tập hợp các tài liệu, một mô hình reranker (ví dụ: Cross-Encoder) có thể được sử dụng để xếp hạng lại chúng dựa trên mức độ liên quan thực tế với câu hỏi, đảm bảo các tài liệu phù hợp nhất được gửi đến LLM.

### 3. Generate (Tạo sinh phản hồi)

Giai đoạn cuối cùng là sử dụng thông tin đã truy xuất được để tạo ra phản hồi.
*   **[[Prompt Engineering]]:** Các tài liệu được truy xuất (context) được chèn vào prompt của LLM cùng với câu hỏi của người dùng.
*   **Chỉ dẫn rõ ràng:** Prompt cần hướng dẫn LLM chỉ trả lời dựa trên ngữ cảnh được cung cấp và thông báo "Tôi không biết" nếu thông tin không có sẵn trong ngữ cảnh đó.
*   **Giảm Hallucination:** Việc cung cấp ngữ cảnh cụ thể giúp LLM không phải "tự bịa" thông tin, tăng cường độ tin cậy của câu trả lời.

## Lợi ích của RAG

*   **Giảm thiểu Hallucination:** Đây là lợi ích chính, giúp LLM cung cấp thông tin chính xác và đáng tin cậy hơn.
*   **Tiếp cận kiến thức cập nhật:** LLM có thể truy cập thông tin mới nhất mà nó không được huấn luyện, giải quyết vấn đề kiến thức bị "đóng băng" tại thời điểm huấn luyện.
*   **Cải thiện tính giải thích được (Explainability):** Bằng cách trích dẫn nguồn thông tin đã truy xuất, RAG giúp người dùng hiểu tại sao LLM đưa ra câu trả lời đó.
*   **Giảm chi phí và tăng hiệu quả:** Thay vì huấn luyện lại toàn bộ LLM trên dữ liệu mới, RAG cho phép cập nhật kiến thức bằng cách thêm tài liệu vào kho tri thức, tiết kiệm tài nguyên tính toán.
*   **Tăng cường khả năng xử lý ngữ cảnh dài:** Thông qua chunking và truy xuất, RAG cho phép LLM xử lý các tài liệu rất dài mà không vượt quá giới hạn token của nó.

## Các Kỹ thuật RAG Nâng cao

*   **Query Transformation:** Cải thiện câu hỏi của người dùng trước khi truy xuất (ví dụ: viết lại câu hỏi, sử dụng [[HyDE (Hypothetical Document Embeddings)]], hoặc tạo nhiều câu hỏi).
*   **Context Optimization:** Tối ưu hóa cách ngữ cảnh được trình bày cho LLM (ví dụ: sắp xếp lại các chunk để thông tin quan trọng nằm ở đầu hoặc cuối prompt, nén ngữ cảnh).
*   **Agentic RAG:** Sử dụng các tác tử (agents) để LLM có thể lên kế hoạch, quan sát, hành động và tự đánh giá, tích hợp các công cụ (tools) bên ngoài để thực hiện các bước trong quy trình RAG.
*   **Corrective RAG (CRAG):** Một vòng lặp kiểm tra và tự sửa lỗi, nơi hệ thống RAG đánh giá chất lượng truy xuất và sử dụng các công cụ bổ sung (ví dụ: tìm kiếm web) nếu kết quả không rõ ràng hoặc không đầy đủ.

RAG là một phương pháp mạnh mẽ để xây dựng các ứng dụng LLM đáng tin cậy và có khả năng truy xuất thông tin, mở rộng đáng kể khả năng của LLMs truyền thống.
