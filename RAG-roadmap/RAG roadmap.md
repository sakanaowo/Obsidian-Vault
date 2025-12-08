# Lộ trình toàn diện để trở thành chuyên gia RAG

**Mục tiêu vai trò:** RAG Architect / AI Engineer  
**Thời gian dự kiến:** 3–6 tháng

---

## Phase 1 | Nền tảng lý thuyết

*Mục tiêu:* Nắm vững các khái niệm cốt lõi về mô hình ngôn ngữ lớn, biểu diễn vector và truy hồi thông tin. Đây là điều kiện tiên quyết để hiểu, phân tích và gỡ lỗi các hệ thống RAG.

### 1. Mô hình ngôn ngữ lớn (Large Language Models – LLMs)

Để thiết kế và tối ưu RAG, cần hiểu cơ chế sinh và giới hạn của mô hình nền.

- **Kiến trúc Transformer**
  - **Cơ chế Self-Attention:**  
    Là thành phần trung tâm của Transformer, sử dụng ba ma trận $Q, K, V$ (Query, Key, Value) để cho phép mô hình “tập trung” vào các vị trí khác nhau trong chuỗi đầu vào khi tạo biểu diễn ngữ nghĩa.
  - **Cửa sổ ngữ cảnh (Context Window):**  
    Số lượng token tối đa mà LLM có thể xử lý trong một lần suy luận. Đây là giới hạn chính khiến ta cần RAG để “mở rộng trí nhớ” cho mô hình.

- **Tokenization**
  - Văn bản được chuyển thành chuỗi số nguyên gọi là **token**.
  - Lưu ý: **Token khác từ**. Một từ có thể là 1 hoặc nhiều token (ví dụ: một số, một từ dài, hoặc từ có ký hiệu đặc biệt).

- **Sinh ngôn ngữ mang tính xác suất**
  - LLM dự đoán token tiếp theo dựa trên phân phối xác suất có điều kiện, học được từ dữ liệu huấn luyện.
  - Mô hình không lưu “sự thật tuyệt đối”, mà mã hóa **tương quan thống kê** giữa các mẫu dữ liệu.

- **Prompt Engineering**
  - **Zero-shot vs Few-shot:**  
    Việc đưa ví dụ (few-shot) trong prompt thường giúp mô hình bám sát dữ liệu RAG hơn so với zero-shot.
  - **Chain-of-Thought (CoT):**  
    Yêu cầu mô hình “suy nghĩ từng bước” giúp giảm hiện tượng ảo giác trong các bài toán suy luận phức tạp.

---

### 2. Vector Embeddings

Embeddings là cầu nối giữa ngôn ngữ tự nhiên và biểu diễn số để tính toán.

- **Khái niệm**
  - Đoạn văn bản (từ, câu, đoạn) được ánh xạ thành một vector thực có kích thước cố định, ví dụ: $[0.12, -0.98, 0.05, \dots]$.

- **Không gian ngữ nghĩa (Semantic Space)**
  - Các từ/đoạn văn có ý nghĩa gần nhau sẽ có embedding nằm gần nhau trong không gian vector.
  - Quan hệ ngữ nghĩa có thể được biểu diễn bằng phép toán vector (ví dụ: $king - man + woman \approx queen$).

- **Metric đo khoảng cách/độ giống nhau**
  - **Cosine Similarity:**  
    Đo **góc** giữa hai vector, giá trị trong khoảng $[-1, 1]$. Là metric phổ biến nhất trong RAG.
  - **Euclidean Distance (L2):**  
    Đo **khoảng cách đường thẳng** giữa hai điểm trong không gian.
  - **Dot Product (tích vô hướng):**  
    Phụ thuộc cả vào hướng và độ lớn vector, hữu ích khi độ lớn embedding mang thông tin về tầm quan trọng.

---

### 3. Cơ bản về Truy hồi Thông tin (Information Retrieval – IR)

- **Tìm kiếm từ khóa (Lexical Search)**
  - Dựa trên khớp chuỗi ký tự/từ (BM25, TF-IDF).
  - Phù hợp với các thực thể chính xác như mã lỗi, số hiệu, ID, tên riêng.

- **Tìm kiếm ngữ nghĩa (Semantic Search)**
  - Dựa trên vector embedding (dense retrieval), tập trung vào **ý định và nghĩa**, thay vì khớp ký tự.
  - Ví dụ: truy vấn “cách sửa màn hình bị vỡ” có thể khớp với tài liệu “hướng dẫn thay thế màn hình hiển thị”.

- **Một số chỉ số đánh giá thường dùng**
  - **Precision:** Tỷ lệ tài liệu được truy hồi là đúng trong tập kết quả.
  - **Recall:** Tỷ lệ tài liệu đúng trong toàn bộ kho đã được truy hồi.
  - **MRR (Mean Reciprocal Rank):**  
    Đánh giá vị trí hạng của kết quả đúng đầu tiên trong danh sách truy hồi.

---

### 4. Cơ sở dữ liệu vector (Vector Databases)

- **HNSW (Hierarchical Navigable Small World)**
  - Cấu trúc chỉ mục đồ thị phân cấp, cho phép tìm kiếm xấp xỉ lân cận gần nhất trong không gian vector với tốc độ cao.
  - Có thể hình dung như một hệ thống “đường cao tốc nhiều tầng” giúp điều hướng nhanh giữa các vector.

- **Một số hệ quản trị phổ biến**
  - **Chuyên dụng cho vector:**
    - Pinecone
    - Weaviate
    - Milvus
    - Chroma
  - **Tích hợp trong hệ DB khác:**
    - PostgreSQL với pgvector
    - Elasticsearch
    - Redis

---

## Phase 2 | Thành phần lõi của hệ thống RAG

*Mục tiêu:* Hiểu, thiết kế và triển khai pipeline RAG chuẩn: **Ingest → Retrieve → Generate**.

### 1. Xử lý tài liệu và Chunking

Chất lượng truy hồi phụ thuộc mạnh vào cách chia và tiền xử lý tài liệu.

- **Chunking cố định theo kích thước**
  - Chia tài liệu theo số token (ví dụ: 512 token) và sử dụng **overlap** (ví dụ: 50 token) để tránh cắt rời câu/đoạn quan trọng.
- **Chunking theo ngữ nghĩa**
  - Sử dụng embedding để phát hiện “điểm gãy ngữ nghĩa” (spike khoảng cách vector), từ đó chia văn bản dựa trên thay đổi chủ đề.
- **Recursive Character Splitting**
  - Chiến lược chia theo thứ tự ưu tiên: đoạn (paragraph) → dòng (newline) → khoảng trắng (space).  
  - Đây là chiến lược mặc định trong một số framework như LangChain.
- **Chunking nhận thức cấu trúc**
  - Khi xử lý HTML/PDF, cần bảo toàn các cấu trúc như bảng, heading, list… nhằm giữ mạch ngữ nghĩa khi chia.

---

### 2. Phân loại truy vấn (Query Classification)

Không phải mọi truy vấn đều cần truy hồi từ kho dữ liệu.

- **Vai trò**
  - Đóng vai trò “bộ định tuyến” (router), quyết định khi nào dùng RAG, khi nào chỉ cần LLM.
- **Ví dụ tuyến đường**
  - Truy vấn dạng chào hỏi, trò chuyện: “Hello, how are you?” → dùng **LLM thuần**, không cần truy hồi.
  - Truy vấn kiến thức nội bộ: “What is the vacation policy?” → truy hồi từ **vector store/RAG**.
- **Cách triển khai**
  - Bộ phân loại nhị phân đơn giản, hoặc một lần gọi LLM nhỏ để phân loại ý định truy vấn.

---

### 3. Tìm kiếm lai (Hybrid Search)

Kết hợp lexical và semantic để đạt độ chính xác cao hơn, đặc biệt với các thuật ngữ chính xác (mã lỗi, ID, số liệu).

- **Quy trình cơ bản**
  1. Thực hiện **vector search** để nắm bắt ý nghĩa tổng thể truy vấn.
  2. Thực hiện **BM25/keyword search** để nắm bắt khớp từ khóa chính xác.
  3. Áp dụng **Reciprocal Rank Fusion (RRF)** để trộn hai danh sách xếp hạng thành một kết quả cuối cùng.

---

### 4. Metadata và Lọc (Filtering)

- **Tiền lọc (Pre-filtering)**
  - Áp dụng điều kiện lọc (ví dụ: `WHERE year = 2024`) trước khi truy vấn vector.
  - Ưu điểm: nhanh, nhưng yêu cầu metadata được gán nhãn chính xác, nhất quán.

- **Hậu lọc (Post-filtering)**
  - Truy hồi trước, sau đó mới lọc trên top-$k$ kết quả.
  - Nhược điểm: có thể dẫn đến không có kết quả nếu toàn bộ top-$k$ bị loại bởi bộ lọc.

- **Tự động suy luận bộ lọc (Auto-retrieval)**
  - Dùng LLM để suy luận metadata từ ngôn ngữ tự nhiên của truy vấn.  
    Ví dụ: “Q3 reports for Tesla” → `{company: "Tesla", quarter: "Q3"}`.

---

### 5. Reranking (Tăng độ chính xác truy hồi)

- **Bi-Encoder (Retriever)**
  - Mã hóa truy vấn và tài liệu **tách biệt**, thích hợp cho bước truy hồi sơ bộ (top 50–100).
  - Nhanh, chi phí tính toán thấp.

- **Cross-Encoder (Reranker)**
  - Nhận **cặp (truy vấn, tài liệu)** làm đầu vào và xuất một score liên quan.
  - Chậm hơn nhưng chính xác cao hơn, phù hợp để rerank tập nhỏ (top 50).

- **Quy trình điển hình**
  - Hybrid Search → lấy top 50.
  - Cross-Encoder rerank top 50 → chọn top 5 tốt nhất gửi cho LLM.

- **Một số mô hình reranker phổ biến**
  - Cohere Rerank
  - BGE-Reranker
  - ColBERT

---

## Phase 3 | Kỹ thuật nâng cao

*Mục tiêu:* Nâng chất lượng từ “chạy được” lên “chạy tốt và ổn định trong thực tế”.

### 1. Biến đổi truy vấn (Query Transformation)

Truy vấn người dùng thường mơ hồ, thiếu ngữ cảnh; cần chuẩn hóa hoặc mở rộng trước khi truy hồi.

- **Viết lại truy vấn (Query Rewriting)**
  - Chuẩn hóa truy vấn ngắn/gãy thành truy vấn rõ nghĩa, giàu ngữ cảnh hơn.  
    Ví dụ: “It’s broken” → “Detailed troubleshooting for device X failure”.

- **HyDE (Hypothetical Document Embeddings)**
  1. Dùng LLM sinh một câu trả lời “lý tưởng” (tài liệu giả) cho truy vấn.
  2. Tính embedding cho tài liệu giả.
  3. Truy hồi các tài liệu thật có embedding gần tài liệu giả.

- **Multi-Query**
  - Tách truy vấn phức tạp thành nhiều truy vấn con, truy hồi riêng lẻ rồi tổng hợp kết quả.

---

### 2. Tối ưu ngữ cảnh (Context Optimization)

- **Hiện tượng “Lost in the Middle”**
  - LLM có xu hướng chú trọng phần đầu và cuối context hơn phần giữa.
  - Cách giảm thiểu:
    - Sắp xếp lại thứ tự chunk, đưa các chunk quan trọng nhất lên đầu hoặc cuối context window.

- **Nén ngữ cảnh (Context Compression)**
  - Tóm tắt hoặc trích yếu các chunk đã truy hồi trước khi đưa vào LLM.
  - Mục đích: tiết kiệm token, giảm chi phí và tăng khả năng mô hình tập trung vào thông tin cốt lõi.

---

### 3. Truy hồi cha–con (Parent-Child Retrieval)

Giải quyết mâu thuẫn giữa **chunk dài có nhiều ngữ cảnh** và **chunk ngắn phù hợp cho truy hồi vector**.

- **Ý tưởng**
  - Tạo **Parent Chunk** (lớn) để giữ ngữ cảnh đầy đủ.
  - Chia tiếp thành **Child Chunk** (nhỏ) để tối ưu độ chính xác embedding.

- **Quy trình**
  1. Chia tài liệu thành các Parent Chunk lớn.
  2. Chia tiếp mỗi Parent thành nhiều Child Chunk nhỏ.
  3. Chỉ index **Child Chunk** để truy hồi.
  4. Khi một Child được truy hồi, lấy **Parent tương ứng** gửi cho LLM.

---

### 4. GraphRAG (RAG với Đồ thị tri thức)

- **Khái niệm**
  - Thay vì chỉ lưu văn bản tuyến tính, GraphRAG lưu trữ **thực thể (Node)** và **quan hệ (Edge)**:
    - Node: “Elon Musk”, “Tesla”, “SpaceX”, …
    - Edge: “CEO of”, “Owns”, “Acquired by”, …

- **Trường hợp sử dụng**
  - Các bài toán cần **multi-hop reasoning** (suy luận qua nhiều bước).  
    Ví dụ: “Who is the CEO of the company that acquired Twitter?”  
    Các bước:
    - Tìm công ty đã mua Twitter.
    - Tìm CEO của công ty đó.
  - Vector search thuần thường gặp khó khăn; graph truy vấn tốt hơn cho dạng này.

- **Ngôn ngữ truy vấn Cypher**
  - Ngôn ngữ truy vấn cho cơ sở dữ liệu đồ thị (ví dụ Neo4j), tương tự vai trò của SQL trong hệ quan hệ.

---

### 5. Agentic RAG

Thay vì pipeline cố định, LLM được cấp “công cụ” và tự chọn hành động.

- **Mẫu ReAct (Reason + Act)**
  - Vòng lặp:
    1. **Thought:** xác định cần loại thông tin nào.
    2. **Action:** gọi công cụ (ví dụ: `search_tool`).
    3. **Observation:** nhận kết quả.
    4. **Thought:** quyết định bước xử lý tiếp theo (tính toán, truy vấn bổ sung,…).
    5. **Action:** gọi công cụ khác (ví dụ: `calculator_tool`).
- **Một số framework hỗ trợ**
  - LangGraph
  - CrewAI
  - AutoGen

---

### 6. Corrective RAG (CRAG)

Bổ sung cơ chế kiểm soát chất lượng dựa trên **độ tin cậy truy hồi**.

- **Logic điển hình**
  - Nếu điểm tin cậy truy hồi **cao** → sinh câu trả lời từ context hiện tại.
  - Nếu điểm tin cậy **trung bình/mơ hồ** → kết hợp thêm nguồn khác (ví dụ: web search).
  - Nếu điểm tin cậy **thấp** → mô hình nên chủ động trả lời “không biết” hoặc yêu cầu người dùng cung cấp thêm thông tin.

---

## Phase 4 | Đánh giá và triển khai sản phẩm

*Mục tiêu:* Thiết kế cơ chế đo lường, giám sát và tối ưu hệ thống RAG trong môi trường thực tế.

### 1. Bộ ba đánh giá RAG (The RAG Triad)

Thường sử dụng LLM làm “trọng tài” (LLM-as-a-judge) để chấm chất lượng.

1. **Context Relevance (độ liên quan của ngữ cảnh):**  
   Đo xem các đoạn văn được truy hồi có phù hợp với truy vấn hay không.
2. **Groundedness / Faithfulness (tính bám nguồn):**  
   Đánh giá việc câu trả lời có chỉ dựa trên thông tin trong context, tránh ảo giác hay suy đoán không có căn cứ.
3. **Answer Relevance (độ phù hợp của câu trả lời):**  
   Câu trả lời có thực sự giải quyết câu hỏi của người dùng hay không.

- **Một số framework đánh giá**
  - RAGAS (Retrieval Augmented Generation Assessment)
  - TruLens
  - Arize Phoenix

---

### 2. Fine-tuning

- **Fine-tuning Embedding**
  - Trong các miền đặc thù (luật cổ, y sinh, kỹ thuật chuyên sâu…), embedding tổng quát có thể hoạt động kém.
  - Fine-tuning embedding bằng **contrastive learning** để tăng khả năng phân biệt các khái niệm chuyên ngành.

- **Fine-tuning LLM**
  - Thường hiệu quả hơn khi dùng để **điều chỉnh phong cách, định dạng đầu ra, quy tắc an toàn**, thay vì nhồi nhét thêm tri thức cụ thể (cái mà RAG đã giải quyết qua truy hồi).

---

### 3. Tối ưu vận hành (Production Optimization)

- **Semantic Caching**
  - Lưu cache câu trả lời dựa trên embedding của truy vấn.
  - Khi hai truy vấn có embedding rất gần nhau (ví dụ: “What is RAG?” và “Define RAG”), có thể tái sử dụng kết quả, giảm chi phí suy luận.

- **Streaming**
  - Trả token theo luồng (streaming) từ mô hình ra giao diện, giúp giảm độ trễ cảm nhận của người dùng và cải thiện trải nghiệm.

---

## Phase 5 | Framework và Công cụ

*Mục tiêu:* Nắm được hệ sinh thái công cụ phục vụ orchestration, lưu trữ, đánh giá và quan sát hệ thống RAG.

| Category        | Tools        | Notes                                                                                          |
| --------------- | ------------ | ---------------------------------------------------------------------------------------------- |
| **Orchestration | ** LangChain | Hệ sinh thái lớn, nhiều tích hợp, thích hợp cho các workflow phức tạp; đường cong học tập dốc. |
|                 | LlamaIndex   | Tập trung mạnh vào ingest dữ liệu và tối ưu RAG.                                               |
|                 | Haystack     | Framework NLP modular, hướng production.                                                       |
| **Vector DBs    | ** Pinecone  | Dịch vụ quản lý (managed), dễ khởi đầu, phù hợp cho sản phẩm.                                  |
|                 | Weaviate     | Mã nguồn mở, hỗ trợ hybrid search tốt.                                                         |
|                 | Milvus       | Hỗ trợ scale lớn, thường dùng trong môi trường doanh nghiệp.                                   |
| **Graph         | ** Neo4j     | Lựa chọn phổ biến cho GraphRAG và truy vấn đồ thị.                                             |
| **Evaluation    | ** RAGAS     | Thư viện chuẩn cho đánh giá hệ thống RAG.                                                      |
| **Observability | ** LangSmith | Hữu ích cho logging, tracing và debug ứng dụng xây bằng LangChain.                             |

---

## Tài liệu tham khảo và Bước tiếp theo

1. **DeepLearning.AI – Khóa học RAG:**  
   Điểm khởi đầu phù hợp cho người muốn tiếp cận RAG từ góc độ thực hành (code-first).
2. **Microsoft – Azure RAG Overview:**  
   Tài liệu định hướng kiến trúc cho các hệ thống RAG quy mô doanh nghiệp.
3. **Neo4j – Tài liệu GraphRAG nâng cao:**  
   Hữu ích khi cần mở rộng từ vector search sang đồ thị tri thức.
4. **Redis – “10 Techniques to Improve RAG”:**  
   Tổng hợp các kỹ thuật tối ưu độ chính xác trong hệ thống RAG thực tế.
5. **Arxiv – Survey Paper về RAG:**  
   Cung cấp nền tảng lý thuyết, phân loại phương pháp và tổng quan nghiên cứu.

### Kế hoạch hành động gợi ý

1. Xây dựng một hệ thống **RAG cơ bản**:  
   Nạp tài liệu (PDF) → Chunking → Lưu vào Vector Store → Truy vấn.
2. Bổ sung **Hybrid Search** và **Reranking**, sau đó đo lường cải thiện.
3. Thêm **Memory (lịch sử hội thoại)** để hỗ trợ hội thoại đa lượt.
4. Chuyển sang **Agentic RAG** bằng cách tích hợp công cụ và khung điều phối (LangGraph, v.v.).

---

Bạn có muốn tôi chuẩn hóa thêm frontmatter YAML (tags, status, estimated_time, level, …) cho note [[RAG roadmap]] và gợi ý liên kết tới các note nền tảng như [[Neural Network]], [[Linear Regression]], [[Logistic Regression]] không?