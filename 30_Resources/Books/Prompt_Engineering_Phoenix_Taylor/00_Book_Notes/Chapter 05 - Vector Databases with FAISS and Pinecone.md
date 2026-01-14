---
tags:
  - Resources/BookNote
  - AI/VectorDatabase
  - AI/RAG
  - AI/Embeddings
created: 2026-01-04
source: [[Prompt Engineering for Generative AI Future-Proof Inputs for Reliable Al Outputs (James Phoenix, Mike Taylor) (Z-Library).pdf]]
author: James Phoenix, Mike Taylor
---

# Chapter 05: Vector Databases with FAISS and Pinecone

## 1. Embeddings: Ngôn ngữ của sự tương đồng

Chương 5 mở đầu bằng việc giới thiệu khái niệm nền tảng: **[[Embeddings]]**. Để máy tính hiểu được ngữ nghĩa của ngôn ngữ, chúng ta cần chuyển đổi văn bản thành các con số.

*   **Vector Space:** Văn bản được biểu diễn dưới dạng các vector trong không gian nhiều chiều (ví dụ: model `ada-002` của OpenAI có 1536 chiều).
*   **Semantic Proximity:** Trong không gian này, các từ hoặc câu có ý nghĩa giống nhau sẽ nằm gần nhau về mặt địa lý.
    ![[fig_5-1_2-D_vector_distances_Each_location_on_the_graph_ca.png]]
    *Figure 5-1: Minh họa đơn giản về khoảng cách vector 2D. "Mouse" sẽ gần "Mickey Mouse" hơn là "Ratatouille".*

### Tại sao Embeddings quan trọng?
Embeddings giải quyết hạn chế của tìm kiếm từ khóa (Keyword Search). Nó cho phép tìm kiếm dựa trên *ý định* (Intent) và *ngữ cảnh* (Context). Ví dụ: Tìm kiếm "phương tiện giao thông công cộng" có thể trả về "xe buýt" dù không có từ nào trùng khớp.

## 2. Vector Databases: Bộ nhớ dài hạn cho AI

Khi số lượng vector tăng lên hàng triệu, việc so sánh từng đôi một (Brute-force) trở nên bất khả thi. **[[Vector Databases]]** ra đời để giải quyết bài toán này.

### Chức năng chính
1.  **Indexing:** Tổ chức dữ liệu vector để tìm kiếm nhanh (sử dụng thuật toán như HNSW, IVF).
2.  **Retrieval:** Truy xuất Top-K kết quả tương đồng nhất với vector truy vấn.
3.  **Storage:** Lưu trữ metadata đi kèm với vector (ví dụ: nội dung văn bản gốc, nguồn, tác giả).

![[fig_5-2_Multidimensional_vector_distances_A_vector_databas.png]]
*Figure 5-2: Vector Database lưu trữ và lập chỉ mục các điểm dữ liệu trong không gian đa chiều.*

## 3. Thực hành: FAISS vs Pinecone

Chương này hướng dẫn thực hành với hai công cụ đại diện cho hai hướng tiếp cận khác nhau:

### [[FAISS]] (Facebook AI Similarity Search)
*   **Loại:** Thư viện (Library).
*   **Đặc điểm:** Chạy local, cực nhanh, miễn phí.
*   **Workflow:**
    1.  Tạo index trong RAM (`IndexFlatL2` hoặc `IndexIVFFlat`).
    2.  Thêm vector vào index.
    3.  Lưu index ra file đĩa để tái sử dụng (`index.write_index`).
*   **Hạn chế:** Không tự quản lý dữ liệu bền vững (persistence), khó scale.

```python
# Ví dụ FAISS
import faiss
index = faiss.IndexFlatL2(dimension)
index.add(vectors)
distances, indices = index.search(query_vector, k=5)
```

### [[Pinecone]]
*   **Loại:** Dịch vụ quản lý (Managed Database).
*   **Đặc điểm:** Dễ dùng (API), tự động scale, hỗ trợ metadata filtering mạnh mẽ.
*   **Workflow:**
    1.  Khởi tạo index trên cloud qua API.
    2.  Upsert (Upload + Insert) vector kèm metadata.
    3.  Query trực tiếp qua API.
*   **Ưu điểm:** Phù hợp cho production, không cần lo về hạ tầng.

## 4. Ứng dụng: RAG Pipeline

Chương này kết nối Vector Database với mô hình [[Retrieval Augmented Generation (RAG)]].

1.  **Loading & Splitting:** Dùng LangChain để đọc tài liệu và chia nhỏ ([[Text Chunking]]).
    > **Lưu ý:** Chiến lược chia nhỏ (Chunking Strategy) ảnh hưởng lớn đến chất lượng tìm kiếm. Chunk quá nhỏ mất ngữ cảnh, chunk quá lớn chứa nhiều nhiễu.
2.  **Embedding:** Tạo vector cho từng chunk.
3.  **Indexing:** Lưu vào Vector DB (FAISS/Pinecone).
4.  **Retrieval:** Khi người dùng hỏi, tìm các chunk liên quan nhất.
5.  **Generation:** Đưa các chunk này vào prompt của LLM để trả lời.

### Self-Querying
Một kỹ thuật nâng cao được đề cập là **Self-Querying Retriever**. Thay vì chỉ tìm kiếm vector thuần túy, LLM sẽ phân tích câu hỏi của người dùng để tách ra phần "nội dung cần tìm" và phần "bộ lọc metadata".
*   *User:* "Tìm phim kinh dị năm 1990".
*   *Self-Query:* Vector Search("phim kinh dị") + Filter(year == 1990).

---
**Liên kết:** [[Chapter 04 - Advanced Techniques for Text Generation with LangChain]] | [[Chapter 06 - Autonomous Agents with Memory and Tools]]
