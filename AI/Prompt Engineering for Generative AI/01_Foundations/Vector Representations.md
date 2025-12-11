---
tags:
  - vector-representations
  - embeddings
  - nlp
  - machine-learning
  - deep-learning
status: in_progress
created: 2025-12-10
source:
  - Prompt Engineering for Generative AI (James Phoenix, Mike Taylor)
  - RAG roadmap.pdf
---

# Vector Representations (Biểu diễn Vector)

## Định nghĩa

**Vector Representations**, hay còn gọi là **Word Embeddings** (nhúng từ) hoặc đơn giản là **embeddings**, là các biểu diễn số học (dưới dạng các mảng số hoặc vector) của các thực thể phi số (non-numerical entities) như từ, cụm từ, câu, đoạn văn, tài liệu, hoặc thậm chí là hình ảnh. Trong lĩnh vực [[Natural Language Processing (NLP)]] và [[Large Language Models]] (LLMs), các vector này đóng vai trò như một "ngôn ngữ" mà máy tính có thể hiểu được, nắm bắt các mối quan hệ ngữ nghĩa và cú pháp.

## Bản chất số học của Ngôn ngữ

Trong NLP, các từ không chỉ là các ký hiệu chữ cái. Chúng được [[Tokenization]] và sau đó được biểu diễn dưới dạng số. Các vector này là các mảng số đa chiều, trong đó mỗi chiều có thể đại diện cho một đặc điểm ngữ nghĩa hoặc cú pháp nào đó của từ.

*   **Tính chất:** Các từ có ý nghĩa tương tự hoặc liên quan sẽ được ánh xạ gần nhau trong không gian đa chiều (gọi là [[Latent Space]]). Ví dụ, các từ như "vua", "hoàng đế", "hoàng hậu" sẽ có các vector gần nhau hơn so với "xe đạp".
*   **Thực hiện:** Các vector này được tạo ra thông qua một quá trình huấn luyện chuyên sâu, trong đó mô hình học cách xác định và mã hóa các mẫu trong ngôn ngữ.

## Cách tạo và Sử dụng

### 1. Tạo Embeddings

*   **Mô hình Embeddings:** Các mô hình chuyên biệt được thiết kế để tạo ra các vector biểu diễn này. Ví dụ phổ biến bao gồm `text-embedding-ada-002` của OpenAI, hoặc các mô hình từ thư viện Sentence Transformers của Hugging Face.
*   **Quá trình:** Một đoạn văn bản (hoặc hình ảnh) được đưa vào mô hình embedding, và mô hình sẽ trả về một vector số học tương ứng.

### 2. Sử dụng Embeddings

Các vector representations được sử dụng rộng rãi trong các tác vụ NLP và AI:

*   **Tìm kiếm ngữ nghĩa (Semantic Search):** Thay vì tìm kiếm chính xác từ khóa, bạn có thể tìm kiếm các tài liệu hoặc đoạn văn bản có ý nghĩa tương tự bằng cách so sánh [[Distance Metrics]] giữa các vector của chúng. Đây là nền tảng của [[Retrieval Augmented Generation (RAG)]].
*   **Phân loại văn bản (Text Classification):** Phân loại tài liệu dựa trên ý nghĩa ngữ nghĩa của chúng.
*   **Đề xuất (Recommendation Systems):** Đề xuất các sản phẩm hoặc mục tương tự dựa trên vector biểu diễn của chúng.
*   **Đánh giá (Evaluation):** Đo lường sự tương đồng giữa các phản hồi của LLM và văn bản tham chiếu.

## Các loại Vector Embeddings

*   **Dense Vectors (Vector dày đặc):** Hầu hết các số trong mảng đều khác 0. Chúng nắm bắt thông tin ngữ nghĩa và ngữ cảnh tốt, phổ biến trong các ứng dụng tìm kiếm ngữ nghĩa. Ví dụ: các embeddings từ OpenAI Ada 002.
*   **Sparse Vectors (Vector thưa thớt):** Phần lớn các số trong mảng là 0. Chúng thường tốt hơn cho các ứng dụng tìm kiếm dựa trên từ khóa chính xác. Ví dụ: TF-IDF (Term Frequency-Inverse Document Frequency).
*   **Hybrid Search:** Kết hợp cả Dense và Sparse Vectors để tận dụng ưu điểm của cả hai, cải thiện khả năng truy xuất.

## Thách thức và Lưu ý

*   **Kích thước Vector:** Số chiều (dimensions) của vector có thể rất lớn (ví dụ: 1536 chiều cho OpenAI `text-embedding-ada-002`), khiến việc trực quan hóa trở nên khó khăn.
*   **Độ chính xác:** Độ chính xác của các vector phụ thuộc vào mô hình embedding được sử dụng. Các thiên vị hoặc khoảng trống kiến thức trong mô hình cơ sở sẽ ảnh hưởng đến chất lượng của vector.
*   **Ngữ cảnh:** Các mô hình transformer hiện đại tạo ra các vector ngữ cảnh, nghĩa là cùng một từ có thể có các vector khác nhau tùy thuộc vào ngữ cảnh xuất hiện (ví dụ: "bank" trong "riverbank" và "financial bank").
*   **Kích thước Chunk:** Kích thước của đoạn văn bản (chunk) được nhúng cũng quan trọng. Nếu chunk quá lớn, vector có thể trở nên quá khái quát. Nếu chunk quá nhỏ, nó có thể mất đi ngữ cảnh quan trọng.
*   **Chi phí:** Việc tạo và lưu trữ các embeddings có thể tốn kém tài nguyên tính toán và bộ nhớ.

Hiểu về Vector Representations là điều cần thiết để làm việc hiệu quả với các hệ thống AI hiện đại, đặc biệt là trong các ứng dụng RAG, nơi chúng là cầu nối giữa ngôn ngữ con người và hiểu biết của máy móc.
