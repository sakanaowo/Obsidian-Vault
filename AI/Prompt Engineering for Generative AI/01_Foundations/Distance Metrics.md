---
tags:
  - distance-metrics
  - vector-similarity
  - embeddings
  - vector-databases
status: in_progress
created: 2025-12-10
source:
  - Prompt Engineering for Generative AI (James Phoenix, Mike Taylor)
  - RAG roadmap.pdf
---

# Distance Metrics (Các độ đo khoảng cách)

## Định nghĩa

**Distance Metrics** (Các độ đo khoảng cách), hoặc **Similarity Metrics** (Các độ đo tương đồng), là các phép đo toán học được sử dụng để định lượng mức độ "gần gũi" hoặc "tương tự" giữa hai hoặc nhiều [[Vector Representations]]. Trong ngữ cảnh của [[Vector Databases]] và [[Embeddings]], các độ đo này là yếu tố cốt lõi để xác định xem hai mục dữ liệu (ví dụ: hai đoạn văn bản, hai hình ảnh) có ý nghĩa tương tự nhau đến mức nào.

Việc chọn độ đo phù hợp là rất quan trọng vì nó ảnh hưởng trực tiếp đến cách hệ thống tìm kiếm và truy xuất thông tin, đặc biệt trong các ứng dụng như [[Retrieval Augmented Generation (RAG)]].

## Các Độ đo phổ biến

### 1. Cosine Similarity (Độ tương đồng Cosine)

*   **Định nghĩa:** Đo lường cosine của góc giữa hai vector trong không gian đa chiều. Giá trị càng gần 1 cho thấy hai vector càng cùng hướng (tương đồng cao), giá trị gần -1 cho thấy chúng ngược hướng (tương đồng thấp), và giá trị gần 0 cho thấy chúng không liên quan.
*   **Công thức:**
    $$ \text{cosine_similarity}(\mathbf{A}, \mathbf{B}) = \frac{\mathbf{A} \cdot \mathbf{B}}{||\mathbf{A}|| \cdot ||\mathbf{B}||} $$
    Trong đó:
    *   $\mathbf{A} \cdot \mathbf{B}$ là tích vô hướng của vector $\mathbf{A}$ và $\mathbf{B}$.
    *   $||\mathbf{A}||$ và $||\mathbf{B}||$ là độ lớn (magnitude) của vector $\mathbf{A}$ và $\mathbf{B}$.
*   **Đặc điểm:**
    *   **Phổ biến nhất trong RAG:** Bởi vì nó chỉ quan tâm đến hướng của các vector, không quan tâm đến độ lớn của chúng. Điều này có nghĩa là nếu hai vector có cùng hướng nhưng một vector dài hơn (tức là "mạnh hơn" hoặc "quan trọng hơn"), Cosine Similarity vẫn sẽ đánh giá chúng là rất tương đồng.
    *   **Phạm vi:** Từ -1 đến 1.
    *   **Ưu điểm:** Hiệu quả khi độ lớn của vector không đại diện cho mức độ quan trọng. Tốt cho văn bản vì độ dài của văn bản không nhất thiết phản ánh mức độ liên quan.

### 2. Euclidean Distance (Khoảng cách Euclidean - L2)

*   **Định nghĩa:** Đo khoảng cách đường thẳng giữa hai điểm (vector) trong không gian Euclidean. Khoảng cách càng nhỏ, hai vector càng giống nhau.
*   **Công thức:**
    $$ \text{euclidean_distance}(\mathbf{A}, \mathbf{B}) = \sqrt{\sum_{i=1}^{n} (A_i - B_i)^2} $$
    Trong đó $A_i$ và $B_i$ là các thành phần của vector $\mathbf{A}$ và $\mathbf{B}$.
*   **Đặc điểm:**
    *   **Ưu điểm:** Trực quan, dễ hiểu.
    *   **Nhược điểm:** Nhạy cảm với độ lớn của vector. Nếu các vector không được chuẩn hóa, một vector có độ lớn lớn hơn có thể bị coi là khác biệt hơn chỉ vì nó "dài hơn", ngay cả khi hướng của nó rất giống.
    *   **Phạm vi:** Từ 0 đến vô cùng.

### 3. Dot Product (Tích vô hướng)

*   **Định nghĩa:** Là tổng của các tích của các thành phần tương ứng của hai vector.
*   **Công thức:**
    $$ \mathbf{A} \cdot \mathbf{B} = \sum_{i=1}^{n} A_i B_i $$
*   **Đặc điểm:**
    *   **Liên quan đến Cosine Similarity:** Nếu các vector đã được chuẩn hóa (độ lớn bằng 1), tích vô hướng chính là Cosine Similarity.
    *   **Độ lớn quan trọng:** Tích vô hướng tăng khi các vector cùng hướng VÀ có độ lớn lớn. Nếu độ lớn của embedding ngụ ý tầm quan trọng, đây có thể là một độ đo hữu ích.
    *   **Phạm vi:** Từ âm vô cùng đến dương vô cùng.

## Lựa chọn Độ đo phù hợp

Việc lựa chọn độ đo khoảng cách phụ thuộc vào đặc điểm của dữ liệu và mục tiêu của ứng dụng:

*   **Văn bản và ý nghĩa:** Cosine Similarity thường là lựa chọn mặc định và phổ biến nhất cho văn bản vì nó tập trung vào mối quan hệ ngữ nghĩa (hướng của vector) hơn là độ dài hay độ lớn tuyệt đối.
*   **Dữ liệu không gian hoặc có độ lớn quan trọng:** Euclidean Distance có thể phù hợp hơn cho dữ liệu không gian hoặc khi sự khác biệt về độ lớn của vector có ý nghĩa.
*   **Tầm quan trọng của Độ lớn:** Dot Product hữu ích khi độ lớn của embedding thể hiện một khía cạnh nào đó của tầm quan trọng.

Trong các hệ thống AI hiện đại, đặc biệt là với [[Vector Embeddings]], các độ đo này là công cụ thiết yếu để định lượng các mối quan hệ ẩn giấu trong dữ liệu, từ đó cung cấp các tìm kiếm và phân tích thông minh hơn.
