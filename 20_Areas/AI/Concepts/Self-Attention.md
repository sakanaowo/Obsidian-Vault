---
tags:
  - self-attention
  - attention-mechanism
  - transformer-architecture
  - nlp
  - deep-learning
status: in_progress
created: 2025-12-10
source:
  - Prompt Engineering for Generative AI (James Phoenix, Mike Taylor)
  - RAG roadmap.pdf
---

# Self-Attention (Cơ chế Tự chú ý)

## Định nghĩa

**Self-Attention** là một cơ chế trong kiến trúc [[Transformer Architecture]] cho phép mô hình gán trọng số khác nhau cho các từ khác nhau trong câu đầu vào khi xử lý mỗi từ. Nói cách khác, nó giúp mô hình đánh giá tầm quan trọng tương đối của các từ khác nhau trong cùng một câu để hiểu ngữ cảnh của một từ nhất định.

## Tầm quan trọng trong LLMs

Trước khi có Self-Attention, các mô hình ngôn ngữ gặp khó khăn trong việc xử lý các mối quan hệ phụ thuộc tầm xa trong văn bản. Cơ chế Self-Attention đã giải quyết vấn đề này bằng cách cho phép mỗi từ trong một câu tương tác với tất cả các từ khác trong câu đó, bất kể vị trí của chúng. Điều này giúp các [[Large Language Models]] (LLMs) có khả năng:

*   **Hiểu ngữ cảnh sâu sắc:** Mỗi từ có thể "nhìn" vào toàn bộ câu để thu thập thông tin ngữ cảnh, điều này rất quan trọng để phân biệt ý nghĩa của các từ đồng âm (ví dụ: "bank" bờ sông và "bank" ngân hàng) hoặc các tham chiếu (ví dụ: đại từ).
*   **Xử lý phụ thuộc tầm xa:** Nắm bắt các mối quan hệ giữa các từ cách xa nhau trong câu một cách hiệu quả, điều mà các mô hình tuần tự trước đây khó thực hiện.
*   **Xử lý song song:** Không như các mô hình tuần tự, Self-Attention cho phép tính toán các mối quan hệ cho tất cả các từ cùng lúc, tăng tốc độ huấn luyện đáng kể.

## Cơ chế hoạt động

Self-Attention hoạt động bằng cách tạo ra ba vector cho mỗi từ trong chuỗi đầu vào:

1.  **Query (Truy vấn - Q):** Biểu diễn từ hiện tại mà chúng ta đang xử lý.
2.  **Key (Khóa - K):** Biểu diễn ý nghĩa của từ đó trong ngữ cảnh của câu.
3.  **Value (Giá trị - V):** Biểu diễn nội dung thực tế của từ đó.

Các bước chính để tính toán Self-Attention cho một từ cụ thể:

1.  **Tính điểm chú ý (Attention Score):**
    *   Đối với mỗi từ trong câu, vector Query của nó được so sánh với vector Key của *tất cả* các từ trong câu (bao gồm cả chính nó).
    *   Phép so sánh này thường được thực hiện bằng cách lấy tích vô hướng (dot product) giữa Query và Key. Kết quả cho biết mức độ liên quan của mỗi từ khác đến từ hiện tại.
2.  **Chuẩn hóa điểm chú ý:**
    *   Các điểm chú ý này sau đó được chia tỷ lệ (scaled) để ổn định gradient trong quá trình huấn luyện.
    *   Tiếp theo, chúng được truyền qua một hàm softmax để chuyển đổi thành phân phối xác suất, đảm bảo tổng của các điểm chú ý bằng 1. Những giá trị này cho biết "mức độ tập trung" của mô hình vào mỗi từ khác.
3.  **Tính tổng trọng số của Value:**
    *   Mỗi vector Value của các từ trong câu được nhân với điểm chú ý đã chuẩn hóa tương ứng của nó.
    *   Tất cả các vector Value có trọng số này được cộng lại để tạo thành vector đầu ra cho từ hiện tại. Vector này là một biểu diễn ngữ cảnh phong phú, tổng hợp thông tin từ tất cả các từ khác trong câu, với sự nhấn mạnh vào các từ liên quan nhất.

### Multi-Head Self-Attention (Tự chú ý đa đầu)

Trong thực tế, các Transformer sử dụng **Multi-Head Self-Attention**. Điều này có nghĩa là quy trình Self-Attention được thực hiện nhiều lần (multiple "heads") một cách song song.

*   Mỗi "head" học cách tập trung vào các loại mối quan hệ khác nhau (ví dụ: một head có thể chú ý đến các mối quan hệ cú pháp, head khác chú ý đến các mối quan hệ ngữ nghĩa).
*   Kết quả từ tất cả các head được nối lại (concatenated) và sau đó được chiếu tuyến tính (linearly projected) để tạo ra một biểu diễn cuối cùng duy nhất.
*   Điều này giúp mô hình thu thập nhiều thông tin ngữ cảnh đa dạng và phong phú hơn.

## Ví dụ minh họa

Trong câu "The animal didn't cross the street because it was too tired", khi mô hình xử lý từ "it":
*   Vector Query của "it" sẽ được so sánh với vector Key của "animal", "street", "tired", v.v.
*   Điểm chú ý sẽ cao hơn cho "animal" và "tired" so với "street", vì "it" có nhiều khả năng ám chỉ "animal" và liên quan đến "tired".
*   Vector đầu ra cho "it" sẽ là sự kết hợp có trọng số của các vector Value của tất cả các từ, nhưng với trọng số lớn hơn cho "animal" và "tired".

Self-Attention là một thành phần quan trọng giúp LLMs đạt được khả năng hiểu ngôn ngữ tinh tế và tạo ra các phản hồi ngữ cảnh phù hợp, là chìa khóa cho hiệu suất vượt trội của chúng.
