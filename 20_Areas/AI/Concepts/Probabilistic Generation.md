---
tags:
  - probabilistic-generation
  - llm
  - generative-ai
  - nlp
status: in_progress
created: 2025-12-10
source:
  - Prompt Engineering for Generative AI (James Phoenix, Mike Taylor)
  - RAG roadmap.pdf
---

# Probabilistic Generation (Tạo sinh Dựa trên Xác suất)

## Định nghĩa

**Probabilistic Generation** là nguyên tắc cơ bản mà các [[Large Language Models]] (LLMs) hoạt động. Về cốt lõi, LLMs là các mô hình thống kê dự đoán **token** tiếp theo có khả năng xuất hiện cao nhất trong một chuỗi, dựa trên các token trước đó và các mẫu đã học được từ dữ liệu huấn luyện khổng lồ.

## Cơ chế hoạt động

Sau khi [[Transformer Architecture]] hiểu được ngữ cảnh của văn bản đã cho, nó sẽ tiến hành tạo ra văn bản mới. Quá trình này được hướng dẫn bởi khái niệm về khả năng xảy ra hoặc xác suất:

1.  **Dự đoán Token tiếp theo:** Mô hình tính toán xác suất cho mỗi token có thể có trong từ vựng của nó để trở thành token tiếp theo trong chuỗi.
2.  **Chọn Token:** Mô hình chọn token có xác suất cao nhất. Trong toán học, điều này được biểu diễn là:
    $$W_{next} = \text{argmax } P(w | w_1, w_2, ..., w_m)$$
    Trong đó:
    *   $W_{next}$ là từ tiếp theo có khả năng cao nhất.
    *   $P(w | w_1, w_2, ..., w_m)$ là xác suất của một từ $w$ cho trước chuỗi các từ $w_1, w_2, ..., w_m$.
    *   $\text{argmax}$ chọn từ có xác suất cao nhất.
3.  **Lặp lại:** Quá trình này được lặp lại liên tục, với mỗi token được tạo ra sẽ trở thành một phần của ngữ cảnh để dự đoán token tiếp theo. Bằng cách lặp lại quy trình này, mô hình tạo ra một chuỗi văn bản mạch lạc và có liên quan về mặt ngữ cảnh.

## Tầm quan trọng và Implication

*   **Không "Biết" sự thật:** Điều quan trọng là phải hiểu rằng LLMs không "biết" các sự kiện theo cách con người hiểu. Chúng chỉ biết các **tương quan thống kê** từ dữ liệu huấn luyện. Điều này có nghĩa là chúng có thể tạo ra văn bản trông có vẻ đúng nhưng thực chất là sai hoặc bịa đặt – đây là hiện tượng [[Hallucination]].
*   **Tính không xác định (Non-deterministic):** Không giống như các thuật toán truyền thống luôn trả về cùng một kết quả cho cùng một đầu vào, phản hồi của AI có tính không xác định. Mặc dù chúng chọn token có xác suất cao nhất, nhưng vẫn có một yếu tố ngẫu nhiên (được kiểm soát bởi tham số `temperature`) trong quá trình này.
*   **Tham số `Temperature`:** Tham số này kiểm soát mức độ "sáng tạo" hoặc "ngẫu nhiên" của mô hình.
    *   `Temperature` thấp ($
ightarrow$ 0): Mô hình sẽ chọn các token có xác suất cao nhất, dẫn đến đầu ra an toàn, ít sáng tạo hơn.
    *   `Temperature` cao ($
ightarrow$ 1): Mô hình sẽ chọn các token ngẫu nhiên hơn, dẫn đến đầu ra đa dạng và sáng tạo hơn, nhưng cũng có nguy cơ cao hơn về Hallucination.

## Ảnh hưởng đến Prompt Engineering

Hiểu bản chất Probabilistic Generation là rất quan trọng đối với [[Prompt Engineering]]:

*   **Thiết kế Prompt hiệu quả:** Prompt cần được tối ưu hóa để hướng dẫn mô hình đến các phân phối xác suất mong muốn, tăng khả năng tạo ra các phản hồi hữu ích.
*   **Giảm Hallucination:** Các kỹ thuật như [[Retrieval Augmented Generation (RAG)]] và [[Chain of Thought (CoT)]] được thiết kế để cung cấp ngữ cảnh rõ ràng và hướng dẫn suy luận, giúp giảm thiểu khả năng mô hình tạo ra thông tin sai lệch.
*   **Đánh giá chất lượng:** Do tính chất không xác định, việc đánh giá các phản hồi của LLM đòi hỏi một hệ thống đo lường hiệu quả để xác định mức độ phù hợp và đáng tin cậy của đầu ra.
*   **Lựa chọn mô hình:** Các mô hình khác nhau có thể có các khả năng tạo sinh và xu hướng Hallucination khác nhau.

Bản chất Probabilistic Generation là chìa khóa để khai thác sức mạnh của LLMs, nhưng cũng đòi hỏi các chiến lược Prompt Engineering cẩn thận và sự giám sát của con người để đảm bảo độ tin cậy của đầu ra.
