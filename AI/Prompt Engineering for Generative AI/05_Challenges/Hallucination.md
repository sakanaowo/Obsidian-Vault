---
tags:
  - hallucination
  - llm
  - generative-ai
  - reliability
status: in_progress
created: 2025-12-10
source:
  - Prompt Engineering for Generative AI (James Phoenix, Mike Taylor)
  - RAG roadmap.pdf
---

# Hallucination (Ảo giác)

## Định nghĩa

**Hallucination** (ảo giác) là một hiện tượng phổ biến trong các mô hình [[Large Language Models]] (LLMs) khi chúng tạo ra thông tin không chính xác, không liên quan, hoặc bịa đặt một cách tự tin, giống như thể đó là sự thật. Điều này có thể bao gồm các sự kiện, tên, số liệu, hoặc giải thích sai lệch hoàn toàn so với thực tế.

## Nguyên nhân của Hallucination

Hiện tượng Hallucination xuất phát từ bản chất [[Probabilistic Generation]] của LLMs và cách chúng được huấn luyện:

1.  **Mô hình thống kê:** LLMs về cơ bản là các công cụ thống kê, được huấn luyện để dự đoán từ tiếp theo có khả năng nhất dựa trên xác suất từ dữ liệu chúng đã thấy. Chúng không "hiểu" thế giới thực hay "biết" sự thật; chúng chỉ học các mẫu và mối quan hệ trong dữ liệu.
2.  **Dữ liệu huấn luyện:**
    *   **Dữ liệu không hoàn chỉnh hoặc mâu thuẫn:** Nếu dữ liệu huấn luyện chứa thông tin sai lệch, lỗi thời hoặc mâu thuẫn, mô hình có thể học được những "sự thật" không chính xác.
    *   **Thiên vị (Biases):** Các thiên vị trong dữ liệu có thể dẫn đến mô hình khái quát hóa sai hoặc tạo ra thông tin không cân bằng.
3.  **Nội suy và Ngoại suy:** Khi được hỏi về thông tin nằm ngoài phạm vi dữ liệu huấn luyện (ngoại suy) hoặc khi phải kết nối các khái niệm rời rạc (nội suy), mô hình có thể "đoán" hoặc tạo ra thông tin hợp lý nhưng không chính xác.
4.  **Prompt không rõ ràng hoặc mơ hồ:** Một [[Prompt Engineering]] kém, thiếu [[Give Direction]] hoặc [[Specify Format]], có thể khiến mô hình phải tự bịa ra thông tin để cố gắng trả lời.
5.  **Áp lực tạo ra phản hồi:** Các LLMs được thiết kế để luôn tạo ra phản hồi, ngay cả khi chúng không có đủ thông tin tin cậy.

## Tác động và Hậu quả

Hallucination là một vấn đề nghiêm trọng, làm giảm độ tin cậy và tiện ích của LLMs trong các ứng dụng thực tế:

*   **Thông tin sai lệch:** Có thể gây hiểu lầm hoặc đưa ra quyết định sai lầm nếu người dùng tin vào thông tin bịa đặt.
*   **Thiếu tin cậy:** Khả năng Hallucination làm giảm lòng tin của người dùng vào các hệ thống AI.
*   **Chi phí:** Việc sửa lỗi Hallucination đòi hỏi sự can thiệp của con người, gây tốn kém thời gian và nguồn lực.

## Các giải pháp để giảm thiểu Hallucination

Để khắc phục hoặc giảm thiểu Hallucination, một số kỹ thuật và phương pháp đã được phát triển:

1.  **[[Retrieval Augmented Generation (RAG)]]:** Đây là một trong những cách tiếp cận hiệu quả nhất. RAG cho phép LLM truy xuất thông tin liên quan từ một cơ sở dữ liệu tri thức bên ngoài đáng tin cậy (ví dụ: các tài liệu, bài báo, dữ liệu đã được xác minh) và sử dụng thông tin đó làm ngữ cảnh để tạo ra phản hồi.
    *   **Lợi ích:** Đảm bảo LLM chỉ trả lời dựa trên thông tin được cung cấp, giảm đáng kể khả năng tạo ra thông tin bịa đặt.
    *   **Yêu cầu:** Hệ thống [[Vector Databases]] hiệu quả để lưu trữ và truy xuất các [[Vector Embeddings]].

2.  **[[Chain of Thought (CoT)]] Prompting:** Yêu cầu mô hình hiển thị các bước suy luận của nó. Điều này buộc mô hình phải logic hóa quá trình giải quyết vấn đề, từ đó tự kiểm tra và giảm thiểu lỗi.

3.  **Kiểm soát đầu ra (Output Control):**
    *   **Xác định định dạng:** Yêu cầu đầu ra ở định dạng có cấu trúc như [[JSON]] hoặc [[YAML]] giúp dễ dàng xác thực và phát hiện lỗi.
    *   **Thêm hướng dẫn rõ ràng:** Chỉ dẫn mô hình rõ ràng về việc không được bịa đặt thông tin và phải trả lời "Tôi không biết" nếu không tìm thấy câu trả lời trong ngữ cảnh được cung cấp.

4.  **[[Few-Shot Learning]]:** Cung cấp các ví dụ chất lượng cao cho mô hình để hướng dẫn nó tạo ra phản hồi chính xác và theo định dạng mong muốn.

5.  **Human Feedback (Phản hồi của con người):** Liên tục đánh giá và cung cấp phản hồi cho mô hình để nó học cách phân biệt thông tin chính xác và bịa đặt. Các phương pháp như [[Reinforcement Learning from Human Feedback (RLHF)]] đã được sử dụng để căn chỉnh mô hình theo ý định của con người.

6.  **Fine-tuning (Tinh chỉnh):** Tinh chỉnh mô hình trên các tập dữ liệu chất lượng cao, cụ thể cho từng tác vụ có thể giúp mô hình trở nên chính xác hơn và ít Hallucination hơn trong các tác vụ đó.

Hallucination vẫn là một thách thức đang diễn ra trong lĩnh vực LLMs, nhưng với sự phát triển của các kỹ thuật như RAG và CoT, chúng ta có thể xây dựng các hệ thống AI đáng tin cậy hơn.
