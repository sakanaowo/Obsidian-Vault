---
tags:
  - llm
  - hallucination
  - ai
  - reliability
  - prompt-engineering
status: pending
created_date: 2025-12-14
---

# Hallucination (Ảo giác) trong LLMs

## Định nghĩa

**Hallucination** trong bối cảnh các Mô hình Ngôn ngữ Lớn (LLMs) đề cập đến xu hướng của mô hình tạo ra thông tin **sai lệch, không chính xác, hoặc không có căn cứ thực tế**, nhưng lại trình bày một cách tự tin như thể đó là sự thật. Đây là một vấn đề nghiêm trọng, làm giảm độ tin cậy và độ chính xác của LLMs, đặc biệt trong các ứng dụng đòi hỏi tính xác thực cao. Thuật ngữ này ám chỉ việc mô hình "tự tin bịa đặt" thông tin không tồn tại trong dữ liệu huấn luyện hoặc ngữ cảnh được cung cấp.

## Cơ chế hoạt động và Nguyên nhân

Hiện tượng hallucination không phải là lỗi ngẫu nhiên mà thường bắt nguồn từ một số yếu tố cốt lõi trong kiến trúc và quá trình huấn luyện của LLMs:

1.  **Dữ liệu huấn luyện:**
    *   **Sai lệch hoặc không nhất quán:** Dữ liệu huấn luyện có thể chứa thông tin không chính xác hoặc không nhất quán, dẫn đến việc mô hình học được các mối quan hệ sai.
    *   **Thiếu thông tin:** Khi gặp các câu hỏi nằm ngoài phạm vi dữ liệu huấn luyện hoặc yêu cầu thông tin quá cụ thể mà mô hình chưa từng thấy, nó có xu hướng "sáng tạo" ra câu trả lời thay vì từ chối hoặc nói rằng không biết.
2.  **Mô hình xác suất:**
    *   **Ưu tiên sự trôi chảy (Fluency over Factuality):** LLMs được thiết kế để tạo ra văn bản có tính liên kết và ngữ pháp chính xác. Khi mô hình phải chọn giữa việc tạo ra một câu trả lời trôi chảy nhưng sai và một câu trả lời chính xác nhưng ngập ngừng, nó thường ưu tiên sự trôi chảy.
    *   **Tham số Temperature và Top-p:** Các tham số này kiểm soát mức độ ngẫu nhiên trong quá trình tạo văn bản. Giá trị cao hơn có thể khuyến khích sự sáng tạo nhưng cũng tăng nguy cơ hallucination.
3.  **Hạn chế của Context Window:** Khi Context Window quá nhỏ, mô hình không thể duy trì ngữ cảnh đủ lâu. Dù thông tin đúng có thể tồn tại ở đâu đó trong quá khứ, mô hình vẫn "quên" nó và bịa đặt.
4.  **Suy luận không đầy đủ:** Các mô hình thường gặp khó khăn trong việc suy luận từ các thông tin phức tạp hoặc đa bước, dẫn đến việc đưa ra kết luận sai.

## Hậu quả và Tác động

Hallucination có thể gây ra nhiều tác động tiêu cực, từ việc đưa ra thông tin y tế sai lệch, tạo ra các sự kiện lịch sử không có thật, đến việc cung cấp dữ liệu tài chính không chính xác, gây tổn hại danh tiếng và thiệt hại tài chính.

## Kỹ thuật giảm thiểu Hallucination

Để giảm thiểu hallucination, các kỹ thuật Prompt Engineering đóng vai trò quan trọng:

1.  **Cung cấp văn bản tham chiếu (Reference Text):**
    *   **Nguyên tắc:** Hướng dẫn mô hình chỉ trả lời dựa trên thông tin được cung cấp trong prompt, và từ chối trả lời ("I could not find an answer") nếu thông tin không có sẵn.
    *   **Cơ chế:** Buộc mô hình sử dụng các nguồn đáng tin cậy đã được xác định trước, giảm sự "tự do sáng tạo".
    *   **Ví dụ:** "Refer to the articles enclosed within triple quotes to respond to queries. In cases where the answer isn't found within these articles, simply return 'I could not find an answer'."

2.  **Yêu cầu trích dẫn (Citations):**
    *   **Nguyên tắc:** Yêu cầu mô hình trích dẫn nguồn (ví dụ: số trang, đoạn văn) từ văn bản tham chiếu khi đưa ra câu trả lời. Điều này giúp dễ dàng xác minh tính chính xác.

3.  **Ask for Context (Hỏi thêm ngữ cảnh):**
    *   **Nguyên tắc:** Hướng dẫn mô hình đặt câu hỏi ngược lại cho người dùng nếu nó cảm thấy không đủ thông tin để đưa ra câu trả lời chính xác và tự tin.
    *   **Cơ chế:** Biến tương tác thành một cuộc đối thoại, nơi mô hình chủ động tìm kiếm thông tin cần thiết, giảm khả năng bịa đặt do thiếu dữ liệu.

4.  **Chain of Thought (CoT):**
    *   **Nguyên tắc:** Khuyến khích mô hình trình bày các bước suy luận từng bước. Quá trình "tự kiểm tra" này giúp mô hình nhận ra các điểm bất hợp lý trong suy nghĩ của mình.

5.  **Fact-Checking Tools (Công cụ kiểm tra thực tế):**
    *   Kết hợp LLMs với các công cụ bên ngoài có thể truy cập dữ liệu đáng tin cậy (ví dụ: cơ sở dữ liệu, tìm kiếm web) để xác minh thông tin được tạo ra.

Hallucination là một thách thức liên tục trong việc phát triển LLMs, nhưng thông qua các kỹ thuật Prompt Engineering và các biện pháp kiểm soát chặt chẽ, chúng ta có thể cải thiện đáng kể độ tin cậy của các mô hình này.
