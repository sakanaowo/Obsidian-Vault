---
tags:
  - ai-safety
  - challenges
  - llm
  - reliability
status: done
created_date: 2025-12-11
---

# AI Hallucination (Ảo giác AI)

**AI Hallucination** là hiện tượng khi một mô hình ngôn ngữ lớn ([[Large Language Models]]) tạo ra thông tin sai lệch, không chính xác, hoặc hoàn toàn bịa đặt nhưng lại trình bày chúng một cách rất tự tin và trôi chảy.

## Nguyên nhân
*   **Cơ chế dự đoán:** LLM được thiết kế để dự đoán từ tiếp theo (next-token prediction) dựa trên xác suất thống kê, không phải để truy xuất sự thật (fact retrieval). Nếu không có dữ liệu thực tế, nó sẽ chọn từ có xác suất xuất hiện cao nhất để "lấp đầy chỗ trống".
*   **Dữ liệu huấn luyện:** Dữ liệu nguồn có thể chứa thông tin sai lệch, lỗi thời hoặc mâu thuẫn.
*   **Áp lực tuân thủ:** Mô hình có xu hướng ưu tiên việc trả lời câu hỏi của người dùng hơn là thừa nhận sự thiếu hiểu biết (trừ khi được huấn luyện đặc biệt để từ chối).

## Các dạng Hallucination
1.  **Fabrication (Bịa đặt hoàn toàn):** Tạo ra tên sách, trích dẫn, hoặc sự kiện lịch sử không hề tồn tại.
2.  **Inconsistency (Mâu thuẫn):** Đưa ra hai thông tin trái ngược nhau trong cùng một câu trả lời.
3.  **Conflict (Sai lệch với nguồn):** Khi sử dụng trong RAG, mô hình trả lời sai so với tài liệu tham chiếu được cung cấp.

## Cách phòng tránh (Mitigation Strategies)
1.  **Reference Text:** Yêu cầu mô hình chỉ trả lời dựa trên văn bản được cung cấp ("Answer using only the provided text").
2.  **Citation:** Yêu cầu trích dẫn nguồn cụ thể.
3.  **Temperature = 0:** Giảm tính ngẫu nhiên của mô hình.
4.  **"I don't know":** Hướng dẫn mô hình nói "Tôi không biết" nếu không tìm thấy thông tin, thay vì cố gắng đoán.
5.  **Retrieval Augmented Generation (RAG):** Cung cấp kiến thức bên ngoài chính xác để làm cơ sở (grounding) cho câu trả lời.
