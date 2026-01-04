---
tags:
  - AI/PromptEngineering
  - AI/Technique
  - Concept
aliases:
  - Sequential Chaining
created: 2026-01-04
---

### Định nghĩa

**Prompt Chaining** là kỹ thuật chia nhỏ một tác vụ phức tạp thành một chuỗi các sub-task (tác vụ con) tuần tự, trong đó output của prompt trước trở thành input của prompt sau. Thay vì cố gắng giải quyết tất cả mọi thứ trong một prompt khổng lồ ("One prompt to rule them all"), Prompt Chaining áp dụng nguyên lý "Divide and Conquer" (Chia để trị).

### Tại sao cần Prompt Chaining?

1.  **Reliability (Độ tin cậy):** LLM thường hoạt động tốt hơn khi tập trung vào một nhiệm vụ cụ thể tại một thời điểm. Giảm thiểu khả năng [[Hallucination]] và bỏ sót chỉ thị.
2.  **Debuggability:** Dễ dàng xác định bước nào bị lỗi trong chuỗi xử lý để tinh chỉnh.
3.  **Context Window:** Giúp xử lý các tài liệu dài vượt quá giới hạn token bằng cách xử lý từng phần (như trong mô hình Map-Reduce).

### Các mô hình Chaining phổ biến

*   **Sequential Chain:** A -> B -> C. Output của A là Input của B.
*   **Map-Reduce:** Xử lý song song nhiều phân đoạn văn bản (Map), sau đó tổng hợp kết quả lại (Reduce). Thường dùng cho tóm tắt văn bản dài.
*   **Refine:** Sinh ra một kết quả sơ khởi, sau đó lặp lại qua các phần dữ liệu tiếp theo để tinh chỉnh và cập nhật kết quả đó.

### Ứng dụng

*   Viết quy trình code: (1) Viết test case -> (2) Viết function -> (3) Review code.
*   Tóm tắt sách: (1) Tóm tắt từng chương -> (2) Tổng hợp tóm tắt các chương thành tóm tắt sách.
