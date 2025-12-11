---
tags:
  - prompt-engineering
  - technique
  - reasoning
status: done
created_date: 2025-12-11
---

# Least-to-Most Prompting

**Least-to-Most Prompting** là một kỹ thuật giải quyết vấn đề phức tạp bằng cách chia nhỏ nó thành một chuỗi các bài toán con đơn giản hơn, sau đó giải quyết tuần tự từng bài toán con đó. Kết quả của bài toán trước được sử dụng làm đầu vào hoặc ngữ cảnh cho bài toán sau.

## Quy trình thực hiện
1.  **Decomposition (Phân rã):** Chia vấn đề chính thành danh sách các vấn đề con.
2.  **Sequential Solving (Giải quyết tuần tự):** Giải quyết từng vấn đề con theo thứ tự.

## So sánh với Chain of Thought
*   **Chain of Thought (CoT):** Yêu cầu mô hình suy nghĩ từng bước trong *cùng một lần* trả lời.
*   **Least-to-Most:** Thường tách biệt rõ ràng giai đoạn "lên kế hoạch" (chia nhỏ vấn đề) và giai đoạn "thực thi" (giải quyết từng phần). Nó hiệu quả hơn CoT đối với các vấn đề dài và phức tạp hơn mức context window cho phép xử lý một lần.

## Ví dụ: Viết một ứng dụng Web
1.  **Prompt 1 (Plan):** "Hãy liệt kê các bước để xây dựng một ứng dụng Todo List bằng React."
    *   *Output:* 1. Thiết kế UI, 2. Tạo Component, 3. Xử lý State, 4. Lưu trữ dữ liệu.
2.  **Prompt 2 (Execute Step 1):** "Dựa trên bước 1, hãy viết code CSS/HTML cho giao diện..."
3.  **Prompt 3 (Execute Step 2):** "Dựa trên giao diện đã có, hãy viết component React..."

Kỹ thuật này giúp mô hình không bị "quá tải" (overwhelmed) bởi độ phức tạp của toàn bộ vấn đề ngay từ đầu.
