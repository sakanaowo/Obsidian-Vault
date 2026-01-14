---
tags:
  - AI/Technique
  - AI/Agent
  - Concept
aliases:
  - Reason and Act
created: 2026-01-04
---

### Định nghĩa

**ReAct** (viết tắt của **Re**ason and **Act**) là một mô hình thiết kế Prompt cho [[Autonomous Agents]], kết hợp giữa khả năng suy luận ([[Chain of Thought]]) và khả năng hành động (Action execution).

Thay vì để model suy nghĩ xong mới hành động, hoặc hành động mù quáng, ReAct tạo ra một vòng lặp liên tục đan xen giữa suy nghĩ và hành động.

### Vòng lặp ReAct

Quy trình chuẩn của một ReAct Agent bao gồm 3 bước lặp lại:

1.  **Thought (Suy nghĩ):** Model phân tích tình huống hiện tại và quyết định bước tiếp theo cần làm gì.
    *   *Ví dụ:* "Người dùng hỏi về tuổi của vợ Barack Obama. Mình cần tìm tên vợ ông ấy trước."
2.  **Action (Hành động):** Model chọn một công cụ (Tool) cụ thể để thực hiện ý định đó.
    *   *Ví dụ:* `GoogleSearch("Barack Obama wife")`
3.  **Observation (Quan sát):** Model đọc kết quả trả về từ công cụ.
    *   *Ví dụ:* "Michelle Obama."
    *   *New Thought:* "Đã có tên là Michelle Obama. Giờ mình cần tìm tuổi của bà ấy." -> *New Action...*

### Lợi ích

*   **Độ chính xác cao:** Giảm thiểu ảo giác ([[Hallucination]]) vì model dựa vào dữ liệu thực tế từ Observation.
*   **Khả năng giải quyết vấn đề phức tạp:** Chia nhỏ vấn đề thành từng bước logic, model có thể tự sửa sai nếu một Action không trả về kết quả mong muốn.
*   **Minh bạch (Interpretability):** Con người có thể đọc log (Thought/Action/Observation) để hiểu tại sao Agent đưa ra quyết định đó.
