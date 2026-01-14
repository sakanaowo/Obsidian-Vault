---
tags:
  - AI/Agent
  - AI/Concept
aliases:
  - AI Agents
  - Intelligent Agents
created: 2026-01-04
---

### Định nghĩa

**Autonomous Agents** (Tác nhân tự hành) là các hệ thống AI có khả năng tự suy luận, lập kế hoạch và thực hiện chuỗi hành động để đạt được mục tiêu cụ thể mà không cần sự can thiệp liên tục của con người.

Khác với LLM thông thường (chỉ thụ động trả lời input), Agent đóng vai trò chủ động, sử dụng LLM làm "bộ não" để điều khiển các công cụ khác.

### Các thành phần cốt lõi

1.  **Brain (LLM):** Trung tâm xử lý logic, chịu trách nhiệm suy luận ([[Chain of Thought]]), phân rã tác vụ và ra quyết định.
2.  **Perception (Inputs):** Tiếp nhận thông tin từ môi trường (User query, API response, Database state).
3.  **Actions (Tools):** Khả năng tác động vào thế giới thực thông qua công cụ (Search Google, Gửi Email, Chạy Code Python, Query Database).
4.  **Planning:** Khả năng chia nhỏ mục tiêu lớn thành các bước nhỏ hơn và tự sửa lỗi (Self-correction) nếu gặp thất bại.
5.  **Memory:**
    *   **Short-term:** Lưu trữ ngữ cảnh hội thoại hiện tại.
    *   **Long-term:** Lưu trữ kiến thức và kinh nghiệm quá khứ (thường dùng [[Vector Databases]]).

### Ví dụ

*   **User:** "Lên kế hoạch du lịch Đà Lạt 3 ngày và đặt vé máy bay giúp tôi."
*   **Agent:**
    1.  *Thought:* Cần tìm thông tin thời tiết -> tìm địa điểm -> tìm vé máy bay.
    2.  *Action 1:* Google Search "Thời tiết Đà Lạt tuần tới".
    3.  *Action 2:* Google Search "Địa điểm du lịch Đà Lạt".
    4.  *Action 3:* Sử dụng công cụ đặt vé (qua API) để kiểm tra giá.
    5.  *Result:* Tổng hợp thông tin và phản hồi người dùng.
