---
tags:
  - AI/Agent
  - AI/Architecture
  - Concept
aliases:
  - BabyAGI Architecture
created: 2026-01-04
---

### Định nghĩa

**Plan-and-Execute Agent** (Agent Lập kế hoạch và Thực thi) là một kiến trúc Agent tách biệt hai quá trình: **Planning** (Lập kế hoạch) và **Execution** (Thực thi).

Khác với [[ReAct Framework]] (suy nghĩ từng bước một), mô hình này lập ra toàn bộ kế hoạch ngay từ đầu, sau đó mới thực hiện.

### Quy trình hoạt động

1.  **Planner:** Một LLM đóng vai trò "kiến trúc sư". Nó nhận mục tiêu từ người dùng và tạo ra một danh sách các bước (Task List) cần thực hiện.
2.  **Executor:** Một Agent khác (thường là Action Agent) lần lượt thực hiện từng task trong danh sách.
3.  **Replanner (Tùy chọn):** Sau khi mỗi task hoàn thành, Planner có thể xem xét lại kết quả và cập nhật kế hoạch (thêm/bớt task) nếu cần thiết (như trong **BabyAGI**).

### So sánh với ReAct

| Đặc điểm | ReAct | Plan-and-Execute |
| :--- | :--- | :--- |
| **Tư duy** | Step-by-step (Nghĩ -> Làm -> Nghĩ -> Làm) | Upfront Planning (Nghĩ hết -> Làm hết) |
| **Ưu điểm** | Linh hoạt, xử lý tình huống bất ngờ tốt. | Tốt cho các quy trình dài hạn, phức tạp cần cái nhìn tổng thể. |
| **Nhược điểm** | Dễ bị lạc hướng (lost focus) nếu chuỗi quá dài. | Khó sửa sai nếu kế hoạch ban đầu sai (trừ khi có Replanner). |

### Ứng dụng

Phù hợp cho các tác vụ phức tạp đòi hỏi nhiều bước phụ thuộc lẫn nhau, ví dụ: "Viết một cuốn sách", "Nghiên cứu thị trường và tạo báo cáo".
