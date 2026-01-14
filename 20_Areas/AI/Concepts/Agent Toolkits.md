---
tags:
  - AI/Agent
  - AI/Tools
  - Concept
aliases:
  - LangChain Toolkits
created: 2026-01-04
---

### Định nghĩa

**Agent Toolkits** là các bộ sưu tập công cụ (Tools) được đóng gói sẵn cho một miền cụ thể, giúp [[Autonomous Agents]] dễ dàng tương tác với các hệ thống bên ngoài.

Trong hệ sinh thái [[LangChain]], Toolkits giúp giảm bớt gánh nặng cho lập trình viên bằng cách cung cấp sẵn các hàm (functions) và prompt cần thiết để làm việc với một API hay Database nào đó.

### Ví dụ về Toolkits phổ biến

1.  **SQL Toolkit:**
    *   *Chức năng:* Cho phép Agent truy vấn Database SQL.
    *   *Công cụ con:* `ListTables`, `SchemaInfo`, `QuerySQL`.
    *   *Use case:* "Cho tôi biết doanh thu tháng trước là bao nhiêu?" -> Agent tự viết câu SQL query.

2.  **Pandas/CSV Toolkit:**
    *   *Chức năng:* Phân tích dữ liệu dạng bảng (Dataframe).
    *   *Công cụ con:* Python REPL (để chạy code pandas).
    *   *Use case:* "Vẽ biểu đồ tương quan giữa giá và số lượng bán."

3.  **Gmail/Office365 Toolkit:**
    *   *Chức năng:* Quản lý email, lịch.
    *   *Công cụ con:* `SearchEmails`, `SendEmail`, `CreateEvent`.

### Lợi ích

*   **Abstraction:** Ẩn đi sự phức tạp của API bên dưới.
*   **Safety:** Các toolkit thường đi kèm các cơ chế an toàn (ví dụ: giới hạn số lượng row trả về từ SQL để tránh tràn context).
*   **Prompt Optimization:** Các tool được mô tả kỹ lưỡng để LLM dễ hiểu và sử dụng đúng cách.
