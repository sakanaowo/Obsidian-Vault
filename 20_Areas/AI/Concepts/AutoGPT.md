---
tags:
  - AI/Agent
  - AI/History
  - Concept
alias:
  - Auto-GPT
created: 2026-01-04
---

### Định nghĩa

**AutoGPT** là một dự án mã nguồn mở đình đám (từng là repo tăng trưởng nhanh nhất lịch sử GitHub), đại diện cho thế hệ [[Autonomous Agents]] đầu tiên có khả năng thực hiện các mục tiêu cấp cao (High-level goals) thông qua việc tự động suy luận và sử dụng công cụ.

Khác với [[BabyAGI]] tập trung vào quản lý task list, AutoGPT nổi bật nhờ khả năng truy cập Internet và File System mạnh mẽ.

### Đặc điểm nổi bật

1.  **Internet Access:** Có thể tự Google Search để tìm thông tin mới nhất (vượt qua cutoff date của model).
2.  **Long-Term Memory:** Sử dụng [[Vector Databases]] (như [[Pinecone]]) để ghi nhớ thông tin qua các phiên làm việc.
3.  **File Management:** Có thể đọc/ghi file, cho phép nó viết code, lưu lại và chạy code đó.
4.  **Continuous Mode:** Có khả năng chạy liên tục mà không cần user nhắc (prompting) cho đến khi hoàn thành mục tiêu (tuy nhiên dễ bị kẹt trong vòng lặp vô hạn).

### Cơ chế

AutoGPT sử dụng mô hình "Thought-Plan-Criticism" (Suy nghĩ - Lập kế hoạch - Phê bình):
*   **Thoughts:** "Tôi cần tìm giá cổ phiếu Tesla."
*   **Reasoning:** "Việc này giúp tôi phân tích thị trường."
*   **Plan:** "- Tìm giá trên Google.
- Lưu vào file csv."
*   **Criticism:** "Tôi cần kiểm tra xem dữ liệu có chính xác không."

### Hạn chế

Dù rất ấn tượng, AutoGPT thời kỳ đầu thường gặp vấn đề về độ ổn định (dễ bị loop), chi phí token cao và đôi khi hành động thiếu kiểm soát. Tuy nhiên, nó đã mở đường cho các framework agent trưởng thành hơn như [[LangChain]] Agents.
