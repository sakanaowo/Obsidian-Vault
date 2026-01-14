---
tags:
  - AI/Memory
  - AI/Agent
  - Concept
aliases:
  - Long-term Memory
  - Short-term Memory
created: 2026-01-04
---

### Định nghĩa

**Agent Memory** là cơ chế cho phép [[Autonomous Agents]] lưu trữ, ghi nhớ và truy xuất thông tin qua các bước thực hiện (steps) hoặc qua các phiên làm việc (sessions). Nếu không có Memory, Agent sẽ bị "mất trí nhớ" sau mỗi lần gọi model.

### Phân loại Memory

1.  **Short-Term Memory (STM):**
    *   **Bản chất:** Lưu trữ trong Context Window của LLM.
    *   **Dạng:** Lịch sử hội thoại (Chat History), các bước suy luận vừa thực hiện (Thought/Observation trước đó).
    *   **Giới hạn:** Bị giới hạn bởi độ dài context (ví dụ: 8k, 32k tokens). Khi đầy, thông tin cũ sẽ bị xóa (trừ khi dùng kỹ thuật tóm tắt).
    *   **LangChain:** `ConversationBufferMemory`, `ConversationSummaryMemory`.

2.  **Long-Term Memory (LTM):**
    *   **Bản chất:** Lưu trữ ngoài (External Storage).
    *   **Dạng:** [[Vector Databases]] (lưu kiến thức ngữ nghĩa), SQL Database (lưu thông tin có cấu trúc).
    *   **Cơ chế:** Agent sử dụng [[Semantic Search]] để "nhớ lại" (retrieve) những thông tin liên quan từ quá khứ khi cần thiết.
    *   **LangChain:** `VectorStoreRetrieverMemory`.

### Vai trò của Memory trong Agent

*   **Duy trì ngữ cảnh:** Hiểu được "cái đó" trong câu "Mua cho tôi cái đó" ám chỉ sản phẩm đã nhắc đến trước đó.
*   **Học hỏi (Learning):** Ghi nhớ phản hồi của người dùng hoặc kết quả của các hành động trước để cải thiện hiệu suất trong tương lai (Reflection).
*   **Cá nhân hóa:** Nhớ sở thích của người dùng qua nhiều phiên làm việc.
