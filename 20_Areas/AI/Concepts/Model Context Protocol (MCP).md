---
tags:
  - AI/Agents
  - AI/Protocol
  - Concept
alias:
  - MCP
  - Model Context Protocol
created: 2026-01-18
---

> [!NOTE] ELI5
> Hãy tưởng tượng bạn có một “bộ não” (LLM) nhưng nó không tự đọc được file hay gọi ứng dụng khác. **MCP** giống như một cái “ổ cắm tiêu chuẩn” để bộ não đó cắm vào nhiều “thiết bị” (tool, dữ liệu) một cách thống nhất. Thay vì mỗi thiết bị dùng một loại đầu cắm khác nhau, MCP làm chúng dùng chung chuẩn nên dễ thay và dễ mở rộng.

## 1) MCP giải bài toán gì (First Principles)

Một [[Large Language Models|LLM]] bản chất là một bộ máy dự đoán token: nó *không* có quyền truy cập mặc định vào file hệ thống, database, hay API nội bộ. Muốn LLM làm việc “thực” (agent), ứng dụng phải cung cấp **ngữ cảnh** (context) và **công cụ** (tools) cho nó. Vấn đề là: nếu mỗi ứng dụng/nhà cung cấp tự nghĩ ra một kiểu “tool interface” khác nhau, hệ sinh thái bị phân mảnh: bạn viết một tool cho A thì không dùng lại được cho B, và câu hỏi “LLM được phép thấy/gọi cái gì” trở thành một mê cung tích hợp.

**Model Context Protocol (MCP)** đúng vì nó tách “điều mà model có thể biết và có thể làm” thành các primitive chuẩn hóa, được thể hiện qua một hợp đồng giao tiếp (protocol) giữa **MCP client** và **MCP server**. Khi chuẩn hóa interface, bạn chuyển chi phí tích hợp từ “N×M” (mỗi app × mỗi tool) về “N+M”: app chỉ cần nói MCP, tool chỉ cần nói MCP, và hai bên *khớp* nhau.

## 2) Mô hình khái niệm: “Context” không chỉ là prompt

Trong MCP, “context” được mô hình hóa thành các loại khả năng có cấu trúc, thường xoay quanh:

- **Resources**: dữ liệu/đối tượng *đọc* được (read-only) mà model có thể yêu cầu cung cấp vào ngữ cảnh (ví dụ: một file, một record, một tài liệu).
- **Tools**: hành động có thể gọi (có thể có side effects) tương tự [[Function Calling]] nhưng được “đóng gói” thành chuẩn chung ở tầng tích hợp.
- (Tùy triển khai) **Prompts/Templates**: các “mẫu” prompt được server cung cấp để client dùng nhất quán trong những tác vụ lặp lại.

Điểm quan trọng về cơ chế: MCP hướng tới việc làm rõ *ranh giới quyền lực* (capability boundary). Model không “tự” truy cập dữ liệu; nó chỉ có thể **yêu cầu** client thực hiện tương tác với server theo những capability đã công bố. Điều này khiến việc audit, sandboxing, và least-privilege trở nên khả thi hơn.

## 3) Luồng tương tác điển hình (Client ↔ Server ↔ Model)

Một vòng làm việc thường có cấu trúc:

1. **Discovery**: client hỏi server “bạn có resources/tools nào?” để tạo *catalog*.
2. **Reasoning & Selection**: model (thông qua client) quyết định cần đọc resource nào hoặc gọi tool nào để đạt mục tiêu.
3. **Read/Call**: client thực thi `resources/read` hoặc `tools/call`, nhận kết quả.
4. **Synthesis**: client đưa kết quả trở lại vào ngữ cảnh để model tổng hợp câu trả lời hoặc quyết định bước tiếp theo.

Mệnh đề “MCP giúp agent mạnh hơn” đúng vì agent mạnh khi có **vòng phản hồi (feedback loop)** giữa (a) mô hình, (b) hành động/quan sát từ thế giới. MCP chuẩn hóa phần (b) để vòng lặp này trở nên có thể tái sử dụng và kiểm soát.

## 4) Ví dụ tối thiểu (minimally concrete)

Giả sử một MCP server cung cấp tool `search_docs` và resource `doc://handbook/oncall`. Một client (ứng dụng agent) có thể:

1) liệt kê tools/resources; 2) model chọn đọc handbook; 3) client đọc và đưa nội dung vào context; 4) model gọi `search_docs` để tìm cụm từ; 5) trả lời.

Nếu bạn quen với [[LangChain]]: MCP đóng vai trò như một “chuẩn cổng” ở tầng tích hợp tool/context; còn LangChain là framework orchestration ở tầng pipeline/agent loop. Hai thứ có thể bổ trợ nhau.

## 5) So sánh nhanh: MCP vs Function Calling

[[Function Calling]] là “khả năng tạo lời gọi tool có cấu trúc” ở tầng model API (thường gắn chặt với một vendor hoặc SDK). MCP là “chuẩn mô tả và truy cập tool/resource” ở tầng *hệ sinh thái tích hợp*. Nói cách khác: Function Calling trả lời “model sẽ biểu diễn lệnh gọi ra sao?”, còn MCP trả lời “các tool/resources được công bố, truy cập, và quản trị như thế nào để nhiều client dùng lại?”

## 6) Nuance quan trọng (thực chiến)

1) **Thiết kế tool**: tool càng “thuần” (đầu vào/đầu ra rõ, ít side effects, có idempotency khi có thể) thì agent càng đáng tin.  
2) **Giới hạn context**: resources cần cơ chế paging/chunking; nếu không, agent sẽ tự tạo “context bloat” và chi phí suy luận tăng.  
3) **Bảo mật**: chuẩn hóa interface không tự động an toàn; an toàn đến từ policy: xác thực server, giới hạn capability, logging, và user consent cho hành động nhạy cảm.

## Liên kết

Xem tiếp các thành phần: [[MCP Server]], [[MCP Client]], [[MCP Resources]], [[MCP Tools]], [[MCP Transports]], [[MCP Security Model]].

