---
tags:
  - AI/Agents
  - AI/Protocol
  - Concept
created: 2026-01-18
---

> [!NOTE] ELI5
> **Tool** là việc bạn có thể “nhờ hệ thống làm giúp”, như bấm nút tìm kiếm hoặc gửi một tin nhắn. AI không tự làm được trong đầu; nó phải yêu cầu tool làm thay. Tool càng rõ ràng (đầu vào/đầu ra), AI càng ít làm sai.

## Tools trong MCP và quan hệ với Function Calling

Trong **[[Model Context Protocol (MCP)]]**, **tools** là danh mục hành động mà server công bố, để client có thể gọi theo chuẩn chung. Ở tầng model, việc model “chọn tool” thường được biểu diễn bằng [[Function Calling]] hoặc một cơ chế tương đương; MCP tập trung vào tầng *hệ sinh thái*: mô tả tool, gọi tool, nhận kết quả theo một contract thống nhất giữa client và server.

Claim: chuẩn hóa tools giúp tăng khả năng tái sử dụng và an toàn. Lý do là khi tools có schema và semantics ổn định, bạn có thể:

- kiểm tra đầu vào (validation),
- ghi log/trace theo chuẩn,
- áp policy (ai được gọi tool nào),
- và test agent bằng “mock server” thay vì gọi hệ thống thật.

## Semantics quan trọng khi thiết kế tool

### 1) Side effects và “điểm không quay lại”

Tool có side effects (ghi dữ liệu, gửi email, thanh toán) cần phân loại và guardrail mạnh hơn tool “read-only”. Nếu không, prompt injection có thể biến một đoạn văn trong tài liệu thành “lệnh” khiến agent gây thiệt hại. Vì vậy, tool interface nên:

1) tách “dry-run/plan” khỏi “commit”, hoặc  
2) yêu cầu user consent ở client, hoặc  
3) có cơ chế policy check trước khi thực thi.

### 2) Idempotency và retry

Trong môi trường phân tán, retry là bình thường. Nếu tool không idempotent, một lần retry có thể tạo side effect lặp. Thiết kế đúng là: nếu có thể, nhận `request_id` hoặc `idempotency_key` để server đảm bảo “gọi lại vẫn như cũ”.

### 3) Đầu ra có cấu trúc (và giới hạn)

Tool output nên có cấu trúc rõ, tối thiểu hóa văn bản tự do, và giới hạn kích thước. Điều này đúng vì:

- model cần cấu trúc để suy luận ổn định,
- client cần giới hạn để tránh context overflow,
- và bạn cần bề mặt tấn công nhỏ hơn trước prompt injection.

