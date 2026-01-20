---
tags:
  - AI/MOC
  - Concept
created: 2026-01-18
---

> [!NOTE] ELI5
> Đây là “mục lục” cho các mảnh ghép giúp một LLM trở thành một agent có thể đọc dữ liệu và làm việc ngoài đời thật. Bạn có thể bắt đầu từ MCP để hiểu “chuẩn cắm tool/dữ liệu”, rồi đi sang Function Calling để hiểu “model biểu diễn lệnh gọi tool”.

## Chuẩn giao tiếp: Model ↔ Context/Tools

### Model Context Protocol (MCP)

- [[Model Context Protocol (MCP)]]
- [[MCP Server]]
- [[MCP Client]]
- [[MCP Resources]]
- [[MCP Tools]]
- [[MCP Transports]]
- [[MCP Security Model]]

### Tool use ở tầng model API

- [[Function Calling]]

## Framework & kiến trúc agent

- [[Autonomous Agents]]
- [[Agent Toolkits]]
- [[Agent Memory]]
- [[LangChain]]

## TODO (hướng mở rộng)

Nếu cần phủ hết bề mặt tấn công và độ tin cậy khi tool-use:

- Tạo/chuẩn hóa concept note [[Prompt Injection]] và liên kết với [[MCP Security Model]].
- Tạo concept note “Tool Reliability” (idempotency, retries, timeouts, partial failure).

