---
tags:
  - AI/Protocol
  - AI/Systems
  - Concept
created: 2026-01-18
---

> [!NOTE] ELI5
> “Transport” là cách hai chương trình nói chuyện với nhau: như gọi điện, nhắn tin, hay nói trực tiếp. MCP không bắt buộc chỉ một kiểu nói; nó có thể dùng nhiều đường khác nhau. Quan trọng là dù đi đường nào, họ vẫn dùng cùng “ngôn ngữ” MCP.

## Transport là lớp “đường truyền”, không phải logic agent

Trong **[[Model Context Protocol (MCP)]]**, transport là cách **[[MCP Client]]** và **[[MCP Server]]** trao đổi message (thường là JSON-RPC). Tách transport ra khỏi semantics đúng vì bạn có thể:

- chạy server như một tiến trình cục bộ (local) khi cần quyền truy cập filesystem,
- hoặc chạy server như dịch vụ mạng (remote) khi cần chia sẻ giữa nhiều client,
- mà không đổi cách client “nghĩ” về tools/resources.

## Hai mô hình triển khai phổ biến

### 1) Local/STDIO (server như subprocess)

Client spawn server và nói chuyện qua STDIO. Điểm mạnh: dễ sandbox theo OS boundary, ít bề mặt mạng, triển khai nhanh trong desktop/IDE. Điểm yếu: quản lý lifecycle và môi trường (PATH, secrets) phức tạp hơn nếu nhiều server.

### 2) Remote/HTTP (server như service)

Client kết nối qua HTTP (đôi khi kết hợp streaming như SSE/WebSocket tùy stack). Điểm mạnh: chia sẻ server cho nhiều client, dễ quan sát (metrics/logs tập trung). Điểm yếu: cần giải bài toán auth, network policy, và rủi ro exfiltration cao hơn.

> [!NOTE] Suy luận thêm (không phải trích dẫn từ một nguồn cụ thể)
> Khi thiết kế hệ thống agent nội bộ, lựa chọn transport thực chất là lựa chọn “trust boundary”: local transport thường gắn với trust ở máy người dùng; remote transport gắn với trust ở mạng và hệ thống IAM.

