---
tags:
  - AI/Agents
  - AI/Protocol
  - Concept
alias:
  - Model Context Protocol Server
created: 2026-01-18
---

> [!NOTE] ELI5
> **MCP server** giống như một “quầy dịch vụ” cung cấp hai thứ: (1) thông tin bạn được đọc và (2) việc bạn được nhờ làm. Bạn không tự vào kho lấy đồ; bạn phải hỏi quầy và quầy chỉ đưa những thứ nó cho phép. Nhờ vậy, nhiều ứng dụng khác nhau có thể đến cùng một quầy theo cùng một cách.

## Vai trò và ranh giới trách nhiệm

Một **MCP server** là tiến trình/dịch vụ đứng sau **[[Model Context Protocol (MCP)]]** để “công bố capability” và thực thi các yêu cầu từ **[[MCP Client]]**. Lập luận cốt lõi: server là nơi *đóng gói tích hợp* với hệ thống thật (filesystem, database, SaaS, nội bộ), còn client là nơi *điều phối* với model và UX. Tách như vậy đúng vì nó cho phép bạn thay model/client mà không viết lại tích hợp, và thay tích hợp mà không đụng vào logic agent.

## MCP server thường cung cấp những gì

### 1) Resources (đọc dữ liệu)

Server công bố danh mục **[[MCP Resources]]** (ví dụ: “tài liệu A”, “bảng B”), và triển khai cách đọc chúng. Resources nên được thiết kế để trả về “đơn vị ngữ cảnh hữu dụng”, không phải dump toàn bộ dữ liệu thô, vì ngữ cảnh của model là hữu hạn.

### 2) Tools (thực thi hành động)

Server công bố **[[MCP Tools]]**: các hàm/hành động mà client có thể gọi. Nếu tool có side effects (gửi email, xoá file, deploy), server cần policy rõ ràng về xác thực và logging, vì “tool call” là nơi agent có thể gây hại nếu bị prompt injection hoặc sai lệch mục tiêu.

### 3) Prompts/Templates (tùy triển khai)

Một số server còn cung cấp prompt template để client dùng nhất quán trong các tác vụ lặp (ví dụ: format báo cáo, checklist). Ý tưởng đúng vì nó biến “prompt” thành artefact có versioning, gần với code hơn là văn bản rải rác.

## Nuance: server là nơi dễ “leak” nhất

MCP server nằm sát dữ liệu thật nên rủi ro chính là **data exfiltration** và **privilege escalation**. Nguyên tắc thực chiến:

- Capability phải tối thiểu: chỉ công bố resources/tools cần thiết.
- Đầu ra cần giảm nhạy cảm: lọc PII/secret nếu phù hợp ngữ cảnh.
- Tách môi trường và khóa: production/dev khác nhau, token scoped theo server.

> [!NOTE] Suy luận thêm (không phải trích dẫn từ một nguồn cụ thể)
> Nếu coi agent là một “OS”, MCP server đóng vai trò tương tự “device driver”: chuẩn hóa cách nói chuyện với phần cứng/dịch vụ. Ẩn dụ này hữu ích để nghĩ về sandboxing và permission boundary.

