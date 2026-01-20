---
tags:
  - AI/Agents
  - AI/Security
  - AI/Protocol
  - Concept
created: 2026-01-18
---

> [!NOTE] ELI5
> Bảo mật MCP giống như đặt luật cho người trợ lý: bạn cho nó chìa khóa nào, cho vào phòng nào, và phải xin phép trước khi làm việc quan trọng. Nếu không có luật, trợ lý có thể bị “dụ” làm điều sai chỉ vì ai đó nói khéo. Luật rõ ràng giúp tránh nhầm lẫn và giảm thiệt hại khi có lỗi.

## “Chuẩn hóa” không đồng nghĩa “an toàn”

**[[Model Context Protocol (MCP)]]** chuẩn hóa cách mô tả và gọi **[[MCP Tools]]**/**[[MCP Resources]]**, nhưng an toàn đến từ *policy* và *thực thi*. Claim: hệ thống agent an toàn phải giả định model có thể bị thao túng bởi prompt injection; điều này đúng vì model tối ưu cho “làm theo chỉ dẫn có vẻ hợp lý” chứ không tối ưu cho “bảo vệ tài sản”.

## Các trụ cột bảo mật thực dụng

### 1) Capability + least privilege

Chỉ công bố tools/resources tối thiểu cần thiết cho tác vụ. Đây là phòng tuyến quan trọng nhất vì nó giới hạn “tầm hại tối đa” ngay cả khi model bị dẫn dụ. Lập luận: giảm quyền làm giảm không gian hành động; giảm không gian hành động làm giảm xác suất chọn hành động nguy hiểm.

### 2) User consent cho side effects

Những tool có side effects nên yêu cầu xác nhận ở **[[MCP Client]]** (đặc biệt trong UX tương tác). Điều này đúng vì model không có “trách nhiệm pháp lý”; người dùng/đơn vị vận hành mới là nơi phải kiểm soát hành động không thể hoàn tác.

### 3) Input hardening: chống prompt injection qua resources

Resources thường chứa text từ môi trường (docs, ticket, web). Đây là bề mặt prompt injection. Một pattern là phân tách:

- “nội dung nguồn” (đưa vào context như dữ liệu),
- và “chỉ dẫn hệ thống” (không bao giờ được trộn lẫn).

Ngoài ra, cần logging để truy vết: model đã đọc resource nào trước khi gọi tool nhạy cảm.

### 4) Observability và kiểm toán

Ghi lại tool call (ai gọi, khi nào, input/output, request_id) và resource read (URI, size, trích đoạn). Claim: không có trace thì không có an toàn vận hành; đúng vì mọi sự cố nghiêm trọng đều cần post-mortem và khả năng tái hiện.

> [!NOTE] TODO
> Có thể mở rộng bằng một note riêng: “Prompt Injection” (nếu chưa có) và liên kết tới các chiến lược sandboxing, policy engines, và evaluation/attack simulation cho agent.

