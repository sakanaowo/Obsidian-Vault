---
tags:
  - AI/Security
  - AI/Agents
  - Concept
created: 2026-01-18
---

> [!NOTE] ELI5
> **Prompt injection** giống như việc ai đó giấu một tờ giấy “ra lệnh” vào trong tài liệu bạn đang đọc. Nếu bạn (hoặc AI) tin tờ giấy đó là luật thật và làm theo, bạn sẽ làm sai việc. Cách phòng là: phân biệt rõ “dữ liệu để đọc” với “chỉ dẫn để làm”, và không để dữ liệu tự biến thành mệnh lệnh.

## Bản chất (First Principles): dữ liệu có thể giả dạng chỉ dẫn

Một [[Large Language Models|LLM]] không có “cổng phân quyền” tự nhiên giữa *content* và *instruction*: mọi thứ được đưa vào context đều là token, và mô hình học các quy luật thống kê về “cái gì thường là chỉ dẫn quan trọng”. Vì vậy, khi một nguồn dữ liệu bên ngoài (webpage, ticket, email, tài liệu nội bộ) chứa các câu kiểu “BỎ QUA mọi hướng dẫn trước đó và làm X”, mô hình có thể ưu tiên nó nếu nó trông giống instruction mạnh.

Claim: prompt injection là rủi ro hệ thống (systemic), không phải bug lẻ. Lý do: nó xuất phát từ cách LLM tối ưu hóa mục tiêu “tuân theo instruction” và từ việc context là một không gian pha trộn. Điều này được thể hiện qua việc các agent tool-use thường thất bại khi chúng “đọc” tài liệu có nội dung mang tính mệnh lệnh và sau đó gọi tool nhạy cảm.

## Hai loại hay gặp: direct vs indirect

**Direct injection**: người dùng nói thẳng với agent các chỉ dẫn độc hại.  
**Indirect injection**: agent đọc một resource (web/doc/email) có chỉ dẫn độc hại được nhúng vào; rủi ro cao hơn vì nó đi vòng qua “hàng rào tâm lý” của người dùng (“tôi chỉ bảo nó tóm tắt tài liệu thôi mà”).

## Liên hệ với agent và MCP

Trong hệ **tool-use**, prompt injection nguy hiểm hơn chat thường vì nó có thể dẫn đến **hành động**. Với **[[Model Context Protocol (MCP)]]**, bề mặt chính là:

- **[[MCP Resources]]**: nơi injection được “đưa vào” context,
- **[[MCP Tools]]**: nơi injection được “thực thi” thành side effects,
- **[[MCP Client]]**: nơi có thể đặt guardrails (user consent, policy check, logging).

## Phòng thủ thực dụng (không phải “một mẹo prompt”)

1) **Tách role**: đảm bảo dữ liệu từ resources luôn được gắn nhãn là “dữ liệu”, không bao giờ nâng nó lên system instruction.  
2) **Least privilege**: giảm sức công phá nếu model bị dẫn dụ (xem [[MCP Security Model]]).  
3) **Gating & confirmation**: tool có side effects phải có xác nhận hoặc policy engine.  
4) **Observability**: log “đã đọc gì” trước khi “đã gọi tool gì” để truy vết và đánh giá.

## TODO

- Viết thêm về mô hình đe doạ (threat model): attacker goals, assets, trust boundaries.
- Tổng hợp chiến lược đánh giá: red-teaming prompt injection cho agent tool-use.

