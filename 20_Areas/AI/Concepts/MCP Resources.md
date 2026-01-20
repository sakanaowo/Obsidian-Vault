---
tags:
  - AI/Agents
  - AI/Protocol
  - Concept
created: 2026-01-18
---

> [!NOTE] ELI5
> **Resource** là thứ bạn được “đọc” giống như mở một trang sách hoặc xem một tờ giấy. Bạn không tự tìm trong kho; bạn yêu cầu đúng resource và hệ thống đưa nội dung cho bạn. Nhờ vậy, AI có thể xem tài liệu đúng lúc mà không phải nhét tất cả vào prompt từ đầu.

## Resource là gì trong MCP

Trong **[[Model Context Protocol (MCP)]]**, **resources** là các “đối tượng ngữ cảnh” mà server công bố để client có thể đọc và đưa vào prompt/context cho model. Đây là một cách formal hóa câu hỏi: “model được phép *biết* những dữ liệu nào, và dữ liệu đó được truy cập bằng cách nào?”

Claim: resource là primitive quan trọng hơn so với việc “dán nguyên văn dữ liệu vào prompt”, vì resource cho phép:

1) **Discovery** (liệt kê cái gì có thể đọc),  
2) **Addressability** (địa chỉ hóa bằng URI/ID),  
3) **Governance** (quyền đọc, logging, và kiểm soát phạm vi).

Lập luận: điều này đúng vì mọi hệ thống lớn đều cần *quản trị truy cập*; prompt thuần văn bản không có affordance tự nhiên cho RBAC/ABAC và audit.

## Resource design: cái bẫy “đưa quá nhiều”

Resource tốt không nhất thiết là “đầy đủ”, mà là “đủ để quyết định”. Nếu bạn trả về quá nhiều văn bản, agent dễ gặp:

- **Context bloat**: token tăng → chi phí tăng → độ chính xác giảm do nhiễu.
- **Prompt injection surface**: càng nhiều text từ nguồn ngoài, càng nhiều cơ hội bị chèn chỉ dẫn độc hại.

Vì vậy resources thường cần *chunking* và *retrieval semantics* (ví dụ: đọc theo đoạn, theo trang, theo query).

## Resource templates (khái niệm liên quan)

Một pattern phổ biến là **resource template**: thay vì công bố từng resource tĩnh, server công bố một “khuôn” để client điền tham số (ví dụ: `doc://kb/{id}`) rồi đọc. Template đúng vì nó cho phép “không gian resource” lớn được mô tả gọn, nhưng vẫn giữ được boundary: client chỉ đọc qua những đường dẫn mà server cho phép.

> [!NOTE] TODO
> Nếu bạn muốn tách sâu hơn, có thể tạo một concept note riêng: `Resource Templates (MCP)` để bàn về parameter validation, enumeration, và chống path traversal.

