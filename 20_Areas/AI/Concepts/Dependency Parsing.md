---
tags:
  - nlp
  - syntax
  - parsing
  - dependency-parsing
status: in_progress
created_date: 2026-01-13
source:
  - assets/Library/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang.pdf
---

# Dependency Parsing

**Dependency parsing** là bài toán gán cấu trúc cú pháp cho câu dưới dạng một đồ thị (thường là cây) gồm các quan hệ phụ thuộc giữa từ–từ. Mỗi quan hệ là một cung có nhãn (dependency label) nối từ “head” (từ trung tâm) đến “dependent” (từ phụ thuộc). Trong nhiều ngôn ngữ, **động từ chính** thường là gốc (ROOT) vì nó chi phối cấu trúc mệnh đề.

Giá trị của dependency parsing không chỉ là “vẽ cây”, mà là cung cấp một cấu trúc giúp trả lời các câu hỏi kiểu “ai làm gì cho ai”: chủ ngữ, tân ngữ, bổ ngữ, trạng ngữ… Khi văn bản được đưa về cấu trúc phụ thuộc, nhiều thao tác rút trích thông tin và chuẩn hóa nghĩa trở nên trực tiếp hơn.

> [!NOTE] Suy luận thêm — Parsing là nơi ambiguity bùng nổ
> Một chuỗi từ có thể có nhiều cây cú pháp hợp lệ. Do đó, parsing là một dạng “chọn cấu trúc hợp lý” dựa trên thống kê/ngữ cảnh, và là nguồn lỗi lan truyền nếu pipeline phụ thuộc mạnh vào kết quả parsing.

