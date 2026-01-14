---
tags:
  - nlp
  - syntax
  - pos-tagging
status: in_progress
created_date: 2026-01-13
source:
  - assets/Library/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang.pdf
---

# Part-of-Speech (POS) Tagging

**Part-of-Speech (POS) tagging** là bài toán gán nhãn loại từ (danh từ, động từ, tính từ, trạng từ, giới từ, liên từ…) cho từng token trong câu. Mục tiêu sâu của POS tagging là cung cấp một lớp “nhãn cú pháp” giúp các bước phía sau (parsing, NER, rút trích quan hệ) giảm mơ hồ và học nhanh hơn, vì mô hình không phải tự phát hiện lại mọi tín hiệu chức năng từ dữ liệu thô.

POS tagging khó vì **một từ có thể mang nhiều loại từ** tùy ngữ cảnh (“record” là danh từ hay động từ), và vì ranh giới từ/tokenization có thể không ổn định theo ngôn ngữ. Do đó, POS tagging thường được xem như một bài toán dự đoán theo chuỗi (sequence labeling), nơi mỗi quyết định phụ thuộc vào ngữ cảnh hai phía.

> [!NOTE] Suy luận thêm — POS tagging trong thời đại Transformer
> Dù nhiều hệ end-to-end không cần POS tagging như một bước riêng, POS tagging vẫn hữu ích như một “tín hiệu kiểm tra” (diagnostic) và như dữ liệu phụ trợ cho các hệ thống cần giải thích hoặc ràng buộc ngôn ngữ học.

