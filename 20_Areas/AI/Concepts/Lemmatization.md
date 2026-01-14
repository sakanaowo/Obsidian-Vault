---
tags:
  - nlp
  - preprocessing
  - lemmatization
status: in_progress
created_date: 2026-01-13
source:
  - assets/Library/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang.pdf
---

# Lemmatization

**Lemmatization** đưa một từ về **lemma** — dạng từ điển/canonical form có nghĩa là một từ hợp lệ. Khác với [[Stemming]] (cắt theo quy tắc hình thức), lemmatization thường cần thông tin ngôn ngữ học: phân tích hình thái, từ điển, và thường kết hợp với [[Part-of-Speech Tagging]] để chọn lemma đúng (vì một từ bề mặt có thể vừa là danh từ vừa là động từ).

Ví dụ trong slide: “playing” và “plays” → “play”. Lợi ích của lemmatization là giảm biến thể hình thái mà vẫn giữ “đúng từ”; nhược điểm là tốn tài nguyên và phụ thuộc mạnh vào ngôn ngữ (cần công cụ/lexicon riêng).

> [!NOTE] Suy luận thêm — Khi nào lemmatization tạo ra lợi thế?
> Trong các tác vụ rút trích tri thức, tìm kiếm theo ý nghĩa, hoặc chuẩn hóa thuật ngữ (domain-specific), lemmatization thường giúp giảm nhiễu mà không phá vỡ nghĩa như stemming.

