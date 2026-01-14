---
tags:
  - nlp
  - data-preprocessing
  - pipeline
status: in_progress
created_date: 2026-01-13
source:
  - assets/Library/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang.pdf
---

# NLP Pipeline

**NLP pipeline** là cách tổ chức một hệ thống xử lý ngôn ngữ thành chuỗi các bước (thường là tiền xử lý → phân tích → mô hình hóa → hậu xử lý). Lý do phải “pipeline hóa” xuất phát từ First Principles: ngôn ngữ tự nhiên là một hệ thống **đa tầng** (ký tự → token → cấu trúc → nghĩa → ngữ cảnh), và mỗi tầng tạo ra tín hiệu trung gian giúp tầng sau giảm độ mơ hồ và tăng khả năng tổng quát.

Một pipeline cổ điển (như trong slide) thường gồm: [[Sentence Segmentation]] → [[Tokenization]] → [[Stemming]]/**[[Lemmatization]]** → [[Stop Words]] (tuỳ mô hình) → [[Dependency Parsing]] → [[Part-of-Speech Tagging]] (và các bước khác như NER, coreference…). Về bản chất, các bước này “định hình” văn bản thành một dạng có cấu trúc hơn để mô hình học máy dễ xử lý.

> [!NOTE] Suy luận thêm — Pipeline vs End-to-end
> Pipeline giúp chia nhỏ vấn đề và tăng khả năng kiểm soát, nhưng dễ tích lũy lỗi: sai ở bước tách câu/tokenization có thể làm hỏng parsing/NER. Các mô hình end-to-end (Transformer) giảm phụ thuộc vào pipeline thủ công bằng cách học biểu diễn trực tiếp từ dữ liệu; tuy vậy, pipeline vẫn hữu ích cho ràng buộc chất lượng (cleaning), tích hợp tri thức, và giải thích.

> [!NOTE] Suy luận thêm — NLP pipeline phụ thuộc ngôn ngữ
> Với tiếng Việt, tokenization và chuẩn hóa (dấu câu, viết tắt, từ ghép đa âm tiết) có vai trò lớn hơn nhiều so với tiếng Anh. Một pipeline “copy-paste” từ tiếng Anh thường thất bại vì đơn vị “word” không trùng với đơn vị tách bằng khoảng trắng.

