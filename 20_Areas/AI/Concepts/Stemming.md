---
tags:
  - nlp
  - preprocessing
  - stemming
status: in_progress
created_date: 2026-01-13
source:
  - assets/Library/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang.pdf
---

# Stemming

**Stemming** là kỹ thuật chuẩn hóa hình thái bằng cách đưa một từ về dạng **gốc hình thức** (stem) thông qua các quy tắc cắt/biến đổi hậu tố/tiền tố. Điểm đặc trưng: stem **không bắt buộc** là một từ hợp lệ trong ngôn ngữ; đó là “móc” hình thức để gộp các biến thể gần nhau, nhằm giảm số lượng từ vựng hiệu dụng và giảm độ thưa của dữ liệu.

Ví dụ trong slide: “intelligently”, “intelligence”, “intelligent” được đưa về “intelligen”. Cái được “mua” ở đây là khả năng gộp các biến thể để mô hình bag-of-words/TF-IDF đếm tần suất ổn định hơn; cái “trả giá” là nguy cơ gộp sai sắc thái nghĩa hoặc gộp các từ khác gốc nhưng trùng stem.

> [!NOTE] Suy luận thêm — Stemming phù hợp nhất với mô hình tuyến tính cổ điển
> Khi dùng BoW/TF-IDF + linear classifier, stemming thường cải thiện recall. Với mô hình ngữ cảnh (Transformer), stemming hiếm khi cần vì subword tokenization đã xử lý biến thể hình thức tốt hơn.

