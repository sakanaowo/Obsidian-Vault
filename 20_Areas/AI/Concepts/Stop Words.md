---
tags:
  - nlp
  - preprocessing
  - stopwords
status: in_progress
created_date: 2026-01-13
source:
  - assets/Library/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang.pdf
---

# Stop Words

**Stop words** là các từ xuất hiện rất thường xuyên trong một ngôn ngữ (ví dụ “the”, “is”, “and”), thường mang ít thông tin phân biệt trong các mô hình dựa trên đếm tần suất. Trong bối cảnh BoW/TF-IDF, việc loại stopwords có thể làm giảm kích thước vector và tăng tỷ lệ tín hiệu/nhiễu, vì mô hình không bị chi phối bởi các từ chức năng quá phổ biến.

Tuy nhiên, stopwords không “vô nghĩa” theo ngôn ngữ học: chúng mã hóa quan hệ cú pháp, mạch lạc, và sắc thái diễn đạt. Vì vậy, với các mô hình ngữ cảnh hiện đại, loại bỏ stopwords một cách thô thường là một sai lầm thiết kế.

> [!NOTE] Suy luận thêm — Stopwords phụ thuộc domain
> Trong các miền chuyên biệt, “từ dừng” có thể là thuật ngữ quan trọng (“may” trong legal text, “charge” trong chemistry vs finance). Danh sách stopwords vì vậy nên được xem như một giả thuyết cần kiểm chứng, không phải chân lý.

