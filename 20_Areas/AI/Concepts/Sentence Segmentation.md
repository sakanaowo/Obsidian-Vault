---
tags:
  - nlp
  - preprocessing
  - sentence-segmentation
status: in_progress
created_date: 2026-01-13
source:
  - assets/Library/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang.pdf
---

# Sentence Segmentation

**Sentence segmentation** (tách câu) là bước xác định ranh giới câu trong một đoạn văn bản. Nếu xem NLP như bài toán biến văn bản thành cấu trúc, thì tách câu là bước “đặt đường biên” để các mô hình phía sau làm việc trên đơn vị có ngữ nghĩa tương đối hoàn chỉnh, thay vì một chuỗi dài dễ trộn lẫn chủ ngữ–vị ngữ giữa các câu.

Về mặt cơ chế, tách câu tưởng đơn giản (dựa dấu “.” “?” “!”) nhưng thực tế khó vì dấu câu không luôn biểu diễn kết câu: “.” có thể là viết tắt (vd., etc.), số thập phân (3.14), tiêu đề, hoặc định dạng liệt kê. 

Các cách làm phổ biến:

- **Rule-based**: luật dựa dấu câu + danh sách viết tắt; nhanh nhưng brittle (dễ vỡ khi domain đổi).
- **Statistical/Neural**: coi mỗi vị trí là một quyết định “có kết câu không” dựa vào đặc trưng/embedding; tổng quát tốt hơn nhưng cần dữ liệu.

> [!NOTE] Suy luận thêm — Tách câu sai là lỗi lan truyền
> Nếu hệ thống cắt sai ranh giới, parsing/NER/sentiment phía sau có thể suy luận sai vì mô hình nhìn thấy một “câu” không đúng ngữ pháp hoặc thiếu thành phần ngữ nghĩa.

