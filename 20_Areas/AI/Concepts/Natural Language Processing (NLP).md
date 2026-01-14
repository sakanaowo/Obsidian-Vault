---
tags:
  - nlp
  - natural-language
  - ai
status: in_progress
created_date: 2026-01-13
aliases:
  - NLP
source:
  - assets/Library/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang.pdf
---

# Natural Language Processing (NLP)

**Natural Language Processing (NLP)** là lĩnh vực xây dựng hệ thống tính toán có khả năng **nhận**, **hiểu**, **suy luận** và/hoặc **tạo** ngôn ngữ tự nhiên (văn bản, tiếng nói). Điểm khó cốt lõi của NLP không nằm ở việc “xử lý chuỗi ký tự”, mà nằm ở việc **ánh xạ giữa bề mặt ngôn ngữ và ý nghĩa** trong điều kiện thông tin luôn thiếu: người nói thường không nói hết mọi thứ vì họ dựa vào tri thức chung, ngữ cảnh, và các quy ước giao tiếp. Vì vậy, NLP bản chất là một bài toán **biểu diễn + suy luận dưới bất định**.

Trong thực hành, NLP thường được tách thành hai trụ:

**Natural Language Understanding (NLU)**: từ câu nói/câu viết → trích xuất cấu trúc/ý định/quan hệ (ví dụ “ai làm gì cho ai”), rồi dùng chúng để ra quyết định. NLU mạnh khi nó không chỉ “nhặt từ khóa” mà còn xử lý **vai trò** (agent/patient), **phạm vi** (scope), và **hàm ý** (implicature).

**Natural Language Generation (NLG)**: từ biểu diễn có cấu trúc (facts, ý định, dữ liệu) → tạo câu tự nhiên phù hợp. NLG luôn đối mặt trade-off giữa **tính trung thực với dữ liệu** (faithfulness) và **tính tự nhiên/đúng phong cách** (fluency, style, politeness).

> [!NOTE] Suy luận thêm — Vì sao “hiểu” không thể chỉ là mapping 1-1?
> Nếu mỗi câu luôn có đúng một nghĩa, NLP sẽ gần như là bài toán dịch “text → logic”. Nhưng ngôn ngữ thực tế đầy [[Ambiguity (NLP)]]: từ có nhiều nghĩa, cấu trúc câu có nhiều phân tích, và mục đích giao tiếp có thể khác với nghĩa đen (“Can you tell me the time?”). Vì vậy, hệ NLP hiện đại phải dùng xác suất và tri thức để chọn diễn giải hợp lý.

NLP hiện diện trong nhiều hệ thống (dịch máy, tìm kiếm, phân loại, trợ lý, QA). Dù bề ngoài khác nhau, chúng thường chia sẻ một khuôn chung: **chuẩn hóa đầu vào → rút trích cấu trúc/đặc trưng → suy luận/ra quyết định → (tuỳ bài toán) tạo đầu ra**. Khuôn này thường được đóng gói dưới dạng [[NLP Pipeline]].

