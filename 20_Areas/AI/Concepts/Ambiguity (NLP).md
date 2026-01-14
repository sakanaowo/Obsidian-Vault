---
tags:
  - nlp
  - linguistics
  - ambiguity
status: in_progress
created_date: 2026-01-13
source:
  - assets/Library/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang.pdf
---

# Ambiguity (NLP)

**Ambiguity** là hiện tượng một biểu thức ngôn ngữ (từ/cụm/câu) có thể có **nhiều cách diễn giải hợp lý**. Trong NLP, ambiguity không phải “case hiếm”, mà là trạng thái mặc định vì ngôn ngữ tự nhiên luôn dựa vào ngữ cảnh và tri thức chung để tiết kiệm công sức giao tiếp.

Ambiguity thường được phân lớp theo tầng:

- **Từ vựng (lexical)**: một từ có nhiều nghĩa (“bank”).
- **Cú pháp (syntactic)**: một câu có nhiều cách phân tích cấu trúc.
- **Ngữ nghĩa (semantic)**: nhiều cách gán vai trò và quan hệ nghĩa.
- **Diễn ngôn (discourse)**: đồng tham chiếu, liên kết câu–câu.
- **Ngữ dụng (pragmatic)**: mục đích giao tiếp khác nghĩa đen.

Ví dụ slide “The chicken is ready to eat” minh họa ambiguity ở giao điểm cú pháp–ngữ nghĩa: “chicken” có thể là tác nhân (sẵn sàng ăn) hoặc bị thể (sẵn sàng để bị ăn). Hệ NLP muốn “hiểu” phải có cơ chế **giải mơ hồ**: chọn diễn giải phù hợp nhất với ngữ cảnh và tri thức thế giới.

> [!NOTE] Suy luận thêm — Giải ambiguity là bài toán xác suất + tri thức
> Các mô hình học từ dữ liệu cung cấp phân phối “diễn giải nào hay xảy ra”, nhưng nhiều diễn giải vô lý chỉ bị loại bỏ khi có tri thức thế giới (“the blue pen ate the ice-cream”). Vì vậy, ambiguity là nơi giao nhau giữa học máy và biểu diễn tri thức.

