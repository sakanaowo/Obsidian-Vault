---
name: deep-learning-notes
description: Tạo ghi chú học tập sâu cho môn Deep Learning (D2L). Tự động trigger khi tạo note trong 10_Projects/D2L, 20_Areas, 30_Resources, hoặc bất kỳ file nào có tag #d2l, #deep-learning, #machine-learning. Khi phát hiện người dùng đang "nhồi nhét" một concept mà không hiểu bản chất (ví dụ: copy công thức ignore_index mà không biết nó là gì, dùng kỹ thuật mà không hiểu tại sao), agent phải dừng lại và đào sâu. Đọc và tuân thủ AGENTS.md cùng các skill files trong thư mục này.
---

# Deep Learning Notes — Hướng Dẫn Tạo Ghi Chú Học Tập Sâu

## Quick Reference

Luôn đọc các file hướng dẫn bổ sung trước khi viết nội dung:

- [PEDAGOGY.md](PEDAGOGY.md) — Quy tắc chống nhồi nhét, cách khoan凿 khái niệm chưa hiểu
- [NOTE_CONVENTIONS.md](NOTE_CONVENTIONS.md) — Cấu trúc folder, frontmatter schema, template

## Tổng Quan

Skill này giúp tạo ghi chú học tập cho môn Deep Learning (D2L — Dive into Deep Learning) tại `10_Projects/D2L/`. Nó bổ sung và mở rộng [AGENTS.md](../AGENTS.md) với trọng tâm **chống nhồi nhét kiến thức** — đây là vấn đề cốt lõi: khi gặp một concept mới (như `ignore_index` trong Buổi 47, hoặc bất kỳ tham số/kỹ thuật nào), bạn có xu hướng copy công thức/code mà không hiểu nó tồn tại để giải quyết vấn đề gì, hoạt động ra sao, và khi nào thì dùng.

## Trigger Conditions

Tự động kích hoạt khi:

- Tạo/sửa file trong `10_Projects/D2L/`
- Tạo/sửa concept note trong `20_Areas/AI/Concepts/`
- Tạo/sửa source note trong `30_Resources/`
- Bất kỳ file nào có frontmatter tag: `#d2l`, `#deep-learning`, `#machine-learning`, `#nlp`
- Khi phát hiện người dùng sử dụng thuật ngữ kỹ thuật mà không có định nghĩa rõ ràng trong vault (concept link để trống, hoặc link chết)

## Workflow Chính

### Bước 1: Đọc Progress và Active Recall

1. Đọc `progress.md` để nắm ngữ cảnh (buổi nào, concept nào đã học).
2. Xác định các concept từ buổi trước liên quan đến nội dung mới.
3. Chuẩn bị bộ câu hỏi Active Recall cho mục `## Active Recall`.

### Bước 2: Phân Tích Nội Dung Mới

Với mỗi concept mới xuất hiện trong bài giảng:

1. **Nhận diện**: Đây là concept gì? (Thuật ngữ, tham số, kỹ thuật, công thức?)
2. **Đặt câu hỏi gốc**: Tại sao nó tồn tại? (XEM [PEDAGOGY.md](PEDAGOGY.md) — Anti-Cramming Rules)
3. **Tìm prerequisite**: Cần hiểu gì TRƯỚC khi hiểu concept này?
4. **Kiểm tra vault**: Concept này đã có note chưa? Link hay tạo mới?

> [!WARNING]- Cảnh báo: Dấu hiệu nhồi nhét
>
> Nếu bạn phát hiện mình đang:
>
> - Copy `ignore_index=-100` vào code mà không giải thích `-100` là gì, tại sao dùng, xử lý cái gì
> - Viết công thức mà không nói rõ mỗi ký hiệu đại diện cho cái gì
> - Dùng "kỹ thuật X" mà không biết nó giải quyết vấn đề cụ thể nào
>
> **DỪNG LẠI NGAY.** Đọc [PEDAGOGY.md](PEDAGOGY.md) phần **Concept Probing** trước khi tiếp tục.

### Bước 3: Viết Nội Dung

Tuân thủ cấu trúc **3 tầng** từ AGENTS.md:

1. **ELI5** (2-5 câu, ẩn dụ đời thường, không thuật ngữ chuyên ngành)
2. **Định nghĩa kỹ thuật** (WHAT/INPUT-OUTPUT/WHY — cầu nối từ ẩn dụ sang toán)
3. **Cơ chế chi tiết** (công thức, code, data flow, so sánh, edge cases)

### Bước 4: Cập Nhật Progress

Sau khi hoàn thành, cập nhật `progress.md` theo format trong AGENTS.md.

## Concept Discovery Checklist

Trước khi viết bất kỳ phần nào về concept mới, tự hỏi:

```text
□ Tôi có hiểu concept này đang GIẢI QUYẾT VẤN ĐỀ GÌ không?
□ Tôi có thể giải thích nó bằng ẩn dụ đời thường không?
□ Tôi có biết mỗi tham số trong công thức/code đại diện cho cái gì không?
□ Nếu tôi bỏ concept này đi, điều gì sẽ sai?
□ Concept này khác gì so với những concept tương tự tôi đã biết?
```

Nếu bất kỳ câu nào không trả lời được → **Đọc PEDAGOGY.md trước.**

## Frontmatter Requirements

Mọi file note buổi học D2L phải có:

```yaml
---
session: "D2L Tuần X, Buổi Y — Tên chủ đề"
tags: [d2l, deep-learning, #topic]
status: seedling|growth|evergreen
source: "D2L Chapter X.Y — Tên section"
created: YYYY-MM-DD
related:
  - "[[Buổi N - Tuần M]]"
  - "[[Concept Note Name]]"
---
```

Xem chi tiết tại [NOTE_CONVENTIONS.md](NOTE_CONVENTIONS.md).

## Quick ELI5 Rules

| Sai (nhồi nhét) | Đúng (hiểu sâu) |
| --- | --- |
| "CrossEntropyLoss có `ignore_index=-100` để bỏ qua nhãn đặc biệt" | "ignore_index=-100: token có giá trị -100 sẽ KHÔNG contribute vào loss. Tại sao cần vậy? Vì trong batch có những padding tokens không phải từ thật..." |
| "Gradient clipping giới hạn gradient để tránh exploding" | "Gradient clipping: khi gradient quá lớn (→∞), ta CHẶT nó về ngưỡng max. ELI5: như cầu an toàn cắt xe quá tải..." |
| "ReLU f(x) = max(0,x)" | "ReLU: bỏ âm, giữ dương. Tại sao? Vì âm → 0 giúp mạng sparse, không có vanishing gradient như sigmoid..." |

## Knowledge Gap Detection

Khi làm việc với vault, chú ý các dấu hiệu sau của **knowledge gaps**:

- Wikilink đến concept chưa tồn tại (link chết)
- Giải thích kiểu "đây là technique X để làm Y" mà không giải thích WHY
- Công thức có ký hiệu không được định nghĩa (ví dụ: dùng `h` mà không nói rõ `h` là hidden size)
- Code có tham số mà không comment ý nghĩa (như `ignore_index=-100` không giải thích)

Khi phát hiện → TẠO hoặc CẬP NHẬT concept note tương ứng.
