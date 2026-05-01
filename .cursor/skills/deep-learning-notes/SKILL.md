---
name: deep-learning-notes
description: Tạo ghi chú học tập sâu cho môn Deep Learning (D2L). Tự động trigger khi tạo/sửa note trong 10_Projects/D2L, 20_Areas/AI/Concepts/, 30_Resources/, hoặc bất kỳ file nào có tag #d2l, #deep-learning, #machine-learning, #nlp. Luôn kiểm tra vault để reference tới concept notes có sẵn thay vì tự định nghĩa lại. Giải thích kĩ từng concept, tránh dùng bừa bãi ngôn ngữ chuyên ngành mà không định nghĩa — nếu dùng thuật ngữ mà người đọc chưa biết, họ sẽ bị rối não. Đọc và tuân thủ AGENTS.md cùng các skill files trong thư mục này.
---

# Deep Learning Notes — Hướng Dẫn Tạo Ghi Chú Học Tập Sâu

## Quick Reference

Luôn đọc các file hướng dẫn bổ sung trước khi viết nội dung:

- [PEDAGOGY.md](PEDAGOGY.md) — Quy tắc giải thích kĩ, kiểm tra concept trước khi reference, tránh ngôn ngữ chuyên ngành gây rối
- [NOTE_CONVENTIONS.md](NOTE_CONVENTIONS.md) — Cấu trúc folder, frontmatter schema, template

## Tổng Quan

Skill này giúp tạo ghi chú học tập cho môn Deep Learning (D2L — Dive into Deep Learning) tại `10_Projects/D2L/`. Nó bổ sung và mở rộng [AGENTS.md](../AGENTS.md) với trọng tâm **giải thích kĩ thay vì nhồi nhét** — đây là vấn đề cốt lõi: khi gặp một concept mới (như `ignore_index` trong Buổi 47, hoặc bất kỳ tham số/kỹ thuật nào), agent có xu hướng **dùng bừa bãi ngôn ngữ chuyên ngành** mà không giải thích, **không kiểm tra vault** xem concept đó đã có note chưa, và **mặc định người đọc tự biết** — dẫn đến:

- Vietlish (tiếng Việt + English lẫn lộn) không cần thiết
- Thuật ngữ chuyên ngành xuất hiện mà không định nghĩa
- Wikilink để trống hoặc chỉ tạo stub mà không giải thích
- Không reference tới concept notes có sẵn trong vault

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

### Bước 2: Kiểm Tra Vault Trước Khi Viết

**QUAN TRỌNG:** Trước khi định nghĩa bất kỳ concept nào, LUÔN kiểm tra:

1. **Concept này đã có note chưa?** → Nếu có, LINK tới nó thay vì định nghĩa lại
2. **Thuật ngữ nào cần giải thích?** → Nếu dùng thuật ngữ mà chưa định nghĩa trong vault, phải giải thích NGAY
3. **Có wikilink chết không?** → Tạo stub hoặc reference đúng

> [!WARNING]- Cảnh báo: Dùng bừa bãi ngôn ngữ chuyên ngành
>
> Nếu bạn phát hiện mình đang:
>
> - Viết "attention weights" mà không giải thích nó là gì (trọng số chú ý = trọng số quyết định mức độ "chú ý" vào mỗi value)
> - Viết "BMM" mà không giải thích = Batch Matrix Multiplication (nhân nhiều ma trận cùng lúc)
> - Viết "QKV" mà không reference tới Buổi 50
> - Viết "hidden states" mà không giải thích = vector biểu diễn của một từ/sentence sau khi qua encoder
> - Dùng Vietlish không cần thiết: "ta sẽ compute cái này" thay vì "ta sẽ tính giá trị này"
>
> **DỪNG LẠI NGAY.** Đọc [PEDAGOGY.md](PEDAGOGY.md) phần **Giải Thích Kĩ Trước Khi Dùng** trước khi tiếp tục.

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
