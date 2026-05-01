# Pedagogy Guide — Giải Thích Kĩ & Tránh Rối Não Người Đọc

## Giới Thiệu

Phần này giải quyết vấn đề cốt lõi: **agent dùng bừa bãi ngôn ngữ chuyên ngành mà không giải thích**. Ví dụ thực tế từ vault:

- Viết "attention weights" mà không giải thích đây là trọng số quyết định mức độ "chú ý" vào mỗi value
- Viết "BMM" mà không giải thích = Batch Matrix Multiplication (nhân nhiều ma trận cùng lúc)
- Viết "hidden states" mà không giải thích = vector biểu diễn của một từ/sentence sau khi qua encoder
- Viết "encoder-decoder attention" mà không giải thích query đến từ đâu, key/value đến từ đâu
- Vietlish không cần thiết: "ta sẽ compute cái này" thay vì "ta sẽ tính giá trị này"

Skill này bắt buộc **giải thích kĩ** mỗi khi dùng thuật ngữ mới.

---

## Giải Thích Kĩ Trước Khi Dùng (5 Quy Tắc Vàng)

### Rule 1: Kiểm Tra Vault Trước

**LUÔN kiểm tra vault trước khi định nghĩa concept mới:**

```text
TRƯỚC KHI VIẾT: Hỏi 3 câu
  1. Concept này đã có note trong vault chưa?
  2. Nếu có → LINK tới nó thay vì định nghĩa lại
  3. Nếu chưa → tạo stub với định nghĩa đầy đủ
```

**Ví dụ sai:**
> Bahdanau Attention sử dụng encoder-decoder attention với hidden states.

**Ví dụ đúng:**
> Bahdanau Attention (xem [[Buổi 53 - Tuần 14]]) sử dụng encoder-decoder attention. Trong đó:
> - **Query**: hidden state hiện tại của decoder (vector biểu diễn vị trí đang generate)
> - **Key/Value**: encoder outputs (vector biểu diễn từng vị trí trong câu nguồn)

### Rule 2: Mỗi Thuật Ngữ Phải Có Định Nghĩa

**Bảng thuật ngữ bắt buộc** khi trình bày nội dung:

| Thuật ngữ | Tiếng Việt | Định nghĩa ngắn gọn |
|-----------|------------|----------------------|
| Attention weights | Trọng số chú ý | Trọng số $\alpha_i$ quyết định mức độ "chú ý" vào mỗi value |
| Hidden state | Trạng thái ẩn | Vector biểu diễn của một từ/sentence sau khi qua mạng |
| Encoder | Bộ mã hóa | Phần mạng xử lý input, sinh ra biểu diễn trung gian |
| Decoder | Bộ giải mã | Phần mạng sinh ra output từng bước |
| Query | Truy vấn | Vector biểu diễn "câu hỏi" — muốn tìm thông tin gì |
| Key | Khóa | Vector biểu diễn "định danh" — mỗi vị trí có một key |
| Value | Giá trị | Vector biểu diễn "nội dung" — thông tin thực sự cần lấy |

**Sai:** Viết thuật ngữ rồi bỏ qua, người đọc tự hiểu.
**Đúng:** Mỗi thuật ngữ phải có định nghĩa bằng tiếng Việt, hoặc link tới concept note.

### Rule 3: ELI5 Trước, Toán Sau

Luôn bắt đầu bằng ẩn dụ đời thường **TRƯỚC KHI** viết bất kỳ công thức nào. Người đọc phải hiểu "cái gì" và "tại sao" trước khi thấy "như thế nào".

Xem [NOTE_CONVENTIONS.md](NOTE_CONVENTIONS.md) phần "Concept Introduction Template" để biết template chi tiết.

### Rule 4: So Sánh Với Cái Đã Biết

Mỗi concept mới phải được so sánh với ít nhất 1 concept đã học:

```text
## So Sánh

| Khía cạnh | [Concept cũ] | [Concept mới] |
|-----------|--------------|---------------|
| Query đến từ đâu | Decoder | Decoder |
| Key/Value đến từ đâu | Encoder | Encoder |
| Mục đích | Gen câu mới | Align input-output |
```

### Rule 5: Giải Thích Rõ Ràng Từng Bước Trong Code

Mỗi dòng code phải có comment giải thích:

```python
# Sai: không comment
outputs = torch.bmm(attn_weights, values)

# Đúng: có comment
# BMM: nhân trọng số chú ý (attn_weights) với values
# attn_weights: (batch, n_queries, m_keys)
# values: (batch, m_keys, v)
# outputs: (batch, n_queries, v)
outputs = torch.bmm(attn_weights, values)
```

---

## Concept Probing (Khoan Đào Khái Niệm)

Khi phát hiện một concept chưa rõ ràng, sử dụng protocol sau:

### Bước 1: Nhận Diện — Đây Là Loại Concept Gì?

| Loại | Câu hỏi cần trả lời | Ví dụ |
| --- | --- | --- |
| **Thuật ngữ** | Đây là gì? Tiếng Việt là gì? | "attention weights" = trọng số chú ý |
| **Cơ chế** | Nó làm gì với input? Tại sao cần làm vậy? | encoder-decoder attention = align source-target |
| **Kiến trúc** | Nó được ghép nối thế nào? Input/output là gì? | Seq2Seq với attention |
| **Thuật ngữ vị trí** | Query/Key/Value đến từ đâu trong mạng? | Decoder/Encoder |

### Bước 2: Tìm Nguồn Gốc — Vấn Đề Nào Sinh Ra Concept Này?

Hỏi: **"Concept này được tạo ra để giải quyết vấn đề gì?"**

**Ví dụ — Encoder-Decoder Attention:**

- Vấn đề gốc: Seq2Seq cũ (Buổi ...) chỉ dùng last encoder hidden state → bottleneck thông tin khi câu nguồn dài.
- Giải pháp: Cho decoder "nhìn" vào tất cả encoder hidden states, nhưng tập trung vào những phần liên quan (attention).
- Query đến từ decoder, Key/Value đến từ encoder.

### Bước 3: Kiểm Tra Hiểu — 5 Câu Hỏi Cuối Cùng

Trước khi kết luận đã hiểu, trả lời đủ 5 câu:

1. **Giải thích bằng ẩn dụ đời thường**: "Encoder-decoder attention giống như... [ẩn dụ]"
2. **Định nghĩa bằng tiếng Việt**: "Đây là cơ chế cho phép [X] nhìn vào [Y] để [Z]"
3. **Ai là Query, Key, Value**: "Query = [đến từ đâu], Key/Value = [đến từ đâu]"
4. **Vấn đề nó giải quyết**: "Nó tồn tại vì [vấn đề A], thay thế cách cũ [B]"
5. **So sánh với concept đã biết**: "Khác self-attention (Buổi 55) ở chỗ..."

Nếu câu nào không trả lời được → tìm hiểu thêm trước khi viết.

---

## Bloom's Taxonomy Checklist

Sau khi hoàn thành note, tự kiểm tra mức độ hiểu theo taxonomy:

- **[Nhớ]** Có định nghĩa? Có công thức?
- **[Hiểu]** Có ELI5? Có giải thích bằng lời?
- **[Áp dụng]** Có ví dụ code hoạt động?
- **[Phân tích]** Có so sánh với cái khác? Có giải thích rõ Query/Key/Value?
- **[Đánh giá]** Có nêu trade-off? Có đưa ra lựa chọn khi nào dùng cái nào?
- **[Sáng tạo]** Có suy luận mở rộng? Có ứng dụng ngoài bài học?

Ít nhất phải đạt **mức 4 (Phân tích)** cho mọi concept mới.

---

## Reader Checklist (Sau Mỗi Phần Giải Thích)

Thêm vào cuối mỗi section quan trọng:

```text
> [!CHECKLIST]-
> Reader tự kiểm tra:
>
> - [ ] Tôi có thể giải thích concept này cho một người bạn không?
> - [ ] Tôi biết Query đến từ đâu, Key/Value đến từ đâu?
> - [ ] Tôi hiểu tại sao cần dùng nó, không phải chỉ "làm vậy thì chạy" chứ?
> - [ ] Tôi phân biệt được concept này với concept đã biết?
```

---

## Common Pitfalls — Những Lỗi Thường Gặp

### Lỗi 1: Không giải thích thuật ngữ

**Sai:**
> Bahdanau Attention sử dụng additive attention để compute alignment scores.

**Đúng:**
> Bahdanau Attention (xem [[Buổi 53]]) sử dụng additive attention — một cách tính điểm tương đồng (alignment score) giữa query và key bằng MLP thay vì dot product.

### Lỗi 2: Không chỉ rõ Query/Key/Value đến từ đâu

**Sai:**
> Encoder-decoder attention cho phép decoder nhìn vào encoder.

**Đúng:**
> Encoder-decoder attention:
> - **Query**: hidden state hiện tại của decoder (vector biểu diễn vị trí đang sinh)
> - **Key/Value**: encoder outputs (vector biểu diễn từng vị trí trong câu nguồn)
> - **Output**: weighted sum của encoder outputs — thông tin "liên quan" từ câu nguồn

### Lỗi 3: Vietlish không cần thiết

**Sai:**
> Ta sẽ compute attention scores rồi softmax để normalize.

**Đúng:**
> Ta sẽ tính các điểm tương đồng (attention scores) giữa query và mỗi key, rồi dùng softmax để normalize thành trọng số chú ý (attention weights).

### Lỗi 4: Wikilink tới concept chưa tồn tại

**Sai:**
> Xem [[Attention Mechanism]] để biết thêm.

**Đúng:**
> Nếu concept note chưa tồn tại → tạo stub ngay lập tức với:
> - Định nghĩa ngắn gọn
> - TODO để phát triển sau
> - Link từ note hiện tại
