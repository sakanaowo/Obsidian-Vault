# Pedagogy Guide — Chống Nhồi Nhét & Đào Sâu Kiến Thức

## Giới Thiệu

Phần này giải quyết vấn đề cốt lõi: **bạn có xu hướng nhồi nhét kiến thức** — copy công thức, tham số, code mà không hiểu bản chất. Ví dụ thực tế từ vault:

- Buổi 47: `ignore_index=-100` xuất hiện trong CrossEntropyLoss mà không hiểu `-100` là gì, tại sao chọn `-100` (không phải `-1` hay `0`), và nó xử lý cái gì.
- Đã dùng các kỹ thuật NLP mà không hiểu tại sao cần chúng, chỉ biết "làm vậy thì chạy được".

Skill này bắt buộc **dừng lại và đào sâu** mỗi khi phát hiện pattern nhồi nhét.

---

## Anti-Cramming Rules (5 Quy Tắc Vàng)

### Rule 1: Không Bao Giờ Copy Mà Không Hỏi "Tại Sao"

Mỗi khi gặp một tham số, công thức, hoặc kỹ thuật mới:

```text
TRƯỚC KHI VIẾT: Hỏi 3 câu
  1. Đây giải quyết vấn đề GÌ?
  2. Nếu bỏ đi, điều gì sẽ SAI?
  3. Tại sao lại là THAM SỐ NÀY (giá trị cụ thể)?
```

**Ví dụ sai:**
> `CrossEntropyLoss(ignore_index=-100)` → copy thẳng vào code, không giải thích.

**Ví dụ đúng:**
> `CrossEntropyLoss(ignore_index=-100)`: Khi token có giá trị -100 xuất hiện trong chuỗi (thường là padding), nó **không đóng góp vào loss**. Tại sao -100? Vì giá trị này nằm ngoài range nhãn hợp lệ [0, num_classes-1], nên PyTorch có thể phân biệt được "nhãn thật" và "padding". Nếu không có `ignore_index`, padding tokens sẽ kéo loss về hướng sai (vì chúng không phải từ thật), và gradient từ chúng sẽ làm hỏng weight updates.

### Rule 2: Mỗi Công Thức Phải Có Từ Điển Ký Hiệu

**Template bắt buộc** khi trình bày công thức:

```text
## Công thức

$$
\mathcal{L} = -\sum_{t=1}^{T} \log \frac{\exp(x_{t,y_t})}{\sum_{c} \exp(x_{t,c})}
$$

**Từ điển ký hiệu:**
- $x_{t,y_t}$: ...
- $T$: ...
- $c$: ...
- $\mathcal{L}$: ...
```

**Sai:** Viết công thức rồi bỏ qua, người đọc tự hiểu.
**Đúng:** Giải thích từng ký hiệu, không ký hiệu nào được bỏ qua.

### Rule 3: ELI5 Trước, Toán Sau

Luôn bắt đầu bằng ẩn dụ đời thường **TRƯỚC KHI** viết bất kỳ công thức nào. Người đọc phải hiểu "cái gì" và "tại sao" trước khi thấy "như thế nào".

Xem [NOTE_CONVENTIONS.md](NOTE_CONVENTIONS.md) phần "Concept Introduction Template" để biết template chi tiết.

### Rule 4: So Sánh Với Cái Đã Biết

Mỗi concept mới phải được so sánh với ít nhất 1 concept đã học:

```text
## So Sánh

| Khía cạnh | [Concept cũ] | [Concept mới] |
|---|---|---|
| Vấn đề giải quyết | ... | ... |
| Cơ chế | ... | ... |
| Khi nào dùng | ... | ... |
```

### Rule 5: Edge Cases và Failure Modes

Mỗi kỹ thuật phải nêu rõ:

- Khi nào nó **không hoạt động**?
- Điều gì xảy ra nếu **bỏ nó đi**?
- Có **trade-off** gì không?

---

## Concept Probing (Khoan Đào Khái Niệm)

Khi phát hiện một concept chưa hiểu rõ, sử dụng protocol sau:

### Bước 1: Nhận Diện — Đây Là Loại Concept Gì?

| Loại | Câu hỏi cần trả lời | Ví dụ |
| --- | --- | --- |
| **Tham số/Hyperparameter** | Nó điều khiển cái gì? Giá trị mặc định có hợp lý không? | `ignore_index`, `dropout`, `learning_rate` |
| **Cơ chế/Thuật toán** | Nó làm gì với input? Tại sao cần làm vậy? | attention, gradient clipping, padding |
| **Kiến trúc/Module** | Nó được ghép nối thế nào? Input/output shape gì? | LSTM cell, embedding layer, linear layer |
| **Mất mát/Loss** | Nó đo lường cái gì? Tại sao dùng thay vì cái khác? | CrossEntropyLoss, BCE |
| **Tập dữ liệu/Data** | Dữ liệu được tổ chức thế nào? Preprocessing gì? | vocabulary, tokenization, padding |

### Bước 2: Tìm Nguồn Gốc — Vấn Đề Nào Sinh Ra Concept Này?

Hỏi: **"Concept này được tạo ra để giải quyết vấn đề gì?"**

- Tìm paper gốc hoặc bài blog giải thích
- Tìm commit history / release notes (với code library)
- Hỏi: "Trước khi có concept này, người ta xử lý vấn đề này thế nào?"

**Ví dụ — `ignore_index`:**

- Vấn đề gốc: CrossEntropyLoss nhận mọi token trong batch, bao gồm padding. Padding không phải từ thật → loss bị noise.
- Giải pháp cũ (thủ công): Mask array, nhân loss với mask.
- Giải pháp hiện tại: `ignore_index` — PyTorch tự bỏ qua.
- Tại sao -100? Convention từ thư viện. Có thể dùng bất kỳ giá trị ngoài range nhãn.

### Bước 3: Kiểm Tra Hiểu — 5 Câu Hỏi Cuối Cùng

Trước khi kết luận đã hiểu, trả lời đủ 5 câu:

1. **Giải thích bằng ẩn dụ đời thường**: "ignore_index giống như... [ẩn dụ]"
2. **Định nghĩa kỹ thuật**: "Nó làm [X] với [input], trả về [output]"
3. **Mục đích tồn tại**: "Nó tồn tại vì [vấn đề A], thay thế cách cũ [B]"
4. **Tham số có ý nghĩa gì**: "Giá trị -100 có nghĩa là [Y], có thể thay bằng [Z] nếu..."
5. **Edge case**: "Nếu tất cả tokens trong batch đều bị ignore, loss sẽ..."

Nếu câu nào không trả lời được → tìm hiểu thêm trước khi viết.

---

## Bloom's Taxonomy Checklist

Sau khi hoàn thành note, tự kiểm tra mức độ hiểu theo taxonomy:

- **[Nhớ]** Có định nghĩa? Có công thức?
- **[Hiểu]** Có ELI5? Có giải thích bằng lời?
- **[Áp dụng]** Có ví dụ code hoạt động?
- **[Phân tích]** Có so sánh với cái khác? Có phân tích failure modes?
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
> - [ ] Tôi biết mỗi tham số/giá trị đại diện cho cái gì chưa?
> - [ ] Tôi hiểu tại sao cần dùng nó, không phải chỉ "làm vậy thì chạy" chứ?
```
