---
session: "D2L Tuần 14, Buổi 53 — 11.4 Bahdanau Attention"
aliases: ["Buổi 53"]
tags: [d2l, deep-learning, attention, bahdanau-attention, encoder-decoder, seq2seq, nlp]
status: growth
source: "D2L Chapter 11.4 — Bahdanau Attention"
created: 2026-05-01
related:
  - "[[Buổi 52 - Tuần 14]]"
  - "[[Buổi 54 - Tuần 14]]"
  - "[[Buổi 50 - Tuần 14]]"
---

# Buổi 53 — 11.4 Bahdanau Attention

> [!NOTE] Mục tiêu buổi học
>
> - [ ] Hiểu encoder-decoder attention: Query đến từ decoder, Key/Value đến từ encoder
> - [ ] Nắm rõ vấn đề mà Bahdanau attention giải quyết (bottleneck từ last hidden state)
> - [ ] Hiểu cơ chế: mỗi bước decode nhìn vào toàn bộ encoder outputs
> - [ ] Phân biệt được encoder-decoder attention vs self-attention (Buổi 55)
> - [ ] Implement Bahdanau attention từ scratch

---

## Bảng thuật ngữ — ĐỌC TRƯỚC KHI TIẾP TỤC

| Thuật ngữ | Tiếng Việt | Giải thích bằng tiếng Việt |
|-----------|------------|------------------------------|
| Encoder | Bộ mã hóa | Phần mạng xử lý câu nguồn, sinh ra các vector biểu diễn |
| Decoder | Bộ giải mã | Phần mạng sinh ra câu đích từng từ một |
| Hidden state | Trạng thái ẩn | Vector số cho biết "đang ở đâu" trong quá trình xử lý |
| Encoder hidden state $\mathbf{h}_i$ | Vector biểu diễn từ nguồn | Vector số biểu diễn từ thứ $i$ trong câu nguồn, sau khi qua encoder |
| Decoder hidden state $\mathbf{s}_t$ | Vector biểu diễn vị trí đang sinh | Vector số biểu diễn "bộ nhớ" của decoder tại bước đang sinh từ $t$ |
| Context vector $\mathbf{c}_t$ | Vector ngữ cảnh | Vector chứa thông tin "liên quan" từ câu nguồn, được tính từ attention |
| Attention weights $\alpha_{ti}$ | Trọng số chú ý | Số cho biết decoder đang "chú ý" vào từ nguồn thứ $i$ bao nhiêu (tại bước $t$) |
| Query | Truy vấn | "Câu hỏi" — trong Bahdanau là decoder hidden state, hỏi "từ nào liên quan?" |
| Key | Khóa | "Định danh" — trong Bahdanau là encoder hidden states, cho biết "tôi ở vị trí nào" |
| Value | Giá trị | "Nội dung" — trong Bahdanau là encoder hidden states, chứa thông tin thực sự |
| Bottleneck | Nút thắt cổ chai | Vấn đề khi thông tin bị nén vào quá ít thông số |
| Alignment | Sự sắp xếp | Việc match từ nguồn với từ đích (ví dụ: "cat" align với "mèo") |
| Additive scoring | Tính điểm bằng cộng | Cách tính attention score dùng mạng neural (MLP) thay vì nhân ma trận |

---

## Active Recall — Ôn lại Buổi 52

### Câu hỏi truy hồi (không nhìn tài liệu)

1. Scaled dot product attention formula là gì? Kể tên từng thành phần (Query, Key, Value).
2. Tại sao cần chia $\sqrt{d}$ trong attention? Cho ví dụ với $d=64$.
3. Masked softmax xử lý vấn đề gì? Giá trị -1e6 có ý nghĩa gì?
4. BMM (Batch Matrix Multiplication) là gì? Tại sao dùng nó thay vì loop?
5. Additive attention khác dot product attention ở điểm nào?

### Trả lời chi tiết

---

**Câu 1: Scaled dot product attention formula là gì? Kể tên từng thành phần (Query, Key, Value).**

**Công thức:**
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}\right)\mathbf{V}$$

**Giải thích từng thành phần:**

| Thành phần | Trong attention là gì? | Vai trò |
|-------------|----------------------|---------|
| **Query ($\mathbf{Q}$)** | "Câu hỏi" muốn tìm thông tin | Vector cho biết "tôi đang tìm gì" |
| **Key ($\mathbf{K}$)** | "Định danh" của mỗi vị trí | Vector cho biết "tôi ở vị trí nào, mang thông tin gì" |
| **Value ($\mathbf{V}$)** | "Nội dung" thực sự cần lấy | Vector chứa thông tin thực sự |

**Cách hoạt động từng bước:**
1. Query nhân với Key (sau khi transpose) → ra ma trận điểm số
2. Chia cho $\sqrt{d}$ → "scale" để kiểm soát độ lớn
3. Softmax → điểm số thành tỷ lệ (0-1, tổng = 1)
4. Nhân với Value → lấy thông tin theo trọng số

---

**Câu 2: Tại sao cần chia $\sqrt{d}$ trong attention? Cho ví dụ với $d=64$.**

**Vấn đề:**
- Khi tính dot product $\mathbf{q}^\top \mathbf{k}$ với vectors chiều $d$
- Mỗi phần tử có mean = 0, variance = 1 (standard initialization)
- Kết quả dot product có **variance = $d$**

**Ví dụ với $d = 64$:**
- Giả sử $\mathbf{q}, \mathbf{k}$ có 64 phần tử, mỗi phần tử ~ N(0,1)
- Dot product = tổng 64 số ngẫu nhiên
- Variance = 64 → độ lệch chuẩn = 8
- Giá trị dot product có thể lên tới ±8 hoặc ±16

**Tại sao đây là vấn đề?**
- Softmax của các số lớn và chênh lệch sẽ "bão hòa"
- Ví dụ: scores = [100, 105, 102] → softmax → [~0, ~1, ~0]
- Một token chiếm ~100%, các token khác gần như bị lãng quên
- Gradient biến mất → model không học được

**Giải pháp: Chia cho $\sqrt{d}$**
- $\text{Var}\left(\frac{\mathbf{q}^\top \mathbf{k}}{\sqrt{d}}\right) = \frac{d}{d} = 1$
- Scores có variance = 1 → độ lệch chuẩn = 1
- Giá trị hợp lý hơn, softmax hoạt động cân bằng
- Gradient flow tốt hơn

---

**Câu 3: Masked softmax xử lý vấn đề gì? Giá trị -1e6 có ý nghĩa gì?**

**Vấn đề được xử lý:**
- Trong thực tế, câu có độ dài khác nhau
- Câu ngắn được "đệm" thêm tokens rỗng (padding) để cùng độ dài
- Padding tokens không mang thông tin, không nên ảnh hưởng đến kết quả

**Cách masked softmax hoạt động:**

```
Ví dụ: Batch gồm 2 câu
- Câu 1: "Hello world" (2 tokens thực, 4 padding)
- Câu 2: "The cat sat" (3 tokens thực, 3 padding)

Attention scores sau khi tính:
Câu 1: [s₁, s₂, -1e6, -1e6, -1e6, -1e6]
Câu 2: [s₁, s₂, s₃, -1e6, -1e6, -1e6]

Sau softmax:
Câu 1: [a₁, a₂, ~0, ~0, ~0, ~0]  ← padding không ảnh hưởng
Câu 2: [b₁, b₂, b₃, ~0, ~0, ~0]  ← padding không ảnh hưởng
```

**Tại sao dùng -1e6 thay vì -infinity?**

| Giá trị | exp(x) | softmax | Vấn đề |
|---------|--------|---------|---------|
| $-\infty$ | 0 | 0 | Gây NaN trong một số implementation |
| -1e6 | ~0 | ~0 | **An toàn** — xấp xỉ 0 với floating point |
| -100 | exp(-100) ≈ 0 | ~0 | Được, nhưng -1e6 "chắc chắn" hơn |

-1e6 là "safe approximation" cho -∞ trong floating point arithmetic.

---

**Câu 4: BMM (Batch Matrix Multiplication) là gì? Tại sao dùng nó thay vì loop?**

**Định nghĩa:**
- BMM = Batch Matrix Multiplication
- Nhân nhiều ma trận cùng lúc trong một batch
- Cú pháp: `torch.bmm(A, B)`

**Ví dụ:**

```python
# Có 3 cặp ma trận cần nhân
A = torch.randn(3, 2, 4)  # 3 ma trận 2x4
B = torch.randn(3, 4, 6)  # 3 ma trận 4x6

# Cách 1: Dùng loop (chậm)
results = []
for i in range(3):
    results.append(A[i] @ B[i])  # A[i]: 2x4, B[i]: 4x6 → 2x6

# Cách 2: Dùng BMM (nhanh)
results = torch.bmm(A, B)  # Shape: (3, 2, 6)
```

**Tại sao BMM nhanh hơn loop?**

| Loop | BMM |
|------|-----|
| 3 lần nhân ma trận riêng lẻ | 1 lần nhân batch |
| Python loop chậm | GPU parallelizes tất cả cùng lúc |
| Memory không liên tục | Memory access tối ưu |

GPU được thiết kế để làm việc với batches — BMM tận dụng được điều này.

---

**Câu 5: Additive attention khác dot product attention ở điểm nào?**

**So sánh công thức:**

| Loại | Công thức scoring |
|------|-------------------|
| **Dot Product** | $a(\mathbf{q}, \mathbf{k}) = \mathbf{q}^\top \mathbf{k}$ |
| **Additive** | $a(\mathbf{q}, \mathbf{k}) = \mathbf{v}^\top \tanh(\mathbf{W}_q\mathbf{q} + \mathbf{W}_k\mathbf{k})$ |

**Điểm khác biệt chi tiết:**

| Khía cạnh | Dot Product | Additive |
|-----------|-------------|----------|
| **Yêu cầu về chiều** | Query và Key phải cùng chiều ($q = k$) | Có thể khác chiều ($q \neq k$) |
| **Số tham số** | 0 (không có) | $W_q \in \mathbb{R}^{h \times q}$, $W_k \in \mathbb{R}^{h \times k}$, $\mathbf{v} \in \mathbb{R}^h$ |
| **Cơ chế** | Nhân ma trận trực tiếp | MLP để tính điểm |
| **Non-linearity** | Không có | Có (tanh) |
| **Tốc độ** | Nhanh | Chậm hơn |
| **Dùng trong** | Transformer (2017) | Bahdanau NMT (2014) |

**Khi nào dùng cái nào?**

- **Dot product**: Khi query và key cùng dimension (như trong Transformer)
- **Additive**: Khi query và key khác dimension (như trong Bahdanau, cross-modal attention)

---

### Liên kết cần ôn lại

- [[Buổi 52 - Tuần 14|Attention Scoring Functions]]
- [[Buổi 50 - Tuần 14|QKV — Queries, Keys, and Values]]

---

# PHẦN I — TỪ SEQ2SEQ CỔ ĐIỂN ĐẾN ATTENTION

## 1.1 Vấn đề: Thông tin bị nén quá nhiều vào một chỗ

### Bảng thuật ngữ phần này

| Thuật ngữ | Giải thích |
|-----------|------------|
| Seq2Seq | Kiến trúc gồm encoder (đọc input) và decoder (sinh output) |
| Context vector $\mathbf{c}$ | Một vector số duy nhất chứa toàn bộ thông tin câu nguồn |
| Last hidden state | Trạng thái ẩn cuối cùng của encoder — thường dùng làm context vector |

### Giải thích bằng ví dụ cụ thể

**Seq2Seq cổ điển hoạt động như thế này:**

```
Câu nguồn: "The cat sat on the mat"
Câu đích muốn dịch: "Con mèo ngồi trên thảm"

Bước 1: ENCODER đọc từng từ
  "The" → encoder → vector số h₁
  "cat" → encoder → vector số h₂
  "sat" → encoder → vector số h₃
  "on"  → encoder → vector số h₄
  "the" → encoder → vector số h₅
  "mat" → encoder → vector số h₆

Bước 2: NÉN thông tin vào MỘT vector duy nhất
  Lấy vector cuối cùng h₆ (gọi là "context vector" c)
  Vector này phải chứa: chủ ngữ="cat", vị ngữ="sat",
  bổ ngữ="mat", quan hệ "on" kết nối sat với mat...

Bước 3: DECODER "GIẢI NÉN" vector đó thành câu đích
  c → decoder → "Con"
  "Con" + c → decoder → "mèo"
  "mèo" + c → decoder → "ngồi"
  ...tiếp tục cho đến hết câu
```

### Tại sao đây là vấn đề?

**Ví dụ đời thường:**

Bạn đọc một cuốn tiểu thuyết 500 trang, sau đó phải tóm tắt nó vào **một câu duy nhất**. Bạn sẽ phải bỏ lỡ hầu hết chi tiết!

Seq2Seq cổ điển gặp vấn đề tương tự:
- **Thông tin từ 50 từ** phải nén vào **một vector** (ví dụ: 256 số)
- Vector có kích thước cố định, không thể chứa đủ mọi thứ
- Khi câu nguồn dài, model bắt đầu "quên" thông tin

**Hệ quả cụ thể:**
- Từ ở đầu câu (đã qua encoder trước) bị "quên" nhiều nhất
- Các từ xa nhau trong câu khó liên kết được với nhau
- Dịch câu dài → chất lượng kém

> [!KEY]- Ghi nhớ
>
> Seq2Seq cổ điển = encoder nén tất cả → một vector → decoder giải nén
> Vấn đề = một vector không đủ chứa mọi thứ

---

## 1.2 Giải pháp: Decoder được "nhìn" vào toàn bộ câu nguồn

### Bảng thuật ngữ phần này

| Thuật ngữ | Giải thích |
|-----------|------------|
| Encoder-decoder attention | Cơ chế cho phép decoder "nhìn" vào tất cả encoder outputs |
| Cross-attention | Tên gọi khác của encoder-decoder attention |
| Attention scores | Điểm số cho biết mức độ "liên quan" giữa query và key |

### Giải thích bằng ví dụ cụ thể

**Thay vì chỉ một vector, encoder-decoder attention làm như sau:**

```
Câu nguồn: "The cat sat on the mat"
Câu đích muốn dịch: "Con mèo ngồi trên thảm"

Bước 1: ENCODER đọc từng từ (giống như trước)
  "The" → h₁
  "cat" → h₂
  "sat" → h₃
  "on"  → h₄
  "the" → h₅
  "mat" → h₆

Bước 2: DECODER NHÌN VÀO TẤT CẢ (điểm khác biệt!)
  Khi đang sinh từ "ngồi":
  - Decoder hỏi: "Từ nào trong câu nguồn liên quan đến hành động?"
  - Trả lời: h₃ ("sat") có liên quan cao nhất
  - Kết quả: Decoder lấy thông tin từ h₃ ("sat") nhiều hơn

  Khi đang sinh từ "mèo":
  - Decoder hỏi: "Từ nào trong câu nguồn là chủ ngữ?"
  - Trả lời: h₂ ("cat") có liên quan cao nhất
  - Kết quả: Decoder lấy thông tin từ h₂ ("cat") nhiều hơn

Bước 3: DECODER SINH TỪNG TỪ
  Mỗi bước có vector ngữ cảnh RIÊNG, không dùng chung một vector
```

**Cơ chế "hỏi-đáp" trong attention:**

1. **Decoder hỏi** (tạo "câu hỏi" từ hidden state hiện tại)
2. **Encoder trả lời** (cho điểm mỗi vị trí)
3. **Tính trọng số** (softmax để điểm thành 0-1, tổng = 1)
4. **Lấy thông tin** (cộng có trọng số các vector)

> [!KEY]- Ghi nhớ
>
> Encoder-decoder attention = decoder được "hỏi" về từng vị trí trong câu nguồn
> Mỗi bước decode có vector ngữ cảnh RIÊNG, không dùng chung

---

# PHẦN II — BAHDAU ATTENTION: CƠ CHẾ CHI TIẾT

## 2.1 Ai là Query, Key, Value trong Bahdanau?

### Bảng thuật ngữ phần này

| Thuật ngữ | Ai là | Giải thích |
|-----------|--------|------------|
| **Query** | Decoder hidden state $\mathbf{s}_{t-1}$ | "Câu hỏi" của decoder — "Từ nào liên quan?" |
| **Key** | Encoder hidden states $\mathbf{h}_i$ | "Định danh" của mỗi vị trí nguồn — "Tôi ở vị trí nào" |
| **Value** | Encoder hidden states $\mathbf{h}_i$ | "Nội dung" của mỗi vị trí nguồn — "Tôi mang thông tin gì" |

### Giải thích trực quan

```
┌─────────────────────────────────────────────────────────────┐
│                    ENCODER-DECODER ATTENTION                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ENCODER (đã xử lý xong câu nguồn)                        │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐         │
│  │ h₁  │ │ h₂  │ │ h₃  │ │ h₄  │ │ h₅  │ │ h₆  │         │
│  │The  │ │cat  │ │ sat │ │ on  │ │the  │ │ mat │         │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘         │
│     │        │        │        │        │        │           │
│     └────────┴────────┼────────┴────────┴────────┘           │
│                      ↓                                      │
│              ┌────────────────┐                             │
│              │ Key = h₁..h₆   │  ← Encoder outputs         │
│              │ Value = h₁..h₆ │    làm Key và Value          │
│              └────────────────┘                             │
│                                                             │
│  DECODER (đang sinh từ thứ t)                              │
│  ┌─────────────────┐                                       │
│  │ Decoder hidden  │                                       │
│  │ state s_{t-1}   │                                       │
│  │                 │                                       │
│  │ "Tôi đang sinh  │                                       │
│  │  từ gì?"        │                                       │
│  └────────┬────────┘                                       │
│           ↓                                                │
│  ┌────────────────┐                                        │
│  │ Query = s_{t-1} │  ← Decoder hidden state               │
│  │                 │    làm Query                          │
│  └────────────────┘                                        │
│                                                             │
│  ATTENTION: Query hỏi Key → được điểm → Value được trộn    │
│                                                             │
│  "Từ nào"  → Query đặt câu hỏi                            │
│  "Tôi ở đây" → Key trả lời định danh                      │
│  "Đây là nội dung" → Value trả l�ời thông tin               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

> [!CRITICAL]- Điểm khác biệt với Buổi 55 (Self-Attention)
>
> | Loại | Query đến từ | Key/Value đến từ |
> |------|---------------|------------------|
> | **Encoder-Decoder (Buổi 53)** | Decoder | Encoder |
> | **Self-Attention (Buổi 55)** | Cùng sequence | Cùng sequence |
>
> Self-attention = "nhìn vào chính mình" (câu nguồn nhìn vào câu nguồn)
> Encoder-decoder = "nhìn sang người khác" (decoder nhìn vào encoder)

---

## 2.2 Công thức từng bước

### Bảng ký hiệu — ĐỌC TRƯỚC KHI XEM CÔNG THỨC

| Ký hiệu | Kích thước | Giải thích bằng tiếng Việt |
|----------|------------|----------------------------|
| $\mathbf{s}_{t-1}$ | $(n, d)$ | Decoder hidden state TRƯỚC khi sinh từ $t$ |
| $\mathbf{h}_i$ | $(n, d)$ | Encoder hidden state tại vị trí $i$ trong câu nguồn |
| $\alpha_{ti}$ | $(n, T)$ | Trọng số chú ý: decoder chú ý vào vị trí $i$ bao nhiêu (tại bước $t$) |
| $\mathbf{c}_t$ | $(n, d)$ | Context vector: kết quả của attention, dùng cho decoder tại bước $t$ |
| $T$ | scalar | Độ dài câu nguồn |
| $n$ | scalar | Batch size |
| $d$ | scalar | Số chiều của hidden state |

### Công thức và giải thích từng bước

**Bước 1: Tính điểm tương đồng (attention scores)**

$$a(\mathbf{s}_{t-1}, \mathbf{h}_i) = \mathbf{v}^\top \tanh(\mathbf{W}_s \mathbf{s}_{t-1} + \mathbf{W}_h \mathbf{h}_i)$$

**Giải thích bằng tiếng Việt:**

```
Ý nghĩa: "Decoder hỏi encoder: vị trí i có liên quan không?"

Cách tính:
  1. Lấy decoder hidden state (s_{t-1}) và encoder hidden state (h_i)
  2. Biến đổi mỗi cái bằng một ma trận (W_s, W_h)
  3. Cộng lại: "kết hợp thông tin từ cả hai"
  4. Đi qua hàm tanh: "tạo ra giá trị mới, không âm cũng không quá lớn"
  5. Nhân với vector v: "tổng hợp thành một số cuối cùng"
  6. Kết quả: một số cho biết mức độ "liên quan"
```

**Tại sao dùng MLP (W_s, W_h, v) thay vì nhân ma trận?**

- Vì decoder hidden state và encoder hidden state có thể có **chiều khác nhau**
- MLP có thể "biến đổi" hai vector về cùng không gian trước khi so sánh

---

**Bước 2: Tính trọng số chú ý (attention weights)**

$$\alpha_{ti} = \text{softmax}(a(\mathbf{s}_{t-1}, \mathbf{h}_i))$$

**Giải thích bằng tiếng Việt:**

```
Ý nghĩa: "Chuyển điểm thành tỷ lệ (0-100%)"

Ví dụ: Decoder đang sinh từ "ngồi"
  - Điểm cho "sat":  8.0
  - Điểm cho "cat": 0.5
  - Điểm cho "mat": 1.0
  - Điểm cho "the": 0.1
  
  Sau softmax:
  - α_ngồi,sat = 0.90 (90%)
  - α_ngồi,cat = 0.02 (2%)
  - α_ngồi,mat = 0.06 (6%)
  - α_ngồi,the = 0.02 (2%)
  
  Tổng = 1.0 (100%)
```

---

**Bước 3: Tính context vector**

$$\mathbf{c}_t = \sum_{i=1}^T \alpha_{ti} \mathbf{h}_i$$

**Giải thích bằng tiếng Việt:**

```
Ý nghĩa: "Lấy thông tin từ các vị trí, có trọng số theo mức độ liên quan"

Ví dụ:
  c_t = 0.90 × h_sat + 0.02 × h_cat + 0.06 × h_mat + 0.02 × h_the
  
  = "Lấy 90% thông tin từ 'sat', 6% từ 'mat', ..."
```

---

**Bước 4: Decoder sử dụng context vector**

$$\mathbf{s}_t = \text{RNN}(\mathbf{s}_{t-1}, [\mathbf{y}_{t-1}; \mathbf{c}_t])$$

**Giải thích bằng tiếng Việt:**

```
Ý nghĩa: "Decoder tạo hidden state mới từ:
  1. Hidden state cũ (s_{t-1})
  2. Từ đã sinh trước đó (y_{t-1})
  3. Thông tin liên quan từ câu nguồn (c_t)"

Tại sao cần cả hai y_{t-1} VÀ c_t?
  - y_{t-1}: "Đã sinh từ gì rồi" (để giữ tính tuần tự)
  - c_t: "Cần lấy thông tin gì từ câu nguồn" (để dịch đúng)
```

---

## 2.3 Minh họa bằng ví dụ dịch câu

```
Câu nguồn: "The cat sat on the mat"
Đang sinh từ thứ 3: "ngồi"

Bước 1: ENCODER tạo hidden states
  h₁("The") = [0.1, 0.2, ...]
  h₂("cat") = [0.3, 0.5, ...]  ← Chủ ngữ
  h₃("sat") = [0.8, 0.1, ...]  ← Vị ngữ (hành động)
  h₄("on")  = [0.2, 0.3, ...]
  h₅("the") = [0.1, 0.1, ...]
  h₆("mat") = [0.4, 0.6, ...]  ← Bổ ngữ

Bước 2: DECODER hidden state
  s₂ = [0.2, 0.7, ...]  ← Đã sinh "Con mèo"

Bước 3: ATTENTION tính điểm
  Query = s₂ = [0.2, 0.7, ...]
  
  Score với h₁: v^T × tanh(W_s×s₂ + W_h×h₁) = 0.5
  Score với h₂: v^T × tanh(W_s×s₂ + W_h×h₂) = 2.0  ← "cat" liên quan
  Score với h₃: v^T × tanh(W_s×s₂ + W_h×h₃) = 9.0  ← "sat" rất liên quan!
  Score với h₄: v^T × tanh(W_s×s₂ + W_h×h₄) = 1.0
  Score với h₅: v^T × tanh(W_s×s₂ + W_h×h₅) = 0.3
  Score với h₆: v^T × tanh(W_s×s₂ + W_h×h₆) = 3.0  ← "mat" khá liên quan

Bước 4: SOFTMAX thành trọng số
  α₁ = exp(0.5) / Z ≈ 0.01
  α₂ = exp(2.0) / Z ≈ 0.05
  α₃ = exp(9.0) / Z ≈ 0.85  ← "sat" chiếm 85%!
  α₄ = exp(1.0) / Z ≈ 0.02
  α₅ = exp(0.3) / Z ≈ 0.01
  α₆ = exp(3.0) / Z ≈ 0.15
  (Z là tổng để tổng = 1)

Bước 5: TÍNH CONTEXT VECTOR
  c₃ = 0.01×h₁ + 0.05×h₂ + 0.85×h₃ + 0.02×h₄ + 0.01×h₅ + 0.15×h₆
     ≈ 85% thông tin từ "sat" (vị ngữ)

Bước 6: DECODER SINH TỪ
  s₃ = RNN(s₂, [y₂; c₃])
  Output: "ngồi" (vị ngữ)
```

---

# PHẦN III — IMPLEMENTATION

## 3.1 Bảng thuật ngữ cho code

| Tên trong code | Giải thích |
|----------------|------------|
| `queries` | Decoder hidden states — "câu hỏi" |
| `keys` | Encoder hidden states — "định danh" |
| `values` | Encoder hidden states — "nội dung" |
| `valid_lens` | Độ dài thực của mỗi câu nguồn (để bỏ qua padding) |
| `attention_weights` | Trọng số chú ý α — mức độ chú ý vào mỗi vị trí |
| `context` | Context vector — kết quả của attention |

## 3.2 Code với comment giải thích từng dòng

```python
class BahdanauAttention(nn.Module):
    """Encoder-decoder attention: Decoder được "nhìn" vào tất cả encoder outputs.

    Hoạt động như một "hệ thống hỏi-đáp":
    - Decoder hỏi: "Từ nào trong câu nguồn liên quan?"
    - Encoder trả lời bằng cách cho điểm mỗi vị trí
    - Lấy thông tin từ các vị trí có điểm cao

    Args:
        num_hiddens: Số units trong hidden layer của MLP dùng để tính điểm
    """
    def __init__(self, num_hiddens):
        super().__init__()

        # Ma trận W_h: biến đổi encoder hidden states
        # Biến "định danh" của mỗi vị trí nguồn về cùng không gian với query
        self.W_h = nn.LazyLinear(num_hiddens, bias=False)

        # Ma trận W_s: biến đổi decoder hidden state
        # Biến "câu hỏi" của decoder về cùng không gian với keys
        self.W_s = nn.LazyLinear(num_hiddens, bias=False)

        # Vector v: tổng hợp thông tin thành một số cuối cùng
        # Nhân với kết quả tanh để ra điểm tương đồng
        self.v = nn.LazyLinear(1, bias=False)

    def forward(self, queries, keys, values, valid_lens=None):
        """
        Args:
            queries: Decoder hidden states (Query)
                    Shape: (batch_size, n_queries, d)
                    - n_queries = 1 trong Seq2Seq (đang sinh từng từ)
                    - Trong Bahdanau: đây là decoder hidden state s_{t-1}

            keys: Encoder hidden states (Key)
                  Shape: (batch_size, m_keys, d)
                  - m_keys = độ dài câu nguồn
                  - Trong Bahdanau: đây là encoder outputs h_i

            values: Encoder hidden states (Value)
                    Shape: (batch_size, m_keys, d)
                    - Thường = keys trong Bahdanau
                    - Trong Bahdanau: đây là encoder outputs h_i

            valid_lens: Độ dài thực của mỗi câu nguồn
                       Shape: (batch_size,)
                       - Dùng để bỏ qua padding tokens

        Returns:
            Context vectors
            Shape: (batch_size, n_queries, d)
            - Trọng số hóa tổng của values, theo attention weights
        """
        # ============================================================
        # Bước 1: Biến đổi queries và keys về cùng không gian
        # ============================================================

        # queries: decoder hidden state s_{t-1}
        # Shape: (batch, n, d) → (batch, n, h) sau projection
        queries = self.W_s(queries)

        # keys: encoder hidden states h_i
        # Shape: (batch, m, d) → (batch, m, h) sau projection
        keys = self.W_h(keys)

        # ============================================================
        # Bước 2: Tính features bằng broadcasting
        # ============================================================

        # Broadcasting để tính tất cả cặp (query, key) cùng lúc
        #
        # queries.unsqueeze(2): (batch, n, 1, h)
        #   - Thêm chiều để broadcast với keys
        # keys.unsqueeze(1): (batch, 1, m, h)
        #   - Thêm chiều để broadcast với queries
        #
        # Result: (batch, n, m, h)
        #   - Với mỗi query, có m điểm cho m keys
        #   - Ví dụ: n=1 (1 từ đang sinh), m=10 (câu nguồn 10 từ)
        #   - Tính điểm cho 10 vị trí cùng lúc
        features = queries.unsqueeze(2) + keys.unsqueeze(1)

        # tanh: activation function
        # - Giữ giá trị trong range [-1, 1]
        # - Cho phép "tăng" hoặc "giảm" điểm tương đương
        features = torch.tanh(features)

        # ============================================================
        # Bước 3: Tính attention scores
        # ============================================================

        # v(features): (batch, n, m, h) → (batch, n, m, 1)
        # - Vector v nhân với mỗi feature
        # - Kết quả squeeze thành (batch, n, m)
        # - Đây là "điểm tương đồng" a(s_{t-1}, h_i)
        scores = self.v(features).squeeze(-1)

        # ============================================================
        # Bước 4: Masked softmax
        # ============================================================

        # Masked softmax để bỏ qua padding tokens
        #
        # Ví dụ:
        # - Câu 1: 5 từ thực + 3 padding
        # - Câu 2: 7 từ thực + 1 padding
        #
        # valid_lens = [5, 7]
        # Scores cho câu 1: vị trí 0-4 tính bình thường,
        #                    vị trí 5-7 set = -1e6
        # Scores cho câu 2: vị trí 0-6 tính bình thường,
        #                    vị trí 7 set = -1e6
        #
        # softmax(-1e6) ≈ 0 → không ảnh hưởng đến context vector
        self.attention_weights = masked_softmax(scores, valid_lens)

        # ============================================================
        # Bước 5: Tính context vector = weighted sum của values
        # ============================================================

        # BMM (Batch Matrix Multiplication):
        # attention_weights: (batch, n, m) — trọng số cho mỗi vị trí
        # values: (batch, m, d) — thông tin tại mỗi vị trí
        #
        # Result: (batch, n, d)
        # - Mỗi query được "trộn" thông tin từ tất cả values
        # - Trọng số cao → lấy nhiều thông tin từ value đó
        return torch.bmm(self.attention_weights, values)
```

---

# PHẦN IV — SO SÁNH

## 4.1 Bảng so sánh các loại Attention

| Khía cạnh | Bahdanau (Encoder-Decoder) | Self-Attention |
|-----------|---------------------------|----------------|
| **Query đến từ** | Decoder hidden state | Cùng sequence |
| **Key đến từ** | Encoder outputs | Cùng sequence |
| **Value đến từ** | Encoder outputs | Cùng sequence |
| **Mục đích** | Align input (nguồn) với output (đích) | Capture dependencies trong cùng sequence |
| **Dùng trong** | Seq2Seq + Attention, NMT | Transformer, BERT |
| **Năm** | 2014 | 2017 |

## 4.2 Bahdanau vs Dot Product Attention

| Khía cạnh | Bahdanau (Additive) | Dot Product (Multiplicative) |
|-----------|---------------------|------------------------------|
| **Cách tính điểm** | MLP: $\mathbf{v}^\top \tanh(\mathbf{W}_q\mathbf{q} + \mathbf{W}_k\mathbf{k})$ | Nhân ma trận: $\mathbf{q}^\top \mathbf{k}$ |
| **Chiều q và k** | Có thể khác nhau | Phải giống nhau |
| **Số tham số** | Nhiều (W_q, W_k, v) | Không có |
| **Tốc độ** | Chậm hơn (MLP) | Nhanh hơn (chỉ nhân ma trận) |
| **Dùng trong** | Bahdanau NMT (2014) | Transformer (2017) |

---

# PHẦN V — TÓM TẮT

## 5.1 Ghi nhớ chính

```
┌─────────────────────────────────────────────────────────────┐
│                 BAHDANAU ATTENTION — TÓM TẮT                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  VẤN ĐỀ:                                                   │
│  - Seq2Seq cổ điển nén tất cả vào MỘT vector               │
│  - Khi câu dài → thông tin bị mất                          │
│                                                             │
│  GIẢI PHÁP:                                                 │
│  - Decoder được "nhìn" vào TẤT CẢ encoder outputs           │
│  - Mỗi bước decode có context vector RIÊNG                   │
│                                                             │
│  CƠ CHẾ:                                                    │
│  - Query = Decoder hidden state (câu hỏi)                    │
│  - Key = Encoder hidden states (định danh)                  │
│  - Value = Encoder hidden states (nội dung)                 │
│  - Attention scores = điểm tương đồng (MLP trong Bahdanau) │
│  - Attention weights = softmax(scores)                      │
│  - Context vector = Σ(weights × values)                    │
│                                                             │
│  SỰ KHÁC BIỆT VỚI SELF-ATTENTION:                          │
│  - Self-attention: Query, Key, Value cùng nguồn             │
│  - Bahdanau: Query từ decoder, Key/Value từ encoder         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 5.2 Bảng thuật ngữ cuối buổi

| Thuật ngữ | Tiếng Việt | Ghi nhớ |
|-----------|------------|---------|
| Encoder-decoder attention | Chú ý bộ mã hóa-giải mã | Decoder hỏi encoder |
| Bahdanau attention | Chú ý Bahdanau | Dùng MLP để tính điểm |
| Cross-attention | Chú ý chéo | Query ≠ Key/Value nguồn |
| Bottleneck | Nút thắt cổ chai | Thông tin bị nén quá nhiều |
| Alignment | Sự sắp xếp | Match từ nguồn với từ đích |

---

## Bài tập

1. **Attention visualization**: Vẽ attention weights cho một câu dịch. Nhận xét pattern.
2. **Different scoring**: Thay MLP bằng dot product. Kết quả thay đổi thế nào?
3. **Complexity**: So sánh tính toán của Seq2Seq cổ điển và có attention.

---

## TODO

- [ ] Visualize attention weights cho các câu dịch khác nhau
- [ ] Implement với dot product scoring để so sánh
- [ ] Đọc paper Bahdanau (2014)

---

## Liên kết

- [[Buổi 50 - Tuần 14|QKV — Queries, Keys, and Values]]
- [[Buổi 51 - Tuần 14|Attention Pooling by Similarity]]
- [[Buổi 52 - Tuần 14|Attention Scoring Functions]]
- [[Buổi 54 - Tuần 14|Multi-Head Attention]]
- [[Buổi 55 - Tuần 14|Self-Attention & Positional Encoding]]
