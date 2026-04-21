---
session: "D2L Tuần 12, Buổi 45 — 10.3 & 10.4 Deep RNN + Bidirectional RNN"
d2l_chapter: "10.3–10.4"
tags:
  - d2l
  - deep-learning
  - rnn
  - deep-rnn
  - bidirectional-rnn
  - stacked-rnn
  - modern-rnn
aliases:
  - Deep RNN
  - Bidirectional RNN
  - BiRNN
date: 2026-04-20
status: complete
---

# Buổi 45 — 10.3 Deep Recurrent Neural Networks & 10.4 Bidirectional RNN

> **Nguồn:** [d2l.ai — 10.3](https://d2l.ai/chapter_recurrent-modern/deep-rnn.html) + [d2l.ai — 10.4](https://d2l.ai/chapter_recurrent-modern/bi-rnn.html)
> **Buổi trước:** [[Buổi 44 - Tuần 12]] — 10.2 Gated Recurrent Units (GRU)
> **Buổi sau:** [[Buổi 46 - Tuần 13]] — 10.5 Machine Translation and the Dataset

---

## Active Recall — Ôn lại Buổi 44 (GRU)

### Câu hỏi (không nhìn tài liệu)

1. GRU có bao nhiêu cổng? Kể tên và giải thích vai trò của từng cổng.
2. Viết công thức cập nhật hidden state của GRU. Giải thích ý nghĩa từng thành phần.
3. Reset gate nhân phần tử với $H_{t-1}$ trước khi tính candidate. Tại sao điều này quan trọng?
4. Update gate tạo ra tổ hợp lồi — điều đó có nghĩa là gì? Cho ví dụ số.
5. GRU có cell state riêng không? Đây là điểm khác biệt gì so với LSTM?
6. Số tham số của GRU ít hơn LSTM bao nhiêu phần trăm? Tính cụ thể.
7. Khi nào nên chọn GRU thay vì LSTM? Cho 3 ví dụ tình huống cụ thể.
8. Trong RNN thường: $H_t = \tanh(X_t W_{xh} + H_{t-1} W_{hh} + b)$. GRU khác ở điểm nào?
9. Nếu $Z_t = 0.8$, $H_{t-1} = 0.9$, $\tilde{H}_t = 0.2$. Tính $H_t$ và giải thích ý nghĩa.
10. GRU có thể khôi phục hành vi RNN thường không? Trong điều kiện nào?

### Tự trả lời

1. **2 cổng**: Reset Gate $R_t$ (quyết định quên bao nhiêu ký ức cũ khi tạo candidate) và Update Gate $Z_t$ (quyết định tỉ lệ giữa trạng thái cũ và candidate mới).
2. $H_t = Z_t \odot H_{t-1} + (1 - Z_t) \odot \tilde{H}_t$. $Z_t$ kiểm soát giữ, $(1-Z_t)$ kiểm soát thêm mới.
3. Khi $R_t \to 0$: phép nhân $R_t \odot H_{t-1}$ xóa gần hết ảnh hưởng của trạng thái cũ, candidate chỉ phụ thuộc vào $X_t$ — mạng "reset" về trạng thái ban đầu.
4. Tổ hợp lồi = convex combination = weighted average giữa hai giá trị. $H_t$ luôn nằm giữa $H_{t-1}$ và $\tilde{H}_t$ theo tỉ lệ $Z_t$. Ví dụ: $Z=0.7$ nghĩa là 70% giữ lại trạng thái cũ, 30% thay bằng candidate.
5. Không. LSTM có 2 trạng thái song song ($C_t$ và $H_t$). GRU chỉ có 1 ($H_t$) — tất cả cơ chế nhớ đều encode trong cùng vector.
6. GRU: $3(dh+h^2+h)$, LSTM: $4(dh+h^2+h)$. GRU ít hơn **25%**.
7. (a) Cần huấn luyện nhanh, tài nguyên hạn chế; (b) Chuỗi ngắn đến trung bình; (c) Ứng dụng real-time, on-device.
8. RNN thường dùng trực tiếp $H_{t-1}$ trong công thức tanh. GRU thêm $R_t \odot H_{t-1}$ (reset) rồi dùng $Z_t$ để trộn kết quả.
9. $H_t = 0.8 \times 0.9 + 0.2 \times 0.2 = 0.72 + 0.04 = 0.76$. Giữ 80% trạng thái cũ (0.9), thêm 20% candidate mới (0.2).
10. Có. Khi $Z_t = 1$ (giữ nguyên hoàn toàn) và $R_t = 1$ (không reset), GRU khôi phục RNN thường: $H_t = H_{t-1}$.

### Ghi chú khái niệm cần ôn lại

- [[Gated Recurrent Unit]]
- [[Long Short-Term Memory]]
- [[Recurrent Neural Network]]
- [[Backpropagation Through Time]]
- [[Sigmoid Function]]

---

# PHẦN I — DEEP RECURRENT NEURAL NETWORKS (10.3)

---

## 1. Tổng quan: Deep RNN là gì?

> [!NOTE] Giải thích đơn giản — ELI5
> Một tầng RNN giống như một người đọc hiểu từng câu một cách tuần tự. Nếu muốn hiểu sâu hơn — như phân tích văn học, so sánh với các tác phẩm khác, đánh giá phong cách — bạn cần nhiều người đọc xếp chồng lên nhau. Người thứ nhất đọc từng câu. Người thứ hai đọc từng đoạn mà người thứ nhất đã tóm tắt, hiểu mối liên hệ giữa các câu. Người thứ ba đọc từng chương mà người thứ hai đã tóm tắt, hiểu cấu trúc toàn bộ văn bản. **Deep RNN chính là nhiều "người đọc" xếp chồng** — mỗi tầng hiểu ở một cấp độ khác nhau.

### 1.1 Định nghĩa kỹ thuật — Deep RNN là gì?

**Deep RNN (còn gọi là Stacked RNN)** là kiến trúc RNN gồm **nhiều tầng RNN xếp chồng** (stacked), trong đó đầu ra của tầng dưới được dùng làm đầu vào cho tầng trên. Khái niệm "chiều sâu" (depth) trong RNN có **hai hướng** khác nhau:

| Hướng chiều sâu | Mô tả | Tương đương trong CNN |
|---|---|---|
| Chiều sâu theo **thời gian** | Thông tin đi qua $T$ bước thời gian trước khi ra output cuối | Nhiều layers trong MLP |
| Chiều sâu theo **không gian/tầng** | Nhiều tầng RNN xếp chồng lên nhau | Nhiều conv layers xếp chồng |

Dù RNN 1 tầng đã có "chiều sâu theo thời gian" (vì input bước đầu tiên ảnh hưởng đến output bước $T$ qua $T$ lần recurrence), ta vẫn cần **chiều sâu theo tầng** để biểu diễn quan hệ phức tạp giữa input và output tại cùng bước thời gian.

### 1.2 Tại sao cần Deep RNN?

Với 1 tầng RNN, mô hình học được quan hệ **tuyến tính** giữa input và hidden state. Nhiều tầng cho phép học quan hệ **phi tuyến tính phức tạp hơn** — tương tự như tại sao MLP cần nhiều hidden layers.

**Ví dụ cụ thể:** Trong phân tích văn bản về "Tôi đi bộ qua công viên":
- **Tầng 1:** Nhận diện từng từ → "Tôi" (pronoun), "đi bộ" (verb-phrase)
- **Tầng 2:** Nhận diện cụm từ → "Tôi đi bộ" (subject + action)
- **Tầng 3:** Hiểu ngữ cảnh → "đi bộ qua công viên" (action + location)

Mỗi tầng trừu tượng hóa (abstract) ở mức cao hơn tầng dưới.

---

## 2. Kiến trúc Deep RNN — Công thức chi tiết

> [!NOTE] Giải thích đơn giản
> Hãy tưởng tượng mỗi tầng RNN là một "máy xử lý thông tin". Đầu vào của máy ở tầng $l$ gồm hai thứ: (1) thông tin từ tầng dưới ($l-1$) tại cùng bước thời gian $t$, và (2) ký ức của chính máy đó từ bước thời gian trước ($t-1$). Máy trộn hai nguồn thông tin này rồi tạo ra output cho tầng trên.

### 2.1 Công thức tầng ẩn thứ $l$

Công thức cốt lõi (D2L Eq. 10.3.1):

$$H_t^{(l)} = \phi_l\left(H_t^{(l-1)} W_{xh}^{(l)} + H_{t-1}^{(l)} W_{hh}^{(l)} + b_h^{(l)}\right)$$

**Giải nghĩa từng ký hiệu:**

| Ký hiệu | Ý nghĩa | Shape |
|---|---|---|
| $H_t^{(l-1)} \in \mathbb{R}^{n \times h}$ | Hidden state của tầng dưới tại bước $t$ | $(n, h)$ |
| $W_{xh}^{(l)} \in \mathbb{R}^{h \times h}$ | Trọng số từ tầng dưới vào tầng $l$ | $(h, h)$ |
| $H_{t-1}^{(l)} \in \mathbb{R}^{n \times h}$ | Hidden state của tầng $l$ tại bước $t-1$ | $(n, h)$ |
| $W_{hh}^{(l)} \in \mathbb{R}^{h \times h}$ | Trọng số recurrence của tầng $l$ | $(h, h)$ |
| $\phi_l$ | Hàm kích hoạt (thường là $\tanh$) | scalar |
| $H_t^{(l)}$ | Hidden state mới của tầng $l$ tại bước $t$ | $(n, h)$ |

**Quy ước:** $H_t^{(0)} = X_t$ — tầng 0 chính là input.

### 2.2 Công thức output layer

$$O_t = H_t^{(L)} W_{hq} + b_q \tag{10.3.2}$$

Chỉ tầng ẩn cuối cùng $L$ được dùng để tính output. Tầng trên không trực tiếp xuất ra mà chỉ cung cấp biểu diễn (representations) cho tầng cao hơn.

### 2.3 Hai nguồn thông tin tại mỗi tầng

Tại mỗi bước thời gian $t$, hidden state $H_t^{(l)}$ phụ thuộc vào **hai nguồn**:

1. **Từ chiều không gian (inter-layer):** $H_t^{(l-1)}$ — thông tin từ tầng dưới cùng bước $t$
2. **Từ chiều thời gian (intra-layer):** $H_{t-1}^{(l)}$ — thông tin từ chính tầng đó ở bước $t-1$

### 2.4 Minh họa kiến trúc Deep RNN (D2L Fig. 10.3.1)

![[assets/attachments/d2l-buoi-45/deep-rnn.png]]

_Fig. 10.3.1 (D2L): Kiến trúc Deep RNN với $L$ tầng ẩn. Mỗi hộp trắng là một RNN cell. Tại mỗi bước thời gian, hidden state phụ thuộc vào cùng tầng bước trước (mũi tên ngang) và tầng dưới cùng bước (mũi tên dọc)._

**Đọc sơ đồ — từ trái sang phải:**
- Cột dọc: các bước thời gian $t=1, 2, \ldots, T$
- Hàng ngang: các tầng $l=1, 2, \ldots, L$
- Mũi tên ngang (gạch chấm): recurrence qua thời gian trong cùng tầng
- Mũi tên dọc: truyền từ tầng dưới lên tầng trên cùng bước

### 2.5 So sánh RNN 1 tầng và Deep RNN

| Khía cạnh | RNN 1 tầng | Deep RNN ($L$ tầng) |
|---|---|---|
| Input của hidden | chỉ $X_t$ và $H_{t-1}$ | thêm $H_t^{(l-1)}$ |
| Biểu diễn | tuyến tính | phi tuyến phức tạp hơn |
| Tham số | $dh + h^2 + h$ | $L \times (dh + h^2 + h)$ |
| Tốc độ | nhanh nhất | chậm hơn $L$ lần |
| Gradient flow | qua $T$ bước | qua $T \times L$ bước |

---

## 3. Siêu tham số của Deep RNN

> [!NOTE] Giải thích đơn giản
> Cũng như xây nhà: có thể xây **cao thêm** (thêm tầng) hoặc xây **rộng thêm** (phòng lớn hơn). Trong Deep RNN, thêm tầng = tăng $L$, tăng số đơn vị ẩn = tăng $h$. Cả hai đều làm nhà "lớn hơn" nhưng theo cách khác nhau.

### 3.1 Khoảng giá trị phổ biến

| Siêu tham số | Khoảng phổ biến | Ý nghĩa |
|---|---|---|
| Số đơn vị ẩn $h$ | 64 – 2056 | Chiều rộng mỗi tầng |
| Số tầng $L$ | 1 – 8 | Độ sâu của mạng |

### 3.2 Chiều rộng ($h$) vs Chiều sâu ($L$)

Nghiên cứu thực nghiệm cho thấy:

- **Tăng chiều rộng ($h$)** thường hiệu quả hơn tăng chiều sâu ($L$) cho các bài toán sequence thông thường
- **Tăng chiều sâu ($L$)** giúp học biểu diễn phân cấp (hierarchical representations) tốt hơn
- Deep RNN với $L \geq 3$ dễ bị vanishing/exploding gradient nghiêm trọng hơn
- LSTM/GRU thay vì vanilla RNN giúp giảm vấn đề này trong Deep RNN

> [!TIP] Quy tắc thực hành
> Bắt đầu với $L=2$ và $h=256$. Nếu cần nhiều hơn, ưu tiên tăng $h$ trước. Chỉ tăng $L$ khi cần biểu diễn phân cấp rõ ràng (ví dụ: từ → cụm từ → câu → đoạn).

---

## 4. Cài đặt từ đầu — Stacked RNN (10.3.1)

### 4.1 Khởi tạo tham số

```python
class StackedRNNScratch(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, num_layers, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        init_weight = lambda *shape: nn.Parameter(
            torch.randn(*shape) * sigma)
        triple = lambda: (init_weight(num_inputs if True else num_hiddens, num_hiddens),
                          init_weight(num_hiddens, num_hiddens),
                          nn.Parameter(torch.zeros(num_hiddens)))
        # Mỗi tầng có bộ trọng số riêng
        self.rnns = nn.Sequential(*[d2l.RNNScratch(
            num_inputs if i == 0 else num_hiddens,
            num_hiddens, sigma)
            for i in range(num_layers)])
```

**Phân tích chi tiết:**

- **`nn.Sequential(*[...])`**: Tạo một chuỗi các modules chạy lần lượt. Dấu `*` unpacks danh sách thành các positional arguments.
- **`num_inputs if i == 0 else num_hiddens`**: Tầng đầu tiên ($i=0$) nhận input gốc $X_t$ (kích thước $d$). Các tầng sau ($i \geq 1$) nhận hidden state của tầng trước (kích thước $h$).
- **Mỗi tầng có bộ trọng số riêng**: tổng số tham số = $L \times (dh + h^2 + h)$.

### 4.2 Forward computation

```python
@d2l.add_to_class(StackedRNNScratch)
def forward(self, inputs, Hs=None):
    outputs = inputs                    # shape: (T, n, d) ban đầu
    if Hs is None:
        Hs = [None] * self.num_layers  # danh sách hidden states
    for i in range(self.num_layers):
        outputs, Hs[i] = self.rnns[i](outputs, Hs[i])
        # outputs: (T, n, h) sau mỗi tầng
        outputs = torch.stack(outputs, 0)
        # shape: (T, n, h) — stack không thay đổi shape ở đây
    return outputs, Hs
```

**Phân tích từng bước:**

```
Bước 0: inputs = (T, n, d)         ← đầu vào gốc
Bước 1: outputs, H_1 = rnn[0](inputs, H_1_init)
         outputs = (T, n, h)        ← tầng 1
Bước 2: outputs, H_2 = rnn[1](outputs, H_2_init)
         outputs = (T, n, h)        ← tầng 2
...
Bước L: outputs, H_L = rnn[L-1](outputs, H_L_init)
         outputs = (T, n, h)        ← tầng L (output cuối)
```

### 4.3 Shape analysis cụ thể

Với $L=2$ tầng, $h=32$:

```
Input X:       (T=32, batch=1024, d=28)
After layer 1:  (T=32, batch=1024, h=32)
After layer 2:  (T=32, batch=1024, h=32)
Output O:       (T=32, batch=1024, q=28)
```

---

## 5. Cài đặt gọn — `nn.GRU` nhiều tầng (10.3.2)

```python
class GRU(d2l.RNN):
    def __init__(self, num_inputs, num_hiddens, num_layers, dropout=0):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = nn.GRU(num_inputs, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, inputs, H=None):
        return self.rnn(inputs, H)
```

**So sánh chi tiết scratch vs concise:**

| Khía cạnh | Từ đầu (`StackedRNNScratch`) | Gọn (`nn.GRU`) |
|---|---|---|
| Dòng code | ~20 dòng | 5 dòng |
| Quản lý hidden states | Tự xử lý danh sách $H_1 \ldots H_L$ | Framework tự quản lý |
| Dropout giữa tầng | Phải thêm thủ công | `dropout=` param |
| Tốc độ | Python loop | cuDNN fused kernel |
| Memory | Thủ công | Tự tối ưu |

```python
# Huấn luyện Deep GRU
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
gru = GRU(num_inputs=len(data.vocab), num_hiddens=32, num_layers=2)
model = d2l.RNNLM(gru, vocab_size=len(data.vocab), lr=2)
trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
trainer.fit(model, data)
```

### 5.1 Dropout giữa các tầng

> [!NOTE] Giải thích đơn giản
> Trong một tòa nhà nhiều tầng, nếu tầng dưới (nền tảng) bị hỏng, các tầng trên sẽ bị ảnh hưởng nghiêm trọng. Dropout giữa tầng giống như xây "cầu dự phòng" giữa các tầng — nếu một tầng bị "tắt" (trong quá trình training), các tầng khác vẫn có thể học được.

Dropout giữa tầng hoạt động khác với dropout thông thường:
- **Dropout thông thường**: tắt ngẫu nhiên một số neurons trong cùng tầng
- **Dropout giữa tầng (variational dropout)**: tắt ngẫu nhiên toàn bộ kết nối từ tầng $l$ sang $l+1$

### 5.2 Thay đổi hyperparameters so với 1 tầng

| Hyperparameter | RNN 1 tầng | Deep RNN |
|---|---|---|
| Learning rate | 4 | **2** (thấp hơn để ổn định) |
| Max epochs | 50 | **100** (hội tụ chậm hơn) |
| Gradient clip | 1 | **1** (bắt buộc, đặc biệt khi $L \geq 3$) |

### 5.3 Minh họa kết quả huấn luyện

![[assets/attachments/d2l-buoi-45/deep-rnn-training.png]]
_Kết quả huấn luyện Deep GRU (2 tầng, $h=32$, $lr=2$) trên bộ dữ liệu Time Machine. So với GRU 1 tầng ($lr=4$), deep RNN cần learning rate thấp hơn và nhiều epochs hơn._

---

## 6. Lưu ý quan trọng khi khởi tạo Deep RNN

Deep RNN đòi hỏi initialization cẩn thận hơn so với shallow RNN:

### 6.1 Vấn đề gradient

Gradient trong Deep RNN phải truyền qua **$T \times L$ tầng** — nhiều hơn RNN 1 tầng ($T$ tầng). Điều này làm tăng nguy cơ vanishing/exploding gradient theo cấp số nhân.

### 6.2 Giải pháp

1. **Gradient clipping bắt buộc**: `gradient_clip_val=1` luôn cần thiết
2. **Learning rate thấp hơn**: giảm từ 4 xuống 2 (hoặc thấp hơn)
3. **Khởi tạo trọng số cẩn thận**: Xavier/orthogonal initialization
4. **Batch Normalization**: có thể giúp normalize activations giữa các tầng (ít phổ biến trong RNN so với CNN)
5. **Layer Normalization**: thay thế phổ biến hơn cho RNN vì nó normalize theo feature dimension thay vì batch dimension

---

# PHẦN II — BIDIRECTIONAL RECURRENT NEURAL NETWORKS (10.4)

---

## 7. Tổng quan: Tại sao cần Bidirectional RNN?

> [!NOTE] Giải thích đơn giản — ELI5
> Đọc câu "I am ___ hungry":
> - Đọc từ trái sang phải: ta không biết điền gì — "hungry" chưa xuất hiện
> - Đọc từ phải sang trái: ta thấy "hungry" → gợi ý "not" hoặc "very"
> - Đọc cả hai hướng: ta hiểu đầy đủ ngữ cảnh
>
> Bidirectional RNN như có **hai người đọc**: một đọc từ trái sang phải, một đọc từ phải sang trái. Cả hai chia sẻ thông tin để hiểu toàn bộ ngữ cảnh tại mỗi từ.

### 7.1 Định nghĩa kỹ thuật

**Bidirectional RNN (BiRNN)** là kiến trúc gồm **hai unidirectional RNN chạy ngược chiều nhau** trên cùng một chuỗi đầu vào:

1. **Forward RNN**: đọc từ $x_1$ đến $x_T$ (trái → phải)
2. **Backward RNN**: đọc từ $x_T$ đến $x_1$ (phải → trái)

Output tại mỗi bước $t$ là **concatenation** của hidden state từ cả hai hướng.

### 7.2 Bài toán nào cần Bidirectional?

| Bài toán | Ví dụ | Tại sao cần 2 hướng |
|---|---|---|
| POS tagging | "book" = verb hay noun? | Cần từ trước và sau |
| Masked language modeling | BERT pretraining | Cần cả 2 hướng để predict từ bị che |
| Named entity recognition | "Paris" = LOC hay ORG? | Cần ngữ cảnh hai phía |
| Machine translation | Dịch câu nguồn | Cần hiểu toàn bộ câu nguồn |
| Sentiment analysis | "not bad" ≠ "bad" | Negation ở cuối đảo ngược nghĩa |

### 7.3 Bài toán nào KHÔNG cần Bidirectional?

**Language modeling** — dự đoán từ tiếp theo — **chỉ cần ngữ cảnh bên trái**. Bidirectional sẽ làm "cháy" thông tin tương lai vào quá khứ → không đúng với mục tiêu.

### 7.4 Ví dụ minh họa từ D2L

D2L minh họa bằng ví dụ masked language modeling:

```
I am ___.
I am ___ hungry.
I am ___ hungry, and I can eat half a pig.
```

- Câu 1: điền "happy" → không có ngữ cảnh
- Câu 2: điền "not" hoặc "very" → "hungry" ở bên phải gợi ý
- Câu 3: điền "very" → "hungry, and I can eat half a pig" → rất hungry

Một model bidirectional có thể nhìn cả bên trái và bên phải để đưa ra dự đoán chính xác.

---

## 8. Kiến trúc Bidirectional RNN — Công thức chi tiết

### 8.1 Công thức hai chiều hidden states

D2L Eq. 10.4.1:

$$\overrightarrow{H}_t = \phi\!\left(X_t W_{xh}^{(f)} + \overrightarrow{H}_{t-1} W_{hh}^{(f)} + b_h^{(f)}\right) \tag{10.4.1a}$$

$$\overleftarrow{H}_t = \phi\!\left(X_t W_{xh}^{(b)} + \overleftarrow{H}_{t+1} W_{hh}^{(b)} + b_h^{(b)}\right) \tag{10.4.1b}$$

**Giải nghĩa từng ký hiệu:**

| Ký hiệu | Ý nghĩa |
|---|---|
| $\overrightarrow{H}_t$ | Forward hidden state tại bước $t$ — chứa thông tin từ $x_1$ đến $x_t$ |
| $\overleftarrow{H}_t$ | Backward hidden state tại bước $t$ — chứa thông tin từ $x_T$ đến $x_t$ |
| $W^{(f)}, W^{(b)}$ | Hai bộ trọng số riêng cho forward và backward |
| $\overrightarrow{H}_{t-1}$ | Forward hidden state bước trước (đọc trái→phải) |
| $\overleftarrow{H}_{t+1}$ | Backward hidden state bước sau (đọc phải→trái) — **lưu ý: $t+1$ chứ không phải $t-1$!** |

### 8.2 Minh họa kiến trúc (D2L Fig. 10.4.1)

![[assets/attachments/d2l-buoi-45/bi-rnn-1.png]]
_Fig. 10.4.1 (D2L): Kiến trúc Bidirectional RNN. Forward RNN (mũi tên xuống) đọc từ trái sang phải. Backward RNN (mũi tên lên) đọc từ phải sang trái. Output tại mỗi bước là concatenation $[\overrightarrow{H}_t; \overleftarrow{H}_t]$._

**Đọc sơ đồ — từng bước:**

1. **Forward direction** (xuống): $x_1 \to \overrightarrow{H}_1$, $x_2 \to \overrightarrow{H}_2$, ... → $\overrightarrow{H}_T$
   - Forward hidden state $\overrightarrow{H}_t$ chứa thông tin từ **đầu chuỗi đến bước $t$**

2. **Backward direction** (lên): $x_T \to \overleftarrow{H}_T$, $x_{T-1} \to \overleftarrow{H}_{T-1}$, ... → $\overleftarrow{H}_1$
   - Backward hidden state $\overleftarrow{H}_t$ chứa thông tin từ **cuối chuỗi đến bước $t$**

3. **Output tại mỗi bước**: concatenation $[\overrightarrow{H}_t; \overleftarrow{H}_t]$
   - Tại bước $t$, ta biết **cả** quá khứ ($x_1 \ldots x_t$) **và** tương lai ($x_t \ldots x_T$)

### 8.3 Công thức output layer

$$O_t = [H_t^{(f)}, H_t^{(b)}] W_{hq} + b_q \tag{10.4.2}$$

**Điểm quan trọng:** Hidden state đầu ra có chiều $2h$ (gấp đôi) vì concatenation của forward và backward. Do đó ma trận trọng số output $W_{hq} \in \mathbb{R}^{2h \times q}$.

### 8.4 So sánh Unidirectional vs Bidirectional

| Khía cạnh | Unidirectional RNN | Bidirectional RNN |
|---|---|---|
| Hidden state | $H_t \in \mathbb{R}^h$ | $[H_t^{(f)}; H_t^{(b)}] \in \mathbb{R}^{2h}$ |
| Ngữ cảnh | Chỉ quá khứ | Cả quá khứ và tương lai |
| Inference | Real-time được | **Cần toàn bộ chuỗi** |
| Tham số | $dh + h^2 + h$ | $2(dh + h^2 + h)$ |
| Ứng dụng | Language modeling, forecasting | Tagging, encoding, MT |

---

## 9. Cài đặt từ đầu — BiRNN (10.4.1)

### 9.1 Khởi tạo

```python
class BiRNNScratch(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        # Hai RNN riêng cho forward và backward
        self.f_rnn = d2l.RNNScratch(num_inputs, num_hiddens, sigma)
        self.b_rnn = d2l.RNNScratch(num_inputs, num_hiddens, sigma)
        # Output dimension = gấp đôi vì concatenation
        self.num_hiddens *= 2
```

**Phân tích:**
- `self.f_rnn`: Forward RNN — nhận input thông thường
- `self.b_rnn`: Backward RNN — nhận **đảo ngược** input
- `self.num_hiddens *= 2`: Vì output là concatenation nên số chiều gấp đôi

### 9.2 Forward computation

```python
@d2l.add_to_class(BiRNNScratch)
def forward(self, inputs, Hs=None):
    f_H, b_H = Hs if Hs is not None else (None, None)

    # Forward: chạy từ trái sang phải (bình thường)
    f_outputs, f_H = self.f_rnn(inputs, f_H)

    # Backward: đảo ngược input trước khi đưa vào
    # reversed(inputs) = [x_T, x_{T-1}, ..., x_1]
    # Backward RNN đọc từ phải sang trái → kết quả tự động đúng thứ tự
    b_outputs, b_H = self.b_rnn(reversed(inputs), b_H)

    # Concatenation: ghép forward và backward tại mỗi bước
    # f_outputs[i] ghép với b_outputs[T-1-i] (vì đã đảo ngược)
    outputs = [torch.cat((f, b), -1) for f, b in zip(
        f_outputs, reversed(b_outputs))]

    return outputs, (f_H, b_H)
```

**Giải thích chi tiết từng bước:**

```
Bước 1 — Forward:
  inputs = [x_1, x_2, x_3, ..., x_T]
  f_outputs = [H→_1, H→_2, H→_3, ..., H→_T]
    trong đó H→_t = forward(x_1..x_t)

Bước 2 — Backward:
  reversed(inputs) = [x_T, x_{T-1}, ..., x_1]
  b_outputs = [H←_T, H←_{T-1}, ..., H←_1]
    trong đó H←_t = backward(x_T..x_t)
  reversed(b_outputs) = [H←_1, H←_2, ..., H←_T]  ← đúng thứ tự!

Bước 3 — Concatenation:
  outputs[t] = concat(f_outputs[t], reversed(b_outputs)[t])
           = concat(H→_t, H←_t)  ← thông tin cả hai hướng tại bước t
```

> [!WARNING] Điểm dễ nhầm: reversed inputs
> Chúng ta đảo ngược input VÌ Backward RNN vẫn chạy từ index 0 đến T trong code, nhưng nội dung đã bị đảo. Điều này giúp tận dụng cùng logic RNN mà không cần viết lại.

---

## 10. Cài đặt gọn — `nn.GRU` Bidirectional (10.4.2)

```python
class BiGRU(d2l.Module):
    def __init__(self, num_inputs, num_hiddens):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = nn.GRU(num_inputs, num_hiddens, bidirectional=True)
        self.num_hiddens *= 2
```

Chỉ cần thêm `bidirectional=True` — framework tự đảo input và quản lý backward pass.

```python
gru = BiGRU(num_inputs=len(data.vocab), num_hiddens=32)
model = d2l.RNNLM(gru, vocab_size=len(data.vocab), lr=2)
trainer.fit(model, data)
# Dự đoán: 'it has for and the time th'
```

---

## 11. Deep Bidirectional RNN

### 11.1 Khái niệm

Kết hợp **Deep** và **Bidirectional**: nhiều tầng bidirectional xếp chồng. Tầng trên nhận concatenation từ tầng dưới ($2h$ chiều) và tạo ra concatenation mới ($2h$ chiều).

### 11.2 Shape analysis

Với $L=2$ tầng bidirectional, $h=32$:

```
Input X:          (T, batch, d=28)
Layer 1 (Bi):    (T, batch, 2h=64)  ← gấp đôi
Layer 2 (Bi):     (T, batch, 4h=128) ← gấp đôi lần nữa
Output O:         (T, batch, 4h=128) ← dùng tầng cuối
```

**Nhận xét:** Mỗi tầng bidirectional gấp đôi chiều. Với $L$ tầng, chiều cuối = $2^L \times h$ — có thể tăng rất nhanh!

---

## 12. So sánh Deep RNN và Bidirectional RNN

### 12.1 Bảng so sánh chi tiết

| Khía cạnh | Deep RNN | Bidirectional RNN |
|---|---|---|
| **Mục đích chính** | Biểu diễn phân cấp (hierarchical) | Ngữ cảnh hai chiều (bidirectional context) |
| **Chiều mở rộng** | Từ trên xuống dưới (stacking) | Từ hai phía vào (forward + backward) |
| **Input cần thiết** | Toàn bộ chuỗi để huấn luyện | Toàn bộ chuỗi để huấn luyện |
| **Inference real-time** | Được (với 1 tầng) | **Không được** |
| **Tham số** | $L \times (dh + h^2 + h)$ | $2 \times (dh + h^2 + h)$ |
| **Hidden state shape** | $h$ mỗi tầng | $2h$ mỗi tầng |
| **Ứng dụng** | Mọi bài toán sequence | Tagging, encoding, BERT |

### 12.2 Khi nào dùng cái nào?

```
Bài toán: Dịch máy (machine translation)
→ Dùng: Bidirectional Encoder (đọc câu nguồn 2 hướng)

Bài toán: Phân tích văn bản (phân cấp từ → đoạn → chương)
→ Dùng: Deep RNN

Bài toán: BERT-style pretraining
→ Dùng: Deep Bidirectional Encoder
```

### 12.3 Minh họa kết hợp Deep + Bidirectional

```python
nn.GRU(input_size=d,
       hidden_size=h,
       num_layers=3,      # 3 tầng
       bidirectional=True) # bidirectional

# Tầng 1: input d → output 2h
# Tầng 2: input 2h → output 4h
# Tầng 3: input 4h → output 8h
# Output: (T, batch, 8h)
```

---

## 13. Hạn chế của Bidirectional RNN

### 13.1 Không dùng được cho real-time

Bidirectional RNN **không thể xử lý streaming** vì:
- Forward RNN cần $\overrightarrow{H}_{t-1}$ (bước trước)
- Backward RNN cần $\overleftarrow{H}_{t+1}$ (bước sau)

Để dự đoán tại bước $t$, ta cần biết toàn bộ chuỗi — từ $x_1$ đến $x_T$.

### 13.2 Chi phí tính toán cao

| Khía cạnh | Unidirectional | Bidirectional |
|---|---|---|
| Forward pass | 1 lần qua chuỗi | 2 lần qua chuỗi |
| Tham số | $dh + h^2 + h$ | $2(dh + h^2 + h)$ |
| Memory | $O(T)$ | $O(T)$ |

---

# PHẦN III — BÀI TẬP (10.3.4 & 10.4.4)

---

## Bài 1: Thay GRU bằng LSTM trong Deep RNN

> _"Thay GRU bằng LSTM và so sánh accuracy và training speed."_

LSTM thường cho perplexity thấp hơn trong Deep RNN vì cell state giúp truyền gradient ổn định qua nhiều tầng. Tuy nhiên tốc độ chậm hơn vì nhiều tham số hơn.

## Bài 2: Nhiều hidden units cho hai hướng khác nhau

> _"Nếu hai hướng sử dụng số hidden units khác nhau, shape của $H_t$ thay đổi thế nào?"_

Nếu forward có $h_f$ units và backward có $h_b$ units:
- $\overrightarrow{H}_t \in \mathbb{R}^{n \times h_f}$
- $\overleftarrow{H}_t \in \mathbb{R}^{n \times h_b}$
- Concatenation: $H_t \in \mathbb{R}^{n \times (h_f + h_b)}$
- Tuy nhiên trong thực tế hiếm khi dùng vì phức tạp và ít cải thiện.

## Bài 3: Thiết kế Deep Bidirectional RNN

> _"Thiết kế một Bidirectional RNN với nhiều hidden layers."_

```python
class DeepBiGRU(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, num_layers, dropout=0):
        self.rnn = nn.GRU(
            num_inputs, num_hiddens,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout
        )
```

## Bài 4: Polysemy (từ đa nghĩa)

> _"Từ 'bank' có thể là ngân hàng hoặc bờ sông. Thiết kế model trả về vector representation phù hợp với ngữ cảnh."_

Bidirectional RNN phù hợp nhất — vì nắm bắt ngữ cảnh hai chiều:
- "I deposited money at the bank" → ngữ cảnh tài chính → vector gần concept MONEY
- "I sat by the bank of the river" → ngữ cảnh địa lý → vector gần concept RIVER

Đây chính là cách BERT tạo contextualized embeddings.

---

## Tổng kết

| Khía cạnh              | Nội dung                                                           |
| ---------------------- | ------------------------------------------------------------------ |
| **Deep RNN là gì**     | Stacking nhiều tầng RNN — mỗi tầng nhận output tầng dưới           |
| **Công thức cốt lõi**  | $H_t^{(l)} = \phi(H_t^{(l-1)}W_{xh} + H_{t-1}^{(l)}W_{hh} + b)$    |
| **Tại sao cần**        | Học biểu diễn phân cấp, quan hệ phức tạp hơn                       |
| **Siêu tham số**       | $L \in [1, 8]$, $h \in [64, 2056]$, lr thấp hơn                    |
| **BiRNN là gì**        | 2 unidirectional chạy ngược chiều, output = concatenation          |
| **Forward + Backward** | $\overrightarrow{H}_t$ (từ trái) + $\overleftarrow{H}_t$ (từ phải) |
| **Shape**              | Bidirectional → $2h$ chiều; Deep Bidirectional → $2^L \times h$    |
| **Hạn chế BiRNN**      | Cần toàn bộ chuỗi — không real-time                                |
| **Ứng dụng BiRNN**     | POS tagging, NER, BERT, machine translation                        |

---

> **Buổi trước:** [[Buổi 44 - Tuần 12]] — 10.2 Gated Recurrent Units (GRU)
> **Buổi sau:** [[Buổi 46 - Tuần 13]] — 10.5 Machine Translation and the Dataset

---

## Thuật ngữ

| Thuật ngữ | Tiếng Anh | Ghi chú |
|---|---|---|
| Deep RNN | Stacked RNN | Nhiều tầng RNN xếp chồng |
| Bidirectional RNN | BiRNN | Hai chiều forward + backward |
| Forward hidden state | $\overrightarrow{H}_t$ | Từ trái sang phải |
| Backward hidden state | $\overleftarrow{H}_t$ | Từ phải sang trái |
| Concatenation | Concatenation | Ghép nối hai vector |
| Stacking | Stacking | Xếp chồng tầng |
| Hidden layer depth | Độ sâu tầng ẩn | Số lượng tầng RNN |
| Inter-layer dependency | Phụ thuộc liên tầng | $H_t^{(l-1)} \to H_t^{(l)}$ |
| Intra-layer dependency | Phụ thuộc trong tầng | $H_{t-1}^{(l)} \to H_t^{(l)}$ |
