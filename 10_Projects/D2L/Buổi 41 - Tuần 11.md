---
session: "D2L Tuần 11, Buổi 41 — 9.6 Concise Implementation of Recurrent Neural Networks"
d2l_chapter: "9.6"
tags:
  - d2l
  - deep-learning
  - rnn
  - nn-rnn
  - high-level-api
  - pytorch
  - cudnn
  - language-model
  - concise-implementation
aliases:
  - RNN Concise Implementation
  - nn.RNN
date: 2026-04-16
status: complete
---

# Buổi 41 — 9.6 Concise Implementation of Recurrent Neural Networks

> **Nguồn:** [d2l.ai — 9.6](https://d2l.ai/chapter_recurrent-neural-networks/rnn-concise.html)
> **Buổi trước:** [[Buổi 40 - Tuần 11]] — 9.5 RNN Implementation from Scratch
> **Buổi sau:** [[Buổi 42 - Tuần 12]] — 9.7 Backpropagation Through Time

---

## Mục tiêu buổi học

1. **Thay thế RNN scratch** bằng `nn.RNN` — hiểu chính xác API thay đổi những gì
2. Hiểu **bên trong `nn.RNN`** — tại sao nhanh hơn scratch dù tính toán tương đương
3. Nắm **`nn.LazyLinear`** — thay thế output layer thủ công
4. Phân tích **4 thay đổi chính** khi chuyển từ scratch sang high-level API
5. Hiểu **`swapaxes(0, 1)`** — tại sao cần hoán trục trong output layer

---

## Active Recall — Kiến thức cũ (Buổi 40)

### Câu hỏi truy hồi (không nhìn tài liệu)

1. `RNNScratch` khởi tạo những tham số nào? Shape của mỗi tham số?
2. Phương thức `forward` của `RNNScratch` thực hiện vòng lặp gì? Trả về cái gì?
3. `one_hot()` trong `RNNLMScratch` biến input shape `(batch, T)` thành shape gì? Tại sao lại transpose?
4. `output_layer()` trong scratch thực hiện phép tính gì? Output shape?
5. Gradient Clipping: công thức chuẩn? Khi nào gradient bị "clip"? Tại sao dùng global norm thay vì per-parameter norm?
6. Training RNN: input `X` có shape gì? Target `Y` liên quan đến `X` như thế nào?
7. Decoding (text generation) gồm 2 phase — đó là gì? Tại sao cần warm-up phase?
8. Perplexity sau training 100 epochs (batch=1024, steps=32, hidden=32) đạt khoảng bao nhiêu? Output text trông như thế nào?
9. Tại sao không dùng teacher forcing khi decoding? Teacher forcing là gì?
10. Trong scratch, `rnn_outputs` là list of T tensors. Mỗi tensor có shape gì?

### Tự trả lời

1. **Claim:** $W_{xh} \in \mathbb{R}^{d \times h}$, $W_{hh} \in \mathbb{R}^{h \times h}$, $b_h \in \mathbb{R}^{1 \times h}$ → **Reasoning:** $W_{xh}$ nhận input $d$-dim, project sang hidden $h$-dim; $W_{hh}$ recurrence giữ thông tin cũ → **Evidence:** `RNNScratch.__init__` Buổi 40.
2. `for X in inputs:` — lặp qua $T$ time steps. Mỗi step: `state = tanh(X @ W_xh + state @ W_hh + b_h)`. Trả về `(outputs, state)` — outputs là list chứa $T$ hidden states.
3. `(batch, T)` → `F.one_hot(X.T, vocab_size).float()` → `(T, batch, |V|)`. Transpose vì RNN xử lý theo time-first: lặp `for X in inputs` lấy từng time step `(batch, |V|)`.
4. `[H @ W_hq + b_q for H in rnn_outputs]` → mỗi H shape `(batch, h)`, kết quả `(batch, |V|)`. Cuối cùng stack thành `(batch, T, |V|)`.
5. $\mathbf{g} \leftarrow \min\left(1, \frac{\theta}{\|\mathbf{g}\|}\right) \cdot \mathbf{g}$. Clip khi $\|\mathbf{g}\| > \theta$. Global norm vì: (a) giữ tỷ lệ gradient giữa layers, (b) per-param clip làm thay đổi hướng tối ưu.
6. `X`: `(batch, T)` — indices. `Y`: `(batch, T)` — dịch phải 1 ký tự so với `X`.
7. (a) **Warm-up**: feed `prefix` ký tự, không lấy output — chỉ để tạo hidden state có context. (b) **Generation**: dùng output step trước làm input step sau (autoregressive). Warm-up cần vì hidden state ban đầu = 0 — không có context.
8. PPL ≈ 8-10 sau 100 epochs. Output: "it has and the time traveller..." — bắt đầu coherent ở cấp từ nhưng chưa có ý nghĩa sâu.
9. Teacher forcing: dùng ground truth (thay vì prediction) làm input step sau. Không dùng khi decode vì không có ground truth. Chỉ dùng khi train.
10. Mỗi tensor shape `(batch, h)` — hidden state tại 1 time step.

### Concept notes cần ôn lại

- [[Recurrent Neural Network]]
- [[Gradient Clipping]]
- [[Perplexity]]
- [[One-Hot Encoding]]
- [[Cross-Entropy Loss]]

---

# PHẦN I — TỔNG QUAN (9.6)

---

## 1. Từ Scratch sang High-Level API: Bức tranh tổng thể

> [!NOTE] ELI5
> Buổi trước ta "nấu ăn từ đầu" — tự xay bột, tự nhào, tự nướng bánh. Hôm nay ta dùng "bột pha sẵn" (nn.RNN) — kết quả tương tự, nhưng nhanh hơn và ít lỗi hơn. Cái hay là ta đã hiểu "bên trong gói bột có gì" nhờ buổi trước.

Mục tiêu chính của D2L 9.6 rất đơn giản: **thay thế phần tự viết bằng API có sẵn của PyTorch**, giữ nguyên logic và kết quả. Section này ngắn vì phần lớn code được kế thừa (`RNNLMScratch`).

**4 thay đổi chính** khi chuyển từ scratch sang high-level:

| #   | Component         | Scratch (Buổi 40)                         | High-level (Buổi 41)                   |
| --- | ----------------- | ----------------------------------------- | -------------------------------------- |
| 1   | RNN core          | `RNNScratch` (3 tham số, Python loop)     | `nn.RNN` (cuDNN optimized)             |
| 2   | Output layer      | `W_hq`, `b_q` (manual `nn.Parameter`)     | `nn.LazyLinear(vocab_size)`            |
| 3   | Return format     | list of $T$ tensors, mỗi cái `(batch, h)` | tensor `(T, batch, h)` — đã stack sẵn  |
| 4   | Gradient clipping | `clip_gradients()` viết tay               | `Trainer(gradient_clip_val=1)` tự động |

![[assets/attachments/d2l-buoi-41/scratch_vs_highlevel.png]]
_Hình 1: So sánh kiến trúc tổng thể Scratch vs High-level. Viền đỏ nét đứt = phần thay đổi._

> [!IMPORTANT] Nguyên tắc "Hiểu trước, dùng API sau"
> D2L có lý do sắp xếp 9.5 (scratch) TRƯỚC 9.6 (concise): khi đã hiểu từng dòng code bên trong, việc dùng API trở thành lựa chọn có ý thức thay vì black box. Đây là phương pháp luận quan trọng trong DL engineering.

---

# PHẦN II — DEFINING THE MODEL (9.6.1)

---

## 2. `RNN` Class — Wrapper cho `nn.RNN`

> [!NOTE] ELI5
> Thay vì tự xây 3 cục gạch (W_xh, W_hh, b_h) rồi tự xếp thành tường, ta mua 1 cục bê-tông đúc sẵn (nn.RNN) — bên trong vẫn có 3 cục gạch đó, nhưng đã được đúc cùng nhau và tối ưu kết cấu.

### 2.1 Định nghĩa kỹ thuật

**`nn.RNN`** là module PyTorch built-in implement Elman RNN (vanilla RNN). Nó thực hiện **chính xác** phép tính:

$$H_t = \tanh(X_t W_{ih}^T + b_{ih} + H_{t-1} W_{hh}^T + b_{hh}) $$

- **Input:** `(T, batch, input_size)` — chuỗi input đã encoded (ví dụ: one-hot)
- **Output:**
  - `output`: `(T, batch, hidden_size)` — hidden states tại **mọi** time step (đã stack)
  - `h_n`: `(num_layers, batch, hidden_size)` — hidden state tại time step **cuối cùng**

> [!WARNING] Khác biệt nhỏ so với D2L notation
> PyTorch dùng **transposed weights** ($W^T$) và **2 bias riêng** ($b_{ih}$, $b_{hh}$) thay vì 1 bias $b_h$ như D2L. Kết quả tương đương vì $b_{ih} + b_{hh}$ tương đương $b_h$, và transpose tương ứng quy ước chiều khác nhau.

### 2.2 Implementation trong D2L

```python
class RNN(d2l.Module):
    """The RNN model implemented with high-level APIs."""
    def __init__(self, num_inputs, num_hiddens):
        super().__init__()
        self.save_hyperparameters()
        self.rnn = nn.RNN(num_inputs, num_hiddens)

    def forward(self, inputs, H=None):
        return self.rnn(inputs, H)
```

**Phân tích code:**

| Dòng                              | Ý nghĩa                                                                                                                      |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `nn.RNN(num_inputs, num_hiddens)` | Tạo RNN với input dim = `num_inputs`, hidden dim = `num_hiddens`. PyTorch tự khởi tạo $W_{ih}$, $W_{hh}$, $b_{ih}$, $b_{hh}$ |
| `forward(self, inputs, H=None)`   | `inputs`: `(T, batch, num_inputs)`, `H`: initial hidden state. Nếu `H=None` → tự tạo tensor zeros                            |
| `return self.rnn(inputs, H)`      | Trả về `(output, h_n)` — tuple gồm toàn bộ hidden states + hidden state cuối cùng                                            |

### 2.3 So sánh trực tiếp: RNNScratch vs RNN

```python
# === SCRATCH (Buổi 40) ===
class RNNScratch(nn.Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        # 3 tham số VIẾT TAY
        self.W_xh = nn.Parameter(
            torch.randn(num_inputs, num_hiddens) * sigma)
        self.W_hh = nn.Parameter(
            torch.randn(num_hiddens, num_hiddens) * sigma)
        self.b_h = nn.Parameter(torch.zeros(num_hiddens))

    def forward(self, inputs, state=None):
        if state is None:
            state = torch.zeros(inputs.shape[1],
                                self.num_hiddens, ...)
        outputs = []
        for X in inputs:  # Python loop qua T steps
            state = torch.tanh(
                X @ self.W_xh + state @ self.W_hh + self.b_h)
            outputs.append(state)
        return outputs, state

# === HIGH-LEVEL (Buổi 41) ===
class RNN(d2l.Module):
    def __init__(self, num_inputs, num_hiddens):
        super().__init__()
        self.rnn = nn.RNN(num_inputs, num_hiddens)
        # PyTorch tự tạo W_ih, W_hh, b_ih, b_hh

    def forward(self, inputs, H=None):
        return self.rnn(inputs, H)
        # 1 dòng thay thế cả Python loop
```

**Khác biệt cốt lõi:**

| Aspect            | Scratch                            | High-level                            |
| ----------------- | ---------------------------------- | ------------------------------------- |
| Tham số           | 3 `nn.Parameter` tự tạo            | PyTorch tự quản lý bên trong `nn.RNN` |
| Forward pass      | Python `for` loop qua $T$ steps    | 1 call duy nhất — cuDNN xử lý toàn bộ |
| Output format     | `list` of $T$ tensors `(batch, h)` | `tensor (T, batch, h)` — đã stack sẵn |
| Hidden state init | Tự tạo `torch.zeros`               | Tự động nếu `H=None`                  |
| Code lines        | ~15 dòng                           | ~5 dòng                               |

---

## 3. Bên trong `nn.RNN`: Tại sao nhanh hơn?

> [!NOTE] ELI5
> Tưởng tượng bạn giặt quần áo: scratch = giặt tay từng cái một (Python loop); nn.RNN = cho hết vào máy giặt (cuDNN kernel) — kết quả giống nhau, nhưng máy giặt tận dụng được cơ chế quay nhanh mà tay không làm được.

Mặc dù RNN về bản chất là **sequential** (step $t$ phụ thuộc step $t-1$), `nn.RNN` vẫn nhanh hơn scratch vì:

### 3.1 Tối ưu hóa cuDNN

1. **Fused CUDA kernel** — Thay vì $T$ lần gọi Python → GPU → Python → GPU..., cuDNN gộp toàn bộ $T$ steps thành 1 kernel launch duy nhất. Overhead Python bị loại bỏ hoàn toàn.

2. **Memory-efficient** — Không tạo $T$ Python objects trung gian. cuDNN quản lý buffer trên GPU trực tiếp.

3. **Optimized matrix operations** — Trong mỗi step, phép nhân ma trận được tối ưu bằng cuBLAS (batched GEMM), memory access pattern được coalesced.

4. **Pre-allocated buffers** — cuDNN pre-allocate output tensor `(T, batch, h)` 1 lần, thay vì append vào list rồi stack.

![[assets/attachments/d2l-buoi-41/nn_rnn_internals.png]]
_Hình 2: Scratch dùng Python loop tuần tự (T lần gọi GPU). nn.RNN fuse thành 1 cuDNN kernel duy nhất._

### 3.2 API Comparison chi tiết

```python
# SCRATCH: trả về list + state
outputs, state = rnn_scratch(inputs, state)
# outputs: list of T tensors, mỗi cái (batch, h)
# state: (batch, h)

# nn.RNN: trả về stacked tensor + state
output, h_n = nn_rnn(inputs, h_0)
# output: (T, batch, h) — ĐÃ STACK SẴN
# h_n: (num_layers, batch, h) — thêm dim num_layers
```

> [!IMPORTANT] Return format thay đổi → ảnh hưởng output_layer
> Scratch trả list → output_layer phải loop: `[H @ W_hq for H in outputs]`.
> nn.RNN trả tensor → output_layer chỉ cần 1 phép tính: `self.linear(hiddens)`.
> Đây chính là lý do `output_layer` cũng cần viết lại.

---

## 4. `RNNLM` Class — Language Model với High-Level API

> [!NOTE] ELI5
> Nếu `RNN` là "bộ não" đọc hiểu chuỗi ký tự, thì `RNNLM` là "người hoàn chỉnh" — có bộ não (RNN) cộng thêm "miệng" (output layer) để nói ra dự đoán. Buổi trước ta tự làm cả não lẫn miệng. Hôm nay ta thay não bằng nn.RNN, còn miệng bằng nn.LazyLinear.

### 4.1 Định nghĩa kỹ thuật

**`RNNLM`** kế thừa từ `RNNLMScratch` (Buổi 40), chỉ **override 2 methods**: `init_params()` và `output_layer()`. Mọi thứ khác — `one_hot()`, `forward()`, `predict()`, training loop — đều giữ nguyên.

### 4.2 Implementation

```python
class RNNLM(d2l.RNNLMScratch):
    """The RNN-based language model implemented with high-level APIs."""
    def init_params(self):
        self.linear = nn.LazyLinear(self.vocab_size)

    def output_layer(self, hiddens):
        return self.linear(hiddens).swapaxes(0, 1)
```

**Phân tích từng dòng:**

#### `init_params`: `nn.LazyLinear` thay thế `W_hq` + `b_q`

```python
# SCRATCH:
self.W_hq = nn.Parameter(
    torch.randn(num_hiddens, vocab_size) * sigma)  # (h, |V|)
self.b_q = nn.Parameter(torch.zeros(vocab_size))   # (|V|,)

# HIGH-LEVEL:
self.linear = nn.LazyLinear(self.vocab_size)
```

**`nn.LazyLinear`** là gì?

- Linear layer mà **không cần chỉ định `in_features`** (input dimension) khi khởi tạo
- PyTorch **tự suy luận** `in_features` từ dữ liệu đầu tiên đi qua (`forward` lần đầu)
- Ở đây: input sẽ có dim = `num_hiddens` (hidden state), PyTorch tự tạo weight `(num_hiddens, vocab_size)` + bias `(vocab_size,)`

> [!NOTE] Tại sao dùng Lazy?
> Vì `RNNLM` không trực tiếp biết `num_hiddens` — thông tin đó nằm bên trong `RNN`. Dùng `LazyLinear` tránh phải truyền `num_hiddens` một cách tường minh, giảm coupling giữa components.

#### `output_layer`: swapaxes explained

```python
# SCRATCH:
def output_layer(self, rnn_outputs):
    outputs = [torch.matmul(H, self.W_hq) + self.b_q
               for H in rnn_outputs]
    return torch.stack(outputs, dim=1)  # (batch, T, |V|)

# HIGH-LEVEL:
def output_layer(self, hiddens):
    return self.linear(hiddens).swapaxes(0, 1)
```

**Tại sao cần `swapaxes(0, 1)`?**

Đây là vấn đề **convention mismatch** giữa `nn.RNN` output và downstream loss:

```
nn.RNN output (hiddens): (T, batch, h)
          ↓ self.linear
      linear output:      (T, batch, |V|)
          ↓ swapaxes(0, 1)
      final output:       (batch, T, |V|)  ← LOSS CẦN FORMAT NÀY
```

**Chi tiết:**

1. `nn.RNN` trả `output` shape `(T, batch, h)` — **time-first** convention
2. `self.linear(hiddens)` áp dụng linear lên last dim → `(T, batch, |V|)`
3. Nhưng `cross_entropy` và phần còn lại của pipeline mong đợi `(batch, T, |V|)` — **batch-first**
4. `swapaxes(0, 1)` hoán đổi trục 0 (T) và trục 1 (batch) → `(batch, T, |V|)` ✓

> [!WARNING] Dễ quên swapaxes → shape mismatch
> Nếu quên `swapaxes`, loss function sẽ nhận `(T, batch, |V|)` thay vì `(batch, T, |V|)` → gradient tính sai → model không học được. Đây là bug phổ biến khi dùng nn.RNN.

---

## 5. So sánh Code: 4 Thay đổi chính

![[assets/attachments/d2l-buoi-41/code_comparison.png]]
_Hình 3: 4 thay đổi cụ thể khi chuyển từ Scratch sang High-level API._

### 5.1 Bảng tổng hợp chi tiết

| Thay đổi          | Scratch                               | High-level                                         | Lý do                                         |
| ----------------- | ------------------------------------- | -------------------------------------------------- | --------------------------------------------- |
| **RNN core**      | `RNNScratch`: 3 params + Python loop  | `nn.RNN`: cuDNN fused kernel                       | Performance + ít code                         |
| **Output layer**  | `W_hq`, `b_q` (manual `nn.Parameter`) | `nn.LazyLinear(vocab_size)`                        | Tự suy luận input dim, quản lý params tự động |
| **Return format** | `list[Tensor(batch, h)]` × $T$        | `Tensor(T, batch, h)` + `h_n(1, batch, h)`         | Pre-allocated, không cần stack                |
| **Grad clipping** | `clip_gradients()` viết tay           | `Trainer(gradient_clip_val=1)` / `clip_grad_norm_` | Framework quản lý, ít bug                     |

### 5.2 Phần KHÔNG thay đổi (kế thừa từ `RNNLMScratch`)

Đây là điểm hay của thiết kế OOP trong D2L — phần lớn logic **giữ nguyên**:

- `one_hot()` — vẫn `F.one_hot(X.T, vocab_size).float()`
- `forward()` — vẫn: one_hot → rnn → output_layer
- `predict()` — vẫn: warm-up → generation (autoregressive)
- Training loop — vẫn: forward → loss → backward → clip → step
- Loss function — vẫn: `cross_entropy(logits.reshape(-1, |V|), Y.reshape(-1))`

> [!NOTE] Thiết kế kế thừa
> D2L dùng inheritance (`RNNLM` kế thừa `RNNLMScratch`) để chỉ override phần cần thay đổi. Ưu điểm: tái sử dụng code, thấy rõ "chỉ 2 methods thay đổi". Nhược điểm: data flow không tường minh nếu chưa đọc class cha.

---

# PHẦN III — TRAINING AND PREDICTING (9.6.2)

---

## 6. Training: Cùng hyperparameters, cùng kết quả, nhanh hơn

### 6.1 Setup giống hệt Buổi 40

```python
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
rnn = RNN(num_inputs=len(data.vocab), num_hiddens=32)
model = RNNLM(rnn, vocab_size=len(data.vocab), lr=1)
trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
trainer.fit(model, data)
```

| Hyperparameter      | Giá trị | Giống Buổi 40? |
| ------------------- | ------- | -------------- |
| `batch_size`        | 1024    | ✓              |
| `num_steps`         | 32      | ✓              |
| `num_hiddens`       | 32      | ✓              |
| `lr`                | 1       | ✓              |
| `gradient_clip_val` | 1       | ✓              |
| `max_epochs`        | 100     | ✓              |

**100% giống** — đây là controlled experiment: chỉ thay implementation, giữ mọi thứ khác.

### 6.2 Kết quả

**Trước training:**

```
model.predict('it has', 20, data.vocab)
# → "it hasoadd dd dd dd dd dd "  (random garbage)
```

**Sau 100 epochs:**

```
model.predict('it has', 20, data.vocab)
# → "it has and the trave the t"
```

**So sánh với scratch (Buổi 40):**

| Metric          | Scratch                | High-level                   |
| --------------- | ---------------------- | ---------------------------- |
| Perplexity      | ~ tương đương          | ~ tương đương                |
| Training speed  | Chậm hơn (Python loop) | **Nhanh hơn** (cuDNN kernel) |
| Output quality  | Tương tự               | Tương tự                     |
| Code complexity | ~50 dòng               | ~10 dòng                     |

> [!IMPORTANT] "Comparable perplexity but runs faster"
> Đây là kết quả kỳ vọng. `nn.RNN` không thay đổi mô hình toán học — nó chỉ thay đổi **cách tính toán** (implementation). Nên kết quả phải tương đương. Nhanh hơn vì cuDNN optimization (xem phần 3).

---

## 7. Data Flow: Scratch vs High-level

![[assets/attachments/d2l-buoi-41/output_shape_flow.png]]
_Hình 4: Data flow và shape transformations. Viền đỏ nét đứt = node thay đổi giữa scratch và high-level._

### 7.1 Phân tích shape step-by-step

**Ví dụ cụ thể:** `batch=1024, T=32, |V|=28, h=32`

| Step          | Scratch Shape              | High-level Shape                           |
| ------------- | -------------------------- | ------------------------------------------ |
| Input X       | `(1024, 32)`               | `(1024, 32)`                               |
| After one_hot | `(32, 1024, 28)`           | `(32, 1024, 28)`                           |
| After RNN     | list of 32 × `(1024, 32)`  | `(32, 1024, 32)` + `(1, 1024, 32)`         |
| After output  | stack → `(1024, 32, 28)`   | `(32, 1024, 28)` → swap → `(1024, 32, 28)` |
| Loss input    | `(32768, 28)` + `(32768,)` | `(32768, 28)` + `(32768,)`                 |

Chú ý: $32768 = 1024 \times 32$ (batch × steps).

**Điểm khác biệt chính:** bước "After RNN" — scratch trả list, high-level trả stacked tensor. Bước "After output" — scratch stack thủ công, high-level cần `swapaxes` do convention mismatch.

---

# PHẦN IV — PHÂN TÍCH SÂU

---

## 8. Khi nào dùng Scratch vs High-Level?

| Tình huống                               | Nên dùng               | Lý do                                                            |
| ---------------------------------------- | ---------------------- | ---------------------------------------------------------------- |
| Production / training tốc độ cao         | High-level             | cuDNN nhanh gấp nhiều lần                                        |
| Nghiên cứu / custom RNN architecture     | Scratch                | Flexibility: sửa bất kỳ phần nào                                 |
| Học / hiểu cơ chế                        | Scratch trước          | Hiểu rồi mới dùng API → tránh black box                          |
| Debug gradient flow                      | Scratch                | Dễ đặt breakpoint, in intermediate states                        |
| Bidirectional / multi-layer / dropout    | High-level             | `nn.RNN(bidirectional=True, num_layers=2, dropout=0.5)` — 1 dòng |
| Custom activation (không phải tanh/relu) | Scratch / `nn.RNNCell` | `nn.RNN` chỉ hỗ trợ tanh và relu                                 |

> [!NOTE] `nn.RNNCell` — Giải pháp trung gian
> Nếu cần custom logic **bên trong** vòng lặp (ví dụ: attention mỗi step) nhưng vẫn muốn PyTorch quản lý params, dùng `nn.RNNCell` — nó implement 1 step duy nhất, bạn tự viết loop. Đây là trung gian giữa scratch hoàn toàn và `nn.RNN`.

---

## 9. `nn.RNN` Parameters deep-dive

```python
rnn = nn.RNN(input_size=28, hidden_size=32)
for name, param in rnn.named_parameters():
    print(name, param.shape)
```

Output:

```
weight_ih_l0  torch.Size([32, 28])   # W_ih: h × input
weight_hh_l0  torch.Size([32, 32])   # W_hh: h × h
bias_ih_l0    torch.Size([32])       # b_ih: h
bias_hh_l0    torch.Size([32])       # b_hh: h
```

**Tổng params:** $h \times d + h \times h + h + h = 32 \times 28 + 32 \times 32 + 32 + 32 = 896 + 1024 + 64 = 1984$

So sánh scratch: $d \times h + h \times h + h + h \times q + q$. Phần RNN core: $d \times h + h \times h + h = 28 \times 32 + 32 \times 32 + 32 = 1952$ (ít hơn 32 vì scratch chỉ có 1 bias).

> [!WARNING] Đếm params: 2 bias vs 1 bias
> `nn.RNN` mặc định có **2 bias** ($b_{ih}$ và $b_{hh}$) — tổng cộng $2h$ bias params. Scratch D2L chỉ có **1 bias** $b_h$ — tổng $h$. Kết quả tương đương vì $b_{ih} + b_{hh}$ collapse thành 1 effective bias. Nhưng **đếm params sẽ khác nhau** nếu so số chính xác.

---

## 10. Exercises (D2L 9.6.4) — Phân tích

### Exercise 1: Overfitting with high-level APIs?

> _"Can you make the model overfit using the high-level APIs?"_

**Trả lời: Có — bằng cách tăng model capacity hoặc giảm data.**

Cách overfit:

1. **Tăng `num_hiddens`**: 32 → 512 hoặc 1024. Capacity lớn → dễ memorize
2. **Dùng multi-layer RNN**: `nn.RNN(num_layers=3)` → nhiều tham số hơn
3. **Giảm data**: batch_size nhỏ, hoặc chỉ dùng 1 đoạn văn bản ngắn
4. **Tăng epochs**: train đủ lâu, model sẽ memorize training data

**Dấu hiệu overfit:** training loss tiếp tục giảm nhưng generated text bắt đầu "copy" đúng nguyên văn từ tập train thay vì generalize.

```python
# Ví dụ: dễ overfit
data = d2l.TimeMachine(batch_size=32, num_steps=64)    # nhỏ data
rnn = RNN(num_inputs=len(data.vocab), num_hiddens=512) # lớn model
model = RNNLM(rnn, vocab_size=len(data.vocab), lr=0.01)
trainer = d2l.Trainer(max_epochs=500)  # train lâu
```

### Exercise 2: Autoregressive model from 9.1 using RNN

> _"Implement the autoregressive model of Section 9.1 using an RNN."_

Section 9.1 định nghĩa autoregressive model: $P(x_t \mid x_{t-1}, \ldots, x_1)$ — dùng $\tau$ observations trước để dự đoán tiếp.

```python
class AutoregressiveRNN(nn.Module):
    """Autoregressive model dùng nn.RNN thay vì MLP."""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, h=None):
        # x: (batch, T, input_size) — chuỗi quan sát
        output, h_n = self.rnn(x, h)
        # output: (batch, T, hidden_size)
        pred = self.linear(output)  # (batch, T, output_size)
        return pred, h_n
```

**Khác biệt so với MLP autoregressive (9.1):**

- MLP: cần fix window size $\tau$ → giới hạn context
- RNN: hidden state mang thông tin **toàn bộ** lịch sử → không giới hạn context (về lý thuyết)
- MLP: mỗi step independent, RNN: sequential dependency qua hidden state

---

## Tổng kết

| Aspect              | Takeaway                                                            |
| ------------------- | ------------------------------------------------------------------- |
| **Core idea**       | `nn.RNN` thay thế scratch RNN — cùng toán, khác implementation      |
| **Performance**     | Nhanh hơn nhờ cuDNN fused kernel — loại bỏ Python loop overhead     |
| **Code reduction**  | ~50 dòng → ~10 dòng, nhưng cần hiểu scratch trước                   |
| **Key gotcha**      | `swapaxes(0, 1)` trong output_layer — time-first vs batch-first     |
| **LazyLinear**      | Tự suy luận input dim, tiện nhưng cần hiểu nó hoạt động thế nào     |
| **When to scratch** | Research, custom arch, learning. When high-level: production, speed |

---

> **Buổi trước:** [[Buổi 40 - Tuần 11]] — 9.5 RNN Implementation from Scratch
> **Buổi sau:** [[Buổi 42 - Tuần 12]] — 9.7 Backpropagation Through Time
