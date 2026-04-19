---
session: "D2L Tuần 12, Buổi 42 — 9.7 Backpropagation Through Time"
d2l_chapter: "9.7"
tags:
  - d2l
  - rnn
  - bptt
  - backpropagation
  - vanishing-gradient
  - exploding-gradient
  - truncation
  - gradient
aliases:
  - BPTT
  - Backpropagation Through Time
date: 2026-04-19
status: complete
---

# Buổi 42 — 9.7 Backpropagation Through Time (BPTT)

> **Mục tiêu:** Hiểu cơ chế lan truyền ngược gradient trong RNN — tại sao nó đặc biệt khó, tại sao gradient "biến mất" hoặc "phát nổ", và cách thực tế xử lý (truncation, gradient clipping). Đây là nền tảng lý thuyết giải thích **mọi hạn chế** của vanilla RNN và motivate cho LSTM/GRU ở Chương 10.

---

## Active Recall — Ôn lại Buổi 41 (9.6 Concise RNN)

### Câu hỏi (không nhìn tài liệu)

1. `nn.RNN` thực hiện phép tính gì? Viết công thức đầy đủ (bao gồm cả bias).
2. `nn.RNN` trả về bao nhiêu giá trị? Shape của từng giá trị là gì (với `batch=1024, T=32, h=32`)?
3. Tại sao `nn.RNN` có **2 bias** ($b_{ih}$, $b_{hh}$) trong khi scratch D2L chỉ có 1 bias $b_h$? Kết quả khác nhau không?
4. `nn.LazyLinear(vocab_size)` khác gì `nn.Linear(hidden_size, vocab_size)`? Ưu điểm?
5. Tại sao cần `swapaxes(0, 1)` trong `RNNLM.output_layer()`? Nếu quên thì sao?
6. Kể tên 4 thay đổi khi chuyển từ scratch sang high-level API.
7. cuDNN fused kernel giúp `nn.RNN` nhanh hơn scratch như thế nào? (hint: loại bỏ cái gì?)
8. Khi nào nên dùng scratch RNN thay vì `nn.RNN`? Cho ít nhất 2 trường hợp.
9. `nn.RNNCell` khác `nn.RNN` ở điểm nào? Dùng khi nào?
10. Perplexity và output text sau training 100 epochs scratch vs high-level khác nhau không? Tại sao?

### Tự trả lời

1. **Claim:** $H_t = \tanh(X_t W_{ih}^T + b_{ih} + H_{t-1} W_{hh}^T + b_{hh})$ → **Reasoning:** Elman RNN chuẩn, PyTorch dùng transposed weights và 2 bias riêng → **Evidence:** `nn.RNN` docs + Buổi 41 phần 2.1.
2. Trả 2 giá trị: `output` shape `(T, batch, h)` = `(32, 1024, 32)` — hidden states mọi step; `h_n` shape `(num_layers, batch, h)` = `(1, 1024, 32)` — hidden state cuối.
3. 2 bias vì convention PyTorch (tương thích cuDNN). Kết quả tương đương: $b_{ih} + b_{hh}$ collapse thành 1 effective bias. Chỉ **đếm params** khác: `nn.RNN` có $2h$ bias params, scratch có $h$.
4. `LazyLinear` **tự suy luận** `in_features` khi forward lần đầu → không cần biết `hidden_size` lúc init. Ưu điểm: code gọn, ít coupling, dễ thay đổi hidden size.
5. `nn.RNN` trả time-first `(T, batch, h)`, nhưng loss function cần batch-first `(batch, T, |V|)`. Nếu quên → shape mismatch → gradient sai → model không học.
6. (i) `RNNScratch` → `nn.RNN`, (ii) `W_hq, b_q` → `nn.LazyLinear`, (iii) list of tensors → stacked tensor, (iv) manual `clip_gradients()` → `Trainer(gradient_clip_val=1)`.
7. cuDNN loại bỏ **Python loop overhead** — fuse toàn bộ T steps thành 1 CUDA kernel call. Không cần chuyển control giữa Python/GPU mỗi step.
8. (a) Nghiên cứu/debug custom architecture; (b) Custom activation (không phải tanh/relu); (c) Muốn hiểu cơ chế khi học.
9. `nn.RNNCell` implement **1 step duy nhất**, bạn tự viết loop. `nn.RNN` implement **toàn bộ sequence**. Dùng `RNNCell` khi cần custom logic mỗi step (ví dụ: attention).
10. **Không khác nhau** đáng kể — cùng mô hình toán học, cùng hyperparameters → cùng kết quả. High-level chỉ nhanh hơn về tốc độ training.

### Concept notes cần ôn lại

- [[Recurrent Neural Network]]
- [[Gradient Clipping]]
- [[Perplexity]]
- [[Cross-Entropy Loss]]
- [[Backpropagation Through Time]]

---

# PHẦN I — TỔNG QUAN: TẠI SAO CẦN HIỂU BPTT?

---

## 1. Bối cảnh và Động lực

> [!NOTE] ELI5
> Hãy tưởng tượng bạn đang chơi trò "truyền tin" qua 100 người. Người cuối phải báo lại cho người đầu tiên rằng tin nhắn bị sai ở đâu (= gradient). Nhưng qua mỗi người, thông tin bị méo đi một chút. Qua 100 người, hoặc tin nhắn biến mất hoàn toàn (vanishing), hoặc bị khuếch đại thành tiếng hét đinh tai (exploding). BPTT chính là "luật chơi" của trò truyền tin này trong RNN.

**Backpropagation Through Time (BPTT)** là thuật toán tính gradient cho RNN. Nó là trường hợp đặc biệt của backpropagation thông thường, nhưng áp dụng lên computational graph **đã unroll theo thời gian**. Sự khác biệt cốt lõi so với feedforward networks:

- Trong MLP: gradient chỉ truyền qua $L$ layers (thường < 100)
- Trong RNN: gradient phải truyền qua $T$ **time steps**, với $T$ có thể lên tới hàng nghìn
- **Weight sharing:** cùng ma trận $W_{hh}$ được nhân lại $T$ lần → gradient là **tích** của $T$ Jacobians

> [!IMPORTANT] Tại sao section 9.7 quan trọng
> Section này **không có code mới** — nó hoàn toàn là lý thuyết. Nhưng nó giải thích:
>
> 1. Tại sao vanilla RNN không học được long-range dependencies (vanishing gradient)
> 2. Tại sao cần gradient clipping (exploding gradient) — đã dùng ở Buổi 40-41
> 3. Tại sao cần LSTM/GRU (Chương 10) — thiết kế để "sửa" vấn đề này
> 4. `detach_()` trong training code thực ra là truncated BPTT — không phải trick ngẫu nhiên

## 1.1 Tại sao gọi là "Backpropagation Through Time"?

Vì khi huấn luyện RNN, ta **mở vòng lặp theo thời gian** (unroll) thành một chuỗi gồm $T$ bản sao của cùng một cell:

- bước 1: $x_1 \to h_1 \to o_1$
- bước 2: $x_2$ đi cùng $h_1$ để tạo $h_2 \to o_2$
- ...
- bước $T$: $x_T$ đi cùng $h_{T-1}$ để tạo $h_T \to o_T$

Ở chiều thuận, thông tin đi **từ trái sang phải theo thời gian**. Ở chiều ngược, gradient đi **từ loss cuối về các bước trước đó**. Vì gradient phải chui qua cả chuỗi thời gian này, nên mới gọi là **lan truyền ngược xuyên thời gian**.

## 1.2 Từ điển các khái niệm xuất hiện trong BPTT

| Khái niệm                 | Nghĩa đơn giản                                                 | Vai trò trong BPTT                                              |
| ------------------------- | -------------------------------------------------------------- | --------------------------------------------------------------- |
| **Time step $t$**         | Vị trí thứ $t$ trong chuỗi                                     | Mỗi bước tạo ra 1 hidden state và có thể tạo 1 loss             |
| **Unroll**                | Mở vòng lặp RNN thành chuỗi các bước                           | Giúp ta nhìn RNN như một mạng rất sâu để áp dụng backprop       |
| **Hidden state $h_t$**    | "bộ nhớ tạm" tại thời điểm $t$                                 | Mang thông tin quá khứ sang hiện tại                            |
| **Shared weights**        | Cùng một bộ trọng số dùng lặp lại ở mọi bước                   | Là lý do gradient phải cộng đóng góp từ nhiều thời điểm         |
| **Loss $\ell(y_t,o_t)$**  | Mức sai ở bước $t$                                             | Tạo tín hiệu lỗi để cập nhật tham số                            |
| **Chain rule**            | Quy tắc đạo hàm theo chuỗi phụ thuộc                           | Công cụ toán học cốt lõi để truyền lỗi ngược                    |
| **Jacobian**              | Ma trận đạo hàm của vector theo vector                         | Đo xem hidden state sau nhạy với hidden state trước tới mức nào |
| **Long-range dependency** | Quan hệ giữa đầu ra hiện tại và thông tin rất xa trong quá khứ | Thứ mà vanilla RNN thường học rất kém                           |
| **Bias của gradient**     | Gradient xấp xỉ bị lệch so với gradient thật                   | Xuất hiện trong truncated BPTT                                  |
| **Variance cao**          | Mỗi lần ước lượng gradient dao động mạnh                       | Là nhược điểm của randomized truncation                         |

> [!NOTE] Phân biệt 2 chữ "backprop"
>
> - **Backpropagation** là quy trình lan truyền lỗi ngược để tính gradient.
> - **Gradient descent** là bước dùng gradient đó để cập nhật tham số.
>
> BPTT chỉ làm nhiệm vụ **tính gradient** cho RNN; còn optimizer như SGD/Adam mới là thứ **cập nhật trọng số**.

---

# PHẦN II — MÔ HÌNH ĐƠN GIẢN HÓA (9.7.1)

---

## 2. Simplified Model: Phân tích Gradient

### 2.1 Setup

D2L bắt đầu bằng mô hình đơn giản (không chỉ định activation cụ thể):

$$h_t = f(x_t, h_{t-1}, w_h) \tag{9.7.1}$$

$$o_t = g(h_t, w_o) \tag{9.7.2}$$

- $f$: hàm tính hidden state (bao gồm cả activation, ví dụ tanh)
- $g$: hàm tính output (linear + softmax)
- $w_h$, $w_o$: tham số (shared across time)

Loss trên toàn bộ sequence:

$$L = \frac{1}{T} \sum_{t=1}^{T} \ell(y_t, o_t) \tag{9.7.3}$$

### 2.2 Gradient đối với $w_o$ — Trường hợp dễ

$$\frac{\partial L}{\partial w_o} = \frac{1}{T} \sum_{t=1}^{T} \frac{\partial \ell(y_t, o_t)}{\partial w_o} = \frac{1}{T} \sum_{t=1}^{T} \frac{\partial \ell(y_t, o_t)}{\partial o_t} \cdot \frac{\partial g(h_t, w_o)}{\partial w_o}$$

Đơn giản vì $o_t$ chỉ phụ thuộc $w_o$ **trực tiếp** tại mỗi step — không có sự phụ thuộc qua thời gian. Tương tự MLP thông thường.

### 2.3 Gradient đối với $w_h$ — Trường hợp KHÓ

Đây là nơi mọi thứ phức tạp. $w_h$ ảnh hưởng đến $h_t$ tại **mọi** time step, và mỗi $h_t$ lại phụ thuộc vào $h_{t-1}$:

$$\frac{\partial L}{\partial w_h} = \frac{1}{T} \sum_{t=1}^{T} \frac{\partial \ell(y_t, o_t)}{\partial w_h} \tag{9.7.4}$$

Tại mỗi step $t$:

$$\frac{\partial \ell(y_t, o_t)}{\partial w_h} = \frac{\partial \ell(y_t, o_t)}{\partial o_t} \cdot \frac{\partial g(h_t, w_o)}{\partial h_t} \cdot \underbrace{\frac{\partial h_t}{\partial w_h}}_{\text{phần khó}} \tag{9.7.5}$$

**Vấn đề:** $\frac{\partial h_t}{\partial w_h}$ không đơn giản vì $h_t = f(x_t, h_{t-1}, w_h)$ — $w_h$ xuất hiện **cả trực tiếp** (qua $f$) **và gián tiếp** (qua $h_{t-1}$, mà $h_{t-1}$ cũng phụ thuộc $w_h$).

Áp dụng chain rule đệ quy:

$$\frac{\partial h_t}{\partial w_h} = \frac{\partial f(x_t, h_{t-1}, w_h)}{\partial w_h} + \frac{\partial f(x_t, h_{t-1}, w_h)}{\partial h_{t-1}} \cdot \frac{\partial h_{t-1}}{\partial w_h} \tag{9.7.6}$$

Mở rộng đệ quy đến tận $h_0$:

$$\frac{\partial h_t}{\partial w_h} = \sum_{\tau=0}^{t} \left(\prod_{i=\tau+1}^{t} \frac{\partial f(x_i, h_{i-1}, w_h)}{\partial h_{i-1}}\right) \frac{\partial f(x_\tau, h_{\tau-1}, w_h)}{\partial w_h} \tag{9.7.7}$$

> [!WARNING] Đây là công thức THEN CHỐT
> Công thức (9.7.7) cho thấy gradient tại step $t$ là **tổng** của $t+1$ số hạng, mỗi số hạng chứa **tích** của các Jacobians $\frac{\partial f}{\partial h}$. Tích này chính là nguyên nhân gây vanishing/exploding gradient:
>
> - Nếu $\left\|\frac{\partial f}{\partial h}\right\| < 1$ → tích → 0 khi $t$ lớn (vanishing)
> - Nếu $\left\|\frac{\partial f}{\partial h}\right\| > 1$ → tích → $\infty$ khi $t$ lớn (exploding)

### 2.4 Cách đọc công thức (9.7.7) theo ngôn ngữ đời thường

Đừng nhìn công thức rồi hoảng. Hãy đọc nó theo 3 ý sau:

1. **$w_h$ xuất hiện ở mọi time step** vì RNN dùng shared weights.
2. Muốn biết loss ở bước $t$ trách nhiệm lên $w_h$ thế nào, ta phải xét **mọi nơi trong quá khứ** mà $w_h$ đã tác động vào hidden state.
3. Từ một thời điểm cũ $\tau$ đi tới hiện tại $t$, tín hiệu lỗi phải đi qua từng mắt xích trung gian, nên xuất hiện tích:
   $$\prod_{i=\tau+1}^{t} \frac{\partial f(x_i,h_{i-1},w_h)}{\partial h_{i-1}}$$

Nói ngắn gọn: **mỗi đường đi ngược đóng góp một ít gradient**, và BPTT cộng tất cả các đường đi đó lại.

### 2.5 Ví dụ cực nhỏ để thấy vanishing/exploding xuất hiện thế nào

Giả sử ta bỏ hết phi tuyến và dùng mô hình 1 chiều:

$$h_t = w h_{t-1}, \quad h_3 = w^3 h_0$$

Khi đó:

$$\frac{\partial h_3}{\partial h_0} = w^3$$

- Nếu $w = 0.5$ thì $w^3 = 0.125$ — tín hiệu đã nhỏ đi rất mạnh chỉ sau 3 bước.
- Nếu $w = 2$ thì $w^3 = 8$ — tín hiệu bị phóng đại rất nhanh.

Đây chính là phiên bản tối giản của vấn đề trong RNN thực tế: thay vì một số $w$, ta có cả ma trận $W_{hh}$ và hiện tượng còn mạnh hơn nhiều khi chuỗi dài.

![[assets/attachments/d2l-buoi-42/gradient_chain.png]]
_Hình 1: Chuỗi nhân gradient trong BPTT — mỗi số hạng chứa tích Jacobian dài, gây vanishing/exploding._

---

## 3. Ba Chiến lược Tính Gradient

Công thức (9.7.7) cho thấy tính **full gradient** rất tốn kém: $O(T)$ thời gian và $O(T)$ bộ nhớ (phải lưu toàn bộ hidden states). D2L đề xuất 3 chiến lược:

### 3.1 Full Computation (Tính toán đầy đủ)

- **Cách làm:** Tính gradient theo đúng công thức (9.7.7) — truyền ngược qua **toàn bộ** $T$ steps
- **Ưu điểm:** Gradient chính xác
- **Nhược điểm:**
  - **$O(T)$ bộ nhớ** — phải lưu mọi intermediate states cho backward pass
  - **$O(T)$ thời gian** — mỗi step backward cần nhân thêm 1 Jacobian
  - **Chậm, không ổn định** — tích Jacobian dài → vanishing/exploding
- **Thực tế:** Hiếm khi dùng cho $T > 100$

### 3.2 Truncated BPTT (Cắt ngắn — Chiến lược chính trong thực tế)

> [!NOTE] ELI5
> Thay vì yêu cầu tin nhắn truyền ngược qua tất cả 100 người, ta chỉ cho nó truyền ngược qua $\tau$ người gần nhất (ví dụ: 20 người). Thông tin từ xa bị bỏ qua, nhưng ta tiết kiệm rất nhiều thời gian và bộ nhớ.

- **Cách làm:** Chỉ truyền gradient ngược $\tau$ steps (thay vì $T$):

$$\frac{\partial h_t}{\partial w_h} \approx \sum_{\tau=\max(0, t-\tau')}^{t} \left(\prod_{i=\tau+1}^{t} \frac{\partial f}{\partial h_{i-1}}\right) \frac{\partial f}{\partial w_h}$$

- **Ưu điểm:**
  - $O(\tau)$ bộ nhớ và thời gian — **kiểm soát được**
  - Tránh vanishing/exploding vì tích Jacobian ngắn
  - Có hiệu ứng **regularization** — giảm overfitting
- **Nhược điểm:**
  - **Biased** — gradient không chính xác, thiếu long-range dependencies
  - Phải chọn $\tau$ phù hợp (hyperparameter)

> [!IMPORTANT] Đây là chiến lược mặc định
> Khi bạn gọi `state.detach_()` trong training loop (Buổi 40), bạn đang thực hiện **truncated BPTT** với $\tau$ = `num_steps`. Cụ thể: `detach_()` cắt computational graph, ngăn gradient truyền ngược qua boundary.

### 3.3 Randomized Truncation (Cắt ngẫu nhiên)

- **Cách làm:** Tại mỗi step, quyết định có truyền gradient tiếp hay không bằng random variable $\xi_t$:

$$z_t = \frac{\partial f(x_t, h_{t-1}, w_h)}{\partial w_h} + \xi_t \frac{\partial f(x_t, h_{t-1}, w_h)}{\partial h_{t-1}} \cdot \frac{\partial h_{t-1}}{\partial w_h}$$

với $\xi_t \sim \text{Bernoulli}$ hoặc Geometric distribution, và **rescale** sao cho $E[z_t] = \frac{\partial h_t}{\partial w_h}$.

- **Ưu điểm:**
  - **Unbiased** — đúng khi lấy kỳ vọng
  - Thỉnh thoảng "may mắn" bắt được long-range dependencies
- **Nhược điểm:**
  - **Variance cao** — gradient rất noisy, training không ổn định
  - Khó tune — cần chọn distribution hợp lý

### 3.4 So sánh 3 chiến lược

![[assets/attachments/d2l-buoi-42/truncation_strategies.png]]
_Hình 2: So sánh 3 chiến lược truncation. Mũi tên đỏ = gradient flow backward. Mũi tên xám nét đứt = gradient bị cắt._

| Chiến lược         | Bias  | Variance | Bộ nhớ    | Thực tế                   |
| ------------------ | ----- | -------- | --------- | ------------------------- |
| Full               | Không | Thấp     | $O(T)$    | Không khả thi khi $T$ lớn |
| Truncated ($\tau$) | Có    | Thấp     | $O(\tau)$ | **Mặc định**              |
| Randomized         | Không | **Cao**  | Random    | Hiếm dùng                 |

> [!NOTE] Truncated BPTT thắng trong thực tế
> D2L nhấn mạnh: dù truncated BPTT có bias (không capture long-range), nó **ổn định hơn** và có hiệu ứng regularization. Randomized tuy unbiased nhưng variance quá cao → training khó hội tụ. Trong thực tế, hầu hết frameworks (PyTorch, TensorFlow) đều dùng truncated BPTT.

---

# PHẦN III — BPTT CHI TIẾT VỚI MA TRẬN (9.7.2)

---

## 4. Model tường minh (Identity Activation)

Để dẫn giải công thức rõ ràng, D2L dùng mô hình **đơn giản hóa** (identity activation, không bias):

$$h_t = W_{hx} x_t + W_{hh} h_{t-1} \tag{9.7.9}$$

$$o_t = W_{qh} h_t \tag{9.7.10}$$

> [!WARNING] Tại sao dùng identity thay vì tanh?
> Đây là **simplification có chủ đích** để công thức gradient trở nên tractable (dạng closed-form). Khi có tanh, Jacobian $\frac{\partial h_t}{\partial h_{t-1}}$ phức tạp hơn (phụ thuộc vào giá trị activation), nhưng **kết luận định tính** (vanishing/exploding) vẫn đúng.

Loss:

$$L = \frac{1}{T} \sum_{t=1}^{T} \ell(y_t, o_t) \tag{9.7.11}$$

### 4.1 Computational Graph

![[assets/attachments/d2l-buoi-42/bptt_computational_graph.png]]
_Hình 3: Computational graph của RNN unrolled qua 3 time steps. Forward pass (xanh), backward pass (đỏ nét đứt). $W_{hh}$ gây ra sự phụ thuộc ngang giữa các hidden states.\_

### 4.2 Gradient $\partial L / \partial W_{qh}$ — Output weights

$$\frac{\partial L}{\partial W_{qh}} = \sum_{t=1}^{T} \text{prod}\left(\frac{\partial L}{\partial o_t}, h_t^T\right) \tag{9.7.12}$$

Đơn giản vì $W_{qh}$ chỉ xuất hiện trong $o_t = W_{qh} h_t$ — không có dependency xuyên thời gian. Ở đây "prod" là phép nhân ma trận phù hợp (outer product).

### 4.3 Gradient $\partial L / \partial h_T$ — Hidden state cuối cùng

$$\frac{\partial L}{\partial h_T} = \text{prod}\left(\frac{\partial L}{\partial o_T}, W_{qh}\right) \tag{9.7.13}$$

$h_T$ chỉ ảnh hưởng đến $o_T$ (và không có step $T+1$ nào sau nó), nên gradient chỉ đến từ loss tại $T$.

### 4.4 Gradient $\partial L / \partial h_t$ — PHẦN QUAN TRỌNG NHẤT

Với $t < T$, $h_t$ ảnh hưởng đến **hai nơi**: (1) $o_t$ (trực tiếp), và (2) $h_{t+1}$ (qua $W_{hh}$). Áp dụng chain rule:

$$\frac{\partial L}{\partial h_t} = \text{prod}\left(\frac{\partial L}{\partial o_t}, W_{qh}\right) + \text{prod}\left(\frac{\partial L}{\partial h_{t+1}}, W_{hh}\right) \tag{9.7.14}$$

Mở rộng đệ quy (từ $T$ ngược về $t$):

$$\frac{\partial L}{\partial h_t} = \sum_{i=t}^{T} \left(W_{hh}^T\right)^{T-i} \cdot W_{qh}^T \cdot \frac{\partial L}{\partial o_i} \tag{9.7.15}$$

> [!WARNING] Đây là nơi vanishing/exploding xuất hiện
> Công thức (9.7.15) chứa $\left(W_{hh}^T\right)^{T-i}$ — **lũy thừa** của ma trận $W_{hh}$. Với eigenvalue decomposition $W_{hh} = Q \Lambda Q^T$:
>
> $$(W_{hh}^T)^k = Q \Lambda^k Q^T$$
>
> - Nếu eigenvalue $|\lambda_j| < 1$ → $\lambda_j^k \to 0$ (vanishing)
> - Nếu eigenvalue $|\lambda_j| > 1$ → $\lambda_j^k \to \infty$ (exploding)
>
> Đây chính là **lý do toán học** tại sao vanilla RNN không học được long-range dependencies.

![[assets/attachments/d2l-buoi-42/vanishing_exploding_gradient.png]]
_Hình 4: Vanishing vs Exploding gradient. Trái: eigenvalue < 1 → gradient giảm theo hàm mũ. Phải: eigenvalue > 1 → gradient tăng theo hàm mũ (lưu ý log scale)._

### 4.5 Gradient $\partial L / \partial W_{hx}$ và $\partial L / \partial W_{hh}$

$$\frac{\partial L}{\partial W_{hx}} = \sum_{t=1}^{T} \text{prod}\left(\frac{\partial L}{\partial h_t}, x_t^T\right) \tag{9.7.16a}$$

$$\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \text{prod}\left(\frac{\partial L}{\partial h_t}, h_{t-1}^T\right) \tag{9.7.16b}$$

Cả hai đều chứa $\frac{\partial L}{\partial h_t}$ (công thức 9.7.15) → **đều bị ảnh hưởng bởi vanishing/exploding**.

---

## 5. Tổng hợp: Roadmap Gradient trong BPTT

Tóm tắt toàn bộ gradient flow:

```
L = (1/T) Σ l(y_t, o_t)
    │
    ├── ∂L/∂W_qh = Σ (∂L/∂o_t) · h_t^T          ← ĐƠN GIẢN (không qua thời gian)
    │
    ├── ∂L/∂h_t = Σ (W_hh^T)^{T-i} · W_qh^T · (∂L/∂o_i)   ← CHỨA LŨY THỪA W_hh
    │       │
    │       ├── ∂L/∂W_hx = Σ (∂L/∂h_t) · x_t^T    ← KẾ THỪA vấn đề từ ∂L/∂h_t
    │       └── ∂L/∂W_hh = Σ (∂L/∂h_t) · h_{t-1}^T ← KẾ THỪA vấn đề từ ∂L/∂h_t
```

**Kết luận:** Gradient đối với $W_{qh}$ (output layer) **không bị** vanishing/exploding. Chỉ gradient đối với $W_{hh}$ và $W_{hx}$ (recurrent layer) mới bị — vì chúng phụ thuộc vào $\frac{\partial L}{\partial h_t}$ chứa lũy thừa $W_{hh}$.

---

# PHẦN IV — KẾT NỐI VỚI CODE THỰC TẾ

---

## 6. `detach_()` chính là Truncated BPTT

> [!NOTE] ELI5
> Mỗi khi bạn gọi `state.detach_()`, bạn đang "cắt dây điện thoại" giữa đoạn cũ và đoạn mới. Gradient không thể "gọi về" quá xa nữa — chỉ truyền được trong đoạn hiện tại.

Trong training code (Buổi 40-41):

```python
# Trong training loop
for X, Y in data_iter:
    if state is not None:
        state.detach_()  # ← TRUNCATED BPTT!
    y_hat, state = model(X, state)
    loss = cross_entropy(y_hat, Y)
    loss.backward()
    ...
```

**`state.detach_()`** làm gì:

1. **Cắt computational graph** tại boundary giữa 2 minibatches
2. Gradient từ minibatch hiện tại **không truyền ngược** qua minibatch trước
3. Hiệu quả: truncation length $\tau$ = `num_steps` (độ dài mỗi minibatch)

> [!NOTE] Một chỗ rất dễ hiểu nhầm
> `detach_()` **không xóa ký ức số học** trong hidden state. Giá trị của `state` vẫn được giữ lại để mô hình tiếp tục dùng ở bước sau. Thứ bị cắt chỉ là **đường lan truyền gradient** về quá khứ.
>
> Nói cách khác:
>
> - **forward memory** vẫn còn,
> - nhưng **backward credit assignment** bị dừng lại.

![[assets/attachments/d2l-buoi-42/detach_truncated_bptt.png]]
_Hình 5: Không detach → Full BPTT (gradient truyền vô hạn). Detach mỗi num_steps → Truncated BPTT (gradient dừng tại checkpoint xanh)._

> [!IMPORTANT] Trả lời câu hỏi "tại sao detach?"
> Buổi 40, ta dùng `detach_()` mà chưa giải thích tường minh. Bây giờ ta biết:
>
> 1. **Lý do thực tế:** Tiết kiệm bộ nhớ — không cần lưu graph toàn bộ corpus
> 2. **Lý do lý thuyết:** Truncated BPTT — gradient xấp xỉ nhưng ổn định hơn full BPTT
> 3. **Hiệu ứng phụ:** Regularization — model không overfit vào long-range patterns

### 6.1 Nếu không `detach_()`?

```python
# Giả sử KHÔNG detach
for X, Y in data_iter:
    # state giữ nguyên graph từ step trước
    y_hat, state = model(X, state)
    loss = cross_entropy(y_hat, Y)
    loss.backward()  # ← GRAPH NGÀY CÀNG DÀI → OOM!
```

- Mỗi forward step **nối thêm** vào graph cũ
- Sau vài nghìn steps → **Out of Memory**
- Gradient phải truyền qua toàn bộ → cực kỳ chậm + vanishing/exploding

---

# PHẦN V — VANISHING & EXPLODING GRADIENTS: PHÂN TÍCH SÂU

---

## 7. Eigenvalue Analysis

### 7.1 Từ lũy thừa ma trận đến eigenvalue

Với $W_{hh} \in \mathbb{R}^{h \times h}$, giả sử diagonalizable:

$$W_{hh} = Q \Lambda Q^{-1}$$

với $\Lambda = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_h)$ là ma trận eigenvalue, $Q$ là ma trận eigenvector.

Khi đó:

$$(W_{hh})^k = Q \Lambda^k Q^{-1} = Q \cdot \text{diag}(\lambda_1^k, \lambda_2^k, \ldots, \lambda_h^k) \cdot Q^{-1}$$

**Hành vi của $\lambda_j^k$: **

| Trường hợp | Hành vi khi $k \to \infty$ | Ý nghĩa cho gradient |
| ---------- | -------------------------- | -------------------- | ------------------------- | ----------------------------------------------------------- |
| Nếu $      | \lambda_j                  | < 1$                 | $\lambda_j^k \to 0$       | Thành phần gradient theo hướng eigenvector $j$ dần biến mất |
| Nếu $      | \lambda_j                  | = 1$                 | Độ lớn gần như giữ nguyên | Gradient tương đối ổn định                                  |
| Nếu $      | \lambda_j                  | > 1$                 | $\lambda_j^k \to \infty$  | Thành phần gradient theo hướng eigenvector $j$ phát nổ      |

### 7.2 Gradient bị "align" với eigenvector lớn nhất

Khi $k$ đủ lớn, $(W_{hh})^k$ bị **dominated** bởi eigenvalue lớn nhất $|\lambda_1|$:

$$(W_{hh})^k \approx \lambda_1^k \cdot q_1 q_1^T$$

Điều này có 2 hệ quả:

1. **Vanishing:** Nếu $|\lambda_1| < 1$, gradient đối với dependencies xa → 0. Model **chỉ học được short-range** patterns.
2. **Exploding:** Nếu $|\lambda_1| > 1$, gradient tăng exponential → training diverge. Gradient clipping là "band-aid" — cắt magnitude nhưng không sửa hướng.

### 7.3 Tại sao Gradient Clipping chỉ là giải pháp tạm?

> [!NOTE] ELI5
> Gradient clipping giống như đặt giới hạn tốc độ trên đường cao tốc. Xe (gradient) không vượt quá tốc độ X, tránh tai nạn (diverge). Nhưng nó không sửa được lý do xe muốn chạy nhanh — con đường (kiến trúc RNN) vẫn có vấn đề.

Gradient clipping (Buổi 40, $\theta = 1$):

$$\mathbf{g} \leftarrow \min\left(1, \frac{\theta}{\|\mathbf{g}\|}\right) \cdot \mathbf{g}$$

- **Xử lý exploding:** Có — cắt magnitude khi quá lớn
- **Xử lý vanishing:** **Không** — gradient đã = 0, clipping không giúp gì
- **Giải pháp thực sự:** Thay đổi **kiến trúc** → LSTM/GRU (Chương 10) với gating mechanism

---

# PHẦN VI — EXERCISES (9.7.4)

---

## 8. Phân tích Exercises

### Exercise 1: Eigenvalue analysis of $M^k$

> _"Show that eigenvalues $\lambda_i$ of an orthogonal matrix $M$ satisfy $|\lambda_i| = 1$."_

**Chứng minh:**

Nếu $M$ là ma trận trực giao: $M^T M = I$.

Gọi $v$ là eigenvector với eigenvalue $\lambda$: $Mv = \lambda v$.

$$\|Mv\|^2 = v^T M^T M v = v^T v = \|v\|^2$$

Mặt khác:

$$\|Mv\|^2 = \|\lambda v\|^2 = |\lambda|^2 \|v\|^2$$

Suy ra: $|\lambda|^2 = 1 \Rightarrow |\lambda| = 1$. $\square$

**Ý nghĩa cho RNN:** Nếu $W_{hh}$ là ma trận trực giao → eigenvalues đều có $|\lambda| = 1$ → gradient **không vanish và không explode**. Đây là motivation cho **Orthogonal RNN** (Arjovsky et al., 2016) — khởi tạo $W_{hh}$ trực giao và thêm constraint giữ nó gần trực giao trong quá trình training.

### Exercise 2: Gradient alignment

> _"Show that for a random vector $v$, $M^k v$ aligns with the eigenvector $v_1$ of $M$ corresponding to the largest eigenvalue."_

**Lập luận:**

Phân tích $v = \sum_i c_i v_i$ (trong cơ sở eigenvector).

$$M^k v = \sum_i c_i \lambda_i^k v_i$$

Chia cho $\lambda_1^k$ (eigenvalue lớn nhất):

$$\frac{M^k v}{\lambda_1^k} = c_1 v_1 + \sum_{i \neq 1} c_i \left(\frac{\lambda_i}{\lambda_1}\right)^k v_i$$

Vì $\left|\frac{\lambda_i}{\lambda_1}\right| < 1$ khi $i \neq 1$, các số hạng $\left(\frac{\lambda_i}{\lambda_1}\right)^k \to 0$. Suy ra:

$$M^k v \approx c_1 \lambda_1^k v_1 \quad \text{khi } k \gg 1$$

**Ý nghĩa:** Gradient trong RNN dài sẽ bị "kéo" về hướng eigenvector dominant → **mất thông tin** về các hướng khác. Đây là lý do vanilla RNN "quên" dependencies xa — gradient bị collapse về 1 chiều.

### Exercise 3: Các phương pháp khác ngoài gradient clipping

> _"Besides gradient clipping, can you think of any other methods to cope with gradient explosion in RNNs?"_

1. **Orthogonal/Unitary initialization:** Khởi tạo $W_{hh}$ trực giao → eigenvalues ban đầu $|\lambda| = 1$ → gradient ổn định (ít nhất ở giai đoạn đầu training)

2. **Gradient penalty (regularization):** Thêm $\alpha \|\nabla\|^2$ vào loss → penalize gradient lớn:
   $$L' = L + \alpha \|\nabla_w L\|^2$$

3. **Architecture changes:**
   - **LSTM:** Gate mechanism kiểm soát gradient flow (Chương 10)
   - **GRU:** Tương tự LSTM nhưng ít params hơn
   - **Residual connections:** Skip connections giúp gradient "đi tắt"

4. **Learning rate scheduling:** Giảm LR khi phát hiện gradient lớn → bước update nhỏ hơn

5. **Weight regularization:** L2 regularization trên $W_{hh}$ → ép eigenvalues về gần 0 → giảm explosion (nhưng tăng vanishing!)

---

## Tổng kết

| Aspect                | Takeaway                                                                                                     |
| --------------------- | ------------------------------------------------------------------------------------------------------------ |
| **BPTT là gì**        | Backprop thông thường áp dụng lên RNN unrolled theo thời gian                                                |
| **Vấn đề cốt lõi**    | Gradient chứa $W_{hh}^k$ → vanishing ($\|\lambda\|<1$) hoặc exploding ($\|\lambda\|>1$)                      |
| **Truncated BPTT**    | Chiến lược mặc định: chỉ truyền gradient $\tau$ steps. Biased nhưng ổn định                                  |
| **`detach_()`**       | Implementation của truncated BPTT trong PyTorch                                                              |
| **Gradient clipping** | Chỉ xử lý exploding, KHÔNG xử lý vanishing                                                                   |
| **Giải pháp thực sự** | LSTM/GRU (Chương 10) — gate mechanism kiểm soát gradient flow                                                |
| **Key formula**       | $\frac{\partial L}{\partial h_t} = \sum_{i=t}^{T} (W_{hh}^T)^{T-i} W_{qh}^T \frac{\partial L}{\partial o_i}$ |

---

> **Buổi trước:** [[Buổi 41 - Tuần 11]] — 9.6 Concise Implementation of RNNs
> **Buổi sau:** [[Buổi 43 - Tuần 12]] — 10.1 Long Short-Term Memory (LSTM)
