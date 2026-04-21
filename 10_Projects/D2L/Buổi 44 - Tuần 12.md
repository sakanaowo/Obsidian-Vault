---
session: "D2L Tuần 12, Buổi 44 — 10.2 Gated Recurrent Units (GRU)"
d2l_chapter: "10.2"
tags:
  - d2l
  - deep-learning
  - rnn
  - gru
  - gating
  - reset-gate
  - update-gate
  - modern-rnn
aliases:
  - GRU
  - Gated Recurrent Unit
date: 2026-04-20
status: in-progress
---

# Buổi 44 — 10.2 Gated Recurrent Units (GRU)

> **Nguồn:** [d2l.ai — 10.2](https://d2l.ai/chapter_recurrent-modern/gru.html)
> **Buổi trước:** [[Buổi 43 - Tuần 12]] — 10.1 Long Short-Term Memory (LSTM)
> **Buổi sau:** [[Buổi 45 - Tuần 12]] — 10.3 Deep Recurrent Neural Networks

---

## Active Recall — Ôn lại Buổi 43 (LSTM)

### Ôn lại từ gốc: LSTM giải quyết vấn đề gì?

> [!NOTE] Giải thích thật đơn giản
> LSTM như một người ghi chép thông minh: có sổ nháp ($C_t$) để lưu ký ức dài hạn, có 3 bút kiểm soát — bút xóa (cổng quên), bút ghi (cổng đầu vào), bút đọc (cổng đầu ra). Khi cổng quên mở gần hết, ký ức cũ truyền nguyên vẹn qua "đường cao tốc" $C_t$ — gradient không bị biến mất.

### Câu hỏi (không nhìn tài liệu)

1. LSTM có bao nhiêu cổng? Kể tên và vai trò của từng cổng.
2. Viết công thức cập nhật trạng thái ô nhớ $C_t$. Giải thích ý nghĩa từng thành phần.
3. Tại sao LSTM giải quyết được gradient biến mất mà RNN thường không?
4. Cổng đầu ra ($O_t$) dùng tanh trước khi nhân với cổng. Tại sao cần tanh?
5. LSTM có bao nhiêu trạng thái song song? Khác gì RNN thường?
6. Số tham số của LSTM gấp bao nhiêu lần RNN thường?
7. Khi $F_t \approx 1$ và $I_t \approx 0$, trạng thái ô nhớ $C_t$ thay đổi thế nào?
8. Tại sao bias của cổng quên thường được khởi tạo bằng 1 trong thực tế?
9. Hai trường hợp đặc biệt của $C_t$ là gì? Khi nào mỗi trường hợp xảy ra?
10. Điểm khác biệt chính giữa cài đặt từ đầu và `nn.LSTM` là gì?

### Tự trả lời

1. LSTM có **3 cổng**: cổng đầu vào $I_t$ (quyết định ghi bao nhiêu thông tin mới), cổng quên $F_t$ (quyết định xóa bao nhiêu ký ức cũ), cổng đầu ra $O_t$ (quyết định xuất bao nhiêu nội dung ô nhớ).
2. $C_t = F_t \odot C_{t-1} + I_t \odot \tilde{C}_t$ — ký ức cũ đã lọc cộng thông tin mới đã lọc.
3. RNN thường: $H_t = \tanh(W_{hh} H_{t-1} + ...)$ — gradient qua tích ma trận → biến mất. LSTM: $C_t$ cập nhật bằng phép cộng → gradient truyền qua đường gần như nguyên vẹn.
4. Vì $C_t$ tích lũy qua phép cộng liên tục, giá trị có thể vượt $(-1,1)$. Tanh chuẩn hóa về $(-1,1)$ trước khi xuất ra.
5. LSTM có **2 trạng thái song song**: $C_t$ (ô nhớ nội bộ) và $H_t$ (trạng thái ẩn xuất ra). RNN thường chỉ có $H_t$.
6. **4 lần** RNN thường — vì 4 bộ trọng số song song (3 cổng + 1 candidate).
7. $C_t \approx C_{t-1}$ — ô nhớ giữ nguyên, không cập nhật thêm.
8. Sigmoid(1) ≈ 0.73 → cổng quên bắt đầu ở trạng thái mở ~73%, giữ lại ký ức cũ theo mặc định, giúp học phụ thuộc xa dễ hơn (Jozefowicz et al., 2015).
9. (a) $C_t = C_{t-1}$ khi $F_t=1, I_t=0$ — giữ nguyên ký ức; (b) $C_t = \tilde{C}_t$ khi $F_t=0, I_t=1$ — reset hoàn toàn.
10. Scratch: tự quản lý 12 tensor tham số, Python loop → chậm. `nn.LSTM`: cuDNN fused kernel → nhanh hơn ~10 lần.

### Ghi chú khái niệm cần ôn lại

- [[Long Short-Term Memory]]
- [[Backpropagation Through Time]]
- [[Recurrent Neural Network]]
- [[Gradient Clipping]]
- [[Sigmoid Function]]

---

# PHẦN I — TỔNG QUAN: GRU LÀ GÌ?

---

## 1. Bối cảnh và Động lực

> [!NOTE] Giải thích đơn giản
> LSTM như một chiếc xe hơi đầy đủ: có ga (cổng đầu vào), phanh (cổng quên), côn (cổng đầu ra), và cả túi khí (cell state riêng). Nhưng xe đầy nặng, tốn xăng. GRU giống như phiên bản xe nhẹ hơn: bỏ túi khí, gộp ga và phanh thành một cần điều khiển (update gate), thêm một công tắc reset đơn giản. Kết quả: chạy gần bằng xe đầy mà tiết kiệm hơn nhiều.

**Gated Recurrent Unit (GRU)** là kiến trúc mạng hồi quy có cơ chế cổng, được đề xuất bởi **Cho et al. (2014)** trong paper "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation". GRU được thiết kế với mục tiêu **đơn giản hóa LSTM** — giữ lại ý tưởng cốt lõi về internal state và gating mechanism, nhưng bỏ bớt một số thành phần để tăng tốc độ tính toán.

**GRU là gì?** GRU gộp 3 cổng của LSTM thành **2 cổng** (reset gate và update gate), đồng thời **bỏ hoàn toàn cell state riêng** — chỉ giữ lại một hidden state $H_t$ duy nhất. Mọi cơ chế điều khiển đều nằm trong hidden state.

**Nó giải quyết vấn đề gì?** Giống LSTM: gradient biến mất trong RNN thường. Nhưng khác LSTM: dùng ít tham số hơn, huấn luyện nhanh hơn, và trong nhiều thực nghiệm đạt hiệu suất tương đương.

**Tại sao cần GRU khi đã có LSTM?**

LSTM với 3 cổng + cell state riêng rất mạnh nhưng phức tạp. Nghiên cứu của Chung et al. (2014) "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling" cho thấy GRU đạt hiệu suất tương đương LSTM trên nhiều bài toán (speech modeling, NLP) trong khi tính toán nhẹ hơn đáng kể.

> [!IMPORTANT] Lịch sử ngắn gọn
> - **2014:** Cho et al. đề xuất GRU trong paper về RNN Encoder-Decoder
> - **2014:** Chung et al. thực nghiệm so sánh GRU vs LSTM trên sequence modeling
> - **2015-2020:** GRU là lựa chọn phổ biến khi cần cân bằng hiệu quả và tốc độ
> - **2020 trở đi:** Transformer dần thay thế, nhưng GRU vẫn được dùng trong các ứng dụng nhẹ (on-device, real-time)

---

# PHẦN II — KIẾN TRÚC GRU (10.2.1–10.2.3)

---

## 2. Reset Gate và Update Gate (10.2.1)

> [!NOTE] Giải thích đơn giản
> Hãy tưởng tượng bạn đang viết một bài luận. Reset gate giống như nút "xóa bản nháp cũ" — quyết định có bắt đầu lại từ đầu hay giữ lại ý đã viết. Update gate giống như nút "ghi đè hoặc thêm" — quyết định có thay thế hoàn toàn hay chỉ bổ sung vào bản cũ.

### 2.1 Đầu vào chung cho cả 2 cổng

Tương tự LSTM, cả 2 cổng đều nhận cùng đầu vào:
- $X_t \in \mathbb{R}^{n \times d}$: dữ liệu tại bước thời gian hiện tại
- $H_{t-1} \in \mathbb{R}^{n \times h}$: trạng thái ẩn bước trước

### 2.2 Công thức 2 cổng

Cả 2 cổng đều có cấu trúc giống nhau: **tầng kết nối đầy đủ + sigmoid**.

$$R_t = \sigma(X_t W_{xr} + H_{t-1} W_{hr} + b_r) \tag{10.2.1a}$$

$$Z_t = \sigma(X_t W_{xz} + H_{t-1} W_{hz} + b_z) \tag{10.2.1b}$$

Trong đó:
- $W_{xr}, W_{xz} \in \mathbb{R}^{d \times h}$: trọng số kết nối từ đầu vào
- $W_{hr}, W_{hz} \in \mathbb{R}^{h \times h}$: trọng số kết nối từ hidden state trước
- $b_r, b_z \in \mathbb{R}^{1 \times h}$: bias

### 2.3 Minh họa trực quan: Hai cổng

![[assets/attachments/d2l-buoi-44/gru-1.png]]
_Fig 10.2.1 (D2L): Reset Gate ($R_t$) và Update Gate ($Z_t$) trong GRU. Cả hai đều nhận $[X_t, H_{t-1}]$ và trả về vector trong $(0, 1)^h$._

**Đọc sơ đồ — từng bước:**

1. **Phía dưới:** Hai đầu vào $X_t$ và $H_{t-1}$ được concatenate thành $[\mathbf{X}_t, \mathbf{H}_{t-1}]$.
2. **Copy 2 lần:** Vector nối được copy ra 2 bản và đưa vào 2 tầng FC song song.
3. **Hai hộp σ:** Mỗi hộp là tầng FC + sigmoid. Output: vector $h$ chiều, giá trị $\in (0, 1)$.
4. **Output:** $R_t$ (Reset Gate) và $Z_t$ (Update Gate).

> [!NOTE] Trực giác về 2 cổng
> **Reset Gate $R_t$**: "Ta có nên quên hết những gì ta nhớ từ trước không?" — Khi $R_t \to 0$, mạng quyết định bỏ qua ký ức cũ, bắt đầu "từ đầu".
>
> **Update Gate $Z_t$**: "Ta nên giữ bao nhiêu phần trăm của trạng thái cũ?" — Khi $Z_t \to 1$, ta giữ gần như nguyên trạng thái cũ. Khi $Z_t \to 0$, ta thay thế hoàn toàn bằng thông tin mới.

### 2.4 Hiểu sâu vai trò từng cổng

| Cổng | Câu hỏi | Khi → 0 | Khi → 1 |
|---|---|---|---|
| Reset Gate ($R_t$) | "Ký ức cũ còn cần không?" | Reset hoàn toàn — bắt đầu mới | Khôi phục RNN thường |
| Update Gate ($Z_t$) | "Giữ hay thay thế?" | Thay thế hoàn toàn bằng mới | Giữ nguyên trạng thái cũ |

---

## 3. Candidate Hidden State — Reset Gate hành động (10.2.2)

> [!NOTE] Giải thích đơn giản
> Reset gate giống như công tắc "bật/tắt tivi". Khi bật ($R_t \approx 1$): tivi hoạt động bình thường, hiển thị cả chương trình cũ lẫn mới. Khi tắt ($R_t \approx 0$): màn hình đen — tất cả nội dung cũ biến mất, chỉ có tín hiệu mới được hiển thị.

Candidate hidden state $\tilde{H}_t$ là trạng thái ẩn **tiềm năng** — được tính với sự điều chỉnh của reset gate:

$$\tilde{H}_t = \tanh(X_t W_{xh} + (R_t \odot H_{t-1}) W_{hh} + b_h) \tag{10.2.2}$$

**Điểm khác biệt quan trọng so với RNN thường:**

Trong RNN thường (Buổi 39, công thức 9.4.5):
$$H_t = \tanh(X_t W_{xh} + H_{t-1} W_{hh} + b_h)$$

Trong GRU:
$$\tilde{H}_t = \tanh(X_t W_{xh} + (R_t \odot H_{t-1}) W_{hh} + b_h)$$

**Reset gate nhân phần tử với $H_{t-1}$ trước khi đưa vào tầng FC!** Nếu $R_t \to 0$: phép nhân này xóa gần hết ảnh hưởng của trạng thái cũ, candidate chỉ phụ thuộc vào $X_t$ — tương đương khởi tạo lại hidden state. Nếu $R_t \to 1$: khôi phục hoàn toàn hành vi RNN thường.

### 3.1 Minh họa trực quan: Reset Gate điều khiển Candidate

![[assets/attachments/d2l-buoi-44/gru-2.png]]
_Fig 10.2.2 (D2L): Tính Candidate Hidden State $\tilde{H}_t$. Reset Gate $R_t$ nhân phần tử với $H_{t-1}$ trước khi cộng với $X_t W_{xh}$._

**Đọc sơ đồ — từng bước:**

1. **$X_t$ và $H_{t-1}$** đi vào từ hai phía.
2. **$R_t$** (đã tính ở bước trước) nhân phần tử với $H_{t-1}$ → **"ký ức đã được lọc"**.
3. **Concatenate** $[X_t, R_t \odot H_{t-1}]$ → đưa vào tầng FC.
4. **Hộp tanh** → tạo $\tilde{H}_t$ — candidate có giá trị trong $(-1, 1)^h$.

> [!NOTE] Tại sao tanh cho candidate?
> Tanh cho giá trị trong $(-1, 1)$ — cho phép cả tăng (+) và giảm (-) khi cộng vào hidden state. Sigmoid chỉ cho giá trị dương → không đủ linh hoạt.

---

## 4. Hidden State — Update Gate hành động (10.2.3)

> [!NOTE] Giải thích đơn giản
> Update gate giống như phép tính trung bình có trọng số. Nếu $Z_t = 0.8$: hidden state mới = 80% trạng thái cũ + 20% candidate mới. Nếu $Z_t = 0.2$: hidden state mới = 20% trạng thái cũ + 80% candidate mới. Đây là **tổ hợp lồi** (convex combination) — tức là ta luôn ở "đâu đó giữa" cũ và mới, không bao giờ nhảy ra ngoài.

Công thức cập nhật hidden state:

$$H_t = Z_t \odot H_{t-1} + (1 - Z_t) \odot \tilde{H}_t \tag{10.2.3}$$

### 4.1 Phân tích công thức

Công thức này là **tổ hợp lồi** (convex combination) giữa trạng thái cũ và candidate:

| Trường hợp | $Z_t$ | $H_t$ | Ý nghĩa |
|---|---|---|---|
| Giữ nguyên hoàn toàn | $Z_t \to 1$ | $H_t \approx H_{t-1}$ | Bỏ qua đầu vào hiện tại, nhảy qua bước này |
| Thay thế hoàn toàn | $Z_t \to 0$ | $H_t \approx \tilde{H}_t$ | Bỏ qua trạng thái cũ, dùng candidate mới |
| Trung gian | $0 < Z_t < 1$ | Tổ hợp lồi | Cân bằng giữa nhớ và học |

### 4.2 Minh họa trực quan: Update Gate hoàn thiện hidden state

![[assets/attachments/d2l-buoi-44/gru-3.png]]
_Fig 10.2.3 (D2L): Tính Hidden State $H_t$. Update Gate $Z_t$ điều khiển tổ hợp lồi giữa $H_{t-1}$ và $\tilde{H}_t$._

**Đọc sơ đồ — từng bước:**

1. **$Z_t$** nhân phần tử với $H_{t-1}$ → **"giữ lại bao nhiêu của trạng thái cũ"**.
2. **$(1 - Z_t)$** nhân phần tử với $\tilde{H}_t$ → **"thêm bao nhiêu của candidate mới"**.
3. **Phép cộng** hai thành phần → $H_t$ mới.
4. **Output:** $H_t$ vừa giữ được một phần ký ức cũ, vừa tích hợp thông tin mới.

> [!NOTE] Công thức giống ResNet!
> Công thức $H_t = Z_t \odot H_{t-1} + (1 - Z_t) \odot \tilde{H}_t$ tương tự residual connection: $H_t = H_{t-1} + \tilde{H}_t$. Khác ở chỗ: GRU có cổng điều khiển $Z_t$ quyết định tỉ lệ — linh hoạt hơn nhiều so với phép cộng cố định.

### 4.3 Hai tính chất đặc trưng của GRU

D2L tóm tắt hai đặc điểm cốt lõi:

> **Reset Gate giúp capture short-term dependencies** — khi cần nhớ chỉ vài bước gần nhất, reset gate đóng lại để bỏ qua ký ức xa.
>
> **Update Gate giúp capture long-term dependencies** — khi cần nhớ thông tin từ rất lâu, update gate giữ giá trị gần 1 để trạng thái cũ truyền qua nhiều bước.

---

## 5. So sánh LSTM và GRU

### 5.1 Bảng so sánh chi tiết

| Khía cạnh | LSTM | GRU |
|---|---|---|
| **Số cổng** | 3 (Input, Forget, Output) | 2 (Reset, Update) |
| **Cell State riêng** | Có ($C_t$ tách biệt) | Không (chỉ $H_t$) |
| **Trạng thái song song** | 2: $C_t$ và $H_t$ | 1: chỉ $H_t$ |
| **Tham số** | $4(dh + h^2 + h)$ | $3(dh + h^2 + h)$ |
| **Phép toán mỗi bước** | 8 phép nhân ma trận | 6 phép nhân ma trận |
| **Cổng đầu ra riêng** | Có ($O_t$) | Không (dùng chung $H_t$) |
| **Tốc độ huấn luyện** | Chậm hơn | Nhanh hơn ~25–30% |
| **Bộ nhớ** | Nhiều hơn | Ít hơn |

### 5.2 Công thức song song: LSTM vs GRU

| Mô hình | Trạng thái nội bộ | Trạng thái ẩn |
|---|---|---|
| **LSTM** | $C_t = F_t \odot C_{t-1} + I_t \odot \tilde{C}_t$ | $H_t = O_t \odot \tanh(C_t)$ |
| **GRU** | Không có (chỉ $H_t$) | $H_t = Z_t \odot H_{t-1} + (1-Z_t) \odot \tilde{H}_t$ |

**Quan sát quan trọng:** Công thức GRU hoàn toàn nằm trên hidden state — không có cell state riêng. Tất cả cơ chế "nhớ" và "quên" đều được encode trong cùng một vector $H_t$.

### 5.3 Khi nào dùng GRU? Khi nào dùng LSTM?

**Chọn GRU khi:**
- Cần huấn luyện nhanh, tài nguyên hạn chế
- Chuỗi ngắn đến trung bình (dưới ~500 bước)
- Ứng dụng real-time, on-device, mobile
- Muốn đơn giản hóa debugging

**Chọn LSTM khi:**
- Chuỗi rất dài và cần kiểm soát tinh vi
- Cần "tắt tiếng" một phần ký ức mà vẫn giữ phần khác (cổng đầu ra riêng)
- Cần độ ổn định gradient cao hơn (cell state riêng保护 tốt hơn)
- Benchmark shows LSTM outperforms on specific task

---

# PHẦN III — CÀI ĐẶT TỪ ĐẦU (10.2.4)

---

## 6. Khởi tạo tham số (10.2.4.1)

```python
import torch
from torch import nn
from d2l import torch as d2l
```

```python
class GRUScratch(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()

        init_weight = lambda *shape: nn.Parameter(
            torch.randn(*shape) * sigma)
        triple = lambda: (init_weight(num_inputs, num_hiddens),
                          init_weight(num_hiddens, num_hiddens),
                          nn.Parameter(torch.zeros(num_hiddens)))

        self.W_xz, self.W_hz, self.b_z = triple()  # Update Gate
        self.W_xr, self.W_hr, self.b_r = triple()  # Reset Gate
        self.W_xh, self.W_hh, self.b_h = triple()  # Candidate Hidden State
```

**Phân tích code:**

- `triple()` trả về 3 tensor: $W_x \in \mathbb{R}^{d \times h}$, $W_h \in \mathbb{R}^{h \times h}$, $b \in \mathbb{R}^{h}$
- Trọng số khởi tạo Gaussian $\mathcal{N}(0, 0.01^2)$
- **Tổng cộng 3 lần gọi `triple()`** = 9 tensor tham số (so với 12 của LSTM)
- Khác LSTM: không có bộ riêng cho cổng đầu ra và candidate cell

---

## 7. Hàm forward (10.2.4.2)

```python
@d2l.add_to_class(GRUScratch)
def forward(self, inputs, H=None):
    if H is None:
        # Initial state with shape: (batch_size, num_hiddens)
        H = torch.zeros((inputs.shape[1], self.num_hiddens),
                      device=inputs.device)
    outputs = []
    for X in inputs:
        Z = torch.sigmoid(torch.matmul(X, self.W_xz) +
                        torch.matmul(H, self.W_hz) + self.b_z)
        R = torch.sigmoid(torch.matmul(X, self.W_xr) +
                        torch.matmul(H, self.W_hr) + self.b_r)
        H_tilde = torch.tanh(torch.matmul(X, self.W_xh) +
                           torch.matmul(R * H, self.W_hh) + self.b_h)
        H = Z * H + (1 - Z) * H_tilde
        outputs.append(H)
    return outputs, H
```

**Phân tích từng bước:**

1. **Khởi tạo:** Nếu không có trạng thái trước, tạo $H$ bằng 0. Lưu ý: GRU chỉ trả **một hidden state** $H$ (không phải bộ đôi $(H, C)$ như LSTM).

2. **Vòng lặp thời gian:** `for X in inputs` — duyệt từng bước thời gian.

3. **Reset Gate:** `R = sigmoid(X @ W_xr + H @ W_hr + b_r)`

4. **Update Gate:** `Z = sigmoid(X @ W_xz + H @ W_hz + b_z)`

5. **Candidate:** `H_tilde = tanh(X @ W_xh + (R * H) @ W_hh + b_h)` — nhân phần tử R và H **trước** phép nhân ma trận với $W_{hh}$. Đây là điểm khác biệt cốt lõi với RNN thường.

6. **Hidden State mới:** `H = Z * H + (1 - Z) * H_tilde` — tổ hợp lồi.

> [!WARNING] So sánh với LSTM
> ```python
> # LSTM: 4 dòng, bộ đôi (H, C)
> I = sigmoid(X @ W_xi + H @ W_hi + b_i)
> F = sigmoid(X @ W_xf + H @ W_hf + b_f)
> O = sigmoid(X @ W_xo + H @ W_ho + b_o)
> C_tilde = tanh(X @ W_xc + H @ W_hc + b_c)
> C = F * C + I * C_tilde        # cell state update
> H = O * tanh(C)                 # hidden state có cổng riêng
>
> # GRU: 3 dòng, chỉ H
> Z = sigmoid(X @ W_xz + H @ W_hz + b_z)       # update gate
> R = sigmoid(X @ W_xr + H @ W_hr + b_r)       # reset gate
> H_tilde = tanh(X @ W_xh + (R * H) @ W_hh + b_h)  # candidate
> H = Z * H + (1 - Z) * H_tilde                  # hidden state
> ```
> GRU gọn hơn đáng kể nhưng vẫn đảm bảo cùng cơ chế gating.

---

## 8. Huấn luyện (10.2.4.3)

```python
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
gru = GRUScratch(num_inputs=len(data.vocab), num_hiddens=32)
model = d2l.RNNLMScratch(gru, vocab_size=len(data.vocab), lr=4)
trainer = d2l.Trainer(max_epochs=50, gradient_clip_val=1, num_gpus=1)
trainer.fit(model, data)
```

**Lưu ý:** Cùng bộ dữ liệu và hyperparameters với LSTM. Sử dụng lại `RNNLMScratch` vì GRU có cùng giao diện với RNN thường (nhận inputs + state, trả outputs + state).

### Kết quả huấn luyện

![[assets/attachments/d2l-buoi-44/gru-training.png]]
_Kết quả huấn luyện GRU từ đầu (scratch) trên bộ dữ liệu Time Machine. Loss giảm dần qua epochs, perplexity cải thiện._

---

# PHẦN IV — CÀI ĐẶT GỌN VỚI API CAO CẤP (10.2.5)

---

## 9. Dùng `nn.GRU`

```python
class GRU(d2l.RNN):
    def __init__(self, num_inputs, num_hiddens):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = nn.GRU(num_inputs, num_hiddens)

    def forward(self, inputs, H=None):
        return self.rnn(inputs, H)
```

```python
gru = GRU(num_inputs=len(data.vocab), num_hiddens=32)
model = d2l.RNNLM(gru, vocab_size=len(data.vocab), lr=4)
trainer.fit(model, data)
```

**So sánh:**

| Khía cạnh | Cài đặt từ đầu | `nn.GRU` |
|---|---|---|
| Dòng code | ~20 dòng | 5 dòng |
| Tốc độ | Chậm (Python loop) | Nhanh (cuDNN fused kernel) |
| Tham số | Tự quản lý 9 tensor | Tự quản lý bên trong |
| Tiện ích | Không | Hỗ trợ đa tầng, dropout |

**Dự đoán:**

```python
model.predict('it has', 20, data.vocab, d2l.try_gpu())
# Kết quả: 'it has so it and the time '
```

![[assets/attachments/d2l-buoi-44/gru-training-concise.png]]
_Kết quả huấn luyện GRU với `nn.GRU` (concise). Tốc độ nhanh hơn đáng kể nhờ cuDNN._

> [!NOTE] Khi nào dùng cài đặt nào?
> - **Cài đặt từ đầu:** Khi học, nghiên cứu, hoặc cần thay đổi kiến trúc bên trong (ví dụ: peephole connections, coupled gates)
> - **`nn.GRU`:** Trong mọi trường hợp thực tế — nhanh hơn, ít lỗi hơn

---

## 10. So sánh RNN thường, GRU, và LSTM

### 10.1 Trên cùng dataset

| Mô hình | Perplexity (sau 50 epochs) | Thời gian huấn luyện | Tham số |
|---|---|---|---|
| RNN thường (Buổi 40) | Cao | Nhanh nhất | $dh + h^2 + h$ |
| GRU (Buổi 44) | Thấp hơn | Trung bình | $3(dh + h^2 + h)$ |
| LSTM (Buổi 43) | Thấp nhất | Chậm nhất | $4(dh + h^2 + h)$ |

### 10.2 Gradient flow

| Mô hình | Gradient flow | Cơ chế |
|---|---|---|
| RNN thường | $W_{hh}^k$ → biến mất/bùng nổ | Phép nhân ma trận liên tiếp |
| GRU | $Z_t$ kiểm soát tỉ lệ giữ lại | Tổ hợp lồi + reset gate |
| LSTM | $F_t$ kiểm soát tỉ lệ giữ lại | Cell state + phép cộng |

---

# PHẦN V — BÀI TẬP (10.2.7)

---

## 11. Phân tích bài tập

### Bài 1: Giá trị tối ưu của 2 cổng

> _"Giả sử ta chỉ muốn dùng đầu vào tại bước $t'$ để dự đoán đầu ra tại bước $t > t'$. Giá trị tối ưu cho reset gate và update gate tại mỗi bước là gì?"_

**Đáp án:**

- **Update Gate $Z_\tau \to 1$** (với mọi $\tau$ từ $t'$ đến $t-1$): Khi $Z_\tau \to 1$, trạng thái $H_\tau \approx H_{\tau-1}$ — không có thông tin mới nào được thêm vào. Đến bước $t'$, trạng thái chứa đúng thông tin từ bước $t'$.

- **Reset Gate $R_{t'} \to 0$** (tại bước $t'$): Khi $R_{t'} \to 0$, candidate $\tilde{H}_{t'}$ chỉ phụ thuộc vào $X_{t'}$ — không bị ảnh hưởng bởi trạng thái cũ, đảm bảo candidate chỉ chứa thông tin từ $t'$.

### Bài 2: Điều chỉnh hyperparameters

Tương tự LSTM — tăng `num_hiddens` → perplexity giảm nhưng tốn thời gian; tăng `lr` → hội tụ nhanh hơn nhưng có thể dao động.

### Bài 3: So sánh RNN thường và GRU

```python
# RNN thường: 2 phép nhân ma trận
H = tanh(X @ W_xh + H @ W_hh + b)

# GRU: 6 phép nhân ma trận
Z = sigmoid(X @ W_xz + H @ W_hz + b_z)
R = sigmoid(X @ W_xr + H @ W_hr + b_r)
H_tilde = tanh(X @ W_xh + (R * H) @ W_hh + b_h)
H = Z * H + (1 - Z) * H_tilde
```

### Bài 4: Chỉ dùng một phần của GRU

- **Chỉ có Reset Gate** ($Z_t = 1$ luôn): $H_t = H_{t-1} + \tilde{H}_t$ — trạng thái cũ luôn được giữ lại hoàn toàn. Candidate được thêm vào nhưng không có cơ chế kiểm soát tỉ lệ. Tương tự phép residual connection cố định.

- **Chỉ có Update Gate** ($R_t = 1$ luôn): Phụ thuộc hoàn toàn vào update gate — khi $Z_t$ điều khiển toàn bộ, reset gate không ảnh hưởng. Khi $Z_t \to 0$, trạng thái reset về candidate.

---

## Tổng kết

| Khía cạnh                | Nội dung                                                    |
| ------------------------ | ----------------------------------------------------------- |
| **GRU là gì**            | LSTM đơn giản hóa — 2 cổng (Reset + Update), 1 hidden state |
| **Giải quyết vấn đề gì** | Gradient biến mất trong RNN, nhưng với ít tham số hơn       |
| **Reset Gate**           | Kiểm soát bao nhiêu ký ức cũ ảnh hưởng đến candidate        |
| **Update Gate**          | Kiểm soát tỉ lệ giữa trạng thái cũ và candidate mới         |
| **Công thức cốt lõi**    | $H_t = Z_t \odot H_{t-1} + (1 - Z_t) \odot \tilde{H}_t$     |
| **Tham số**              | $3(dh + h^2 + h)$ — ít hơn LSTM 25%                         |
| **Khi nào dùng**         | Chuỗi ngắn-trung bình, cần tốc độ, tài nguyên hạn chế       |
| **Kế thừa**              | LSTM (phức tạp hơn), Transformer (song song hơn)            |

---

> **Buổi trước:** [[Buổi 43 - Tuần 12]] — 10.1 Long Short-Term Memory (LSTM)
> **Buổi sau:** [[Buổi 45 - Tuần 12]] — 10.3 Deep Recurrent Neural Networks

---

## Thuật ngữ

| Thuật ngữ | Tiếng Anh | Ghi chú |
|---|---|---|
| Cổng reset | Reset Gate | Kiểm soát quên ký ức cũ |
| Cổng cập nhật | Update Gate | Kiểm soát giữ/thay thế |
| Trạng thái ẩn ứng viên | Candidate Hidden State | Nội dung mới tiềm năng |
| Tổ hợp lồi | Convex Combination | Trộn cũ và mới theo tỉ lệ |
| GRU đầy đủ | Fully Gated Unit | Phiên bản chuẩn của GRU |
| GRU tối thiểu | Minimal Gated Unit | Biến thể đơn giản hơn |
