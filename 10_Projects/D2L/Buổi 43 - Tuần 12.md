---
session: "D2L Tuần 12, Buổi 43 — 10.1 Long Short-Term Memory (LSTM)"
d2l_chapter: "10.1"
tags:
  - d2l
  - deep-learning
  - rnn
  - lstm
  - gating
  - memory-cell
  - vanishing-gradient
  - sequence-model
aliases:
  - LSTM
  - Long Short-Term Memory
date: 2026-04-20
status: complete
---

# Buổi 43 — 10.1 Long Short-Term Memory (LSTM)

> **Nguồn:** [d2l.ai — 10.1](https://d2l.ai/chapter_recurrent-modern/lstm.html)
> **Buổi trước:** [[Buổi 42 - Tuần 12]] — 9.7 Backpropagation Through Time
> **Buổi sau:** [[Buổi 44 - Tuần 13]] — 10.2 Gated Recurrent Units (GRU)

---

## Active Recall — Ôn lại Buổi 42 (9.7 BPTT)

### Ôn lại từ gốc: các đối tượng trong RNN thực chất là gì?

> [!NOTE] Giải thích thật đơn giản
> Nếu coi chuỗi là một bộ phim, thì $x_t$ là một khung hình, $h_t$ là điều người xem còn nhớ sau khi xem đến khung hình đó, $o_t$ là dự đoán người xem đưa ra ở thời điểm đó, còn gradient là tín hiệu chỉ ra "ta đã nhớ sai ở đâu và phải sửa cách nhớ thế nào".

- **$x_t$**: đầu vào tại thời điểm $t$; có thể là ký tự, từ, embedding, hoặc vector đặc trưng.
- **$h_t$**: bản tóm tắt ngữ cảnh từ đầu chuỗi đến thời điểm $t$.
- **$o_t$**: đầu ra hoặc dự đoán tại bước $t$.
- **$y_t$**: nhãn đúng tại bước $t$.
- **loss**: đo độ sai giữa $o_t$ và $y_t$.
- **gradient**: tín hiệu sửa lỗi cho biết tham số nào đã làm mô hình dự đoán tệ đi.

### Câu hỏi (không nhìn tài liệu)

1. BPTT khác gì so với backpropagation trong mạng feedforward thông thường?
2. Viết công thức gradient $\frac{\partial h_t}{\partial w_h}$ dạng khai triển đệ quy. Giải thích tại sao nó chứa **tích** của các Jacobian.
3. Trong mô hình đơn giản hóa (không có hàm kích hoạt), gradient $\frac{\partial L}{\partial h_t}$ chứa thành phần $(W_{hh}^T)^{T-i}$. Tại sao thành phần này gây ra hiện tượng gradient biến mất hoặc bùng nổ?
4. Giải thích vai trò của phân tích trị riêng (eigenvalue) đối với $W_{hh}$. Khi nào gradient biến mất? Khi nào bùng nổ?
5. Kể tên 3 chiến lược tính gradient trong BPTT. Chiến lược nào được dùng trong thực tế?
6. Truncated BPTT có ưu điểm và nhược điểm gì so với full BPTT?
7. Lệnh `state.detach_()` trong PyTorch thực chất đang thực hiện chiến lược nào? Nó cắt cái gì?
8. Tại sao gradient clipping chỉ giải quyết được gradient bùng nổ mà không giải quyết được gradient biến mất?
9. Nếu $W_{hh}$ là ma trận trực giao thì trị riêng có tính chất gì? Điều đó giúp gì cho gradient?
10. D2L gợi ý rằng giải pháp thực sự cho gradient biến mất là gì? (Gợi ý: chương 10)

### Tự trả lời

1. **Khẳng định:** BPTT thực chất là backpropagation thường, nhưng áp dụng lên đồ thị tính toán đã "trải ra" theo thời gian → **Lập luận:** RNN dùng chung trọng số $W_{hh}$ ở mọi bước thời gian, nên gradient phải lan truyền ngược qua chuỗi $T$ bước, tạo ra tích Jacobian dài → **Bằng chứng:** Công thức (9.7.7) của D2L.
2. $\frac{\partial h_t}{\partial w_h} = \sum_{\tau=0}^{t} \left(\prod_{i=\tau+1}^{t} \frac{\partial f}{\partial h_{i-1}}\right) \frac{\partial f}{\partial w_h}$. Mỗi số hạng chứa tích Jacobian vì mỗi $h_i$ phụ thuộc $h_{i-1}$, và $w_h$ ảnh hưởng gián tiếp qua toàn bộ chuỗi.
3. Nếu trị riêng $|\lambda| < 1$ thì $|\lambda|^k \to 0$ khi $k$ lớn → gradient biến mất. Nếu $|\lambda| > 1$ thì $|\lambda|^k \to \infty$ → gradient bùng nổ. Đây là hệ quả trực tiếp của phép lũy thừa ma trận.
4. Phân tích trị riêng: $W_{hh} = Q\Lambda Q^{-1}$, nên $(W_{hh})^k = Q\Lambda^k Q^{-1}$. Hành vi phụ thuộc hoàn toàn vào $|\lambda_j|$ so với 1.
5. Ba chiến lược: (a) Full BPTT — chính xác nhưng tốn bộ nhớ $O(T)$; (b) Truncated BPTT — xấp xỉ, chỉ truyền ngược $\tau$ bước → **mặc định trong thực tế**; (c) Randomized truncation — không thiên lệch nhưng phương sai cao → hiếm dùng.
6. Ưu điểm: $O(\tau)$ bộ nhớ, ổn định, có hiệu ứng điều chuẩn. Nhược điểm: thiên lệch (không bắt được phụ thuộc xa), cần chọn $\tau$ phù hợp.
7. `detach_()` thực hiện truncated BPTT — nó cắt đồ thị tính toán tại ranh giới hai minibatch, ngăn gradient truyền ngược qua minibatch trước. Độ dài truncation = `num_steps`.
8. Gradient clipping chỉ cắt ngắn gradient khi nó quá lớn ($\|\mathbf{g}\| > \theta$). Khi gradient đã bằng 0 (biến mất), không có gì để cắt — clipping không giúp được.
9. Ma trận trực giao có trị riêng $|\lambda| = 1$. Gradient không biến mất cũng không bùng nổ — trạng thái lý tưởng.
10. Thay đổi kiến trúc — dùng cơ chế cổng (gating mechanism) trong LSTM và GRU để kiểm soát dòng gradient.

### Ghi chú khái niệm cần ôn lại

- [[Backpropagation Through Time]]
- [[Recurrent Neural Network]]
- [[Gradient Clipping]]
- [[Vanishing and Exploding Gradients]]

---

# PHẦN I — TỔNG QUAN: TẠI SAO CẦN LSTM?

---

## 1. Bối cảnh và Động lực

> [!NOTE] Giải thích đơn giản
> Hãy tưởng tượng bạn đang đọc một cuốn tiểu thuyết dài 500 trang. Ở trang 10, tác giả giới thiệu nhân vật chính tên Lan. Ở trang 490, tác giả nhắc lại "cô ấy" — bạn cần nhớ "cô ấy" là Lan.
>
> RNN thường giống như một người đọc sách nhưng bị mất trí nhớ — chỉ nhớ vài trang gần nhất. LSTM giống như người đọc sách **có sổ tay ghi chú**: họ quyết định ghi gì vào sổ (cổng đầu vào), xóa gì khỏi sổ (cổng quên), và khi nào mở sổ ra đọc (cổng đầu ra).

**Long Short-Term Memory (LSTM)** là kiến trúc mạng hồi quy được đề xuất bởi Hochreiter và Schmidhuber năm 1997, với mục tiêu giải quyết vấn đề gradient biến mất trong RNN thường.

**LSTM là gì?** LSTM thay thế mỗi nút ẩn của RNN thường bằng một **ô nhớ** (memory cell) phức tạp hơn. Ô nhớ này có **trạng thái nội bộ** (internal state) riêng, được bảo vệ bởi 3 cổng (gate) — cho phép mạng **chọn lọc** thông tin nào cần nhớ, cần quên, và cần xuất ra.

**Nó giải quyết vấn đề gì?** Như đã phân tích ở Buổi 42, gradient trong RNN thường chứa lũy thừa $W_{hh}^k$ — gây ra gradient biến mất khi $|\lambda| < 1$. LSTM giải quyết bằng cách tạo một "đường cao tốc" (highway) cho gradient — trạng thái ô nhớ $C_t$ được cập nhật bằng **phép cộng** (không phải phép nhân ma trận), nên gradient có thể truyền qua nhiều bước thời gian mà không bị suy giảm.

**Tại sao tên là "Long Short-Term Memory"?**

Cái tên này nghe mâu thuẫn nhưng rất có ý nghĩa:

| Loại bộ nhớ | Tương ứng trong RNN | Đặc điểm |
|---|---|---|
| Bộ nhớ dài hạn (long-term) | Trọng số $W$ | Thay đổi chậm trong quá trình huấn luyện |
| Bộ nhớ ngắn hạn (short-term) | Kích hoạt ẩn $H_t$ | Thay đổi liên tục mỗi bước thời gian |
| **Bộ nhớ "ngắn hạn dài"** | Trạng thái ô nhớ $C_t$ | **Trung gian** — tồn tại nhiều bước nhưng có thể bị xóa bất cứ lúc nào |

LSTM giới thiệu một loại bộ nhớ trung gian — tồn tại **lâu hơn** kích hoạt tức thời (short-term) nhưng **linh hoạt hơn** trọng số cố định (long-term). Đó là "Long Short-Term" Memory — bộ nhớ ngắn hạn nhưng có khả năng kéo dài.

> [!IMPORTANT] Lịch sử ngắn gọn
> - **1997:** Hochreiter & Schmidhuber đề xuất LSTM
> - **Giữa 2000:** LSTM thắng nhiều cuộc thi dự đoán chuỗi
> - **2011-2017:** LSTM là kiến trúc chủ đạo cho mọi bài toán chuỗi (dịch máy, nhận dạng giọng nói, sinh văn bản)
> - **2017 trở đi:** Transformer dần thay thế, nhưng nhiều ý tưởng cốt lõi của LSTM (cơ chế cổng, trạng thái ô nhớ) vẫn ảnh hưởng đến thiết kế Transformer

---

# PHẦN II — Ô NHỚ CÓ CỔNG (10.1.1)

---

## 2. Tổng quan kiến trúc ô nhớ

> [!NOTE] Giải thích đơn giản
> Mỗi ô nhớ LSTM giống như một **két sắt thông minh**:
> - **Cổng quên** = nút "xóa": quyết định xóa bao nhiêu phần trăm nội dung cũ trong két
> - **Cổng đầu vào** = nút "ghi": quyết định ghi bao nhiêu phần trăm thông tin mới vào két
> - **Cổng đầu ra** = nút "đọc": quyết định lấy bao nhiêu phần trăm nội dung két ra ngoài
>
> Cả 3 nút đều tự động điều chỉnh dựa trên dữ liệu đầu vào hiện tại và trạng thái trước đó.

**Ô nhớ có cổng (gated memory cell)** là thành phần cốt lõi của LSTM. Nó khác RNN thường ở chỗ: RNN thường chỉ có 1 trạng thái ẩn $H_t$, còn LSTM có **2 trạng thái** song song:

1. **Trạng thái ô nhớ** $C_t$ (cell state) — "bộ nhớ dài hạn", thông tin được bảo vệ và truyền qua nhiều bước
2. **Trạng thái ẩn** $H_t$ (hidden state) — "bộ nhớ làm việc", được xuất ra cho các tầng tiếp theo

Ba cổng kiểm soát dòng thông tin:

- **Cổng đầu vào** $I_t$: dữ liệu mới có quan trọng không? Có nên ghi vào ô nhớ không?
- **Cổng quên** $F_t$: ký ức cũ còn cần thiết không? Có nên xóa bớt không?
- **Cổng đầu ra** $O_t$: nội dung ô nhớ có nên ảnh hưởng đến output không?

> [!TIP] Phương pháp tiếp cận D2L: Xây dựng dần dần
> Thay vì trình bày toàn bộ LSTM cùng lúc, D2L sử dụng **4 sơ đồ tiến dần** (progressive diagrams) — mỗi sơ đồ thêm đúng một thành phần mới. Cách này giúp hiểu **tại sao** mỗi phần tồn tại trước khi chuyển sang phần tiếp theo:
> 1. **Hình 10.1.1:** Ba cổng (Forget, Input, Output) — "bộ điều khiển"
> 2. **Hình 10.1.2:** Thêm nút đầu vào (Input Node) — "nội dung mới"
> 3. **Hình 10.1.3:** Cập nhật trạng thái ô nhớ — "đường cao tốc ký ức"
> 4. **Hình 10.1.4:** Tính trạng thái ẩn — kiến trúc LSTM hoàn chỉnh

---

## 3. Trạng thái ẩn có cổng (10.1.1.1)

Điểm khác biệt cốt lõi giữa RNN thường và LSTM:

| Đặc điểm | RNN thường | LSTM |
|---|---|---|
| Trạng thái ẩn | $H_t = \tanh(X_t W_{xh} + H_{t-1} W_{hh} + b_h)$ | $H_t$ được tính qua nhiều bước có kiểm soát |
| Cập nhật | Ghi đè toàn bộ mỗi bước | **Chọn lọc**: giữ cái cũ + thêm cái mới |
| Cơ chế | Không có cổng | 3 cổng kiểm soát |
| Gradient | $W_{hh}^k$ → biến mất/bùng nổ | Đường $C_t$ → gradient ổn định |

D2L đặt ra 3 nhu cầu thực tế mà RNN thường không đáp ứng được:

1. **Nhớ lâu:** Nếu thông tin đầu tiên rất quan trọng (ví dụ: checksum ở đầu chuỗi), ta cần cơ chế **lưu giữ** nó suốt chuỗi → cổng quên gần 1
2. **Bỏ qua nhiễu:** Một số bước thời gian không chứa thông tin hữu ích (ví dụ: thẻ HTML trong phân tích cảm xúc) → cổng đầu vào gần 0
3. **Reset khi cần:** Khi có ranh giới logic (hết chương sách, đổi chủ đề) → cổng quên gần 0 để xóa ký ức cũ

---

## 4. Ba cổng: Đầu vào, Quên, và Đầu ra (10.1.1.2)

### 4.1 Đầu vào chung cho cả 3 cổng

Cả 3 cổng đều nhận cùng 2 đầu vào:
- $X_t \in \mathbb{R}^{n \times d}$: dữ liệu tại bước thời gian hiện tại
- $H_{t-1} \in \mathbb{R}^{n \times h}$: trạng thái ẩn của bước trước

(Trong đó $n$ = kích thước batch, $d$ = số chiều đầu vào, $h$ = số đơn vị ẩn)

### 4.2 Công thức 3 cổng

Cả 3 cổng đều có cấu trúc giống nhau: **tầng kết nối đầy đủ + sigmoid**.

$$I_t = \sigma(X_t W_{xi} + H_{t-1} W_{hi} + b_i) \tag{10.1.1a}$$

$$F_t = \sigma(X_t W_{xf} + H_{t-1} W_{hf} + b_f) \tag{10.1.1b}$$

$$O_t = \sigma(X_t W_{xo} + H_{t-1} W_{ho} + b_o) \tag{10.1.1c}$$

Trong đó:
- $W_{xi}, W_{xf}, W_{xo} \in \mathbb{R}^{d \times h}$: trọng số kết nối từ đầu vào
- $W_{hi}, W_{hf}, W_{ho} \in \mathbb{R}^{h \times h}$: trọng số kết nối từ trạng thái ẩn trước
- $b_i, b_f, b_o \in \mathbb{R}^{1 \times h}$: hệ số thiên lệch (bias dùng broadcasting)

> [!IMPORTANT] Tại sao dùng sigmoid?
> Sigmoid có miền giá trị $(0, 1)$ — hoàn hảo cho vai trò "công tắc dimmer" (điều chỉnh cường độ). Giá trị 0 = tắt hoàn toàn, giá trị 1 = mở hoàn toàn. Khi nhân phần tử với phần tử (Hadamard product) với tín hiệu cần kiểm soát, sigmoid hoạt động như bộ lọc: giữ lại $x\%$ thông tin.
>
> Xem thêm: [[Sigmoid Function]]

### 4.3 Minh họa trực quan: Ba cổng trong sơ đồ D2L

![[assets/attachments/d2l-buoi-43/lstm-0.png]]
_Hình 10.1.1 (D2L): Tính toán Cổng quên ($F_t$), Cổng đầu vào ($I_t$), và Cổng đầu ra ($O_t$)._

**Đọc sơ đồ — từng bước:**

1. **Phía dưới cùng:** Hai đầu vào $\mathbf{X}_t$ (Input) và $\mathbf{H}_{t-1}$ (Hidden state) được **nối (concatenate)** thành vector $[\mathbf{X}_t, \mathbf{H}_{t-1}]$ — ký hiệu mũi tên cong (⌒) trong sơ đồ.
2. **Sao chép 3 lần:** Vector nối được **copy** ra 3 bản (ký hiệu mũi tên rẽ nhánh ↕) và đưa vào 3 tầng FC song song.
3. **3 hộp xanh $\sigma$:** Mỗi hộp là một tầng FC (Fully Connected) + sigmoid. Output: vector $h$ chiều, mỗi phần tử $\in (0,1)$.
4. **Ba output phía trên:** $\mathbf{F}_t$ (Forget gate), $\mathbf{I}_t$ (Input gate), $\mathbf{O}_t$ (Output gate).

> [!NOTE] Quan sát quan trọng từ sơ đồ
> Ba cổng có **kiến trúc giống hệt nhau** — cùng đầu vào $[\mathbf{X}_t, \mathbf{H}_{t-1}]$, cùng FC+σ — chỉ khác bộ trọng số $(W, b)$. Mạng tự học bộ trọng số nào ứng với vai trò nào thông qua huấn luyện. Ở sơ đồ này, ta **chưa thấy** 3 cổng được dùng vào đâu — các sơ đồ tiếp theo sẽ cho thấy.

### 4.4 Hiểu trực quan từng cổng

**Cổng quên** $F_t$ — "Ký ức cũ có còn cần không?"

Ví dụ thực tế: Trong dịch máy, khi gặp dấu chấm kết thúc câu, cổng quên nên gần 0 để xóa ngữ cảnh câu cũ, chuẩn bị cho câu mới. Khi đang giữa câu, cổng quên nên gần 1 để giữ ngữ cảnh.

**Cổng đầu vào** $I_t$ — "Thông tin mới có quan trọng không?"

Ví dụ: Khi đọc văn bản và gặp từ "the" (mạo từ), cổng đầu vào nên gần 0 vì từ này ít mang ý nghĩa. Khi gặp tên riêng quan trọng, cổng đầu vào nên gần 1.

**Cổng đầu ra** $O_t$ — "Có nên xuất nội dung ô nhớ ra ngoài không?"

Ví dụ: Ô nhớ có thể đang lưu giới tính của chủ ngữ (nam/nữ) để dùng sau này cho đại từ. Nhưng ở bước hiện tại, thông tin đó chưa cần xuất → cổng đầu ra gần 0. Khi đến bước cần chọn "anh ấy" hay "cô ấy", cổng đầu ra mở ra.

---

## 5. Nút đầu vào — Ứng viên (10.1.1.3)

> [!NOTE] Giải thích đơn giản
> Nếu 3 cổng là "bộ điều khiển" (quyết định **bao nhiêu**), thì nút đầu vào là "nội dung" (quyết định **cái gì**). Nó tạo ra thông tin mới ứng viên — chờ cổng đầu vào quyết định có ghi vào ô nhớ hay không.

**Nút đầu vào** (input node) tạo ra giá trị ứng viên $\tilde{C}_t$ — thông tin mới **tiềm năng** có thể được thêm vào ô nhớ:

$$\tilde{C}_t = \tanh(X_t W_{xc} + H_{t-1} W_{hc} + b_c) \tag{10.1.2}$$

Trong đó:
- $W_{xc} \in \mathbb{R}^{d \times h}$, $W_{hc} \in \mathbb{R}^{h \times h}$: trọng số
- $b_c \in \mathbb{R}^{1 \times h}$: bias

> [!IMPORTANT] Tại sao dùng tanh thay vì sigmoid?
> - **Sigmoid** cho giá trị trong $(0, 1)$ — phù hợp làm "công tắc"
> - **Tanh** cho giá trị trong $(-1, 1)$ — phù hợp làm "nội dung" vì có thể **cộng hoặc trừ** giá trị vào ô nhớ
>
> Nếu dùng sigmoid cho ứng viên, ô nhớ chỉ có thể tăng (vì sigmoid luôn dương). Tanh cho phép cả tăng (+) và giảm (-), linh hoạt hơn nhiều.

**So sánh cấu trúc:** Nút đầu vào có cấu trúc giống hệt 3 cổng — cùng nhận $X_t$ và $H_{t-1}$, cùng dùng tầng kết nối đầy đủ — chỉ khác hàm kích hoạt (tanh thay vì sigmoid). Tổng cộng, LSTM có **4 "tầng con"** song song, tất cả đều xử lý cùng đầu vào.

### 5.1 Minh họa trực quan: Thêm nút đầu vào vào sơ đồ

![[assets/attachments/d2l-buoi-43/lstm-1.png]]
_Hình 10.1.2 (D2L): Tính toán nút đầu vào $\tilde{C}_t$. So với Hình 10.1.1, thêm đúng một hộp mới (tanh) giữa $I_t$ và $O_t$._

**So sánh với sơ đồ trước (Hình 10.1.1 → 10.1.2):**

- **Giống:** Vẫn cùng 2 đầu vào $\mathbf{X}_t$ và $\mathbf{H}_{t-1}$ được nối và sao chép.
- **Khác:** Bây giờ có **4 nhánh** thay vì 3 — thêm hộp **tanh** ($\tilde{C}_t$) nằm giữa $I_t$ và $O_t$.
- **Nhận diện nhanh:** 3 hộp σ (sigmoid) = 3 cổng kiểm soát "bao nhiêu". 1 hộp tanh = nút đầu vào tạo ra "nội dung gì".

> [!NOTE] Vị trí tanh trong sơ đồ
> D2L đặt $\tilde{C}_t$ ngay cạnh $I_t$ vì chúng sẽ được **nhân phần tử** trực tiếp với nhau: $I_t \odot \tilde{C}_t$ = "bao nhiêu" × "cái gì" = thông tin mới đã lọc. Sơ đồ tiếp theo sẽ cho thấy phép nhân này.

---

## 6. Trạng thái nội bộ ô nhớ (10.1.1.4)

> [!NOTE] Giải thích đơn giản
> Đây là bước "cập nhật sổ ghi chú":
> 1. Xóa bớt ghi chú cũ (cổng quên × ký ức cũ)
> 2. Viết thêm ghi chú mới (cổng đầu vào × ứng viên)
> 3. Kết quả = sổ ghi chú đã được cập nhật

Công thức cập nhật trạng thái ô nhớ là **trái tim** của LSTM:

$$C_t = F_t \odot C_{t-1} + I_t \odot \tilde{C}_t \tag{10.1.3}$$

Trong đó $\odot$ là phép nhân phần tử với phần tử (Hadamard product).

**Phân tích từng phần:**

| Thành phần | Ý nghĩa | Ví dụ |
|---|---|---|
| $F_t \odot C_{t-1}$ | "Giữ lại bao nhiêu ký ức cũ" | $F_t = 0.9$ → giữ 90% ký ức cũ |
| $I_t \odot \tilde{C}_t$ | "Thêm bao nhiêu thông tin mới" | $I_t = 0.1$ → chỉ thêm 10% ứng viên |
| $C_t$ | Ký ức mới = cũ đã lọc + mới đã lọc | Kết hợp cả hai |

### 6.1 Minh họa trực quan: Sơ đồ cập nhật ô nhớ

![[assets/attachments/d2l-buoi-43/lstm-2.png]]
_Hình 10.1.3 (D2L): Tính trạng thái nội bộ ô nhớ $C_t$. Đường ngang phía trên là "đường cao tốc" của ký ức._

**Đọc sơ đồ — tập trung đường ngang phía trên (cell state highway):**

1. **$\mathbf{C}_{t-1}$ đi vào từ trái:** Ký ức cũ (Memory cell internal state) từ bước thời gian trước.
2. **Nút ⊙ đầu tiên:** $C_{t-1}$ nhân phần tử với $F_t$ (cổng quên — mũi tên đi lên từ hộp σ bên trái) → "lọc bỏ ký ức không cần".
3. **Nút ⊙ ở giữa (phía dưới):** $I_t$ nhân phần tử với $\tilde{C}_t$ → "chọn lọc thông tin mới". Kết quả đi lên nút +.
4. **Nút + (phép cộng):** Ký ức cũ đã lọc + thông tin mới đã lọc → $\mathbf{C}_t$ đi ra bên phải.
5. **Cổng $O_t$ (hộp σ bên phải):** Đã được tính nhưng **chưa kết nối** vào đâu — nó sẽ được dùng ở Hình 10.1.4.

> [!NOTE] "Đường cao tốc" Cell State
> Đường ngang $C_{t-1} \to C_t$ chỉ qua phép **nhân phần tử** (⊙) rồi **cộng** (+) — không có nhân ma trận hay hàm phi tuyến nào. Khi $F_t \approx 1$ và $I_t \approx 0$: $C_t \approx C_{t-1}$ — thông tin truyền nguyên vẹn qua hàng trăm bước. Đây chính là cơ chế giúp LSTM giải quyết gradient biến mất, tương tự skip connection trong [[Residual Block|ResNet]].

### 6.2 Hai trường hợp đặc biệt

**Trường hợp 1: $F_t = 1$, $I_t = 0$ (giữ nguyên, không thêm mới)**

$$C_t = 1 \cdot C_{t-1} + 0 \cdot \tilde{C}_t = C_{t-1}$$

Ô nhớ giữ nguyên giá trị — thông tin được bảo toàn vô thời hạn. Đây chính là cơ chế giúp LSTM nhớ được phụ thuộc xa.

**Trường hợp 2: $F_t = 0$, $I_t = 1$ (xóa sạch, ghi mới hoàn toàn)**

$$C_t = 0 \cdot C_{t-1} + 1 \cdot \tilde{C}_t = \tilde{C}_t$$

Ô nhớ bị reset hoàn toàn — phù hợp khi gặp ranh giới logic (đổi chủ đề, hết câu).

> [!WARNING] Đây là lý do LSTM giải quyết gradient biến mất
> Trong RNN thường: $H_t = \tanh(W_{hh} H_{t-1} + ...)$ — trạng thái mới là **phép biến đổi phi tuyến** của trạng thái cũ. Gradient phải đi qua chuỗi nhân ma trận → biến mất.
>
> Trong LSTM: $C_t = F_t \odot C_{t-1} + ...$ — trạng thái mới là **phép cộng tuyến tính** với trạng thái cũ (nhân phần tử bởi $F_t$ rồi cộng). Khi $F_t \approx 1$, gradient truyền qua gần như nguyên vẹn — giống skip connection trong ResNet!

---

## 7. Trạng thái ẩn — Đầu ra cuối cùng (10.1.1.5)

Cuối cùng, ta cần tính trạng thái ẩn $H_t$ — giá trị mà các tầng tiếp theo "nhìn thấy":

$$H_t = O_t \odot \tanh(C_t) \tag{10.1.4}$$

**Hai bước:**
1. Áp dụng tanh lên trạng thái ô nhớ → ép giá trị về khoảng $(-1, 1)$
2. Nhân phần tử với cổng đầu ra → kiểm soát bao nhiêu phần trăm được xuất ra

> [!IMPORTANT] Tại sao cần tanh thêm một lần nữa?
> Trạng thái ô nhớ $C_t$ được cập nhật bằng phép cộng liên tục, nên giá trị có thể tích lũy và vượt ra ngoài khoảng $(-1, 1)$. Áp dụng tanh giúp:
> 1. **Chuẩn hóa** giá trị về khoảng $(-1, 1)$ — tránh lan truyền giá trị quá lớn
> 2. **Duy trì gradient** — tanh có vùng gradient mạnh quanh 0
>
> Lưu ý: tanh này chỉ áp dụng lên output ($H_t$), **không** ảnh hưởng đến $C_t$ — trạng thái ô nhớ vẫn được truyền nguyên bản qua "đường cao tốc".

**Tính chất quan trọng:** Ô nhớ có thể tích lũy thông tin qua nhiều bước mà không ảnh hưởng đến mạng (khi $O_t \approx 0$), rồi đột ngột "phát tín hiệu" khi cổng đầu ra mở ($O_t \approx 1$). Điều này cho phép LSTM "im lặng tích lũy" rồi "hành động đúng lúc".

### 7.1 Minh họa trực quan: Kiến trúc LSTM hoàn chỉnh

![[assets/attachments/d2l-buoi-43/lstm-3.png]]
_Hình 10.1.4 (D2L): Tính trạng thái ẩn $H_t$ — kiến trúc LSTM hoàn chỉnh._

**So sánh với sơ đồ trước (Hình 10.1.3 → 10.1.4) — thêm gì?**

Phía bên phải, $C_t$ được **copy** xuống dưới, đi qua hộp **tanh** (oval), rồi nhân phần tử (⊙) với cổng đầu ra $O_t$ → tạo ra $\mathbf{H}_t$. Bây giờ mọi thành phần đều đã kết nối.

**Đọc toàn bộ dòng dữ liệu — từ đầu vào đến đầu ra:**

```
Đầu vào: X_t (dưới), H_{t-1} (trái dưới), C_{t-1} (trái trên)

① [X_t, H_{t-1}] → concatenate → copy 4 lần
② Nhánh 1: FC + σ → F_t (cổng quên)
③ Nhánh 2: FC + σ → I_t (cổng đầu vào)
④ Nhánh 3: FC + tanh → C̃_t (ứng viên)
⑤ Nhánh 4: FC + σ → O_t (cổng đầu ra)
⑥ I_t ⊙ C̃_t → thông tin mới đã được lọc
⑦ C_{t-1} ⊙ F_t → ký ức cũ đã được lọc
⑧ (⑦) + (⑥) → C_t (trạng thái ô nhớ mới)
⑨ tanh(C_t) ⊙ O_t → H_t (trạng thái ẩn)

Đầu ra: C_t (phải trên), H_t (phải dưới)
```

**Hai đầu ra:**
- $\mathbf{C}_t$ → truyền cho bước thời gian tiếp theo (đi ngang sang phải phía trên)
- $\mathbf{H}_t$ → output cho tầng tiếp theo VÀ truyền cho bước thời gian tiếp theo (đi sang phải phía dưới)

> [!NOTE] 4 sơ đồ = 1 câu chuyện hoàn chỉnh
> Hình 10.1.1 → 10.1.4 kể câu chuyện từng bước: "3 van kiểm soát" → "thêm nội dung mới" → "lắp vào đường ống ký ức" → "nối đầu ra cho thế giới bên ngoài". Mỗi sơ đồ thêm đúng một lớp phức tạp — người đọc không bao giờ bị quá tải.

---

## 8. Tóm tắt công thức

![[assets/attachments/d2l-buoi-43/lstm_formulas_summary.png]]
_Hình 4: Tóm tắt 6 công thức cốt lõi của LSTM. Mỗi công thức tương ứng một vai trò cụ thể._

**Dòng dữ liệu hoàn chỉnh:**

```
Đầu vào: X_t, H_{t-1}, C_{t-1}
    │
    ├─→ Cổng quên:    F_t = σ(X_t W_xf + H_{t-1} W_hf + b_f)
    ├─→ Cổng đầu vào: I_t = σ(X_t W_xi + H_{t-1} W_hi + b_i)
    ├─→ Ứng viên:     C̃_t = tanh(X_t W_xc + H_{t-1} W_hc + b_c)
    ├─→ Cổng đầu ra:  O_t = σ(X_t W_xo + H_{t-1} W_ho + b_o)
    │
    ├─→ Cập nhật ô nhớ: C_t = F_t ⊙ C_{t-1} + I_t ⊙ C̃_t
    └─→ Trạng thái ẩn:  H_t = O_t ⊙ tanh(C_t)

Đầu ra: H_t (cho tầng tiếp theo), C_t (cho bước thời gian tiếp theo)
```

**Đếm tham số:** LSTM có **4 bộ trọng số** song song (cho 3 cổng + 1 ứng viên), mỗi bộ gồm $W_x \in \mathbb{R}^{d \times h}$, $W_h \in \mathbb{R}^{h \times h}$, $b \in \mathbb{R}^{1 \times h}$. Tổng: $4(dh + h^2 + h)$ tham số — gấp ~4 lần RNN thường ($dh + h^2 + h$).

---

# PHẦN III — CÀI ĐẶT TỪ ĐẦU (10.1.2)

---

## 9. Khởi tạo tham số (10.1.2.1)

```python
import torch
from torch import nn
from d2l import torch as d2l
```

```python
class LSTMScratch(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()

        init_weight = lambda *shape: nn.Parameter(
            torch.randn(*shape) * sigma)
        triple = lambda: (init_weight(num_inputs, num_hiddens),
                          init_weight(num_hiddens, num_hiddens),
                          nn.Parameter(torch.zeros(num_hiddens)))

        self.W_xi, self.W_hi, self.b_i = triple()  # Cổng đầu vào
        self.W_xf, self.W_hf, self.b_f = triple()  # Cổng quên
        self.W_xo, self.W_ho, self.b_o = triple()  # Cổng đầu ra
        self.W_xc, self.W_hc, self.b_c = triple()  # Nút đầu vào (ứng viên)
```

**Phân tích code:**

- `triple()` trả về 3 tensor: $W_x \in \mathbb{R}^{d \times h}$, $W_h \in \mathbb{R}^{h \times h}$, $b \in \mathbb{R}^{h}$
- Trọng số khởi tạo Gaussian $\mathcal{N}(0, 0.01^2)$ — giá trị nhỏ để tránh bão hòa sigmoid/tanh khi bắt đầu
- Bias khởi tạo bằng 0
- Tổng cộng 4 lần gọi `triple()` = 12 tensor tham số

> [!NOTE] Mẹo thực tế: bias của cổng quên
> Nhiều cài đặt thực tế khởi tạo $b_f = 1$ (thay vì 0) để cổng quên bắt đầu ở trạng thái "mở" — giữ lại ký ức cũ theo mặc định. Nếu $b_f = 0$, sigmoid ban đầu sẽ cho $F_t \approx 0.5$ — mất 50% ký ức ngay từ đầu, gây khó khăn cho việc học phụ thuộc xa. Xem: Jozefowicz et al. (2015).

---

## 10. Hàm forward (10.1.2.1 tiếp)

```python
@d2l.add_to_class(LSTMScratch)
def forward(self, inputs, H_C=None):
    if H_C is None:
        # Khởi tạo trạng thái ban đầu
        H = torch.zeros((inputs.shape[1], self.num_hiddens),
                        device=inputs.device)
        C = torch.zeros((inputs.shape[1], self.num_hiddens),
                        device=inputs.device)
    else:
        H, C = H_C

    outputs = []
    for X in inputs:
        # Bước 1: Tính 3 cổng
        I = torch.sigmoid(torch.matmul(X, self.W_xi) +
                          torch.matmul(H, self.W_hi) + self.b_i)
        F = torch.sigmoid(torch.matmul(X, self.W_xf) +
                          torch.matmul(H, self.W_hf) + self.b_f)
        O = torch.sigmoid(torch.matmul(X, self.W_xo) +
                          torch.matmul(H, self.W_ho) + self.b_o)

        # Bước 2: Tính ứng viên
        C_tilde = torch.tanh(torch.matmul(X, self.W_xc) +
                             torch.matmul(H, self.W_hc) + self.b_c)

        # Bước 3: Cập nhật trạng thái ô nhớ
        C = F * C + I * C_tilde

        # Bước 4: Tính trạng thái ẩn
        H = O * torch.tanh(C)

        outputs.append(H)

    return outputs, (H, C)
```

**Phân tích từng bước:**

1. **Khởi tạo:** Nếu không có trạng thái trước, tạo $H$ và $C$ bằng 0. Lưu ý: LSTM trả về **bộ đôi** $(H, C)$ thay vì chỉ $H$ như RNN thường.

2. **Vòng lặp thời gian:** `for X in inputs` — duyệt qua từng bước thời gian. `inputs` có shape `(T, batch, d)`.

3. **Tính song song 4 phép toán:** Trong thực tế, 4 phép nhân ma trận (3 cổng + 1 ứng viên) có thể được gộp thành 1 phép nhân lớn rồi tách ra — tối ưu hiệu suất GPU. D2L viết riêng cho dễ hiểu.

4. **Cập nhật:** `F * C` = Hadamard product (nhân phần tử) — mỗi chiều ẩn được kiểm soát **độc lập** bởi cổng tương ứng.

5. **Output:** Chỉ trả `outputs` (danh sách $H_t$) và trạng thái cuối $(H, C)$. Lưu ý: **chỉ $H_t$ được đưa vào tầng output**, $C_t$ hoàn toàn nội bộ.

> [!WARNING] So sánh với RNN thường
> ```python
> # RNN thường: 1 dòng, 1 trạng thái
> H = torch.tanh(X @ W_xh + H @ W_hh + b_h)
>
> # LSTM: 4 dòng, 2 trạng thái
> I = sigmoid(X @ W_xi + H @ W_hi + b_i)   # +3 dòng
> F = sigmoid(X @ W_xf + H @ W_hf + b_f)
> O = sigmoid(X @ W_xo + H @ W_ho + b_o)
> C_tilde = tanh(X @ W_xc + H @ W_hc + b_c)
> C = F * C + I * C_tilde                    # mới
> H = O * tanh(C)                            # mới
> ```
> LSTM phức tạp hơn nhiều nhưng đổi lại khả năng nhớ xa hơn đáng kể.

---

## 11. Huấn luyện và Dự đoán (10.1.2.2)

```python
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
lstm = LSTMScratch(num_inputs=len(data.vocab), num_hiddens=32)
model = d2l.RNNLMScratch(lstm, vocab_size=len(data.vocab), lr=4)
trainer = d2l.Trainer(max_epochs=50, gradient_clip_val=1, num_gpus=1)
trainer.fit(model, data)
```

**Lưu ý:**

- Dùng cùng bộ dữ liệu Time Machine, cùng cấu hình (`batch_size=1024`, `num_steps=32`, `num_hiddens=32`)
- `gradient_clip_val=1` — vẫn cần gradient clipping vì LSTM giảm thiểu nhưng **không loại bỏ hoàn toàn** vấn đề gradient bùng nổ
- Learning rate = 4 — tương đối cao, nhưng gradient clipping giữ cho cập nhật ổn định
- Tái sử dụng `RNNLMScratch` từ Buổi 40 — LSTM tương thích vì cùng giao diện (nhận inputs + state, trả outputs + state)

---

# PHẦN IV — CÀI ĐẶT GỌN VỚI API CAO CẤP (10.1.3)

---

## 12. Dùng `nn.LSTM`

```python
class LSTM(d2l.RNN):
    def __init__(self, num_inputs, num_hiddens):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = nn.LSTM(num_inputs, num_hiddens)

    def forward(self, inputs, H_C=None):
        return self.rnn(inputs, H_C)
```

**So sánh:**

| Khía cạnh | Cài đặt từ đầu | `nn.LSTM` |
|---|---|---|
| Dòng code | ~30 dòng | 5 dòng |
| Tham số | 12 tensor riêng lẻ | Tự quản lý bên trong |
| Tốc độ | Chậm (Python loop) | Nhanh (cuDNN fused kernel) |
| Trạng thái | Bộ đôi $(H, C)$ | Bộ đôi $(H, C)$ — giống |
| Tiện ích | Không | Hỗ trợ đa tầng, dropout giữa tầng |

```python
lstm = LSTM(num_inputs=len(data.vocab), num_hiddens=32)
model = d2l.RNNLM(lstm, vocab_size=len(data.vocab), lr=4)
trainer.fit(model, data)
```

Kết quả tương đương cài đặt từ đầu nhưng nhanh hơn đáng kể.

**Dự đoán:**

```python
model.predict('it has', 20, data.vocab, d2l.try_gpu())
# Kết quả: 'it has a the time travelly'
```

> [!NOTE] Khi nào dùng cài đặt nào?
> - **Cài đặt từ đầu:** Khi học, nghiên cứu, hoặc cần thay đổi kiến trúc bên trong (ví dụ: peephole connections, coupled forget-input gates)
> - **`nn.LSTM`:** Trong mọi trường hợp thực tế khác — nhanh hơn, ít lỗi hơn, tối ưu hóa bởi cuDNN

---

# PHẦN V — SO SÁNH RNN THƯỜNG VÀ LSTM

---

## 13. Tại sao LSTM giải quyết gradient biến mất?

> [!NOTE] Giải thích đơn giản
> Trong RNN thường, thông tin phải "đi bộ" qua từng bước thời gian — qua mỗi bước, nó bị biến đổi bởi phép nhân ma trận. Sau nhiều bước, thông tin bị méo hoàn toàn.
>
> LSTM xây một "đường cao tốc" song song (trạng thái $C_t$). Thông tin có thể "lái xe" trên đường cao tốc này mà chỉ bị ảnh hưởng nhẹ bởi cổng quên (nhân phần tử, không phải nhân ma trận). Gradient cũng "đi nhờ" đường cao tốc này.

**Phân tích toán học:**

Trong RNN thường, gradient lan truyền ngược qua:

$$\frac{\partial C_t}{\partial C_{t-1}} = W_{hh}^T \cdot \text{diag}(\text{tanh}'(...))$$

Đây là phép nhân ma trận — eigenvalue có thể < 1 hoặc > 1 → gradient biến mất hoặc bùng nổ.

Trong LSTM, gradient lan truyền qua trạng thái ô nhớ:

$$\frac{\partial C_t}{\partial C_{t-1}} = \text{diag}(F_t)$$

Đây chỉ là ma trận chéo với giá trị trong $(0, 1)$! Khi $F_t \approx 1$:

$$\frac{\partial C_t}{\partial C_{t-1}} \approx I \quad \text{(ma trận đơn vị)}$$

Gradient truyền qua gần như nguyên vẹn — **không biến mất**.

![[assets/attachments/d2l-buoi-43/rnn_vs_lstm_gradient.png]]
_Hình 5: So sánh gradient trong RNN thường (trái) và LSTM (phải). RNN thường: gradient phụ thuộc vào eigenvalue của $W_{hh}$ — dễ biến mất hoặc bùng nổ. LSTM: cổng quên kiểm soát tốc độ suy giảm — khi $F \approx 1$, gradient gần như ổn định._

---

## 14. Nhận xét tổng hợp

LSTM là mô hình tự hồi quy biến ẩn đầu tiên có cơ chế kiểm soát trạng thái phức tạp. Nhiều biến thể đã được đề xuất: kết nối dư (residual connections), điều chuẩn (regularization), đa tầng (stacking). Tuy nhiên, chi phí huấn luyện LSTM cao vì tính tuần tự — mỗi bước phải đợi bước trước hoàn thành. Sau này, kiến trúc [[Transformer Architecture|Transformer]] sẽ giải quyết hạn chế này bằng cơ chế chú ý (attention) cho phép tính toán song song.

---

# PHẦN VI — BÀI TẬP (10.1.5)

---

## 15. Phân tích bài tập

### Bài 1: Điều chỉnh siêu tham số

> _"Điều chỉnh các siêu tham số và phân tích ảnh hưởng đến thời gian chạy, perplexity, và chuỗi đầu ra."_

Các siêu tham số quan trọng cần thử nghiệm:

| Siêu tham số | Tăng lên | Giảm xuống |
|---|---|---|
| `num_hiddens` | Perplexity giảm (mô hình mạnh hơn), nhưng chậm hơn $O(h^2)$ | Nhanh hơn, nhưng perplexity cao hơn |
| `lr` | Hội tụ nhanh hơn ban đầu, nhưng có thể dao động | Hội tụ chậm nhưng ổn định |
| `num_steps` | Bắt được phụ thuộc xa hơn, nhưng tốn bộ nhớ | Ít bộ nhớ, nhưng bỏ lỡ phụ thuộc xa |
| `gradient_clip_val` | Cập nhật lớn hơn (rủi ro hơn) | Ổn định hơn nhưng hội tụ chậm |

### Bài 3: So sánh chi phí tính toán

> _"So sánh chi phí cho GRU, LSTM, và RNN thường với cùng số đơn vị ẩn."_

| Mô hình | Số phép nhân ma trận mỗi bước | Tham số (xấp xỉ) |
|---|---|---|
| RNN thường | 2 ($X W_{xh}$, $H W_{hh}$) | $dh + h^2 + h$ |
| GRU | 6 (3 cổng × 2) | $3(dh + h^2 + h)$ |
| LSTM | 8 (4 bộ × 2) | $4(dh + h^2 + h)$ |

LSTM tốn khoảng **4 lần** chi phí so với RNN thường. GRU tốn khoảng **3 lần** — đây là lý do GRU được ưa chuộng khi cần cân bằng hiệu quả và tốc độ.

### Bài 4: Tại sao cần tanh thêm một lần nữa cho $H_t$?

> _"Ứng viên $\tilde{C}_t$ đã dùng tanh đảm bảo giá trị trong $(-1, 1)$. Tại sao $H_t$ lại cần tanh nữa?"_

Vì trạng thái ô nhớ $C_t$ được tích lũy qua **phép cộng** nhiều bước:

$$C_t = F_t \odot C_{t-1} + I_t \odot \tilde{C}_t$$

Dù $\tilde{C}_t \in (-1, 1)$, phép cộng liên tục có thể khiến $C_t$ vượt ra ngoài khoảng $(-1, 1)$. Ví dụ: nếu $F_t = 0.9$ và $I_t = 0.5$ qua 100 bước, $C_t$ có thể lớn hơn 1. Tanh chuẩn hóa lại về $(-1, 1)$ trước khi xuất ra.

### Bài 5: Dự đoán chuỗi thời gian

> _"Cài đặt LSTM cho bài toán dự đoán chuỗi thời gian thay vì chuỗi ký tự."_

Thay đổi chính: đầu vào là giá trị liên tục (thay vì one-hot ký tự), tầng output dùng `nn.Linear` (thay vì softmax + cross-entropy), loss function dùng MSE.

---

## Tổng kết

| Khía cạnh | Nội dung |
|---|---|
| **LSTM là gì** | RNN có ô nhớ được kiểm soát bởi 3 cổng (quên, đầu vào, đầu ra) |
| **Giải quyết vấn đề gì** | Gradient biến mất — nhờ "đường cao tốc" $C_t$ cập nhật bằng phép cộng |
| **2 trạng thái** | $C_t$ (ô nhớ nội bộ) + $H_t$ (trạng thái ẩn cho output) |
| **Công thức cốt lõi** | $C_t = F_t \odot C_{t-1} + I_t \odot \tilde{C}_t$ |
| **Chi phí** | ~4 lần RNN thường (4 bộ trọng số song song) |
| **Hạn chế** | Huấn luyện tuần tự (chậm), nhiều tham số |
| **Kế thừa** | GRU (đơn giản hơn), Transformer (song song hơn) |

---

> **Buổi trước:** [[Buổi 42 - Tuần 12]] — 9.7 Backpropagation Through Time
> **Buổi sau:** [[Buổi 44 - Tuần 13]] — 10.2 Gated Recurrent Units (GRU)
