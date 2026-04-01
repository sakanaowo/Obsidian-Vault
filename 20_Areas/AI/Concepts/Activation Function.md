---
title: "Activation Function"
aliases: [hàm kích hoạt, activation, nonlinearity, phi tuyến, nút bẻ cong]
tags: [concept, deep-learning, neural-network, activation]
created: 2026-03-24
modified: 2026-03-25
---

# Activation Function (Hàm kích hoạt)

> [!NOTE] ELI5
> Mạng neuron hoạt động bằng cách **nhân** rồi **cộng** — tức là chỉ kẻ đường thẳng. Nhưng thế giới thật không thẳng: **nhiệt cơ thể** 37°C bình thường, cao hơn HOẶC thấp hơn đều nguy hiểm — đường thẳng không biểu diễn được điều này.
>
> **Activation function** = "nút bẻ cong". Sau mỗi lần nhân + cộng, ta đưa kết quả qua hàm này để **bẻ cong** đường thẳng → mạng có thể vẽ được đường cong, đường gấp khúc, bất kỳ hình gì.

> [!question]- ❓ Bài này nói "mạng", "tầng" — MLP là gì?
> **MLP** ([[Multilayer Perceptron]]) = mạng neuron có **nhiều tầng xếp chồng**:
> 
> ```
> Input (dữ liệu thô) → Tầng ẩn 1 → Tầng ẩn 2 → ... → Output (kết quả)
> ```
> 
> Mỗi "tầng" nhận dữ liệu từ tầng trước, **nhân với trọng số** + **cộng bias** → ra kết quả → chuyển cho tầng sau.
> 
> **Activation function** nằm **giữa các tầng** — nó là lý do mạng nhiều tầng mạnh hơn 1 tầng. Không có activation → nhiều tầng = 1 tầng (xem phần 1 bên dưới).
>
> Đọc chi tiết: [[Multilayer Perceptron]]

---

## 1. Tại sao cần Activation Function?

### Vấn đề: Chồng tuyến tính = vẫn tuyến tính

Giả sử mạng ([[Multilayer Perceptron|MLP]]) có 2 tầng, **không có** activation:

- Tầng 1: $\mathbf{H} = \mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)}$ → nhân + cộng
- Tầng 2: $\mathbf{O} = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}$ → nhân + cộng nữa

Thế vào:

$$\mathbf{O} = \mathbf{X}\underbrace{\mathbf{W}^{(1)}\mathbf{W}^{(2)}}_{\text{gộp thành 1 ma trận}} + \underbrace{\mathbf{b}^{(1)}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}}_{\text{gộp thành 1 bias}}$$

→ **Kết quả giống hệt 1 tầng duy nhất!** Thêm 100 tầng tuyến tính cũng vô nghĩa — bao nhiêu tầng cũng rút gọn thành 1.

### Giải pháp: Chèn "nút bẻ cong" vào giữa

$$\mathbf{H} = \sigma(\mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)})$$

Hàm $\sigma$ (activation) **biến đổi phi tuyến** kết quả → **không thể rút gọn** nữa → mạng thật sự "sâu", có thể học được những quy luật phức tạp.

> [!TIP] Hình dung
> Không có activation: mạng chỉ "kéo giãn" và "xoay" dữ liệu (như kéo tờ giấy).
> Có activation: mạng **gập**, **bẻ**, **uốn** dữ liệu (như gấp tờ giấy) → tách được những nhóm dữ liệu lồng vào nhau.

---

## 2. ReLU — "Siêu sao" hiện đại

$$\text{ReLU}(x) = \max(0, x)$$

**Bằng lời**: Nếu x **âm** → trả về 0 (tắt). Nếu x **dương** → giữ nguyên (bật). Đơn giản vậy thôi.

![[assets/attachments/D2L/Buoi18/relu_plot.png]]

**Ví dụ**: ReLU(−5) = 0, ReLU(3) = 3, ReLU(0) = 0

| Tính chất | Giá trị | Nghĩa dễ hiểu |
| --- | --- | --- |
| Kết quả | $[0, +\infty)$ | Luôn ≥ 0 |
| Đạo hàm | 0 (khi $x<0$), 1 (khi $x>0$) | "Tắt" hoặc "bật hết cỡ" |
| Ưu điểm chính | Đơn giản, gradient không biến mất | Chỉ so sánh với 0 → cực nhanh |
| Nhược điểm | "Neuron chết" (dead neuron) | Nếu input luôn âm → neuron tắt vĩnh viễn |

### Tại sao ReLU phổ biến nhất?

1. **Cực nhanh**: chỉ cần so sánh với 0 — nhanh hơn nhiều so với tính $e^x$
2. **Gradient mạnh**: khi $x > 0$, gradient = 1 → tín hiệu truyền ngược không bị yếu đi theo từng tầng (xem: **vanishing gradient** ở phần Sigmoid)
3. **Tự động tiết kiệm**: ~50% neurons nhận input âm → output = 0 → mạng chỉ "bật" những neurons thực sự cần thiết

### "Neuron chết" (Dead Neuron) — nhược điểm duy nhất

Nếu trọng số khởi tạo xấu hoặc learning rate quá lớn → input của 1 neuron **luôn âm**:
- ReLU output = 0, gradient = 0 **mãi mãi** → neuron đó không bao giờ được cập nhật → **"chết"**

**Giải pháp**: dùng biến thể cho gradient nhỏ khi $x < 0$:

### Biến thể của ReLU

| Biến thể | Công thức | Đặc điểm |
| --- | --- | --- |
| **Leaky ReLU** | $\max(0.01x, x)$ | Cho gradient nhỏ (0.01) khi $x<0$ → neuron không chết |
| **pReLU** | $\max(\alpha x, x)$ | Giống Leaky ReLU nhưng $\alpha$ là tham số tự học |
| **GELU** | $x \cdot \Phi(x)$ | Mượt hơn ReLU — dùng trong Transformer (GPT, BERT) |

---

## 3. Sigmoid — "Ông tổ" (vai trò niche)

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**Bằng lời**: Ép **mọi số** về khoảng (0, 1). Số rất âm → gần 0. Số rất dương → gần 1. Số = 0 → đúng 0.5.

![[assets/attachments/D2L/Buoi18/sigmoid_plot.png]]

**Ví dụ**: Sigmoid(0) = 0.5, Sigmoid(10) ≈ 1, Sigmoid(−10) ≈ 0

| Tính chất | Giá trị | Nghĩa dễ hiểu |
| --- | --- | --- |
| Kết quả | $(0, 1)$ | Luôn nằm giữa 0 và 1 — giống xác suất |
| Đạo hàm | $\sigma(x)(1 - \sigma(x))$, max = 0.25 | Gradient rất nhỏ → yếu |
| Ưu điểm | Output giống xác suất | Tốt cho phân loại đúng/sai |
| Nhược điểm | **Vanishing gradient** | Gradient qua nhiều tầng → gần bằng 0 |

### Tại sao Sigmoid gây "gradient biến mất" (vanishing gradient)?

Gradient max của Sigmoid = **0.25**. Nghĩa là mỗi khi tín hiệu đi qua 1 tầng Sigmoid, nó bị **nhân tối đa 0.25** (mất 75%).

Qua nhiều tầng:

| Số tầng | Gradient còn tối đa | Nghĩa |
| --- | --- | --- |
| 1 | 0.25 | Mất 75% |
| 5 | $0.25^5 ≈ 0.001$ | Mất 99.9% |
| 10 | $0.25^{10} ≈ 10^{-6}$ | Gần bằng 0 |

→ Tầng đầu tiên **không nhận được tín hiệu sửa lỗi** → **không học được gì**. Đây là lý do Sigmoid **không dùng** cho tầng ẩn nữa.

### Vẫn dùng ở đâu?

- **Output layer** cho phân loại **nhị phân** (đúng/sai, mèo/không mèo) — vì cần output trong (0,1) giống xác suất
- **Cổng** (gate) trong [[LSTM]]/[[GRU]] — cần giá trị 0→1 để "đóng" hoặc "mở" luồng thông tin

---

## 4. Tanh — "Sigmoid đối xứng"

$$\tanh(x) = \frac{1 - e^{-2x}}{1 + e^{-2x}}$$

**Bằng lời**: Giống Sigmoid nhưng ép về khoảng **(-1, 1)** thay vì (0, 1). Output có cả **âm lẫn dương**, cân bằng quanh 0.

![[assets/attachments/D2L/Buoi18/tanh_plot.png]]

**Ví dụ**: Tanh(0) = 0, Tanh(5) ≈ 1, Tanh(−5) ≈ −1

| Tính chất | Giá trị | Nghĩa dễ hiểu |
| --- | --- | --- |
| Kết quả | $(-1, 1)$ | Đối xứng — có cả âm lẫn dương |
| Đạo hàm | $1 - \tanh^2(x)$, max = 1 | Mạnh hơn Sigmoid (max = 1 thay vì 0.25) |
| Ưu điểm | **Zero-centered** | Output trung bình = 0 → tối ưu mượt hơn |
| Nhược điểm | Vanishing gradient (nhẹ hơn Sigmoid) | Khi $|x|$ lớn, gradient vẫn → 0 |

### "Zero-centered" nghĩa là gì?

Output trung bình bằng 0 (có cả giá trị âm lẫn dương). **Sigmoid** luôn dương (0→1, trung bình ≈ 0.5) → gradient cập nhật trọng số **luôn cùng dấu** → quá trình tối ưu đi **zig-zag** chậm. Tanh khắc phục vì output cân bằng quanh 0.

### Mối quan hệ Tanh — Sigmoid

$$\tanh(x) = 2\sigma(2x) - 1$$

Tanh thực ra là Sigmoid **co giãn + dịch chuyển**: nhân input ×2, nhân output ×2, trừ đi 1. Dùng chủ yếu trong **LSTM/GRU** (mạng hồi quy).

---

## 5. Bảng so sánh tổng hợp

| | ReLU | Sigmoid | Tanh |
| --- | --- | --- | --- |
| **Công thức** | $\max(0, x)$ | $\frac{1}{1+e^{-x}}$ | $\frac{1-e^{-2x}}{1+e^{-2x}}$ |
| **Kết quả** | $[0, \infty)$ | $(0, 1)$ | $(-1, 1)$ |
| **Gradient max** | 1 | 0.25 | 1 |
| **Vanishing?** | Không (khi $x>0$) | Có, nghiêm trọng | Có, nhẹ hơn |
| **Zero-centered?** | Không | Không | Có |
| **Thường dùng ở** | Tầng ẩn (**mặc định**) | Output nhị phân | LSTM/GRU |
| **Tốc độ** | Rất nhanh | Chậm (tính $e^x$) | Chậm (tính $e^x$) |

![[assets/attachments/D2L/Buoi18/activation_comparison.png]]

![[assets/attachments/D2L/Buoi18/gradient_comparison.png]]
*Đạo hàm (gradient) — chú ý Sigmoid max chỉ 0.25, còn ReLU và Tanh max = 1.0*

![[assets/attachments/D2L/Buoi18/vanishing_gradient.png]]
*Sigmoid gradient tụt dốc qua từng tầng → tầng đầu gần như không học được gì*

> [!TIP] Quy tắc chọn nhanh
> - **Tầng ẩn**: dùng **ReLU** (hoặc GELU cho Transformer)
> - **Output nhị phân** (đúng/sai): dùng **Sigmoid**
> - **Cổng LSTM/GRU**: dùng **Tanh** + **Sigmoid**
> - **Không chắc?** → **ReLU**. Luôn an toàn để bắt đầu.

---

## 6. Code trực quan hóa

```python
import torch
import matplotlib.pyplot as plt

x = torch.arange(-5, 5, 0.01)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].plot(x, torch.relu(x), color='#2196F3', linewidth=2)
axes[0].set_title('ReLU: max(0, x)')
axes[0].axhline(0, color='gray', linewidth=0.5)
axes[0].axvline(0, color='gray', linewidth=0.5)

axes[1].plot(x, torch.sigmoid(x), color='#FF9800', linewidth=2)
axes[1].set_title('Sigmoid: 1/(1+e⁻ˣ)')
axes[1].axhline(0.5, color='gray', linewidth=0.5, linestyle='--')

axes[2].plot(x, torch.tanh(x), color='#4CAF50', linewidth=2)
axes[2].set_title('Tanh')
axes[2].axhline(0, color='gray', linewidth=0.5)

plt.tight_layout()
plt.show()
```

---

## TODO

- [ ] Thêm GELU, Swish, Mish (activation hiện đại dùng trong Transformer)
- [ ] Code visualization đạo hàm của từng hàm
- [ ] Thêm decision boundary visualization: so sánh linear vs ReLU vs Sigmoid
