---
title: "Buổi 22 - Tuần 6: Numerical Stability & Weight Initialization (D2L)"
tags: [d2l, numerical-stability, vanishing-gradient, exploding-gradient, xavier, weight-init, study-note]
created: 2026-03-27
session: "D2L Tuần 6, Buổi 22 — Numerical Stability & Weight Initialization"
source:
  - "30_Resources/Books/Dive Into Deep Learning/d2l-en/chapter_multilayer-perceptrons/numerical-stability-and-init.md"
related:
  - "[[Buổi 21 - Tuần 6]]"
  - "[[Buổi 18 - Tuần 5]]"
  - "[[Activation Function]]"
  - "[[Multilayer Perceptron]]"
---

# Buổi 22 — Numerical Stability & Weight Initialization: Tại sao khởi tạo quyết định sống chết?

> [!NOTE] ELI5
> Bạn xây tháp 100 tầng bằng gạch. Nếu tầng đầu tiên bị **lệch** dù chỉ 1mm → tầng 100 sẽ lệch **hàng mét** → tháp đổ.
>
> Neural network cũng vậy: mạng 100 tầng, tín hiệu (gradient) phải truyền ngược qua **từng tầng**. Nếu mỗi tầng gradient bị **nhân đôi** → tầng 100 = $2^{100}$ = số khổng lồ (**exploding**). Nếu mỗi tầng bị **giảm nửa** → tầng 100 = $0.5^{100}$ ≈ 0 (**vanishing**).
>
> **Weight Initialization** = cách đặt viên gạch đầu tiên cho đúng. Xavier Init giải quyết bằng cách chọn sigma **vừa đủ** → gradient không nổ, không biến mất.

---

## 🎯 Mục tiêu buổi học

1. Hiểu **Vanishing Gradient** — tại sao Sigmoid gây ra, ReLU giải quyết
2. Hiểu **Exploding Gradient** — tại sao W random quá lớn → gradient nổ
3. Hiểu **Symmetry Breaking** — tại sao **PHẢI** khởi tạo random (không được dùng hằng số)
4. Hiểu và áp dụng **Xavier Initialization** — công thức, ý nghĩa, code PyTorch

---

## Phần 1: Vanishing & Exploding Gradients

### 1.1 Gradient qua nhiều tầng = tích nhiều ma trận

Mạng $L$ tầng:

$$\mathbf{o} = f_L \circ f_{L-1} \circ \cdots \circ f_1(\mathbf{x})$$

Gradient của output theo trọng số tầng $l$:

$$\frac{\partial \mathbf{o}}{\partial \mathbf{W}^{(l)}} = \underbrace{\mathbf{M}^{(L)} \cdots \mathbf{M}^{(l+1)}}_{L-l \text{ ma trận}} \cdot \mathbf{v}^{(l)}$$

Trong đó $\mathbf{M}^{(i)} = \frac{\partial \mathbf{h}^{(i)}}{\partial \mathbf{h}^{(i-1)}}$ = Jacobian (đạo hàm) của mỗi tầng.

> [!NOTE] ELI5
> Gradient = **tích** (nhân liên tiếp) nhiều ma trận. Giống nhân dãy phân số:
> - Mỗi số > 1 → tích **nổ** (exploding)
> - Mỗi số < 1 → tích **biến mất** (vanishing)
> - Mỗi số ≈ 1 → tích **ổn định** ← MỤC TIÊU!

### 1.2 Vanishing Gradient — Sigmoid là thủ phạm

Sigmoid: $\sigma(x) = \frac{1}{1+e^{-x}}$, đạo hàm max = **0.25** (tại $x=0$).

Qua $L$ tầng, gradient bị nhân $\leq 0.25^L$:

| Số tầng $L$ | Gradient max | Ý nghĩa |
| --- | --- | --- |
| 1 | 0.25 | OK |
| 5 | $0.25^5 ≈ 10^{-3}$ | Nhỏ |
| 10 | $0.25^{10} ≈ 10^{-6}$ | **Rất nhỏ** |
| 20 | $0.25^{20} ≈ 10^{-12}$ | **Gần như 0** → tầng đầu KHÔNG HỌC! |

ReLU: đạo hàm = **1** (khi x > 0) → gradient **giữ nguyên** qua các tầng → không vanish!

![[assets/attachments/D2L/Buoi22/vanishing_exploding.png]]
*Trái: Sigmoid gradient tụt dốc theo cấp số nhân, ReLU ổn định. Phải: nhân 100 ma trận random → norm nổ!*

> [!question]- ❓ Nếu ReLU giải quyết vanishing, tại sao người ta vẫn dùng Sigmoid/Tanh?
> ReLU giải quyết vanishing gradient **ở hidden layers**, nhưng:
> - **Output layer** cho bài toán nhị phân → vẫn cần Sigmoid (ép 0→1)
> - **LSTM/GRU** (mạng hồi quy) → dùng Sigmoid cho gates, Tanh cho state
> - **Attention** (Transformer) → dùng Softmax (họ hàng Sigmoid)
>
> Tóm lại: **hidden layers → ReLU** (hoặc GELU/SiLU). Các vị trí đặc biệt → Sigmoid/Tanh theo nhu cầu.

### 1.3 Exploding Gradient — W quá lớn

```python
import torch
M = torch.normal(0, 1, size=(4, 4))
print('Ma trận ban đầu:\n', M)

for i in range(100):
    M = M @ torch.normal(0, 1, size=(4, 4))

print('Sau 100 lần nhân:\n', M)
# → Giá trị CỰC LỚN (10^20+) hoặc NaN/Inf
```

> [!question]- ❓ Exploding gradient gây hại thế nào khi train?
> - **Update quá lớn**: $W \leftarrow W - \eta \cdot (\text{gradient CỰC LỚN})$ → W nhảy sang giá trị vô nghĩa
> - **NaN/Inf**: loss bỗng = `nan` → train crash
> - **Không hội tụ**: model dao động không ổn định
>
> **Cách phát hiện**: loss đột ngột tăng vọt, hoặc xuất hiện `nan` trong output/loss.
>
> **Giải pháp ngoài init**: **Gradient Clipping** — cắt gradient nếu quá lớn:
> ```python
> torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
> ```

---

## Phần 2: Symmetry Breaking — Tại sao PHẢI random?

> [!NOTE] ELI5
> 2 đầu bếp cùng nhận **đúng cùng nguyên liệu**, dùng **đúng cùng công thức** → nấu ra **đúng cùng món**. Kết quả? 2 đầu bếp = **vô nghĩa** — chỉ cần 1 thôi.
>
> Nếu tất cả neurons cùng tầng có **cùng trọng số** → chúng tính ra **cùng output** → gradient **cùng giá trị** → update **cùng cách** → **MÃI MÃI giống nhau**. 256 neurons = 1 neuron.

![[assets/attachments/D2L/Buoi22/symmetry_breaking.png]]
*Trái: W = hằng số → h1 = h2 luôn → lãng phí. Phải: W random → mỗi neuron tính khác → mạng mạnh hơn.*

### 2.1 Chứng minh toán học đơn giản

MLP 1 hidden layer, 2 hidden units, weight = $c$ (hằng số):

- Forward: $h_1 = h_2 = \sigma(cx_1 + cx_2 + b)$ → **giống hệt!**
- Gradient: $\frac{\partial L}{\partial w_{1j}} = \frac{\partial L}{\partial w_{2j}}$ → **giống hệt!**
- Update: cả hai nhận cùng update → W mới **vẫn giống nhau**

→ **Vòng lặp vĩnh viễn**: không bao giờ phá được đối xứng bằng SGD thuần.

> [!question]- ❓ Dropout có phá đối xứng không?
> **Có!** Dropout random tắt neurons **khác nhau** mỗi iteration → gradient **khác nhau** cho mỗi neuron → phá đối xứng. Đây là 1 lợi ích phụ của Dropout (ngoài chống overfitting).
>
> Tuy nhiên, **không ai dùng Dropout để phá đối xứng** — random init đã giải quyết từ đầu.

---

## Phần 3: Xavier Initialization

### 3.1 Bài toán: Chọn sigma bao nhiêu?

Khi init $W \sim \mathcal{N}(0, \sigma^2)$, chọn sigma **bao nhiêu** cho vừa?

- $\sigma$ quá nhỏ → output mỗi tầng **teo lại** → cuối cùng ≈ 0 → vanishing
- $\sigma$ quá lớn → output mỗi tầng **phình ra** → cuối cùng ≈ ∞ → exploding
- $\sigma$ vừa đủ → output **giữ nguyên scale** qua mỗi tầng → **ổn định**!

![[assets/attachments/D2L/Buoi22/xavier_comparison.png]]
*Quá nhỏ: output gần 0 (teo). Xavier: output phân bố đẹp (std≈1). Quá lớn: output rất lớn (phình).*

### 3.2 Suy luận Xavier

Xét 1 tầng linear (chưa activation), $n_{\text{in}}$ inputs:

$$o_i = \sum_{j=1}^{n_{\text{in}}} w_{ij} x_j$$

Giả sử: $E[w] = 0, \text{Var}[w] = \sigma^2$ và $E[x] = 0, \text{Var}[x] = \gamma^2$, các biến độc lập.

Tính variance output:

$$\text{Var}[o_i] = n_{\text{in}} \cdot \sigma^2 \cdot \gamma^2$$

Muốn **variance không đổi** qua tầng ($\text{Var}[o] = \text{Var}[x] = \gamma^2$):

$$n_{\text{in}} \cdot \sigma^2 = 1 \implies \sigma^2 = \frac{1}{n_{\text{in}}}$$

Tương tự, lập luận cho **backpropagation** (gradient truyền ngược):

$$n_{\text{out}} \cdot \sigma^2 = 1 \implies \sigma^2 = \frac{1}{n_{\text{out}}}$$

Không thể thỏa mãn cùng lúc cả hai! Xavier **lấy trung bình**:

$$\boxed{\sigma^2 = \frac{2}{n_{\text{in}} + n_{\text{out}}}}$$

> [!question]- ❓ Giải thích trực giác: tại sao chia cho $n_{\text{in}} + n_{\text{out}}$?
> - Tầng có **nhiều inputs** ($n_{\text{in}}$ lớn) → mỗi trọng số đóng góp 1 phần nhỏ → W phải nhỏ (chia cho $n_{\text{in}}$)
> - Tầng có **nhiều outputs** ($n_{\text{out}}$ lớn) → gradient từ nhiều neurons → W phải nhỏ (chia cho $n_{\text{out}}$)
> - Xavier **cân bằng cả hai** bằng trung bình
>
> **Ví dụ**: tầng 784 → 256
> - $\sigma = \sqrt{\frac{2}{784 + 256}} = \sqrt{\frac{2}{1040}} ≈ 0.044$
> - Nhỏ hơn nhiều so với $\sigma = 1$ (quá lớn!) nhưng lớn hơn $\sigma = 0.001$ (quá nhỏ)

### 3.3 Kết quả khi truyền qua nhiều tầng

![[assets/attachments/D2L/Buoi22/variance_propagation.png]]
*Xavier (xanh lá) giữ variance ≈ 1 qua 10 tầng. Sigma quá nhỏ → tắt. Sigma quá lớn → nổ.*

### 3.4 He Initialization (cho ReLU)

Xavier giả sử activation **đối xứng** (Tanh, linear). Nhưng ReLU **tắt nửa âm** → output variance giảm nửa! Cần bù lại:

$$\sigma^2_{\text{He}} = \frac{2}{n_{\text{in}}} \quad \text{(gấp đôi Xavier theo } n_{\text{in}} \text{)}$$

> [!question]- ❓ Khi nào dùng Xavier, khi nào dùng He?
> | Activation | Init | Công thức |
> | --- | --- | --- |
> | **Tanh / Sigmoid / Linear** | Xavier (Glorot) | $\sigma^2 = \frac{2}{n_{\text{in}} + n_{\text{out}}}$ |
> | **ReLU / Leaky ReLU** | **He (Kaiming)** | $\sigma^2 = \frac{2}{n_{\text{in}}}$ |
>
> **Rule**: dùng activation nào → dùng init tương ứng. Sai init → train chậm hoặc không hội tụ.

### 3.5 Code PyTorch

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# ====== Xavier init ======
def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)  # Xavier (Glorot) Normal
        nn.init.zeros_(m.bias)

# ====== He (Kaiming) init — cho ReLU ======
def he_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)

# Áp dụng:
model.apply(he_init)  # ← Dùng He vì dùng ReLU
```

> [!question]- ❓ `model.apply()` hoạt động thế nào?
> `model.apply(fn)` gọi hàm `fn` lên **mỗi module** (layer) trong model:
> - Module 1: `nn.Flatten()` → `fn(Flatten)` → không phải Linear → bỏ qua
> - Module 2: `nn.Linear(784, 256)` → `fn(Linear)` → init weight!
> - Module 3: `nn.ReLU()` → bỏ qua
> - ...
>
> `isinstance(m, nn.Linear)` đảm bảo chỉ init cho **Linear layers**, không động vào ReLU/Flatten.

> [!question]- ❓ PyTorch mặc định init gì? Có cần tự init không?
> `nn.Linear` mặc định dùng **Kaiming Uniform** (He init, uniform distribution) — đã tốt cho ReLU!
>
> Bạn **không bắt buộc** phải tự init nếu dùng ReLU. Nhưng nên tự init khi:
> - Dùng **Tanh/Sigmoid** trong hidden layers → cần Xavier
> - Dùng kiến trúc phức tạp (Transformer, GAN, …) → init đặc biệt
> - Muốn **reproduce** kết quả → set seed + init thống nhất

---

## 📖 Từ điển thuật ngữ Buổi 22

| Thuật ngữ | Dịch sang tiếng Việt | Nghĩa dễ hiểu |
| --- | --- | --- |
| **Vanishing gradient** | Gradient biến mất | Gradient ≈ 0 → tầng đầu không học |
| **Exploding gradient** | Gradient phát nổ | Gradient = ∞ → update quá lớn, model crash |
| **Numerical stability** | Ổn định số học | Gradient không quá lớn / quá nhỏ |
| **Symmetry breaking** | Phá đối xứng | Init random để mỗi neuron học khác nhau |
| **Xavier (Glorot) init** | Khởi tạo Xavier | $\sigma^2 = \frac{2}{n_{\text{in}}+n_{\text{out}}}$ — cho Tanh/Sigmoid |
| **He (Kaiming) init** | Khởi tạo He | $\sigma^2 = \frac{2}{n_{\text{in}}}$ — cho ReLU |
| **Jacobian** | Ma trận Jacobi | Ma trận đạo hàm riêng, kích thước $n_{\text{out}} \times n_{\text{in}}$ |
| **Gradient clipping** | Cắt gradient | Giới hạn max norm gradient, chống explode |
| **model.apply()** | Áp dụng hàm | Gọi function lên mọi module trong model |
| **nn.init** | Module init | Chứa các hàm init: `xavier_normal_`, `kaiming_normal_`, ... |

---

## ✅ Bài tự kiểm tra

1. Sigmoid đạo hàm max = ? Qua 10 tầng Sigmoid, gradient tối đa = ?
2. Tại sao init $W = 0$ (hoặc $W = c$) cho tất cả neurons là **SAI**?
3. Xavier init cho tầng 512 → 256: $\sigma = ?$
4. He init cho tầng 512 → 256 (ReLU): $\sigma = ?$
5. PyTorch `nn.Linear` mặc định dùng init gì?

> [!NOTE]- 📝 Đáp án
> 1. Max = **0.25**. Qua 10 tầng: $0.25^{10} ≈ 9.5 \times 10^{-7}$ — gần như 0!
> 2. Tất cả neurons cùng tầng tính ra **cùng output** → gradient **cùng giá trị** → update **giống nhau** → MÃI MÃI giống nhau. 256 neurons hoạt động như **1 neuron** duy nhất. Đây gọi là **symmetry problem**.
> 3. $\sigma = \sqrt{\frac{2}{512 + 256}} = \sqrt{\frac{2}{768}} ≈ 0.051$
> 4. $\sigma = \sqrt{\frac{2}{512}} ≈ 0.0625$
> 5. **Kaiming Uniform** (He init, uniform distribution). Đặc biệt: `fan_in` mode, `nonlinearity='leaky_relu'` (tương thích ReLU). Nên đủ tốt cho hầu hết trường hợp dùng ReLU.

---

## 🔗 Liên kết

- **Buổi trước**: [[Buổi 21 - Tuần 6]] — Weight Decay & Dropout (code chi tiết)
- **Buổi sau**: [[Buổi 23 - Tuần 6]] — Backpropagation (cơ chế tính gradient tự động)
- **Concept notes**: [[Activation Function]], [[Multilayer Perceptron]]

## 📝 Kết luận

Buổi 22 giải thích **tại sao init quan trọng** cho deep networks:
- **Vanishing gradient**: Sigmoid $\text{grad}_{\max} = 0.25$ → nhân chuỗi → tầng đầu "chết". ReLU giải quyết.
- **Exploding gradient**: W quá lớn → gradient nổ tung → NaN. Gradient clipping giúp kiểm soát.
- **Symmetry breaking**: PHẢI init random. W = 0 hoặc hằng số → mọi neuron giống nhau → lãng phí.
- **Xavier init**: $\sigma^2 = \frac{2}{n_{\text{in}} + n_{\text{out}}}$ → giữ variance ổn định qua tầng (cho Tanh).
- **He init**: $\sigma^2 = \frac{2}{n_{\text{in}}}$ → cho ReLU (PyTorch default).

Buổi 23 sẽ giải đáp: gradient được **tính tự động** như thế nào? → **Backpropagation** — thuật toán nền tảng của deep learning.
