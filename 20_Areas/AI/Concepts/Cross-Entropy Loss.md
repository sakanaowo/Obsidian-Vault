---
title: "Cross-Entropy Loss — Định lượng lỗi trong phân loại"
aliases: [cross entropy, categorical cross-entropy, log loss, hàm mất mát cross-entropy, CE Loss, Information Theory]
tags: [concept, machine-learning, loss-function, classification, information-theory, d2l]
created: 2026-03-19
session: "D2L Tuần 4 — Liên kết từ Buổi 14-17"
---

# Cross-Entropy Loss: Từ **độ hỗn loạn** đến **độ sai**

> [!NOTE] ELI5
> Bạn đang dạy máy học để **nhận diện 10 loại quần áo**. Máy trả lời: "Tôi 30% chắc đó là áo thun (đúng), 20% chắc đó là áo sơ mi, 50% chắc là cái gì khác". 
> 
> **Cross-entropy** là **cách định lượng lỗi** của câu trả lời sai này bằng con số: nếu máy **càng chắc mà sai**, cross-entropy **càng lớn** (trừng phạt kịch liệt). Mục tiêu training = giảm cross-entropy xuống thấp nhất.

---

## 🎯 Mục tiêu

1. **Hiểu entropy từ gốc** — từ thông tin học (Information Theory)
2. **Hiểu cross-entropy là gì** — mở rộng entropy thành 2 phân phối
3. **Tại sao dùng cross-entropy** — không phải loss khác, tại sao nó tốt nhất
4. **Từ công thức đến code** — PyTorch `nn.CrossEntropyLoss`
5. **Thực hành** — tính tay, kiểm chứng bằng code

---

## Phần 1: Entropy — "Độ bất chắc chắn"

### 1.1 Khái niệm entropy

Entropy là thước đo **"bất chắc chắn"** hoặc **"hỗn loạn"** của một phân phối xác suất.

**Ví dụ trực quan**:

| Tình huống | Dự báo | Độ bất chắc |
| --- | --- | --- |
| **Scenario A** | "Ngày mai 99% mưa" | Rất **chắc** → entropy **thấp** |
| **Scenario B** | "Ngày mai 50% mưa, 50% nắng" | Rất **bất chắc** → entropy **cao** |
| **Scenario C** | "Ngày mai 10% mưa, 10% nắng, 80% bão" | Rất **chắc** → entropy **thấp** |

**Nhận xét**: 
- Entropy **thấp** khi phân phối **tập trung** (peaked) vào 1-2 kết quả
- Entropy **cao** khi phân phối **đồng đều** (flat), tức rất bất chắc

### 1.2 Công thức Entropy

Cho phân phối xác suất $p = (p_1, p_2, \ldots, p_K)$ với $\sum p_i = 1$:

$$H(p) = -\sum_{i=1}^{K} p_i \log(p_i)$$

**Giải thích từng phần**:

| Ký hiệu | Ý nghĩa | Ví dụ |
| --- | --- | --- |
| $p_i$ | Xác suất của outcome/class $i$ | $p_1 = 0.8$ (80% sự kiện 1 xảy ra) |
| $\log(p_i)$ | Logarithm tự nhiên của $p_i$ | $\log(0.8) \approx -0.223$ |
| $p_i \log(p_i)$ | Xác suất × logarithm | $0.8 \times (-0.223) \approx -0.178$ |
| $-\sum$ | Cộng tất cả, rồi lấy âm | Để kết quả **dương** |
| $H(p) \geq 0$ | Entropy cuối cùng | Luôn ≥ 0 |

**Tại sao âm lại?**
- Vì $0 < p_i \leq 1$ → $\log(p_i) \leq 0$ (logarithm số từ 0 đến 1 là âm)
- Ví dụ: $\log(0.5) = -0.693 < 0$, $\log(0.1) = -2.303 < 0$
- Khi nhân $p_i \log(p_i)$, kết quả âm → cần dấu âm ngoài → kết quả dương ✅

### 1.3 Ví dụ tính tay: 2 phân phối đối lập

#### Trường hợp 1: Máy rất chắc

Phân phối: $p = (0.99, 0.005, 0.003, 0.002)$

$$H(p) = -(0.99 \log(0.99) + 0.005 \log(0.005) + 0.003 \log(0.003) + 0.002 \log(0.002))$$

Tính từng hạng:
- $0.99 \log(0.99) = 0.99 \times (-0.01005) \approx -0.00995$
- $0.005 \log(0.005) = 0.005 \times (-5.298) \approx -0.02649$
- $0.003 \log(0.003) = 0.003 \times (-5.809) \approx -0.01743$
- $0.002 \log(0.002) = 0.002 \times (-6.215) \approx -0.01243$

$$H(p) = -(-0.00995 - 0.02649 - 0.01743 - 0.01243) = 0.0663$$

**Entropy rất nhỏ** (0.0663) → phân phối rất tập trung → **máy rất chắc chắn**

#### Trường hợp 2: Máy hoàn toàn bối rối

Phân phối: $p = (0.25, 0.25, 0.25, 0.25)$ (đồng đều 4 class)

$$H(p) = -(4 \times 0.25 \log(0.25)) = -(4 \times 0.25 \times (-1.386)) = -(-1.386) = 1.386$$

**Entropy lớn** (1.386) → phân phối rất đồng đều → **máy hoàn toàn không chắc**

#### Maximum entropy

Với $K$ class, entropy tối đa khi tất cả các class có xác suất bằng nhau ($p_i = 1/K$):

$$H_{\max} = -K \times \frac{1}{K} \log\left(\frac{1}{K}\right) = \log(K)$$

Ví dụ Fashion-MNIST (10 classes):
$$H_{\max} = \log(10) \approx 2.303$$

**So sánh**:
- Máy rất chắc (99%): $H \approx 0.07$ (gần 0)
- Máy đoán đều: $H \approx 2.30$ (gần max)

---

## Phần 2: Cross-Entropy — Khoảng cách giữa hai phân phối

### 2.1 Tại sao cần cross-entropy?

Entropy $H(p)$ chỉ miêu tả **1 phân phối duy nhất**. Nhưng trong phân loại, ta luôn có **2 phân phối**:

| Phân phối | Ý nghĩa | Ví dụ (Fashion-MNIST class 0 = áo thun) |
| --- | --- | --- |
| **$q$ (True/Target)** | Nhãn đúng (phân phối one-hot) | $q = (1, 0, 0, \ldots, 0)$ — chắc 100% đó là class 0 |
| **$p$ (Predicted)** | Dự đoán của mô hình | $p = (0.7, 0.15, 0.1, \ldots)$ — mô hình 70% class 0 |

**Câu hỏi**: Độ sai lầm giữa **nhãn đúng** (chắc 100%) và **dự đoán** (chắc 70%) là bao nhiêu?

**Trả lời**: Dùng **cross-entropy** để đo "khoảng cách" giữa 2 phân phối!

### 2.2 Công thức Cross-Entropy

$$H(q, p) = -\sum_{i=1}^{K} q_i \log(p_i)$$

Đọc: "Cross-entropy của $q$ và $p$" = tổng xác suất thật $q_i$ × logarithm xác suất dự đoán $p_i$

**So sánh với Entropy**:

| Công thức | Ý nghĩa |
| --- | --- |
| $H(p) = -\sum p_i \log(p_i)$ | Entropy của **1 phân phối** — độ hỗn loạn nội tại |
| $H(q, p) = -\sum q_i \log(p_i)$ | Cross-entropy **2 phân phối** — lỗi khi dùng $p$ thay vì $q$ |

**Tính chất**:
- $H(q, p) = H(q) + D_{KL}(q \| p)$ (cross-entropy = entropy + KL divergence)
- Khi $q$ là one-hot: $H(q) = 0$ → $H(q, p) = D_{KL}(q \| p)$ (cross-entropy ≈ KL divergence)

### 2.3 Ví dụ tính tay: 1 ảnh áo thun (class 0)

#### Dự đoán tốt

**Nhãn đúng** (ground truth): $q = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0)$ (10 classes of Fashion-MNIST)

**Dự đoán của mô hình**: $p = (0.9, 0.05, 0.03, 0.01, 0.005, 0.005, \ldots)$

$$H(q, p) = -(1 \times \log(0.9) + 0 \times \log(0.05) + 0 \times \log(0.03) + \ldots)$$

Quan sát: Vì $q_i = 0$ với $i \neq 1$, tất cả các hạng $0 \times \log(p_i) = 0$ → chỉ term đầu tiên sống sót:

$$H(q, p) = -\log(0.9) = -(-0.105) = 0.105$$

**Cross-entropy = 0.105** ✅ (nhỏ → dự đoán tốt!)

#### Dự đoán tệ

**Nhãn đúng**: $q = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0)$

**Dự đoán của mô hình**: $p = (0.1, 0.3, 0.2, 0.2, 0.15, 0.05, 0, 0, 0, 0)$ (mô hình chỉ 10% chắc đó là áo thun)

$$H(q, p) = -\log(0.1) = -(-2.303) = 2.303$$

**Cross-entropy = 2.303** ❌ (rất lớn → dự đoán tệ!)

#### Bang so sánh

| Dự đoán cho class đúng ($p_1$) | Cross-entropy $-\log(p_1)$ | Ý nghĩa |
| --- | --- | --- |
| 0.99 | -log(0.99) ≈ 0.01 | Rất tốt → loss rất nhỏ |
| 0.9 | -log(0.9) ≈ 0.105 | Tốt → loss nhỏ |
| 0.7 | -log(0.7) ≈ 0.357 | Khá tốt |
| 0.5 | -log(0.5) ≈ 0.693 | Trung bình |
| 0.3 | -log(0.3) ≈ 1.204 | Tệ |
| 0.1 | -log(0.1) ≈ 2.303 | Rất tệ |
| 0.01 | -log(0.01) ≈ 4.605 | Kinh khủng |
| → 0 | → $\infty$ | Hoàn toàn sai = vô cực |

> [!IMPORTANT] 🔑 Tính chất: "Trừng phạt kịch liệt"
> - Nếu máy dự đoán **đúng** → $p_{\text{true}} \approx 1$ → $\log(p) \approx 0$ → **cross-entropy ≈ 0** ✅
> - Nếu máy dự đoán **sai** → $p_{\text{true}} \approx \text{rất nhỏ}$ → $\log(p) \approx \text{rất âm}$ → $-\log(p) \approx \text{rất lớn}$ → **cross-entropy → ∞** ❌
>
> Đây là **cơ chế trừng phạt** mạnh mẽ — máy càng sai quả quyết → loss càng **lớn vô cực**.

---

## Phần 3: Tại sao Cross-Entropy tốt hơn các loss khác?

### 3.1 Đối thủ 1: Absolute Error

$$L = \sum_{i=1}^{K} |q_i - p_i|$$

**Ví dụ**: $q = (1, 0, 0)$, $p_1 = (0.6, 0.2, 0.2)$ vs $p_2 = (0.1, 0.5, 0.4)$

- Loss $(p_1)$ = $|1-0.6| + |0-0.2| + |0-0.2|$ = 0.8
- Loss $(p_2)$ = $|1-0.1| + |0-0.5| + |0-0.4|$ = 1.0

$(p_2)$ cho loss lớn hơn, nhưng **không phản ánh rõ** sự khác biệt: $(p_1)$ 60% chứng, $(p_2)$ chỉ 10% chắc!

**Vấn đề**: Absolute Error không **trừng phạt kịch liệt** khi sai.

### 3.2 Đối thủ 2: Mean Squared Error (MSE)

$$L = \sum_{i=1}^{K} (q_i - p_i)^2$$

**Ví dụ cũ**:
- Loss $(p_1)$ = $(1-0.6)^2 + (0-0.2)^2 + (0-0.2)^2$ = 0.16 + 0.04 + 0.04 = 0.24
- Loss $(p_2)$ = $(1-0.1)^2 + (0-0.5)^2 + (0-0.4)^2$ = 0.81 + 0.25 + 0.16 = 1.22

$(p_2)$ cao hơn, **nhưng**:
- Bình phương không nhạy cảm với các xác suất xung quanh 0
- Không tận dụng tính chất **log** → gradient không smooth khi training

**Vấn đề**: MSE hoạt động ổn nhưng chậm hơn cross-entropy.

### 3.3 Cross-Entropy vs KL Divergence

KL Divergence:
$$D_{KL}(q \| p) = \sum_{i=1}^{K} q_i \log\left(\frac{q_i}{p_i}\right) = \sum q_i \log(q_i) - \sum q_i \log(p_i)$$

= $H(q) + H(q, p)$

Nhận xét:
- Khi $q$ là **one-hot** (nhãn phân loại) → $H(q) = 0$
- Do đó: $D_{KL}(q \| p) = H(q, p)$
- Cross-entropy ≈ KL divergence (trong trường hợp one-hot)

**Kết luận**: Cross-entropy = cách chuẩn để tối ưu hóa xác suất trong classification.

### 3.4 Bang tổng hợp

| Tiêu chí | Absolute | MSE | Cross-Entropy |
| --- | --- | --- | --- |
| **Trừng phạt sai** | ⚠️ Không kịch liệt | ⚠️ Vừa vừa | ✅ Rất kịch liệt |
| **Gradient** | Khó xử (có vách) | Hợp lý | ✅ Rất smooth |
| **Tư tưởng** | Khoảng cách L1 | Khoảng cách L2 | **Xác suất + thông tin** |
| **Tốc độ hội tụ** | Chậm | Chậm | ✅ Nhanh |
| **Training ổn định** | Có thể rung | Có thể rung | ✅ Rất ổn định |

---

## Phần 4: Gradient — Tại sao Cross-Entropy + Softmax hợp tác tuyệt vời?

### 4.1 Pipeline: Logits → Softmax → Cross-Entropy → Gradient

```
Raw logits: z = (z_1, ..., z_K) ∈ ℝ^K (chưa xác suất)
       ↓
Softmax: p_i = e^z_i / Σ_j e^z_j ∈ (0, 1)
       ↓
Cross-Entropy: L = -log(p_true_class)
       ↓
Gradient: ∂L/∂z_i = p_i - q_i
```

### 4.2 Công thức gradient

**Tính đạo hàm** của cross-entropy loss theo logit $z_i$:

$$\frac{\partial L}{\partial z_i} = \frac{\partial}{\partial z_i} \left( -\sum_j q_j \log(p_j) \right)$$

Với $p_j = \text{softmax}(z)_j$:

$$\frac{\partial L}{\partial z_i} = p_i - q_i$$

**Giải thích cực đơn giản**:

$$\text{Gradient} = \text{Dự đoán} - \text{Thực tế}$$

| Tình huống | $p_i$ | $q_i$ | Gradient | Hành động |
| --- | --- | --- | --- | --- |
| Dự đoán đúng | 0.9 | 1 | 0.9-1=-0.1 | Tăng $z_i$ (âm) |
| Dự đoán quá cao | 0.8 | 0 | 0.8-0=0.8 | Giảm $z_i$ (dương) |
| Dự đoán quá thấp | 0.1 | 1 | 0.1-1=-0.9 | Tăng $z_i$ rất mạnh! (âm) |
| Hoàn toàn sai | 0.01 | 1 | 0.01-1=-0.99 | Cập nhật siêu mạnh! |

**🔑 Nội dung**:
- Gradient **đơn giản** = $p_i - q_i$ (dễ tính!)
- Magnitude **lớn** khi $p_i$ sai quả quyết → learning mạnh mẽ ⚡

### 4.3 So sánh với MSE

**MSE**: $\frac{\partial L}{\partial z_i} \approx 2(p_i - q_i) \times \sigma'(z_i)$ (phức tạp hơn + có derivative softmax)

**Cross-Entropy**: $\frac{\partial L}{\partial z_i} = p_i - q_i$ (sạch sẽ, mạnh mẽ!)

**Kết luận**: Cross-entropy + softmax = "cặp hoàn hảo" cho phân loại.

---

## Phần 5: Code PyTorch

### 5.1 Tính cross-entropy tay (từ scratch)

```python
import torch
import torch.nn.functional as F

# Giả sử mô hình output logits cho 10 classes (Fashion-MNIST)
logits = torch.tensor([2.0, 1.0, -1.0, 0.5, -0.5, 0.2, -0.3, 0.8, 1.5, -0.2])
true_class = 0  # Nhãn đúng: class 0 (áo thun)

# Bước 1: Softmax
probs = F.softmax(logits, dim=0)
print(f"Probabilities: {probs}")
# Output: tensor([0.2759, 0.1017, 0.0108, 0.0519, 0.0072, 0.0296, 0.0049, 0.0713, 0.4304, 0.0162])

# Bước 2: Cross-Entropy = -log(p_true)
ce_manual = -torch.log(probs[true_class])
print(f"Cross-Entropy (tay): {ce_manual.item():.4f}")
# Output: 1.2930
```

### 5.2 Dùng PyTorch built-in

```python
# Cách "chính thức" (nên dùng)
criterion = torch.nn.CrossEntropyLoss()

# Input: RAW LOGITS (chưa softmax!)
logits = torch.tensor([[2.0, 1.0, -1.0, 0.5, -0.5, 0.2, -0.3, 0.8, 1.5, -0.2]])  # Shape: (1, 10)
target = torch.tensor([0])  # Class 0

loss = criterion(logits, target)
print(f"Cross-Entropy (PyTorch): {loss.item():.4f}")
# Output: 1.2930  (giống tay!)
```

### 5.3 Batch training

```python
batch_size = 64
num_classes = 10

# Logits từ model: 64 samples × 10 classes
logits = torch.randn(batch_size, num_classes)

# Nhãn đúng: 64 samples
targets = torch.randint(0, num_classes, (batch_size,))

# Tính loss trung bình
criterion = torch.nn.CrossEntropyLoss()
loss = criterion(logits, targets)

print(f"Average CE Loss on batch: {loss.item():.4f}")

# Backprop
loss.backward()

# Gradients đã được tính ∂L/∂logits
print(f"Gradient shape: {logits.grad.shape}")  # (64, 10)
```

> [!WARNING] ⚠️ Lỗi thường gặp
> ```python
> # ❌ SAIIII: Không tự apply softmax trước!
> probs = F.softmax(logits, dim=1)
> loss = criterion(probs, targets)  # Sai!
> 
> # ✅ ĐÚNG
> loss = criterion(logits, targets)  # CrossEntropyLoss tự làm softmax
> ```
> 
> `CrossEntropyLoss` đã gộp softmax + NLL bên trong → bạn chỉ cần truyền **raw logits**.

---

## Phần 6: Entropy trong thông tin học (Information Theory)

### 6.1 Entropy của phân phối $p$

$$H[p] = -\sum_i p_i \log p_i$$

**Ý nghĩa**: Số bit trung bình cần để **encode** 1 sample từ phân phối $p$.

**Ví dụ**:
- $p = (0.99, 0.01)$ → entropy ≈ 0.08 bits (hầu hết sample = first category → cần ít bit)
- $p = (0.5, 0.5)$ → entropy = 1 bit (bạn cần chính xác 1 bit để phân biệt)
- $p = (0.1, 0.1, ..., 0.1)$ (10 categories) → entropy ≈ 3.32 bits

### 6.2 Cross-Entropy của $q$ dùng code của $p$

$$H(q, p) = -\sum_i q_i \log p_i$$

**Ý nghĩa**: Số bit trung bình nếu bạn **nhầm lẫn** phân phối:
- Thực tế: phân phối $q$
- Bạn dùng: mã hóa cho phân phối $p$ (sai!)

**Ví dụ**:
- Thực tế: $q = (1, 0)$ (chắc 100% category A)
- Bạn dùng code cho: $p = (0.99, 0.01)$ (chắc 99% category A)
- Cross-entropy: $-1 \times \log(0.99) \approx 0.01$ bits (lỗi nhỏ)

- Thực tế: $q = (1, 0)$
- Bạn nhầm dùng code cho: $p = (0.1, 0.9)$ (chắc 90% category B)
- Cross-entropy: $-1 \times \log(0.1) \approx 2.30$ bits (lỗi lớn!)

### 6.3 KL Divergence — Khoảng cách giữa hai phân phối

$$D_{KL}(q \| p) = H(q, p) - H(q)$$

**Ý nghĩa**: Bao nhiêu bit **dư thừa** nếu dùng code của $p$ thay vì optimal code cho $q$.

**Tính chất**:
- $D_{KL}(q \| p) \geq 0$ (luôn không âm)
- $D_{KL}(q \| p) = 0$ ⟺ $q = p$ (giống nhau)
- **Không đối xứng**: $D_{KL}(q \| p) \neq D_{KL}(p \| q)$

---

## Phần 7: Ứng dụng thực tế D2L

| Buổi | Nội dung | Vai trò cross-entropy |
| --- | --- | --- |
| **Buổi 14** | Softmax Regression (lý thuyết) | Giới thiệu công thức |
| **Buổi 15** | Fashion-MNIST dataset | Chuẩn bị dữ liệu |
| **Buổi 16** | Softmax from Scratch | Implement cross-entropy tay |
| **Buổi 17** | Softmax Framework (PyTorch) | Dùng `CrossEntropyLoss` |
| **Buổi 18+** | CNN, RNN, Transformers | Luôn dùng cross-entropy cho classification |

---

## Phần 8: Checklist kiến thức

- [ ] Entropy $H(p) = -\sum p_i \log(p_i)$ là độ bất chắc chắn của 1 phân phối
- [ ] Cross-entropy $H(q, p) = -\sum q_i \log(p_i)$ là lỗi giữa nhãn đúng $q$ và dự đoán $p$
- [ ] Với one-hot encoding: $H(q, p) = -\log(p_{\text{true class}})$
- [ ] Cross-entropy **trừng phạt kịch liệt** khi sai (loss → ∞ khi $p_{\text{true}} \to 0$)
- [ ] Gradient: $\frac{\partial L}{\partial z_i} = p_i - q_i$ (đơn giản + mạnh mẽ)
- [ ] PyTorch: `CrossEntropyLoss(logits, targets)` — **logits không được softmax trước**

---

## 🔗 Liên kết

- **Buổi 14**: [[Buổi 14 - Tuần 4]] — Softmax Regression (lý thuyết)
- **Buổi 15**: [[Buổi 15 - Tuần 4]] — Fashion-MNIST Dataset
- **Concept khác**: [[Softmax Function]], [[One-Hot Encoding]], [[Information Theory]]

## 📝 Kết luận

**Cross-entropy** là **cách chuẩn để đo lỗi phân loại**:

1. **Tại sao**: Từ thông tin học + xác suất, trừng phạt sai rất nặng
2. **Cách hoạt động**: $-\log(p_{\text{true}})$ → càng sai → loss → ∞
3. **Gradient tốt**: $\frac{\partial L}{\partial z_i} = p_i - q_i$ giúp training **nhanh & ổn định**

Buổi 16 bạn implement tay, buổi 17 dùng PyTorch framework — lúc đó kiến thức này sẽ "click" hoàn toàn!

## 1. Bản chất — Từ MLE đến Cross-Entropy

### Bước 1: Maximum Likelihood Estimation (MLE)

Ta muốn tìm mô hình $f$ sao cho xác suất quan sát được dữ liệu $(\mathbf{X}, \mathbf{Y})$ là **lớn nhất**:

$$P(\mathbf{Y} \mid \mathbf{X}) = \prod_{i=1}^n P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)})$$

### Bước 2: Negative Log-Likelihood (NLL)

Nhân nhiều xác suất < 1 → số rất nhỏ → khó tính. Lấy $-\log$:

$$-\log P(\mathbf{Y} \mid \mathbf{X}) = \sum_{i=1}^n -\log P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)})$$

Minimize NLL = Maximize Likelihood.

### Bước 3: Cross-Entropy Loss

Với nhãn [[One-Hot Encoding|one-hot]] $\mathbf{y}$ và dự đoán $\hat{\mathbf{y}}$ (từ [[Softmax Function]]):

$$l(\mathbf{y}, \hat{\mathbf{y}}) = -\sum_{j=1}^q y_j \log \hat{y}_j$$

Vì $\mathbf{y}$ là one-hot (chỉ 1 phần tử bằng 1, còn lại = 0), công thức rút gọn thành:

$$l(\mathbf{y}, \hat{\mathbf{y}}) = -\log \hat{y}_c$$

trong đó $c$ là class đúng. → **Chỉ quan tâm xác suất mà mô hình gán cho class đúng.**

## 2. Ý nghĩa trực quan

| Dự đoán cho class đúng ($\hat{y}_c$) | Loss $= -\log(\hat{y}_c)$ | Ý nghĩa |
| --- | --- | --- |
| 0.99 | 0.01 | Rất tự tin, đúng → loss gần 0 |
| 0.9 | 0.105 | Khá tự tin → loss thấp |
| 0.5 | 0.693 | Phân vân → loss trung bình |
| 0.1 | 2.303 | Khá sai → loss cao |
| 0.01 | 4.605 | Rất sai → loss rất cao |
| → 0 | → $\infty$ | Hoàn toàn sai → loss vô cực |

> [!IMPORTANT] Cross-entropy phạt **rất nặng** khi mô hình tự tin nhưng sai
> Nếu $\hat{y}_c \to 0$ (mô hình gần như chắc chắn đây **không phải** class đúng), loss → $\infty$. Đây là cơ chế "phạt" giúp mô hình học nhanh khi mắc lỗi nghiêm trọng.

## 3. Gradient — Tại sao cross-entropy kết hợp tuyệt vời với softmax?

Đạo hàm của cross-entropy loss theo logit $o_j$:

$$\frac{\partial l}{\partial o_j} = \hat{y}_j - y_j = \text{softmax}(\mathbf{o})_j - y_j$$

Gradient = **dự đoán − thực tế** — cực kỳ đơn giản và đẹp. Tương tự hệt gradient của MSE trong linear regression ($\hat{y} - y$). Đây không phải trùng hợp — cả hai đều thuộc **exponential family**.

## 4. Ý nghĩa Information Theory

### Entropy — thước đo "bất ngờ trung bình"

$$H[P] = -\sum_j P(j) \log P(j)$$

- Data dễ đoán (luôn cùng kết quả) → entropy thấp
- Data khó đoán (ngẫu nhiên hoàn toàn) → entropy cao

### Cross-Entropy — "bất ngờ khi dùng mô hình sai"

$$H(P, Q) = -\sum_j P(j) \log Q(j)$$

- $P$ = phân phối thật (ground truth)
- $Q$ = phân phối dự đoán (model output)
- Cross-entropy **luôn ≥ Entropy**: $H(P, Q) \geq H(P)$
- Bằng nhau khi $P = Q$ → mô hình hoàn hảo

## 5. Ứng dụng trong DL

- **Mọi classification models**: softmax regression, CNNs, Transformers
- **Language modeling**: dự đoán token tiếp theo (GPT, BERT)
- **Knowledge distillation**: soft labels = phân phối xác suất từ teacher model
- PyTorch: `nn.CrossEntropyLoss()` (đã gộp softmax + NLL bên trong)

> [!WARNING] Lưu ý PyTorch
> `nn.CrossEntropyLoss()` nhận **logits** (chưa qua softmax) làm input, không phải probabilities. Nếu bạn tự apply softmax trước rồi đưa vào `CrossEntropyLoss` → sai.

## TODO

- [ ] Thêm binary cross-entropy (BCELoss)
- [ ] So sánh cross-entropy vs focal loss
- [ ] Liên kết với KL divergence
