---
title: "Cross-Entropy Loss — Định lượng lỗi trong phân loại"
aliases:
  [
    cross entropy,
    categorical cross-entropy,
    log loss,
    hàm mất mát cross-entropy,
    CE Loss,
    Information Theory,
  ]
tags:
  [
    concept,
    machine-learning,
    loss-function,
    classification,
    information-theory,
    d2l,
  ]
created: 2026-03-19
updated: 2026-04-14
session: "D2L Tuần 4 — Liên kết từ Buổi 14-17"
---

# Cross-Entropy Loss

> [!NOTE] ELI5 — Giải thích như tôi 5 tuổi
> Tưởng tượng bạn chơi trò đố hình ảnh với một cái máy. Máy nhìn ảnh con mèo, rồi phải đoán: "Đây là mèo, chó, hay thỏ?"
>
> - Máy trả lời: **"Tôi 90% chắc đây là mèo"** → Đúng! Phần thưởng lớn, phạt nhỏ (0.1).
> - Máy trả lời: **"Tôi 10% chắc đây là mèo"** → Sai tự tin! Phạt rất nặng (2.3).
> - Máy trả lời: **"Tôi 1% chắc đây là mèo"** → Sai kinh khủng! Phạt cực nặng (4.6).
>
> **Cross-entropy** chính là "mức phạt" đó. Máy càng tự tin mà sai → mức phạt càng tăng mạnh (phi tuyến). Mục tiêu training = giảm mức phạt này xuống thấp nhất.

---

## Tầng 2 — Định nghĩa kỹ thuật

**Cross-Entropy Loss** là một **hàm mất mát** (loss function) dùng để đo độ sai lệch giữa **phân phối xác suất dự đoán** của mô hình và **nhãn thật** của dữ liệu.

- **Input:** Logits từ model (chưa qua softmax) + nhãn đúng (one-hot hoặc class index)
- **Output:** Một số thực dương — loss càng nhỏ = dự đoán càng gần đúng
- **Giải quyết vấn đề gì?** MSE và Absolute Error không phạt đủ nặng khi model sai có sự tự tin cao. Cross-entropy, qua hàm $-\log$, tạo ra **penalty phi tuyến cực mạnh** cho những lần sai **quá tự tin**.

---

## Tầng 3 — Cơ chế chi tiết

### 1. Nền tảng: Hàm $-\log$ là "bộ khuếch đại hình phạt"

Để hiểu cross-entropy, trước tiên cần hiểu hàm $-\log(p)$:

| Xác suất $p$ (máy đoán đúng class) | $-\log(p)$ (mức phạt) | Cảm nhận                                 |
| ---------------------------------- | --------------------- | ---------------------------------------- |
| 0.99                               | ≈ 0.01                | Gần như chắc chắn đúng → phạt gần bằng 0 |
| 0.90                               | ≈ 0.11                | Khá tự tin đúng → phạt nhỏ               |
| 0.70                               | ≈ 0.36                | Nghiêng về đúng                          |
| 0.50                               | ≈ 0.69                | Không biết gì → phạt trung bình          |
| 0.30                               | ≈ 1.20                | Nghiêng về sai                           |
| 0.10                               | ≈ 2.30                | Khá tự tin sai → phạt nặng               |
| 0.01                               | ≈ 4.61                | Gần như chắc chắn sai → phạt rất nặng    |
| → 0                                | → ∞                   | Hoàn toàn sai → phạt vô cực              |

> [!IMPORTANT] Đây là điểm khác biệt cốt lõi
> Với MSE, sai từ 0.3 xuống 0.1 tăng penalty thêm $(0.7^2 - 0.9^2) \approx 0.32$.
> Với Cross-Entropy, sai từ 0.3 xuống 0.1 tăng penalty thêm $(2.30 - 1.20) = 1.10$ — **gấp 3.4 lần**.
> Càng sai nhiều, cross-entropy càng **trừng phạt mạnh hơn theo hàm mũ**.

---

### 2. Công thức đầy đủ

#### Trường hợp tổng quát (multi-class)

Cho nhãn thật $q = (q_1, \ldots, q_K)$ và dự đoán $p = (p_1, \ldots, p_K)$:

$$H(q, p) = -\sum_{i=1}^{K} q_i \log(p_i)$$

Đọc là: **"tổng của xác suất thật × log của xác suất dự đoán, đổi dấu"**

#### Trường hợp one-hot (phổ biến nhất trong Classification)

Vì nhãn là one-hot — chỉ duy nhất $q_c = 1$, còn lại $q_i = 0$ — tất cả các hạng $0 \times \log(\cdot) = 0$ triệt tiêu, chỉ còn 1 hạng sống sót:

$$L = -\log(p_c)$$

trong đó $c$ là **class đúng**. Cực kỳ đơn giản: **chỉ nhìn vào xác suất mà model gán cho class đúng, lấy $-\log$.**

> [!NOTE] Insight quan trọng
> Cross-entropy **không quan tâm** model phân bổ xác suất sai cho những class nào khác. Chỉ cần $p_c$ (xác suất của class đúng) cao là loss thấp. Đây là lý do tại sao training phân loại rất hiệu quả.

---

### 3. Ví dụ tính tay từng bước

**Bài toán**: Nhận diện quần áo (10 loại). Ảnh thực là "Áo thun" (class 0).

**Dự đoán của model** (sau softmax):

```
p = [0.7,  0.05, 0.05, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.02]
     ↑class 0   (áo thun)
```

**Nhãn thật** (one-hot):

```
q = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

**Tính cross-entropy**:

$$L = -\sum_{i=0}^{9} q_i \log(p_i)$$

$$= -(1 \cdot \log(0.7) + 0 \cdot \log(0.05) + \ldots + 0 \cdot \log(0.02))$$

$$= -\log(0.7) = -(-0.357) = \mathbf{0.357}$$

**Nếu model dự đoán sai** (chỉ 10% chắc là áo thun):

$$p_0 = 0.10 \quad \Rightarrow \quad L = -\log(0.10) = 2.303$$

**Và nếu model sai hoàn toàn** (1% chắc là áo thun):

$$p_0 = 0.01 \quad \Rightarrow \quad L = -\log(0.01) = 4.605$$

---

### 4. Tại sao không dùng MSE hoặc Absolute Error?

#### So sánh trực tiếp trên cùng ví dụ

Cho $q = (1, 0, 0)$, hai mô hình:

- $p_A = (0.70, 0.20, 0.10)$ — nghiêng về đúng (70%)
- $p_B = (0.10, 0.60, 0.30)$ — nghiêng về sai (10%)

**Absolute Error (L1)**:

- $|1-0.7| + |0-0.2| + |0-0.1| = 0.6$ cho $p_A$
- $|1-0.1| + |0-0.6| + |0-0.3| = 1.8$ cho $p_B$

**MSE (L2)**:

- $(0.3)^2 + (0.2)^2 + (0.1)^2 = 0.14$ cho $p_A$
- $(0.9)^2 + (0.6)^2 + (0.3)^2 = 1.26$ cho $p_B$

**Cross-Entropy**:

- $-\log(0.70) = 0.357$ cho $p_A$
- $-\log(0.10) = 2.303$ cho $p_B$ ← **khoảng cách lớn hơn nhiều!**

| Metric        | $p_A$ (70% đúng) | $p_B$ (10% đúng) | Tỷ lệ $B/A$ |
| ------------- | ---------------- | ---------------- | ----------- |
| L1            | 0.60             | 1.80             | 3x          |
| MSE           | 0.14             | 1.26             | 9x          |
| Cross-Entropy | 0.36             | 2.30             | **6.4x**    |

Cross-entropy **phân biệt rõ ràng hơn** giữa dự đoán tốt/tệ, đặc biệt khi $p_c$ tiến dần về 0 (penalty tăng theo hàm mũ). MSE đã khá tốt nhờ bình phương, nhưng cross-entropy **phù hợp hơn về mặt lý thuyết xác suất** (xem phần MLE bên dưới).

---

### 5. Kết nối với Entropy & Information Theory

Entropy là **độ không chắc chắn trung bình** của một phân phối:

$$H(p) = -\sum_{i} p_i \log(p_i)$$

| Phân phối $p$                        | Entropy $H(p)$         | Ý nghĩa              |
| ------------------------------------ | ---------------------- | -------------------- |
| $(1, 0, 0, 0)$ — 100% chắc           | 0                      | Không có bất ngờ nào |
| $(0.25, 0.25, 0.25, 0.25)$ — mù tịt  | $\log(4) \approx 1.39$ | Bất ngờ tối đa       |
| $(0.9, 0.05, 0.03, 0.02)$ — khá chắc | ≈ 0.38                 | Một chút bất ngờ     |

**Cross-entropy** = "entropy khi dùng **bản đồ dự đoán sai** ($p$) để mã hóa dữ liệu thật ($q$)":

$$H(q, p) = -\sum_i q_i \log(p_i) = H(q) + D_{KL}(q \| p)$$

- $H(q)$: entropy của nhãn thật (hằng số, không phụ thuộc model)
- $D_{KL}(q \| p) \geq 0$: khoảng cách giữa 2 phân phối

**Minimize Cross-Entropy = Minimize KL Divergence** — tức là đưa dự đoán $p$ gần nhất có thể về nhãn thật $q$.

---

### 6. Kết nối với Maximum Likelihood Estimation (MLE)

Ta muốn mô hình **cực đại hóa** xác suất dự đoán đúng trên toàn tập data:

$$\max_\theta \prod_{i=1}^{n} p_c^{(i)}$$

Lấy $\log$ (đổi nhân thành cộng) và đổi dấu (từ max sang min):

$$\min_\theta \sum_{i=1}^{n} -\log(p_c^{(i)})$$

Đây chính xác là **cross-entropy loss**. Cross-entropy không phải tùy tiện — nó **trực tiếp xuất phát từ nguyên lý MLE**: training bằng cross-entropy = training để tối đa hóa khả năng mô hình đoán đúng nhãn thật.

---

### 7. Gradient — Vì sao Cross-Entropy + Softmax là "cặp hoàn hảo"

**Pipeline**:

```
Raw logits z ∈ R^K  -->  Softmax  -->  p ∈ (0,1)^K  -->  Cross-Entropy  -->  Loss L
```

Khi tính đạo hàm theo logit $z_i$:

$$\frac{\partial L}{\partial z_i} = p_i - q_i$$

Gradient = **dự đoán trừ thực tế** — cực kỳ đơn giản và our.

| Tình huống    | $p_i$ | $q_i$ | Gradient $= p_i - q_i$ | Hành động            |
| ------------- | ----- | ----- | ---------------------- | -------------------- |
| Đúng, tự tin  | 0.90  | 1     | $-0.10$                | Nhẹ nhàng tăng $z_i$ |
| Sai nhẹ       | 0.40  | 1     | $-0.60$                | Tăng $z_i$ mạnh hơn  |
| Rất sai       | 0.05  | 1     | $-0.95$                | Tăng $z_i$ rất mạnh  |
| Sai nhãn khác | 0.60  | 0     | $+0.60$                | Hạ $z_i$ xuống mạnh  |

Gradient **tự động tỷ lệ dengan mức độ sai** — sai nhiều → cập nhật nhanh; sai ít → cập nhật nhẹ. Đây là lý do cross-entropy hội tụ nhanh hơn MSE.

MSE sẽ cần thêm một hạng $\sigma'(z_i)$ (Jacobian của softmax) trong gradient, làm phức tạp và xảy ra **vanishing gradient** khi $p$ gần 0 hoặc 1.

---

### 8. Code PyTorch

```python
import torch
import torch.nn.functional as F

# ===== Cách 1: Tính tay =====
logits = torch.tensor([2.0, 1.0, -1.0, 0.5, -0.5, 0.2, -0.3, 0.8, 1.5, -0.2])
true_class = 0  # "Áo thun" = class 0

probs = F.softmax(logits, dim=0)          # Bước 1: Softmax → xác suất
loss_manual = -torch.log(probs[true_class])  # Bước 2: -log(p_true)
print(f"CE tay: {loss_manual.item():.4f}")   # → 1.2930

# ===== Cách 2: PyTorch built-in (nên dùng) =====
criterion = torch.nn.CrossEntropyLoss()

# QUAN TRỌNG: truyền RAW LOGITS (chưa softmax)
logits_batch = torch.tensor([[2.0, 1.0, -1.0, 0.5, -0.5, 0.2, -0.3, 0.8, 1.5, -0.2]])
target = torch.tensor([0])

loss = criterion(logits_batch, target)
print(f"CE PyTorch: {loss.item():.4f}")   # → 1.2930 (giống tay!)
```

> [!WARNING] Lỗi rất phổ biến: "Double softmax"
>
> ```python
> # SAI: tự apply softmax trước, rồi đưa vào CrossEntropyLoss
> probs = F.softmax(logits, dim=1)
> loss = criterion(probs, target)   # CrossEntropyLoss sẽ softmax lần NỮA → sai hoàn toàn!
>
> # ĐÚNG: truyền thẳng logits
> loss = criterion(logits, target)  # CrossEntropyLoss = softmax + NLL tích hợp
> ```
>
> `nn.CrossEntropyLoss` = `log_softmax` + `NLLLoss` được gộp lại vì lý do **numerical stability** (tránh $\log(0)$).

---

### 9. Ứng dụng trong thực tế

| Bài toán                          | Cách dùng Cross-Entropy                                  |
| --------------------------------- | -------------------------------------------------------- |
| **Classification** (ảnh, văn bản) | $L = -\log(p_c)$, mỗi sample 1 nhãn                      |
| **Language Modeling** (GPT, BERT) | Dự đoán token tiếp theo — 1 token = 1 bài toán phân loại |
| **Machine Translation**           | CE trên từng token output (teacher forcing)              |
| **Knowledge Distillation**        | $q$ = soft labels từ teacher, không phải one-hot         |
| **Semantic Segmentation**         | CE từng pixel                                            |

**Perplexity** — metric dùng trong Language Models:

$$\text{PPL} = e^H = e^{\text{average CE loss}}$$

Perplexity thấp = model ít bất ngờ khi gặp dữ liệu = model tốt.

---

## Checklist tự kiểm tra

- [ ] Cross-entropy = $-\log(p_{\text{class đúng}})$ khi nhãn là one-hot
- [ ] $p_c \to 0$ (sai tự tin) → loss → $\infty$ (phạt vô cực)
- [ ] Minimize CE = Minimize KL Divergence = Maximize Likelihood
- [ ] Gradient: $\partial L / \partial z_i = p_i - q_i$ — đơn giản, mạnh
- [ ] PyTorch: truyền **raw logits** vào `CrossEntropyLoss`, không softmax trước
- [ ] Language models dùng CE → Perplexity = $e^{\text{CE}}$

---

## Liên kết

- [[Softmax Function]] — biến logits thành xác suất trước khi tính CE
- [[One-Hot Encoding]] — dạng nhãn đầu vào phổ biến nhất
- [[KL Divergence]] — $H(q, p) = H(q) + D_{KL}(q \| p)$
- [[Maximum Likelihood Estimation]] — nền tảng lý thuyết của CE
- [[Perplexity]] — ứng dụng CE trong Language Modeling
- [[Buổi 14 - Tuần 4]] — Softmax Regression, CE loss lý thuyết
- [[Buổi 16 - Tuần 4]] — Implement CE từ scratch
- [[Buổi 17 - Tuần 4]] — Dùng `nn.CrossEntropyLoss` trong PyTorch

## TODO

- [ ] Thêm binary cross-entropy (BCELoss) — cho bài toán 2 class hoặc multi-label
- [ ] So sánh cross-entropy vs focal loss
- [ ] Liên kết với KL divergence
