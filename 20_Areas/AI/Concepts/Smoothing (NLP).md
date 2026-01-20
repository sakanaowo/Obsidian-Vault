---
tags:
  - ai
  - nlp
  - language-model
  - regularization
aliases:
  - Smoothing
  - Language Model Smoothing
  - Add-one Smoothing
  - Laplace Smoothing
  - Kneser-Ney
status: evergreen
related:
  - "[[N-gram Language Model]]"
  - "[[Perplexity]]"
  - "[[Maximum Likelihood Estimation]]"
---

# Smoothing (NLP)

> [!NOTE] ELI5
> Tưởng tượng bạn chưa bao giờ thấy ai ăn "pizza với mật ong". Điều đó không có nghĩa là nó **không thể xảy ra** — chỉ là bạn chưa gặp. **Smoothing** như việc bạn nói: "Những thứ tôi chưa thấy vẫn có thể xảy ra, tôi sẽ dành một chút xác suất cho chúng." Đây là cách để model không nói "impossible" với những gì nó chưa biết.

---

## 1. Vấn Đề: Zero Probabilities

### 1.1 MLE và Data Sparsity

Với Maximum Likelihood Estimation, nếu N-gram không xuất hiện trong training:

$$P_{MLE}(w_n | context) = \frac{C(context, w_n)}{C(context)} = \frac{0}{C(context)} = 0$$

### 1.2 Hậu Quả

- **Xác suất câu = 0:** Vì $P(W) = \prod P(w_i|context)$, một zero → cả tích = 0
- **Perplexity = ∞:** Model "từ chối" test data
- **Overconfidence:** Model quá chắc chắn về những gì nó thấy

### 1.3 Tại Sao Sparsity Là Inevitable?

Số N-grams khả dĩ tăng **theo hàm mũ** với N:
- Vocab $V = 50,000$
- Bigrams: $V^2 = 2.5 \times 10^9$
- Trigrams: $V^3 = 1.25 \times 10^{14}$

Không corpus nào đủ lớn để cover tất cả!

---

## 2. Nguyên Lý Smoothing

**Ý tưởng:** "Steal" một ít probability mass từ seen events, redistribute cho unseen events.

**Constraints:**
- Tổng xác suất vẫn = 1: $\sum_w P(w|context) = 1$
- Unseen events có $P > 0$
- Seen events có $P$ giảm nhưng vẫn reflect relative frequency

---

## 3. Các Kỹ Thuật Smoothing

### 3.1 Laplace (Add-One) Smoothing

**Ý tưởng:** Cộng 1 vào mọi count.

$$P_{Laplace}(w_n | w_{n-1}) = \frac{C(w_{n-1}, w_n) + 1}{C(w_{n-1}) + V}$$

**Ví dụ:**
- $C(\text{chinese}) = 100$, $C(\text{chinese food}) = 52$, $V = 50,000$
- MLE: $P(\text{food}|\text{chinese}) = 52/100 = 0.52$
- Laplace: $P(\text{food}|\text{chinese}) = 53/50,100 = 0.00106$

**Nhược điểm:** Reassign **quá nhiều** mass cho unseen → seen probabilities bị crush.

### 3.2 Add-k Smoothing

**Ý tưởng:** Cộng $k < 1$ thay vì 1.

$$P_{Add-k}(w_n | w_{n-1}) = \frac{C(w_{n-1}, w_n) + k}{C(w_{n-1}) + k \cdot V}$$

**Chọn k:** Optimize trên development set.

**Ứng dụng:** Hiệu quả cho text classification, nhưng vẫn suboptimal cho LM.

### 3.3 Good-Turing Smoothing

**Ý tưởng:** Ước lượng probability của unseen dựa trên **frequency of frequencies**.

Gọi $N_c$ = số N-grams xuất hiện đúng $c$ lần.

**Adjusted count:**
$$c^* = (c+1) \frac{N_{c+1}}{N_c}$$

**Intuition:** "Những gì xuất hiện 0 lần giống như những gì xuất hiện 1 lần, chỉ là ta chưa thấy."

### 3.4 Backoff

**Ý tưởng:** Nếu không có evidence cho N-gram, **fall back** về (N-1)-gram.

**Katz Backoff:**
$$P_{BO}(w_n | w_{n-2}, w_{n-1}) = \begin{cases} P^*(w_n | w_{n-2}, w_{n-1}) & \text{if } C(w_{n-2}, w_{n-1}, w_n) > 0 \\ \alpha(w_{n-2}, w_{n-1}) \cdot P_{BO}(w_n | w_{n-1}) & \text{otherwise} \end{cases}$$

Trong đó:
- $P^*$ là discounted probability
- $\alpha$ là backoff weight (để đảm bảo tổng = 1)

### 3.5 Interpolation

**Ý tưởng:** **Combine** tất cả N-gram levels với weights.

$$P_{interp}(w_n | w_{n-2}, w_{n-1}) = \lambda_3 P_3(w_n | w_{n-2}, w_{n-1}) + \lambda_2 P_2(w_n | w_{n-1}) + \lambda_1 P_1(w_n)$$

**Constraints:** $\lambda_1 + \lambda_2 + \lambda_3 = 1$

**Ưu điểm so với Backoff:** Luôn sử dụng evidence từ tất cả levels, không chỉ khi cần.

**Chọn $\lambda$:** 
- Grid search trên held-out data
- EM algorithm (deleted interpolation)

### 3.6 Kneser-Ney Smoothing (State-of-the-Art)

**Insight:** Lower-order distributions nên phản ánh **versatility**, không chỉ frequency.

**Vấn đề với standard backoff:**
- "San Francisco" xuất hiện nhiều
- Nhưng "Francisco" gần như **chỉ** xuất hiện sau "San"
- Unigram $P(\text{Francisco})$ cao → sai!

**Kneser-Ney solution:** Dùng **continuation count** cho lower-order:

$$P_{continuation}(w) = \frac{|\{v : C(v, w) > 0\}|}{|\{(v, w') : C(v, w') > 0\}|}$$

= "Số contexts khác nhau mà $w$ xuất hiện" / "Tổng số bigram types"

**Full Kneser-Ney:**
$$P_{KN}(w_n | w_{n-1}) = \frac{\max(C(w_{n-1}, w_n) - d, 0)}{C(w_{n-1})} + \lambda(w_{n-1}) P_{continuation}(w_n)$$

Trong đó $d$ là discount (thường 0.75) và $\lambda$ được tính để normalize.

---

## 4. So Sánh Các Phương Pháp

| Method | Pros | Cons |
|--------|------|------|
| Add-one | Simple | Too much mass to unseen |
| Add-k | Better than add-one | Still suboptimal |
| Good-Turing | Principled | Complex, unstable for high counts |
| Backoff | Uses lower-order when needed | Discontinuities |
| Interpolation | Always uses all evidence | Need to tune weights |
| Kneser-Ney | Best performance | More complex |

**Thực tế:** Kneser-Ney (hoặc Modified Kneser-Ney) là standard cho N-gram LMs.

---

## 5. Kết Nối Với Regularization

Smoothing trong NLP tương tự **regularization** trong ML:
- Cả hai giảm overfitting
- Cả hai "pull" model về prior/default
- Add-k ↔ L2 regularization (add to all parameters)
- Backoff ↔ Model simplification

---

## TODO

- [ ] Thêm mathematical derivation của Kneser-Ney
- [ ] Code example so sánh các smoothing methods
- [ ] Liên kết với [[Regularization]] concept note
