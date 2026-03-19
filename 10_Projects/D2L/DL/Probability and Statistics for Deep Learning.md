---
title: "Probability and Statistics for Deep Learning"
aliases:
  ["probability", "xác suất", "thống kê", "Bayes theorem", "random variable"]
tags: [concept, deep-learning, d2l, math, fundamentals, probability]
created: 2026-03-09
session: "D2L Tuần 1, Buổi 6 — Probability & Statistics"
source: "D2L Chapter Preliminaries - sec_prob"
related:
  - "[[Calculus for Deep Learning]]"
  - "[[Linear Algebra for Deep Learning]]"
  - "[[Automatic Differentiation]]"
---

# Probability and Statistics for Deep Learning

> [!NOTE] ELI5
> Mọi thứ trong Machine Learning đều liên quan đến **sự không chắc chắn**. Bạn không _biết chắc_ ảnh này là mèo hay chó — bạn chỉ có thể _ước lượng xác suất_. Xác suất là ngôn ngữ toán học để nói về sự không chắc chắn đó. Thống kê là công cụ để **học** từ data và **cập nhật** niềm tin của bạn về thế giới.

---

## 1. Tại sao Probability quan trọng trong DL?

| Tình huống DL                            | Xác suất tương ứng                                                                        |
| ---------------------------------------- | ----------------------------------------------------------------------------------------- |
| Phân loại ảnh (Image Classification)     | $P(\text{class} \mid \text{image})$ — **Posterior** (Xác suất hậu nghiệm)                 |
| Mô hình ngôn ngữ (Language Model)        | $P(\text{next\_word} \mid \text{context})$ — **Conditional prob** (Xác suất có điều kiện) |
| Phát hiện bất thường (Anomaly Detection) | $P(\mathbf{x})$ thấp → Data point bất thường                                              |
| Học tăng cường (Reinforcement Learning)  | $P(\text{reward} \mid \text{action, state})$ — **Expected reward** (Kỳ vọng phần thưởng)  |
| Hàm mất mát (Loss Function)              | CrossEntropy = $-\log P(y \mid \mathbf{x})$ — **Log-likelihood** (Log hợp lý)             |

## 2. Các Khái Niệm Cơ Bản

### Sample Space & Events (Không gian mẫu & Biến cố)

- **Sample space** $\mathcal{S}$ **(Không gian mẫu)**: Tập hợp tất cả kết quả có thể. VD tung xúc xắc: $\mathcal{S} = \{1,2,3,4,5,6\}$
- **Event** $\mathcal{A}$ **(Biến cố)**: Tập con của $\mathcal{S}$. VD "số lẻ" = $\{1,3,5\}$
- **Probability** $P(\mathcal{A}) \in [0, 1]$ **(Xác suất)**: Mức độ tin tưởng event sẽ xảy ra.

### 3 Tiên đề Kolmogorov (1933)

| Tiên đề                            | Phát biểu logic                                                                                                        |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **Non-negativity (Tính không âm)** | $P(\mathcal{A}) \geq 0$ với mọi event.                                                                                 |
| **Normalization (Tính chuẩn hóa)** | $P(\mathcal{S}) = 1$ — Tổng xác suất toàn bộ các trường hợp = 1.                                                       |
| **Additivity (Tính cộng)**         | Nếu $\mathcal{A}_i$ rời nhau (mutually exclusive): $P\!\left(\bigcup_i \mathcal{A}_i\right) = \sum_i P(\mathcal{A}_i)$ |

Từ 3 tiên đề này suy ra: $P(\emptyset) = 0$, $P(\mathcal{A}') = 1 - P(\mathcal{A})$.

## 3. Random Variables (Biến ngẫu nhiên)

> [!NOTE] ELI5 **Random variable** là một "hàm đặt tên" cho kết quả thí nghiệm. Thay vì nói "xảy ra sự kiện tung ra mặt ngửa", ta đặt "biến X nhận giá trị 1". Việc này cho phép dùng ngôn ngữ toán học linh hoạt hơn nhiều.

### Discrete (Rời rạc) vs Continuous (Liên tục)

|           | Discrete                                                 | Continuous                                                 |
| --------- | -------------------------------------------------------- | ---------------------------------------------------------- |
| Ví dụ     | Tung coin, lắc xúc xắc                                   | Chiều cao người, nhiệt độ                                  |
| Đặc trưng | $P(X = v)$ có ý nghĩa cụ thể                             | Xác suất tại 1 điểm chính xác $P(X = v) = 0$               |
| Phân phối | **PMF** (Probability Mass Function): $\sum_v P(X=v) = 1$ | **PDF** (Probability Density Function): $\int p(x) dx = 1$ |

**Trong DL:** Weights (trọng số) là continuous, class labels (nhãn phân loại) thường là discrete (one-hot encoding).

### Simulation bằng PyTorch (Mô phỏng)

```python
import torch
from torch.distributions.multinomial import Multinomial

# Mô phỏng 1000 lần tung coin công bằng
fair_probs = torch.tensor([0.5, 0.5])
counts = Multinomial(1000, fair_probs).sample()
print(counts / 1000)  # → xấp xỉ [0.5, 0.5] — Law of Large Numbers!

# Convergence (Sự hội tụ): tần suất → xác suất khi n → ∞
counts = Multinomial(1, fair_probs).sample((10000,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
# estimates[:, 0] dần dần hội tụ về 0.5
```

**Law of Large Numbers (Luật số lớn):** Khi $n \to \infty$, tần suất mẫu $\to$ xác suất thực. Tốc độ hội tụ: $O(1/\sqrt{n})$ — tức là tăng 100x mẫu thì giảm 10x sai số.

## 4. Joint, Marginal, Conditional Probability

### Joint Probability (Xác suất đồng thời)

$$P(A=a, B=b) \leq \min(P(A=a),\; P(B=b))$$

**Ý nghĩa:** Xác suất cả $A=a$ **và** $B=b$ đồng thời xảy ra.

### Marginal Probability (Xác suất biên)

$$P(A=a) = \sum_v P(A=a, B=v)$$

"Sum out" (lấy tổng để loại bỏ) biến $B$ để lấy lại phân phối của riêng $A$.

### Conditional Probability (Xác suất có điều kiện)

$$P(B=b \mid A=a) = \frac{P(A=a, B=b)}{P(A=a)}$$

> [!NOTE] ELI5 $P(B \mid A)$ = "Nếu tôi **đã biết** A xảy ra, thì xác suất B xảy ra là bao nhiêu?" Ví dụ: $P(\text{bệnh} \mid \text{test dương tính})$.

**Product rule (Quy tắc nhân):**

$$P(A, B) = P(B \mid A) P(A) = P(A \mid B) P(B)$$

## 5. Bayes' Theorem (Định lý Bayes) — Nền tảng của ML

$$\boxed{P(A \mid B) = \frac{P(B \mid A)\, P(A)}{P(B)}}$$

### Bayesian Interpretation (Diễn giải Bayes)

| Thuật ngữ     | Ý nghĩa                                                                       |
| ------------- | ----------------------------------------------------------------------------- |
| $P(A)$        | **Prior (Tiên nghiệm)**: Niềm tin ban đầu về $A$ (trước khi có data).         |
| $P(B \mid A)$ | **Likelihood (Khả năng/Độ hợp lý)**: Xác suất quan sát được $B$ nếu $A$ đúng. |
| $P(B)$        | **Evidence (Chứng cứ)**: Xác suất tổng quát của data $B$.                     |
| $P(A \mid B)$ | **Posterior (Hậu nghiệm)**: Niềm tin về $A$ **sau khi** quan sát $B$.         |

**Informal:** _Posterior ∝ Likelihood × Prior_

### Ví dụ HIV Test — Bayes rất phản trực giác!

Test có độ chính xác cao: $P(D_1=1 \mid H=0) = 0.01$ (False positive - Dương tính giả), $P(D_1=1 \mid H=1) = 1$ (True positive).

Prevalence (Tỷ lệ lưu hành bệnh): $P(H=1) = 0.0015$ (Prior).

Xác suất nhận kết quả dương tính (Evidence):

$$P(D_1=1) = P(D_1=1 \mid H=0)P(H=0) + P(D_1=1 \mid H=1)P(H=1)$$$$= 0.01 \times 0.9985 + 1 \times 0.0015 = 0.011485$$

Xác suất thực sự có bệnh (Posterior):

$$P(H=1 \mid D_1=1) = \frac{1 \times 0.0015}{0.011485} \approx \mathbf{13\%}$$

> [!WARNING] Base Rate Neglect (Lỗi phớt lờ tỷ lệ nền) Test dương tính nhưng chỉ 13% thực sự bệnh! Vì bệnh rất hiếm (Prior thấp), phần lớn ca dương tính là **False positives**. Đây là lý do cần **Prior** trong Bayes.

Nếu test lần 2 độc lập vẫn dương tính:

$$P(H=1 \mid D_1=1, D_2=1) \approx \mathbf{83\%}$$

Prior yếu có thể bị overcome (vượt qua) bởi nhiều evidence độc lập.

## 6. Independence (Tính độc lập)

### Independence (Độc lập tuyệt đối)

$$A \perp B \iff P(A, B) = P(A) P(B) \iff P(A \mid B) = P(A)$$

**Trong DL:** Mô hình Naive Bayes giả định các features độc lập — sai thực tế nhưng vẫn hoạt động tốt!

### Conditional Independence (Độc lập có điều kiện)

$$A \perp B \mid C \iff P(A, B \mid C) = P(A \mid C) P(B \mid C)$$

> [!NOTE] Paradox quan trọng: Hai biến **có thể correlation (tương quan) cao** nhưng **become independent (trở nên độc lập)** khi condition on (cố định) một biến thứ ba:
>
> - Shoe size và reading level — tương quan cao!
> - Nhưng nếu **condition on age (cố định độ tuổi)** → correlation biến mất.
>
> Nguyên nhân: **Age** là common cause (**Confounding variable** - Biến gây nhiễu). Đây là nền tảng của **Causal inference** (Suy luận nhân quả).

## 7. Expectation & Variance — Bản chất sâu

### Expectation (Kỳ vọng)

> [!NOTE] ELI5
> Kỳ vọng là **giá trị trung bình trong dài hạn** nếu lặp lại thí nghiệm vô số lần. Tung xúc xắc 1 lần không đoán được kết quả — nhưng tung 1 triệu lần, trung bình sẽ xấp xỉ 3.5. Đó là kỳ vọng.

$$E[X] = \sum_x x\, P(X=x) \quad \text{(discrete)} \qquad E[X] = \int x\, p(x)\, dx \quad \text{(continuous)}$$

**Bản chất:** Kỳ vọng là **trung tâm trọng lực** của phân phối — điểm mà nếu đặt phân phối lên một đòn bẩy, nó sẽ cân bằng.

**Linearity (Tính tuyến tính):** $E[aX + bY] = aE[X] + bE[Y]$ — luôn đúng dù $X, Y$ có independent hay không! Property cực mạnh vì rất nhiều phép tính trong DL đúng nhờ nó.

**Ứng dụng trực tiếp trong DL:**

- **Loss function** thực chất là _expected loss_ trên data distribution: $\mathcal{L} = E_{(x,y) \sim P}[\ell(f(x), y)]$
- Ta không biết $P$ nên dùng **sample mean** trên batch làm approximation → đây chính là lý do cộng dồn loss rồi chia cho batch size.
- **Expected reward** trong RL: $Q(s,a) = E[\sum_t \gamma^t r_t \mid s_0=s, a_0=a]$

### Variance (Phương sai) — Tại sao lại bình phương?

> [!NOTE] ELI5
> Phương sai đo **mức độ "không ổn định"** của một đại lượng ngẫu nhiên. Model dự đoán giá nhà với variance thấp đáng tin hơn model cùng kết quả nhưng variance cao. Trong DL, variance cao ↔ model không ổn định (overfit, nhạy cảm với input).

$$\text{Var}[X] = E\!\left[(X - E[X])^2\right] = E[X^2] - (E[X])^2$$

**Bản chất — Tại sao không dùng $E[|X - E[X]|]$ (absolute deviation)?**

Ta _có thể_ dùng absolute deviation, nhưng bình phương có 3 ưu điểm kỹ thuật quyết định:

1. **Differentiable everywhere** — $x^2$ có đạo hàm tại mọi điểm; $|x|$ không có đạo hàm tại $x=0$. Rất quan trọng khi optimize bằng gradient.
2. **Penalizes outliers mạnh hơn** — bình phương phạt nặng các điểm xa trung tâm hơn → phản ánh rủi ro tốt hơn trong nhiều ứng dụng.
3. **Algebraic convenience** — công thức $E[X^2] - E[X]^2$ dễ tính và dễ thao tác đại số hơn nhiều.

**Tại sao $E[X - E[X]] = 0$? (và tại sao đó là vấn đề)**
$$E[X - E[X]] = E[X] - E[E[X]] = \mu - \mu = 0$$
Các độ lệch dương và âm luôn triệt tiêu nhau — không đo được gì! Bình phương loại bỏ dấu âm trước khi cộng.

**Chứng minh công thức rút gọn:**
$$\text{Var}[X] = E[(X-\mu)^2] = E[X^2 - 2\mu X + \mu^2] = E[X^2] - 2\mu^2 + \mu^2 = E[X^2] - \mu^2$$

**Ví dụ đầu tư (từ D2L):**

- 50% mất trắng (return=0), 40% lãi 2×, 10% lãi 10×
- $E[\text{return}] = 0.5 \times 0 + 0.4 \times 2 + 0.1 \times 10 = 1.8$
- $E[\text{return}^2] = 0.5 \times 0 + 0.4 \times 4 + 0.1 \times 100 = 11.6$
- $\text{Var} = 11.6 - 1.8^2 = 11.6 - 3.24 = 8.36$ — rủi ro rất cao so với expected return 1.8!

**Ứng dụng trong DL:**

- **Batch Normalization:** chuẩn hóa theo $\mu$ và $\sigma$ từng layer → giữ distribution ổn định khi training
- **Weight Initialization (Xavier/He):** chọn $\sigma$ của init distribution để gradient không vanish/explode qua nhiều layers
- **Uncertainty estimation:** dự đoán ± $k\sigma$ trong probabilistic models

### Standard Deviation (Độ lệch chuẩn) — Tại sao cần sqrt?

$$\sigma = \sqrt{\text{Var}[X]}$$

Variance có **đơn vị bình phương** — nếu $X$ là điểm thi (điểm), thì $\text{Var}[X]$ có đơn vị điểm². Vô nghĩa để interpret! Lấy $\sqrt{}$ trả đơn vị gốc:

- Model dự đoán giá nhà với $\sigma = \$20{,}000$ → dự đoán thường lệch ~$20K — _dễ hiểu_
- Nếu chỉ nói $\sigma^2 = 4 \times 10^8\ \$^2$ → không ai hình dung được

Ký hiệu chuẩn: mean = $\mu$, variance = $\sigma^2$ — xuất hiện trong mọi phân phối Gaussian.

### Covariance Matrix (Ma trận Hiệp phương sai)

> [!NOTE] ELI5
> Nếu variance đo "feature này spread bao nhiêu", thì covariance đo "khi feature A tăng, feature B có xu hướng tăng hay giảm hay không liên quan?" Covariance matrix gói gọn **tất cả mối quan hệ** đó thành một ma trận.

$$\boldsymbol{\Sigma} = E\!\left[(\mathbf{x} - \boldsymbol{\mu})(\mathbf{x} - \boldsymbol{\mu})^\top\right] \in \mathbb{R}^{n \times n}$$

**Đọc covariance matrix:**

| Vị trí                     | Ý nghĩa                    | Ví dụ                                      |
| -------------------------- | -------------------------- | ------------------------------------------ |
| $\Sigma_{ii}$ (đường chéo) | Variance của feature $i$   | Spread của chiều cao trong dataset         |
| $\Sigma_{ij} > 0$          | Hai feature cùng tăng/giảm | Chiều cao & cân nặng tương quan dương      |
| $\Sigma_{ij} < 0$          | Một tăng thì kia giảm      | Tốc độ xe & thời gian đến nơi nghịch chiều |
| $\Sigma_{ij} \approx 0$    | Không tương quan           | Cân nặng & điểm toán                       |

**Ứng dụng trong ML:**

- **PCA:** diagonalize $\boldsymbol{\Sigma}$ → tìm principal components (hướng variance lớn nhất) để giảm chiều dữ liệu
- **Gaussian Mixture Models:** mỗi cluster có covariance matrix riêng, mô tả hình dạng cluster
- **Markowitz Portfolio:** minimize $\mathbf{\alpha}^\top \boldsymbol{\Sigma} \mathbf{\alpha}$ (variance của portfolio) subject to expected return constraint
- **Feature decorrelation:** nếu $\boldsymbol{\Sigma}$ nhiều off-diagonal lớn → features redundant, có thể giảm chiều

## 8. Aleatoric vs Epistemic Uncertainty

|              | **Aleatoric Uncertainty** (Bất định nội tại)            | **Epistemic Uncertainty** (Bất định nhận thức) |
| ------------ | ------------------------------------------------------- | ---------------------------------------------- |
| Nguồn gốc    | Bản chất ngẫu nhiên, noise (nhiễu) của thế giới.        | Thiếu kiến thức/data.                          |
| Ví dụ        | Biết coin fair 50/50 nhưng không đoán được mặt kế tiếp. | Chưa biết xác suất $p$ của 1 đồng xu lạ.       |
| Có thể giảm? | **Không** — luôn tồn tại.                               | **Có** — thêm data sẽ giúp model học được.     |
| Trong DL     | Noise trong data, overlapping classes.                  | **Model uncertainty**, **Distribution shift**. |

> [!NOTE] Ứng dụng **Bayesian Deep Learning** cố gắng ước lượng cả hai loại uncertainty này. Rất quan trọng trong AI Y tế, xe tự lái — nơi mà "tôi không chắc" cũng là thông tin có giá trị.

## 9. Chebyshev's Inequality (Bất đẳng thức Chebyshev)

$$P(|X - \mu| \geq k\sigma) \leq \frac{1}{k^2}$$

- Với $k=2$: xác suất data point lệch khỏi mean $> 2\sigma$ là $\leq 25\%$.
- Không giả định phân phối cụ thể (như Gaussian) — **universally applicable** (luôn áp dụng được)!

## 10. Kết nối với Deep Learning — Bức tranh toàn cảnh

```
Maximum Likelihood Estimation (MLE - Ước lượng hợp lý cực đại):
    θ* = argmax_θ ∏ P(xᵢ | θ)
       = argmin_θ -∑ log P(xᵢ | θ)
       = Chính là việc minimize Cross-Entropy Loss!

Bayesian Inference (Suy luận Bayes):
    P(θ | data) ∝ P(data | θ) × P(θ)
    (Posterior)   (Likelihood)  (Prior = Regularization)

L2 regularization ↔ Tương đương Gaussian prior on weights
L1 regularization ↔ Tương đương Laplace prior on weights
```

## Exercises (Bài tập từ D2L)

1. Tính variance của estimator $\hat{p} = n_H/n$ (số heads chia n). Làm thế nào variance scale theo $n$?
2. Dùng Chebyshev's inequality để bound deviation từ kỳ vọng.
3. Chứng minh: $E[X - E[X]] = 0$ (không thể dùng để đo spread!)
4. Cho $A, B, C$ là chuỗi Markov (B chỉ phụ thuộc A, C chỉ phụ thuộc B). Đơn giản hóa $P(A,B,C)$.

---

> [!TODO] Mở rộng
>
> - [[Distributions in Deep Learning]] — Gaussian, Bernoulli, Categorical, Dirichlet
> - [[Maximum Likelihood Estimation]] — liên hệ với loss functions
> - [[Bayesian Deep Learning]] — uncertainty quantification
> - [[Information Theory for ML]] — entropy, KL divergence, mutual information
