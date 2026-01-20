---
tags:
  - ai
  - machine-learning
  - classification
  - probabilistic-model
  - supervised-learning
aliases:
  - Naive Bayes Classifier
  - NB
  - Multinomial Naive Bayes
status: evergreen
related:
  - "[[Bayes Theorem]]"
  - "[[Text Classification]]"
  - "[[Sentiment Analysis]]"
  - "[[Maximum Likelihood Estimation]]"
---

# Naive Bayes

> [!NOTE] ELI5
> Tưởng tượng bạn là bác sĩ và bệnh nhân có triệu chứng: sốt + ho + mệt mỏi. Bạn nghĩ: "Cảm cúm thường có sốt, ho, và mệt. COVID cũng vậy. Nhưng cảm cúm phổ biến hơn..." Bạn kết hợp **mức độ phổ biến** của bệnh với **mức độ phù hợp** của triệu chứng để đoán. **Naive Bayes** làm tương tự: tính xác suất từng class dựa trên features, rồi chọn class có xác suất cao nhất.

---

## 1. Nền Tảng: Bayes' Theorem

### 1.1 Công Thức Bayes

$$P(C|X) = \frac{P(X|C) \cdot P(C)}{P(X)}$$

Trong đó:
- $P(C)$: **Prior** — xác suất của class C trước khi thấy data
- $P(X|C)$: **Likelihood** — xác suất thấy features X nếu class là C
- $P(X)$: **Evidence** — xác suất của features (constant cho tất cả classes)
- $P(C|X)$: **Posterior** — xác suất của class C sau khi thấy features X

### 1.2 Classification Rule

Chọn class có posterior cao nhất:

$$C_{best} = \argmax_C P(C|X) = \argmax_C P(X|C) \cdot P(C)$$

(Bỏ $P(X)$ vì không phụ thuộc vào $C$)

---

## 2. "Naive" Assumption

### 2.1 Vấn Đề Với Joint Probability

Với $n$ features $X = (x_1, x_2, ..., x_n)$:

$$P(X|C) = P(x_1, x_2, ..., x_n | C)$$

Ước lượng trực tiếp: cần $O(|X|^n)$ parameters — **exponential**!

### 2.2 Conditional Independence Assumption

**Giả định:** Các features **độc lập với nhau given the class**:

$$P(X|C) = \prod_{i=1}^{n} P(x_i | C)$$

**Hệ quả:** Chỉ cần $O(n \cdot |X|)$ parameters — **linear**!

### 2.3 Tại Sao "Naive"?

Giả định này **hầu như luôn sai** trong thực tế:
- Trong text: "New" và "York" highly correlated
- Trong medical: Symptoms often co-occur

**Nhưng vẫn hiệu quả vì:**
1. Chỉ cần ranking đúng, không cần probability chính xác
2. Errors tend to cancel out
3. Robust với ít data

---

## 3. Naive Bayes Classifier

### 3.1 Decision Rule

$$C_{NB} = \argmax_C P(C) \prod_{i=1}^{n} P(x_i | C)$$

### 3.2 Log-space (Numerical Stability)

Tích nhiều xác suất nhỏ → underflow. Chuyển sang log:

$$C_{NB} = \argmax_C \left[ \log P(C) + \sum_{i=1}^{n} \log P(x_i | C) \right]$$

---

## 4. Variants

### 4.1 Multinomial Naive Bayes

Dùng cho **text classification** với word counts:

$$P(x_i | C) = \frac{count(x_i, C) + \alpha}{\sum_w count(w, C) + \alpha |V|}$$

Trong đó $\alpha$ là smoothing parameter (thường = 1 cho Laplace).

### 4.2 Bernoulli Naive Bayes

Dùng **binary features** (word presence/absence):

$$P(x|C) = P(x_i = 1|C)^{x_i} \cdot P(x_i = 0|C)^{1-x_i}$$

### 4.3 Gaussian Naive Bayes

Dùng cho **continuous features**, giả định Gaussian distribution:

$$P(x_i | C) = \frac{1}{\sqrt{2\pi\sigma_C^2}} \exp\left(-\frac{(x_i - \mu_C)^2}{2\sigma_C^2}\right)$$

---

## 5. Training

### 5.1 Maximum Likelihood Estimation

**Prior:**
$$P(C) = \frac{N_C}{N}$$

**Likelihood (Multinomial):**
$$P(x_i | C) = \frac{count(x_i, C)}{\sum_w count(w, C)}$$

### 5.2 Smoothing

Để tránh zero probabilities:

$$P(x_i | C) = \frac{count(x_i, C) + \alpha}{\sum_w count(w, C) + \alpha |V|}$$

$\alpha = 1$: Laplace smoothing
$\alpha < 1$: Lidstone smoothing

---

## 6. Ưu và Nhược Điểm

### 6.1 Ưu Điểm

| Aspect | Description |
|--------|-------------|
| **Fast** | Training $O(n)$, inference $O(n)$ |
| **Simple** | Chỉ đếm và normalize |
| **Scalable** | Handle high-dimensional data |
| **Robust** | Ít overfitting với small data |
| **Interpretable** | Có thể inspect feature probabilities |
| **Baseline** | Strong baseline for many tasks |

### 6.2 Nhược Điểm

| Aspect | Description |
|--------|-------------|
| **Independence** | Giả định thường sai |
| **Probability calibration** | Probabilities không reliable |
| **Feature correlations** | Không capture interactions |
| **Continuous features** | Gaussian assumption có thể sai |

---

## 7. Ứng Dụng

- **Spam filtering:** Classify email as spam/not spam
- **Sentiment analysis:** Positive/negative classification
- **Document categorization:** Topic classification
- **Medical diagnosis:** Disease prediction from symptoms
- **Recommendation systems:** User preference prediction

---

## 8. So Sánh Với Các Classifiers Khác

| Aspect | Naive Bayes | Logistic Regression | SVM |
|--------|-------------|--------------------|----|
| Model type | Generative | Discriminative | Discriminative |
| Training speed | Very fast | Fast | Slow |
| Feature interactions | No | Limited | With kernels |
| Probability output | Yes (uncalibrated) | Yes (calibrated) | No (need calibration) |
| Small data performance | Good | Moderate | Good |

---

## 9. Code Example

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Training data
texts = ["I love this movie", "This is terrible", 
         "Great film", "Worst ever"]
labels = [1, 0, 1, 0]  # 1 = positive, 0 = negative

# Feature extraction
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train Naive Bayes
clf = MultinomialNB()
clf.fit(X, labels)

# Predict
new_text = ["This movie is great"]
new_X = vectorizer.transform(new_text)
prediction = clf.predict(new_X)  # [1] = positive
```

---

## TODO

- [ ] Thêm mathematical derivation của MLE
- [ ] So sánh calibration với Platt scaling
- [ ] Liên kết với [[Generative vs Discriminative Models]]
