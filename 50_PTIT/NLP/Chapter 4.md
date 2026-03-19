---
tags:
  - nlp
  - ptit
  - source-note
  - classification
  - sentiment-analysis
  - naive-bayes
status: completed
source:
  - assets/Library/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang.pdf
aliases:
  - Sentiment Chapter
  - Chapter 4 NLP PTIT
---

# Chapter 4 — Text Classification and Sentiment Analysis

> [!NOTE] Source
> Tài liệu gốc: `assets/Library/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang.pdf` (Chapter 4, slide 1-28). Nội dung dưới đây bám sát 100% cấu trúc gốc với phần ELI5 và giải thích mục đích/ứng dụng bổ sung.

---

## 1. Text Classification Overview

> [!NOTE] ELI5
> Phân loại văn bản giống như việc sắp xếp thư: bạn nhìn phong bì và quyết định "cái này là hóa đơn", "cái này là quảng cáo", "cái này là thư từ bạn". Máy tính cũng làm tương tự với emails, reviews, và documents.

### 1.1 Định nghĩa

[[Text Classification]] là tác vụ gán **category/label** cho một văn bản.

**Formal definition:**
- **Input:** Document $d$ (sentence, paragraph, hoặc full document)
- **Output:** Label $y$ từ tập categories đã định nghĩa

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_4/img-003.jpg]]

### 1.2 Ứng dụng

- **Spam Detection:** Email là spam hay không?
- **Sentiment Analysis:** Review là positive hay negative?
- **Topic Classification:** Bài báo thuộc category nào? (sports, politics, technology...)
- **Language Identification:** Văn bản viết bằng ngôn ngữ gì?

---

## 2. Bayes' Theorem

> [!NOTE] ELI5
> Bạn thấy ai đó mặc áo bóng đá và nghĩ "chắc người này thích bóng đá". Đây là suy luận ngược: từ **quan sát** (áo) bạn đoán **nguyên nhân** (sở thích). Bayes' theorem là công thức toán học cho việc suy luận ngược này.

### 2.1 Công thức

[[Bayes' Theorem]]:

$$P(h|D) = \frac{P(D|h) \cdot P(h)}{P(D)}$$

| Thành phần | Tên            | Ý nghĩa                                         |
| ---------- | -------------- | ----------------------------------------------- |
| $P(h)$     | **Prior**      | Xác suất của hypothesis $h$ trước khi thấy data |
| $P(D\|h)$  | **Likelihood** | Xác suất thấy data $D$ nếu $h$ đúng             |
| $P(D)$     | **Evidence**   | Xác suất của data (thường ignore vì constant)   |
| $P(h\|D)$  | **Posterior**  | Xác suất của $h$ sau khi thấy data              |

**Mục đích:** Cho phép cập nhật beliefs dựa trên evidence mới.

### 2.2 Maximum A Posteriori (MAP)

Chọn hypothesis có posterior cao nhất:

$$h_{MAP} = \argmax_{h \in H} P(h|D) = \argmax_{h \in H} P(D|h) \cdot P(h)$$

(Bỏ $P(D)$ vì không phụ thuộc vào $h$)

**Ứng dụng cho classification:** Tìm class $C$ có probability cao nhất given document features.

---

## 3. Naive Bayes Classifier

> [!NOTE] ELI5
> Naive Bayes đếm: "Trong các reviews tích cực, từ 'great' xuất hiện bao nhiêu lần? Còn trong reviews tiêu cực?" Nếu review mới có từ 'great', và 'great' xuất hiện nhiều trong positive → khả năng cao review là positive. **Naive** vì nó giả định các từ **độc lập** với nhau.

### 3.1 Áp dụng Bayes cho Text Classification

**Bài toán:** Cho document $D$ với features $X = (x_1, x_2, ..., x_n)$, tìm class $C_i$ tốt nhất:

$$C_{best} = \argmax_{C_i} P(C_i|X) = \argmax_{C_i} P(X|C_i) \cdot P(C_i)$$

**Vấn đề:** $P(X|C_i) = P(x_1, x_2, ..., x_n | C_i)$ rất khó ước lượng trực tiếp (exponential combinations).

### 3.2 Naive Assumption — Conditional Independence

**Giả định Naive:** Các features **độc lập với nhau given the class**:

$$P(X|C_i) = P(x_1, x_2, ..., x_n | C_i) = \prod_{k=1}^{n} P(x_k | C_i)$$

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_4/img-001.png]]

**Naive Bayes Classifier:**

$$C_{NB} = \argmax_{C_i} P(C_i) \prod_{k=1}^{n} P(x_k | C_i)$$

> [!NOTE] Tại sao "naive" nhưng hiệu quả?
> Giả định independence rõ ràng **sai** (từ trong câu liên quan mạnh!). Nhưng:
> 1. **Argmax không cần probability chính xác:** Chỉ cần ranking đúng
> 2. **Dependencies "cancel out":** Errors có thể bù trừ
> 3. **Robust với ít data:** Ít parameters cần estimate
>
> Trong thực tế, Naive Bayes thường competitive với complex models, đặc biệt khi data ít.

### 3.3 Training the Model

#### Prior Probabilities

$$P(C_i) = \frac{\text{Số documents thuộc class } C_i}{\text{Tổng số documents}}$$

#### Conditional Probabilities (Likelihoods)

$$P(x_k | C_i) = \frac{\text{Count của từ } x_k \text{ trong class } C_i}{\text{Tổng số từ trong class } C_i}$$

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_4/img-002.png]]

#### Handling Continuous Attributes

Với **continuous-valued attributes**, giả định Gaussian distribution:

$$P(x_k | C_i) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x_k - \mu)^2}{2\sigma^2}}$$

Trong đó $\mu$ và $\sigma$ được estimate từ training data.

### 3.4 Zero Problem

**Vấn đề:** Nếu từ $x_k$ không xuất hiện trong class $C_i$ training data:

$$P(x_k | C_i) = 0 \Rightarrow P(X|C_i) = 0$$

Cả product trở thành 0, dù các từ khác có probability cao.

**Giải pháp:** [[Smoothing (NLP)|Smoothing]] — thường dùng Add-one (Laplace):

$$P(x_k | C_i) = \frac{C(x_k, C_i) + 1}{\sum_w C(w, C_i) + V}$$

Trong đó $V$ là vocabulary size.

---

## 4. Sentiment Classification với Naive Bayes

> [!NOTE] ELI5
> Khi bạn đọc review "Khách sạn này tuyệt vời!" bạn biết ngay đó là **khen**. "Dịch vụ tệ quá!" là **chê**. Sentiment classification dạy máy làm điều tương tự.

### 4.1 Định nghĩa Sentiment Classification

[[Sentiment Analysis]] (Opinion Mining) là tác vụ xác định **sentiment orientation** của văn bản:

- **Binary:** Positive vs Negative
- **Multi-class:** Positive / Neutral / Negative
- **Fine-grained:** Scale 1-5 sao

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_4/img-004.jpg]]

### 4.2 Data Preprocessing

**Tại sao preprocessing quan trọng?**
- Giảm noise, cải thiện accuracy
- Chuẩn hóa dữ liệu

**Các bước phổ biến:**
1. **Tokenization:** Tách văn bản thành tokens
2. **Lowercasing:** Chuyển về chữ thường
3. **Removing punctuation:** Loại dấu câu
4. **Removing stop words:** Loại từ không mang nghĩa (the, is, a...)
5. **Stemming/Lemmatization:** Chuẩn hóa hình thái từ

**Ví dụ từ tài liệu gốc:**
```
Input: "I love this movie! It's amazing."
After preprocessing: "love movie amazing"
```

### 4.3 Ví dụ chi tiết — Step by Step

**Training Data:**

| Text | Sentiment |
|------|-----------|
| "I love this movie!" | Positive |
| "This movie is terrible" | Negative |
| "It was boring." | Negative |
| "This movie is great!" | Positive |

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_4/img-005.jpg]]

**Step 1: Data Preprocessing**

Tokenize và remove punctuation:
- "I love this movie"
- "This movie is terrible"
- "It was boring"
- "This movie is great"

**Step 2: Prior Probabilities**

$$P(Positive) = \frac{2}{4} = 0.5$$
$$P(Negative) = \frac{2}{4} = 0.5$$

**Step 3: Conditional Probabilities (Likelihoods)**

| Class | Words | Total |
|-------|-------|-------|
| Positive | I, love, this, movie, this, movie, is, great | 8 |
| Negative | this, movie, is, terrible, it, was, boring | 7 |

Word counts:

| Word | Positive count | Negative count |
|------|---------------|----------------|
| this | 2 | 1 |
| movie | 2 | 1 |
| is | 1 | 1 |
| love | 1 | 0 |
| great | 1 | 0 |
| terrible | 0 | 1 |
| boring | 0 | 1 |

**Step 4: Classify "This movie is boring."**

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_4/img-006.jpg]]

$$P(Positive | \text{"this movie is boring"}) \propto P(Pos) \times P(\text{this}|Pos) \times P(\text{movie}|Pos) \times P(\text{is}|Pos) \times P(\text{boring}|Pos)$$

$$= 0.5 \times \frac{2}{8} \times \frac{2}{8} \times \frac{1}{8} \times \frac{0+1}{8+V} = \text{small value}$$

$$P(Negative | \text{"this movie is boring"}) \propto P(Neg) \times P(\text{this}|Neg) \times P(\text{movie}|Neg) \times P(\text{is}|Neg) \times P(\text{boring}|Neg)$$

$$= 0.5 \times \frac{1}{7} \times \frac{1}{7} \times \frac{1}{7} \times \frac{1}{7} = 0.0002$$

**Kết quả:** $P(Negative) > P(Positive)$ → Classify as **Negative** ✓

---

## 5. Evaluation Metrics

### 5.1 Confusion Matrix

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_4/img-007.jpg]]

|  | Predicted Positive | Predicted Negative |
|--|-------------------|-------------------|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

### 5.2 Precision

**"Trong những cái ta predict positive, bao nhiêu % thực sự positive?"**

$$Precision = \frac{TP}{TP + FP}$$

**Mục đích:** Đo chất lượng của positive predictions.

**Ứng dụng:** Quan trọng khi false positives tốn kém (ví dụ: spam filter không nên đánh dấu email quan trọng là spam).

### 5.3 Recall (Sensitivity)

**"Trong những cái thực sự positive, ta tìm được bao nhiêu %?"**

$$Recall = \frac{TP}{TP + FN}$$

**Mục đích:** Đo khả năng "tìm ra" tất cả positives.

**Ứng dụng:** Quan trọng khi missing positives tốn kém (ví dụ: medical diagnosis không nên bỏ sót bệnh).

### 5.4 F1-Score

**Harmonic mean** của Precision và Recall:

$$F_1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

**Tại sao harmonic mean?** Penalize extreme imbalance giữa P và R.

**Ví dụ:**
- $P = 1.0, R = 0.1$ → $F_1 = 0.18$ (low, vì R quá thấp)
- $P = 0.5, R = 0.5$ → $F_1 = 0.50$ (balanced)

**Generalized F-score** với factor $\beta$ (recall là $\beta$ lần quan trọng hơn precision):

$$F_\beta = (1 + \beta^2) \times \frac{Precision \times Recall}{\beta^2 \times Precision + Recall}$$

### 5.5 Accuracy

$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

**Caveat:** Misleading khi class imbalanced! 

Ví dụ: Nếu 95% data là negative, predict "negative" cho mọi thứ → 95% accuracy nhưng **useless classifier**.

---

## 6. Test Sets và Cross-Validation

### 6.1 Importance of Test Sets

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_4/img-008.png]]

**Test Sets:**
- Separate dataset để evaluate model
- Simulate cách model sẽ perform trên new, unseen data
- Prevent overfitting, cung cấp unbiased evaluation

**Why separate train/test?**
- Đo **generalization**, không phải **memorization**
- Model tốt trên training nhưng tệ trên test → **overfitting**

### 6.2 K-Fold Cross-Validation

**Quy trình:**
1. Chia data thành $k$ folds (thường $k = 5$ hoặc $10$)
2. Lặp $k$ lần:
   - Dùng fold $i$ làm test
   - Dùng $k-1$ folds còn lại làm train
   - Tính metrics
3. Average metrics qua $k$ lần

**Ưu điểm:**
- Sử dụng tất cả data cho cả train và test
- Estimate variance của performance
- Robust hơn single train/test split

> [!NOTE] Development Set
> Trong thực tế còn có **development (validation) set** để tune hyperparameters:
> - **Train:** Train model
> - **Dev:** Tune hyperparameters
> - **Test:** Final evaluation (chỉ dùng **một lần cuối**)

---

## 7. Naive Bayes: Ưu và Nhược Điểm

### 7.1 Ưu Điểm

| Ưu điểm | Giải thích |
|---------|------------|
| **Simple** | Chỉ cần đếm và normalize |
| **Fast** | Training và inference đều $O(n)$ |
| **Robust với ít data** | Ít parameters, ít overfitting |
| **Baseline mạnh** | Thường khó beat với small datasets |
| **Interpretable** | Có thể inspect $P(\text{word}\|class)$ |

### 7.2 Nhược Điểm

| Nhược điểm | Giải thích |
|------------|------------|
| **Independence assumption** | Sai về mặt ngôn ngữ |
| **Không capture word order** | "not good" vs "good" đều có "good" |
| **Sensitive to feature selection** | Cần careful preprocessing |
| **Probability estimates unreliable** | Chỉ ranking đáng tin, không phải calibrated probabilities |

---

## 8. Beyond Naive Bayes

### 8.1 Các Classifiers khác cho Text

- **Logistic Regression:** Discriminative model, nhiều features
- **Support Vector Machines (SVM):** Strong baseline cho text
- **Neural Networks:** LSTM, CNN, Transformer-based (BERT, etc.)

### 8.2 Advanced Sentiment Analysis

- **Aspect-based:** Sentiment cho từng aspect (food: positive, service: negative)
- **Fine-grained:** 5-class scale thay vì binary
- **Domain adaptation:** Transfer learning giữa domains
- **Multimodal:** Kết hợp text + image + audio

---

## 9. Kết Luận

Chapter này giới thiệu **Text Classification** và **Sentiment Analysis** với **Naive Bayes**:

1. **Bayes' Theorem** là nền tảng để suy luận từ evidence

2. **Naive Bayes** giả định independence, đơn giản nhưng hiệu quả

3. **Training:** Estimate priors và likelihoods từ labeled data

4. **Zero problem:** Giải quyết bằng smoothing

5. **Evaluation:** Precision, Recall, F1 — quan trọng hơn accuracy cho imbalanced data

6. **Cross-validation:** Đánh giá robust hơn single split

Naive Bayes là **baseline quan trọng**, nhưng neural approaches (BERT, etc.) thường cho kết quả tốt hơn cho sentiment analysis hiện đại.

---

## TODO

- [ ] Liên kết với [[Logistic Regression]] và [[Support Vector Machine]]
- [ ] Thêm code example với scikit-learn
- [ ] So sánh với BERT-based sentiment classifiers
