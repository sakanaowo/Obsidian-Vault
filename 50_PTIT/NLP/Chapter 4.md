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

# Chapter 4 — Sentiment Classification

> [!NOTE] Source
> Tài liệu gốc: `assets/Library/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang.pdf` (Chapter 4). Nội dung dưới đây là **dịch + diễn giải có phê bình** dựa trên slide; các đoạn được đánh dấu "Suy luận thêm" là phần mở rộng từ kiến thức nền.

---

## 1. Opinion Mining và Sentiment Classification

> [!NOTE] ELI5
> Khi bạn đọc review "Khách sạn này tuyệt vời!" bạn biết ngay đó là **khen**. Còn "Dịch vụ tệ quá!" là **chê**. **Sentiment classification** dạy máy tính làm điều tương tự: đọc văn bản và phân loại nó là tích cực (positive) hay tiêu cực (negative). Điều này hữu ích cho việc phân tích hàng ngàn reviews tự động.

### 1.1 Định Nghĩa Bài Toán

[[Sentiment Analysis]] (còn gọi là Opinion Mining) là tác vụ xác định **sentiment orientation** của một văn bản:
- **Binary classification:** Positive vs Negative
- **Multi-class:** Positive / Neutral / Negative
- **Fine-grained:** Scale 1-5 sao

**Formal definition:**
- **Input:** Document $d$ (có thể là sentence, paragraph, hoặc full document)
- **Output:** Label $y \in \{positive, negative\}$ (hoặc multi-class)

**Ví dụ:**
> *"This is by far the worst hotel experience i've ever had. The owner overbooked while i was staying there (even though i booked the room two months in advance) and made me move to another room, but that room wasn't even a hotel room!"*

**Label:** Negative ✓

### 1.2 Challenges Của Data-Driven Approaches

**1. Domain Dependence:**
- Từ "unpredictable" là negative cho car reviews, nhưng có thể positive cho movie reviews (thriller)
- Model train trên restaurant reviews có thể fail trên electronics reviews

**2. Sarcasm và Irony:**
- *"Great, another bug in the software!"* — bề mặt positive, thực tế negative

**3. Negation:**
- *"This movie is not bad"* — "bad" là negative word, nhưng "not bad" = positive

**4. Aspect-based Sentiment:**
- *"The food was great but the service was terrible"* — mixed sentiment

---

## 2. Supervised Learning For Text Classification

> [!NOTE] ELI5
> Để dạy máy phân loại, ta cho nó xem **nhiều ví dụ đã có nhãn**: "Đây là review tích cực, đây là tiêu cực, đây là tích cực...". Máy học từ những ví dụ này để tự phân loại reviews mới. Giống như dạy trẻ nhận biết màu sắc bằng cách chỉ nhiều đồ vật và nói "cái này đỏ, cái kia xanh".

### 2.1 Quy Trình Tổng Quát

```
Raw Text → Preprocessing → Feature Extraction → Train Classifier → Predict
```

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_4/img-003.jpg]]

### 2.2 Text Representation: Bag of Words (BoW)

Văn bản cần được chuyển thành **vector số** để classifier xử lý. Cách đơn giản nhất: **Bag of Words**.

**Ý tưởng:** Biểu diễn document bằng các từ nó chứa, **bỏ qua thứ tự**.

**Feature types:**
- **Binary:** 1 nếu từ xuất hiện, 0 nếu không
- **Term Frequency (TF):** Số lần từ xuất hiện
- **TF-IDF:** TF × Inverse Document Frequency

**Ví dụ:**

| Document | "ant" | "book" | "car" | "food" | ... |
|----------|-------|--------|-------|--------|-----|
| Doc 1 | 0 | 1 | 0 | 0 | ... |
| Doc 2 | 1 | 0 | 0 | 0 | ... |
| Doc 3 | 0 | 0 | 0 | 1 | ... |

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_4/img-004.jpg]]

### 2.3 Preprocessing Steps

1. **Tokenization:** Tách văn bản thành tokens
2. **Lowercasing:** Chuyển về chữ thường (thường)
3. **Stopwords Removal:** Loại bỏ từ không mang nội dung (the, is, a...)
4. **Stemming/Lemmatization:** Chuẩn hóa hình thái từ (running → run)

**Trade-off:** Mỗi bước preprocessing có thể mất thông tin hữu ích!

---

## 3. Naive Bayes Classifier

> [!NOTE] ELI5
> Naive Bayes đếm: "Trong các reviews tích cực, từ 'great' xuất hiện bao nhiêu lần? Còn trong reviews tiêu cực?" Nếu review mới có từ 'great', và 'great' xuất hiện nhiều trong positive hơn negative → khả năng cao review mới là positive. **Naive** vì nó giả định các từ **độc lập** với nhau (giả định đơn giản nhưng hoạt động tốt!).

### 3.1 Bayes' Theorem

Nền tảng của [[Naive Bayes]] là **Bayes' Theorem**:

$$P(h|D) = \frac{P(D|h) \cdot P(h)}{P(D)}$$

Trong đó:
- $P(h)$: **Prior probability** — xác suất của hypothesis h trước khi thấy data
- $P(D|h)$: **Likelihood** — xác suất thấy data D nếu h đúng
- $P(D)$: **Evidence** — xác suất của data (thường ignore vì constant)
- $P(h|D)$: **Posterior probability** — xác suất của h sau khi thấy data

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_4/img-000.jpg]]

### 3.2 Maximum A Posteriori (MAP)

Chọn hypothesis có posterior cao nhất:

$$h_{MAP} = \argmax_{h \in H} P(h|D) = \argmax_{h \in H} P(D|h) \cdot P(h)$$

(Bỏ $P(D)$ vì không phụ thuộc vào $h$)

### 3.3 Áp Dụng Cho Text Classification

**Bài toán:** Cho document $D$ với features $X = (x_1, x_2, ..., x_n)$, tìm class $C_i$ tốt nhất.

$$C_{best} = \argmax_{C_i} P(C_i|X) = \argmax_{C_i} P(X|C_i) \cdot P(C_i)$$

**Vấn đề:** $P(X|C_i) = P(x_1, x_2, ..., x_n | C_i)$ rất khó ước lượng trực tiếp (exponential nhiều combinations).

### 3.4 Naive Assumption: Conditional Independence

**Giả định naive:** Các features độc lập với nhau **given the class**:

$$P(X|C_i) = P(x_1, x_2, ..., x_n | C_i) = \prod_{k=1}^{n} P(x_k | C_i)$$

**Hệ quả:** Chỉ cần ước lượng $P(x_k | C_i)$ cho mỗi feature riêng lẻ!

**Naive Bayes Classifier:**
$$C_{NB} = \argmax_{C_i} P(C_i) \prod_{k=1}^{n} P(x_k | C_i)$$

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_4/img-001.png]]

> [!NOTE] Suy luận thêm — Tại sao "naive" nhưng hiệu quả?
> Giả định independence rõ ràng sai (từ trong câu liên quan mạnh!). Nhưng:
> 1. **Argmax không cần probability chính xác:** Chỉ cần ranking đúng
> 2. **Dependencies "cancel out":** Errors từ independence có thể bù trừ
> 3. **Robust với ít data:** Ít parameters cần estimate
> 
> Trong thực tế, Naive Bayes thường competitive với complex models, đặc biệt khi data ít.

### 3.5 Training: Ước Lượng Parameters

**Prior probability:**
$$P(C_i) = \frac{\text{Số documents thuộc class } C_i}{\text{Tổng số documents}}$$

**Likelihood (với text):**
$$P(x_k | C_i) = \frac{\text{Số lần từ } x_k \text{ xuất hiện trong documents của class } C_i}{\text{Tổng số từ trong documents của class } C_i}$$

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_4/img-002.png]]

### 3.6 Zero Probability Problem

Nếu từ $x_k$ không xuất hiện trong class $C_i$ training data:
$$P(x_k | C_i) = 0 \Rightarrow P(X|C_i) = 0$$

**Giải pháp:** [[Smoothing (NLP)|Smoothing]] — thường dùng Add-one (Laplace):

$$P(x_k | C_i) = \frac{C(x_k, C_i) + 1}{\sum_w C(w, C_i) + V}$$

Trong đó $V$ là vocabulary size.

---

## 4. Ví Dụ Minh Họa: Sentiment Classification

### 4.1 Training Data

| Text | Sentiment |
|------|-----------|
| "I love this movie!" | Positive |
| "This movie is terrible" | Negative |
| "It was boring." | Negative |
| "This movie is great!" | Positive |

### 4.2 Tính Prior

$$P(Positive) = \frac{2}{4} = 0.5$$
$$P(Negative) = \frac{2}{4} = 0.5$$

### 4.3 Tính Likelihoods

**Preprocessing:** Tokenize và remove punctuation:
- Positive: ["I", "love", "this", "movie", "this", "movie", "is", "great"] → 8 words
- Negative: ["this", "movie", "is", "terrible", "it", "was", "boring"] → 7 words

**Counts:**

| Word | Positive count | Negative count |
|------|---------------|----------------|
| this | 2 | 1 |
| movie | 2 | 1 |
| is | 1 | 1 |
| love | 1 | 0 |
| great | 1 | 0 |
| terrible | 0 | 1 |
| boring | 0 | 1 |
| ... | ... | ... |

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_4/img-005.jpg]]

### 4.4 Classify New Document

**Test:** "This movie is boring."

$$P(Positive | \text{"this movie is boring"}) \propto P(Positive) \times P(\text{this}|Pos) \times P(\text{movie}|Pos) \times P(\text{is}|Pos) \times P(\text{boring}|Pos)$$

$$= 0.5 \times \frac{2}{8} \times \frac{2}{8} \times \frac{1}{8} \times \frac{0+1}{8+V}$$

(Dùng smoothing cho "boring" vì count = 0)

$$P(Negative | \text{"this movie is boring"}) \propto P(Negative) \times P(\text{this}|Neg) \times P(\text{movie}|Neg) \times P(\text{is}|Neg) \times P(\text{boring}|Neg)$$

$$= 0.5 \times \frac{1}{7} \times \frac{1}{7} \times \frac{1}{7} \times \frac{1}{7}$$

So sánh và chọn class có probability cao hơn → **Negative** ✓

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_4/img-006.jpg]]

---

## 5. Evaluation Metrics

### 5.1 Confusion Matrix

|  | Predicted Positive | Predicted Negative |
|--|-------------------|-------------------|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_4/img-007.jpg]]

### 5.2 Precision

**"Trong những cái ta predict positive, bao nhiêu % thực sự positive?"**

$$Precision = \frac{TP}{TP + FP}$$

**High precision:** Ít false alarms, nhưng có thể miss nhiều true positives.

### 5.3 Recall (Sensitivity)

**"Trong những cái thực sự positive, ta tìm được bao nhiêu %?"**

$$Recall = \frac{TP}{TP + FN}$$

**High recall:** Tìm được hầu hết positives, nhưng có thể nhiều false alarms.

### 5.4 F1-Score

**Harmonic mean** của Precision và Recall:

$$F_1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

**Tại sao harmonic mean?** Penalize extreme imbalance giữa P và R.

**Ví dụ:**
- $P = 1.0, R = 0.1$ → $F_1 = 0.18$ (low, vì R quá thấp)
- $P = 0.5, R = 0.5$ → $F_1 = 0.50$

### 5.5 Accuracy

$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

**Caveat:** Misleading khi class imbalanced! Nếu 95% data là negative, predict "negative" cho mọi thứ → 95% accuracy nhưng useless.

---

## 6. Test Sets và Cross-Validation

### 6.1 Train/Test Split

- **Training set:** Dùng để train model
- **Test set:** Dùng để evaluate (KHÔNG được dùng trong training!)

**Why separate?** Đo generalization, không phải memorization.

### 6.2 K-Fold Cross-Validation

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_4/img-008.png]]

**Quy trình:**
1. Chia data thành $k$ folds (thường $k = 5$ hoặc $10$)
2. Lặp $k$ lần:
   - Dùng fold $i$ làm test, $k-1$ folds còn lại làm train
   - Tính metrics
3. Average metrics qua $k$ lần

**Ưu điểm:**
- Sử dụng tất cả data cho cả train và test
- Estimate variance của performance
- Robust hơn single train/test split

> [!NOTE] Suy luận thêm — Development Set
> Trong thực tế còn có **development (validation) set** để tune hyperparameters. Quy trình đầy đủ: Train → Tune on Dev → Final evaluation on Test. Test set chỉ dùng **một lần cuối** để report results.

---

## 7. Naive Bayes: Ưu và Nhược Điểm

### 7.1 Ưu Điểm

- **Simple:** Chỉ cần đếm và normalize
- **Fast:** Training và inference đều $O(n)$
- **Robust với ít data:** Ít parameters, ít overfitting
- **Baseline mạnh:** Thường khó beat với small datasets
- **Interpretable:** Có thể inspect $P(word|class)$ để hiểu model

### 7.2 Nhược Điểm

- **Independence assumption:** Sai về mặt ngôn ngữ
- **Không capture word order:** "not good" vs "good" đều có "good"
- **Sensitive to feature selection:** Cần careful preprocessing
- **Probability estimates unreliable:** Chỉ ranking đáng tin, không phải calibrated probabilities

---

## 8. Beyond Naive Bayes

### 8.1 Các Classifier Khác Cho Text

- **Logistic Regression:** Discriminative, có thể dùng nhiều features
- **Support Vector Machines (SVM):** Strong baseline cho text classification
- **Neural Networks:** LSTM, CNN, Transformer-based (BERT, etc.)

### 8.2 Advanced Sentiment Analysis

- **Aspect-based:** Xác định sentiment cho từng aspect (food: positive, service: negative)
- **Fine-grained:** 5-class scale thay vì binary
- **Domain adaptation:** Transfer learning giữa domains
- **Multimodal:** Kết hợp text + image + audio

---

## 9. Kết Luận

Chapter này giới thiệu **Sentiment Classification** như một case study của **text classification** với **Naive Bayes**:

1. **Supervised learning** yêu cầu labeled data, feature extraction, và proper evaluation
2. **Naive Bayes** dựa trên Bayes' theorem với independence assumption — simple but effective
3. **Evaluation** phải xét nhiều metrics (Precision, Recall, F1) và dùng proper methodology (cross-validation)

Naive Bayes là baseline quan trọng, nhưng các approaches hiện đại (neural networks, pre-trained models) thường cho kết quả tốt hơn cho sentiment analysis.

---

## TODO

- [ ] Liên kết với [[Logistic Regression]] và [[Support Vector Machine]]
- [ ] Thêm code example với scikit-learn
- [ ] So sánh với BERT-based sentiment classifiers
