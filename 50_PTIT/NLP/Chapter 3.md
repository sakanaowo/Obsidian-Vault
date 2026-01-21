---
tags:
  - nlp
  - ptit
  - source-note
  - language-model
  - n-gram
status: completed
source:
  - assets/Library/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang.pdf
aliases:
  - LM Chapter
  - Chapter 3 NLP PTIT
---

# Chapter 3 — Statistical Language Models

> [!NOTE] Source
> Tài liệu gốc: `assets/Library/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang.pdf` (Chapter 3, slide 1-21). Nội dung dưới đây bám sát 100% cấu trúc gốc với phần ELI5 và giải thích mục đích/ứng dụng bổ sung.

---

## 1. Language Model là gì?

> [!NOTE] ELI5
> Tưởng tượng bạn đang nhắn tin và điện thoại **gợi ý từ tiếp theo**. Làm sao nó biết "Tôi muốn ăn..." nên gợi ý "cơm" thay vì "ghế"? Đó là nhờ **Language Model** — một chương trình đã "đọc" hàng triệu câu và học được rằng sau "muốn ăn" thường là đồ ăn.

### 1.1 Định nghĩa

[[Language Model]] (LM) là mô hình xác suất gán xác suất cho các chuỗi từ:

$$P(W) = P(w_1, w_2, ..., w_n)$$

**Mục đích:** Đánh giá mức độ "tự nhiên" hay "likely" của một câu — câu nào có xác suất cao hơn thì tự nhiên hơn.

### 1.2 Ứng dụng trực tiếp

- **Speech Recognition:** Chọn transcription có $P(W)$ cao nhất từ các ứng viên acoustic
- **Machine Translation:** Chọn bản dịch tự nhiên nhất
- **Spelling/Grammar Correction:** So sánh $P(\text{câu gốc})$ vs $P(\text{câu sửa})$
- **Text Generation:** Chọn từ tiếp theo có xác suất cao

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_3/img-001.png]]

### 1.3 Khác biệt với Formal Grammar

**Formal grammars** (regular, context-free) đưa ra quyết định "binary": câu hợp lệ hoặc không.

Trong thực tế, chúng ta cần đánh giá **mức độ**:
- *"Colorless green ideas sleep furiously"* — ngữ pháp đúng nhưng vô nghĩa
- *"The cat sat on the mat"* — ngữ pháp đúng và tự nhiên

LM cho phép **ranking** các câu theo likelihood, thay vì chỉ accept/reject.

---

## 2. N-gram Language Models

> [!NOTE] ELI5
> Để đoán từ tiếp theo, bạn không cần nhớ **cả câu** — chỉ cần nhớ **vài từ gần nhất**. Nếu tôi nói "Con mèo đang...", bạn đoán "ngủ" hoặc "chạy" — không cần biết 10 phút trước tôi nói gì. **N-gram model** làm đúng như vậy.

### 2.1 Chain Rule và Markov Assumption

Bắt đầu từ **chain rule** của xác suất:

$$P(w_1, w_2, ..., w_n) = P(w_1) \cdot P(w_2|w_1) \cdot P(w_3|w_1,w_2) \cdots P(w_n|w_1,...,w_{n-1})$$

$$= \prod_{k=1}^{n} P(w_k | w_1, ..., w_{k-1})$$

**Vấn đề:** Số lượng context histories tăng **hàm mũ** — không đủ data để ước lượng!

**Giải pháp: Markov Assumption**

Xác suất từ tiếp theo chỉ phụ thuộc vào $N-1$ từ gần nhất:

$$P(w_k | w_1, ..., w_{k-1}) \approx P(w_k | w_{k-N+1}, ..., w_{k-1})$$

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_3/img-003.jpg]]

### 2.2 Các loại N-gram

| N | Tên | Context | Ví dụ |
|---|-----|---------|-------|
| 1 | Unigram | Không context | $P(\text{phone})$ |
| 2 | Bigram | 1 từ trước | $P(\text{phone} \mid \text{cell})$ |
| 3 | Trigram | 2 từ trước | $P(\text{phone} \mid \text{your cell})$ |
| 4 | 4-gram | 3 từ trước | $P(\text{phone} \mid \text{off your cell})$ |

**Trade-off:**
- **N lớn:** Capture context dài hơn, chính xác hơn
- **N nhỏ:** Ít sparse data, ước lượng robust hơn

Trong thực tế, **trigram** (N=3) là sweet spot phổ biến.

---

## 3. Estimating Probabilities

### 3.1 Maximum Likelihood Estimation (MLE)

N-gram conditional probabilities được ước lượng từ raw text bằng **counting và normalizing**:

**Bigram:**
$$P(w_n | w_{n-1}) = \frac{C(w_{n-1}, w_n)}{C(w_{n-1})}$$

**N-gram:**
$$P(w_n | w_{n-N+1}^{n-1}) = \frac{C(w_{n-N+1}^n)}{C(w_{n-N+1}^{n-1})}$$

Trong đó $C(\cdot)$ là count (số lần xuất hiện) trong training corpus.

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_3/img-005.png]]

### 3.2 Xử lý ranh giới câu

Để có mô hình xác suất consistent, thêm **special tokens**:
- `<s>`: Start-of-sentence token
- `</s>`: End-of-sentence token

**Ví dụ từ tài liệu gốc:**

```
P(<s> i want english food </s>)
= P(i|<s>) × P(want|i) × P(english|want) × P(food|english) × P(</s>|food)
= 0.25 × 0.33 × 0.0011 × 0.5 × 0.68 = 0.000031

P(<s> i want chinese food </s>)
= P(i|<s>) × P(want|i) × P(chinese|want) × P(food|chinese) × P(</s>|food)
= 0.25 × 0.33 × 0.0065 × 0.52 × 0.68 = 0.00019
```

**Mục đích:** Special tokens cho phép mô hình học:
- Từ nào thường bắt đầu câu: $P(w|<s>)$
- Từ nào thường kết thúc câu: $P(</s>|w)$

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_3/img-002.png]]

### 3.3 Generative Model

N-gram LM có thể được xem như một **probabilistic automata** để sinh câu:

```
1. Initialize sentence with N-1 <s> symbols
2. Until </s> is generated do:
   - Stochastically pick next word based on P(w|context)
3. Output: generated word sequence
```

**MLE interpretation:** Relative frequency estimates maximize probability mà model $M$ sẽ generate training corpus $T$:

$$\hat{\lambda} = \argmax_\lambda P(T | M(\lambda))$$

---

## 4. Evaluating Language Models

### 4.1 Extrinsic vs Intrinsic Evaluation

**Extrinsic (in vivo):**
- Đánh giá LM qua hiệu quả của end application (speech recognition accuracy, translation BLEU)
- **Ưu điểm:** Realistic
- **Nhược điểm:** Expensive, slow

**Intrinsic:**
- Đánh giá LM trực tiếp qua khả năng "fit" test data
- **Ưu điểm:** Faster, cheaper
- **Nhược điểm:** May not correlate with extrinsic task

**Best practice:** Verify intrinsic metric correlates với extrinsic ít nhất một lần, rồi dùng intrinsic để phát triển.

### 4.2 Perplexity

> [!NOTE] ELI5
> Perplexity đo "sự bất ngờ" của mô hình khi đọc văn bản mới. Nếu mô hình "hiểu" ngôn ngữ tốt, nó sẽ ít "ngạc nhiên" khi thấy các câu thông thường → perplexity thấp.

[[Perplexity]] đo mức độ "fit" của model với test data:

$$PP(W) = P(w_1, w_2, ..., w_N)^{-\frac{1}{N}} = \sqrt[N]{\frac{1}{P(w_1, w_2, ..., w_N)}}$$

**Diễn giải:**
- Perplexity = **weighted average branching factor**: trung bình mô hình phải "chọn" giữa bao nhiêu từ tại mỗi vị trí
- **Lower is better**: Model tốt gán xác suất cao cho test data → perplexity thấp

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_3/img-004.png]]

**Ví dụ thực nghiệm từ tài liệu gốc (WSJ corpus):**

| Model | Perplexity |
|-------|------------|
| Unigram | 962 |
| Bigram | 170 |
| Trigram | 109 |

**Mục đích:** Trigram tốt hơn bigram, bigram tốt hơn unigram — context giúp dự đoán tốt hơn.

> [!NOTE] Mối quan hệ với Entropy
> Perplexity liên hệ với [[Entropy (Information Theory)]]:
> $$PP = 2^{H(W)}$$
> Trong đó $H(W)$ là cross-entropy. Perplexity 109 ≈ entropy ~6.77 bits/word.

---

## 5. Generalization

> [!NOTE] ELI5
> Nếu bạn chỉ học thuộc lòng 100 câu, bạn sẽ nói tốt 100 câu đó nhưng **không biết nói câu mới**. Mô hình tốt phải **generalize** — học từ dữ liệu cũ để xử lý dữ liệu mới.

### 5.1 Định nghĩa

**Generalization** trong Machine Learning là khả năng của mô hình perform well trên **new, unseen data**.

**Factors ảnh hưởng:**
- Data quality và quantity
- Model complexity và regularization
- Hyperparameters

### 5.2 Balancing Bias and Variance

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_3/img-000.jpg]]

| Aspect | High Bias (Underfitting) | High Variance (Overfitting) |
|--------|--------------------------|----------------------------|
| Model | Quá đơn giản | Quá phức tạp |
| Training error | Cao | Thấp |
| Test error | Cao | Cao |
| N-gram example | Unigram | 5-gram không smoothing |

**Goal:** Tìm sweet spot với cả bias và variance đều acceptable.

### 5.3 Strategies

- **Cross-Validation:** Chia data thành train, validation, test sets
- **Transfer Learning:** Sử dụng pre-trained models, adapt với less data

---

## 6. Smoothing

> [!NOTE] ELI5
> Giả sử bạn chưa bao giờ thấy cụm "unicorn pizza" trong sách. Điều đó không có nghĩa là "unicorn pizza" **không thể tồn tại**. **Smoothing** là cách "dành chỗ" một chút xác suất cho những thứ chưa thấy.

### 6.1 Vấn đề Zero Probability

Với MLE thuần túy, nếu N-gram không xuất hiện trong training:

$$P(w_n | context) = 0$$

**Hậu quả:**
- Xác suất của cả câu = 0 (vì là tích)
- Perplexity = ∞
- Model "từ chối" mọi câu có N-gram unseen

**Vấn đề:** Vì số lượng N-gram khả dĩ là **combinatorial**, phần lớn N-grams sẽ **không xuất hiện** trong training dù training data lớn.

### 6.2 Laplace (Add-One) Smoothing

**Ý tưởng:** "Hallucinate" rằng mỗi N-gram xuất hiện thêm 1 lần.

$$P_{Laplace}(w_n | w_{n-1}) = \frac{C(w_{n-1}, w_n) + 1}{C(w_{n-1}) + V}$$

Trong đó $V$ là vocabulary size.

**Ưu điểm:** Đơn giản, không zero probabilities

**Nhược điểm:** Reassign **quá nhiều** probability mass cho unseen events

### 6.3 Add-k Smoothing

Thay vì add 1, add một giá trị nhỏ hơn $k < 1$:

$$P_{Add-k}(w_n | w_{n-1}) = \frac{C(w_{n-1}, w_n) + k}{C(w_{n-1}) + k \cdot V}$$

**$k$ được chọn** bằng cách optimize trên development set.

**Ứng dụng:** Add-k hiệu quả cho text classification, nhưng vẫn có limitations cho language modeling.

### 6.4 Advanced Smoothing Techniques

Nhiều kỹ thuật nâng cao đã được phát triển:

- **Good-Turing:** Ước lượng xác suất của unseen events dựa trên frequency of frequencies
- **Interpolation:** Kết hợp các N-gram levels
- **Backoff:** Sử dụng lower-order model khi higher-order không có data
- **Kneser-Ney:** State-of-the-art cho N-gram LM, sử dụng continuation probability
- **Class-based (cluster) N-grams:** Nhóm từ thành classes để giảm sparsity

---

## 7. Model Combination

### 7.1 Vấn đề với N lớn

Khi N tăng:
- **Expressiveness** tăng (capture longer dependencies)
- **Smoothing problem** tệ hơn (data sparsity)

**Giải pháp:** Combine kết quả của multiple N-gram models.

### 7.2 Interpolation

Linearly combine estimates của N-gram models ở các orders khác nhau:

**Interpolated Trigram Model:**

$$P_{interp}(w_n | w_{n-2}, w_{n-1}) = \lambda_1 P(w_n) + \lambda_2 P(w_n|w_{n-1}) + \lambda_3 P(w_n|w_{n-2}, w_{n-1})$$

với $\lambda_1 + \lambda_2 + \lambda_3 = 1$.

**Learn $\lambda$ values:** Train để maximize likelihood của development (tuning) corpus.

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_3/img-005.jpg]]

### 7.3 Backoff

Chỉ sử dụng lower-order model khi higher-order không có data (count = 0):

$$P_{BO}(w_n | w_{n-2}, w_{n-1}) = \begin{cases}
P^*(w_n | w_{n-2}, w_{n-1}) & \text{if } C(w_{n-2}, w_{n-1}, w_n) > 0 \\
\alpha \cdot P_{BO}(w_n | w_{n-1}) & \text{otherwise}
\end{cases}$$

Trong đó:
- $P^*$ là discounted probability (để reserve mass cho unseen events)
- $\alpha$ là back-off weight

---

## 8. Long Distance Dependencies

### 8.1 Vấn đề với N-gram

N-gram models có **fixed context window**, không capture được **long-distance dependencies**.

**Ví dụ syntactic dependencies từ tài liệu gốc:**

> "The man next to the large oak tree near the grocery store on the corner **is** tall."
> "The men next to the large oak tree near the grocery store on the corner **are** tall."

Subject-verb agreement phụ thuộc vào "man/men" cách rất xa "is/are".

**Ví dụ semantic dependencies:**

> "The bird next to the large oak tree near the grocery store on the corner **flies** rapidly."
> "The man next to the large oak tree near the grocery store on the corner **talks** rapidly."

**Mục đích:** Đây là motivation cho các models phức tạp hơn (RNN, LSTM, Transformer).

### 8.2 Beyond N-grams

- **RNN/LSTM LMs:** Encode variable-length history trong hidden state
- **Transformer LMs (GPT, BERT):** Self-attention capture arbitrary-length dependencies

**Tuy nhiên:** N-gram vẫn relevant cho:
- Fast, lightweight models
- Interpretable baselines
- Combination với neural models
- Resource-constrained environments

---

## 9. Kết Luận

Chapter này thiết lập **N-gram Language Model** như framework cơ bản:

1. **LM** gán xác suất cho sentences, đánh giá mức độ "tự nhiên"

2. **N-gram** sử dụng Markov assumption để giảm complexity từ exponential xuống tractable

3. **MLE** ước lượng probabilities bằng counting

4. **Perplexity** là metric chuẩn để evaluate LM (lower is better)

5. **Smoothing** giải quyết zero probability problem cho unseen N-grams

6. **Interpolation/Backoff** kết hợp models ở nhiều orders

7. **Long-distance dependencies** là limitation chính của N-gram → motivate neural LMs

---

## TODO

- [ ] Liên kết với [[Neural Language Models]]
- [ ] Thêm code example tính perplexity
- [ ] So sánh N-gram với GPT-style models
