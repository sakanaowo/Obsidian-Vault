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
> Tài liệu gốc: `assets/Library/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang.pdf` (Chapter 3). Nội dung dưới đây là **dịch + diễn giải có phê bình** dựa trên slide; các đoạn được đánh dấu "Suy luận thêm" là phần mở rộng từ kiến thức nền.

---

## 1. Language Model Là Gì?

> [!NOTE] ELI5
> Tưởng tượng bạn đang nhắn tin và điện thoại **gợi ý từ tiếp theo**. Làm sao nó biết "Tôi muốn ăn..." nên gợi ý "cơm" thay vì "ghế"? Đó là nhờ **Language Model** — một chương trình đã "đọc" hàng triệu câu và học được rằng sau "muốn ăn" thường là đồ ăn, không phải đồ vật. Language Model cho điểm "khả năng xảy ra" của các câu.

### 1.1 Định Nghĩa Formal

**Language Model (LM)** là một mô hình xác suất gán xác suất cho các chuỗi từ. Formally, cho một câu $W = w_1, w_2, ..., w_n$, LM ước lượng:

$$P(W) = P(w_1, w_2, ..., w_n)$$

**Ứng dụng trực tiếp:**
- **Sinh văn bản:** Chọn từ tiếp theo có xác suất cao
- **Speech Recognition:** Chọn transcription có $P(W)$ cao nhất từ các ứng viên acoustic
- **Machine Translation:** Chọn bản dịch tự nhiên nhất
- **Spelling/Grammar Correction:** So sánh $P(\text{câu gốc})$ vs $P(\text{câu sửa})$

### 1.2 Sự Khác Biệt Với Formal Grammar

**Formal grammars** (regular, context-free) đưa ra quyết định "binary": câu hợp lệ hoặc không. Trong thực tế, chúng ta cần đánh giá **mức độ** — câu nào tự nhiên hơn, fluent hơn, likely hơn.

> *"Colorless green ideas sleep furiously"* — ngữ pháp đúng nhưng vô nghĩa
> *"The cat sat on the mat"* — ngữ pháp đúng và tự nhiên

LM cho phép **ranking** các câu theo likelihood, thay vì chỉ accept/reject.

---

## 2. N-gram Language Models

> [!NOTE] ELI5
> Để đoán từ tiếp theo, bạn không cần nhớ **cả câu** — chỉ cần nhớ **vài từ gần nhất**. Nếu tôi nói "Con mèo đang...", bạn đoán "ngủ" hoặc "chạy" — không cần biết 10 phút trước tôi nói gì. **N-gram model** làm đúng như vậy: dùng $N-1$ từ gần nhất để đoán từ tiếp theo.

### 2.1 Chain Rule và Markov Assumption

Bắt đầu từ **chain rule** của xác suất:

$$P(w_1, w_2, ..., w_n) = P(w_1) \cdot P(w_2|w_1) \cdot P(w_3|w_1,w_2) \cdots P(w_n|w_1,...,w_{n-1})$$

$$= \prod_{k=1}^{n} P(w_k | w_1, ..., w_{k-1})$$

Vấn đề: số lượng context histories $w_1, ..., w_{k-1}$ tăng **theo hàm mũ** — không đủ data để ước lượng!

**Giải pháp: Markov Assumption (N-gram)**

Giả định xác suất từ tiếp theo chỉ phụ thuộc vào $N-1$ từ gần nhất:

$$P(w_k | w_1, ..., w_{k-1}) \approx P(w_k | w_{k-N+1}, ..., w_{k-1})$$

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_3/img-003.jpg]]

### 2.2 Các Loại N-gram

| N | Tên | Context | Ví dụ |
|---|-----|---------|-------|
| 1 | Unigram | Không có context | $P(\text{phone})$ |
| 2 | Bigram | 1 từ trước | $P(\text{phone} \mid \text{cell})$ |
| 3 | Trigram | 2 từ trước | $P(\text{phone} \mid \text{your cell})$ |
| 4 | 4-gram | 3 từ trước | $P(\text{phone} \mid \text{off your cell})$ |

**Trade-off:**
- **N lớn hơn:** Capture được context dài hơn, chính xác hơn về mặt ngôn ngữ
- **N nhỏ hơn:** Ít sparse data, ước lượng robust hơn

Trong thực tế, **trigram** (N=3) là sweet spot phổ biến cho các hệ thống truyền thống.

### 2.3 Ước Lượng Xác Suất: Maximum Likelihood Estimation (MLE)

Xác suất N-gram được ước lượng bằng **counting và normalizing**:

**Bigram:**
$$P(w_n | w_{n-1}) = \frac{C(w_{n-1}, w_n)}{C(w_{n-1})}$$

**General N-gram:**
$$P(w_n | w_{n-N+1}, ..., w_{n-1}) = \frac{C(w_{n-N+1}, ..., w_n)}{C(w_{n-N+1}, ..., w_{n-1})}$$

Trong đó $C(\cdot)$ là count (số lần xuất hiện) trong training corpus.

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_3/img-005.jpg]]

### 2.4 Xử Lý Ranh Giới Câu

Để có mô hình xác suất consistent, ta thêm **special tokens**:
- `<s>`: Start-of-sentence token
- `</s>`: End-of-sentence token

Điều này cho phép mô hình học:
- Từ nào thường bắt đầu câu: $P(w|\text{<s>})$
- Từ nào thường kết thúc câu: $P(\text{</s>}|w)$

**Ví dụ:**
```
Câu: "I want english food"
Với bigram: "<s> I want english food </s>"

P(<s> I want english food </s>) 
= P(I|<s>) × P(want|I) × P(english|want) × P(food|english) × P(</s>|food)
```

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_3/img-002.png]]

### 2.5 N-gram như Generative Model

N-gram LM có thể được xem như một **probabilistic automaton** để sinh câu:

```
1. Khởi tạo với N-1 tokens <s>
2. Lặp:
   a. Dựa trên N-1 từ trước đó, sample từ tiếp theo theo phân phối P(w|context)
   b. Nếu từ được sample là </s>, dừng lại
3. Output: chuỗi từ đã sinh
```

> [!NOTE] Suy luận thêm — MLE là gì về mặt toán học?
> MLE (Maximum Likelihood Estimation) tìm tham số mô hình $\hat{\theta}$ sao cho xác suất sinh ra dữ liệu quan sát được là lớn nhất: $\hat{\theta} = \argmax_\theta P(D|\theta)$. Với N-gram, "tham số" là các xác suất $P(w|context)$, và MLE leads đến công thức đếm ở trên.

---

## 3. Đánh Giá Language Models

### 3.1 Extrinsic vs Intrinsic Evaluation

**Extrinsic (in vivo):** Đánh giá LM qua hiệu quả của task cuối (speech recognition accuracy, translation BLEU, ...).
- **Ưu điểm:** Realistic, đo đúng cái ta quan tâm
- **Nhược điểm:** Tốn kém, chậm, khó isolate ảnh hưởng của LM

**Intrinsic:** Đánh giá LM trực tiếp qua khả năng "fit" test data.
- **Ưu điểm:** Nhanh, rẻ, có thể iterate nhanh
- **Nhược điểm:** Không đảm bảo correlate với downstream task

**Best practice:** Verify intrinsic metric correlate với extrinsic ít nhất một lần, rồi dùng intrinsic để phát triển.

### 3.2 Perplexity: Metric Intrinsic Chuẩn

[[Perplexity]] đo **trung bình geometric của inverse probability** mà model gán cho test corpus:

$$PP(W) = P(w_1, w_2, ..., w_N)^{-\frac{1}{N}} = \sqrt[N]{\frac{1}{P(w_1, w_2, ..., w_N)}}$$

Hoặc tương đương:
$$PP(W) = \sqrt[N]{\prod_{i=1}^{N} \frac{1}{P(w_i | w_1, ..., w_{i-1})}}$$

**Diễn giải:**
- Perplexity = **weighted average branching factor**: trung bình mô hình phải "chọn" giữa bao nhiêu từ tại mỗi vị trí
- **Lower is better**: Model tốt gán xác suất cao cho test data → perplexity thấp

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_3/img-004.png]]

**Ví dụ thực nghiệm (WSJ corpus):**

| Model | Perplexity |
|-------|------------|
| Unigram | 962 |
| Bigram | 170 |
| Trigram | 109 |

Trigram tốt hơn bigram, bigram tốt hơn unigram — phù hợp với trực giác rằng context giúp dự đoán.

> [!NOTE] Suy luận thêm — Perplexity và Entropy
> Perplexity liên hệ với [[Entropy (Information Theory)]]:
> $$PP = 2^{H(W)}$$
> Trong đó $H(W)$ là cross-entropy giữa true distribution và model. Perplexity 109 tương đương entropy ~6.77 bits/word.

---

## 4. Generalization: Bài Toán Cốt Lõi

> [!NOTE] ELI5
> Nếu bạn chỉ học thuộc lòng 100 câu, bạn sẽ nói tốt 100 câu đó nhưng **không biết nói câu mới**. Mô hình tốt phải **generalize** — học từ dữ liệu cũ để xử lý dữ liệu mới chưa từng thấy. Đây là thách thức lớn nhất của Machine Learning.

### 4.1 Overfitting và Underfitting

**Overfitting:** Model quá phức tạp, "nhớ" training data nhưng không generalize.
- Dấu hiệu: Train perplexity rất thấp, test perplexity cao
- N-gram: N quá lớn → nhiều context không bao giờ xuất hiện trong test

**Underfitting:** Model quá đơn giản, không capture được patterns.
- Dấu hiệu: Cả train và test perplexity đều cao
- N-gram: Unigram bỏ qua mọi context → poor predictions

### 4.2 Bias-Variance Trade-off

| Aspect | High Bias (Underfitting) | High Variance (Overfitting) |
|--------|--------------------------|----------------------------|
| Model | Quá đơn giản | Quá phức tạp |
| Training error | Cao | Thấp |
| Test error | Cao | Cao |
| N-gram example | Unigram | 5-gram không smoothing |

**Goal:** Tìm sweet spot với cả bias và variance đều acceptable.

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_3/img-000.jpg]]

---

## 5. Smoothing: Giải Quyết Zero Probabilities

> [!NOTE] ELI5
> Giả sử bạn chưa bao giờ thấy cụm "unicorn pizza" trong sách. Điều đó không có nghĩa là "unicorn pizza" **không thể tồn tại** — chỉ là bạn chưa gặp. **Smoothing** là cách "dành chỗ" một chút xác suất cho những thứ chưa thấy, để model không nói "impossible" với những gì nó chưa biết.

### 5.1 Vấn Đề Zero Probability

Với MLE thuần túy, nếu N-gram $(w_{n-N+1}, ..., w_n)$ không xuất hiện trong training:

$$P(w_n | w_{n-N+1}, ..., w_{n-1}) = 0$$

**Hậu quả:**
- Xác suất của cả câu = 0 (vì là tích)
- Perplexity = ∞
- Model "từ chối" mọi câu có N-gram unseen

Vì **số lượng N-gram khả dĩ là combinatorial**, phần lớn N-grams sẽ **không xuất hiện** trong training dù training data lớn.

### 5.2 Laplace (Add-One) Smoothing

Ý tưởng: "Hallucinate" rằng mỗi N-gram xuất hiện thêm 1 lần.

**Công thức (Bigram):**
$$P_{Laplace}(w_n | w_{n-1}) = \frac{C(w_{n-1}, w_n) + 1}{C(w_{n-1}) + V}$$

Trong đó $V$ là vocabulary size (số từ distinct).

**Ưu điểm:** Đơn giản, không có zero probabilities

**Nhược điểm:** Reassign **quá nhiều** probability mass cho unseen events, làm model quá "phẳng".

### 5.3 Add-k Smoothing

Thay vì add 1, add một giá trị nhỏ hơn $k < 1$:

$$P_{Add-k}(w_n | w_{n-1}) = \frac{C(w_{n-1}, w_n) + k}{C(w_{n-1}) + k \cdot V}$$

**$k$ được chọn** bằng cách optimize trên development set.

**Ứng dụng:** Add-k hiệu quả cho text classification, nhưng vẫn có vấn đề cho language modeling.

### 5.4 Backoff và Interpolation

**Intuition:** Nếu không có đủ evidence cho N-gram, "fall back" về (N-1)-gram.

**Backoff (Katz Backoff):**
- Nếu $C(w_{n-2}, w_{n-1}, w_n) > 0$: dùng trigram probability (có discount)
- Nếu không: backoff về bigram $P(w_n | w_{n-1})$
- Nếu vẫn không: backoff về unigram $P(w_n)$

**Interpolation:**
Kết hợp các N-gram levels với weights:

$$P_{interp}(w_n | w_{n-2}, w_{n-1}) = \lambda_3 \cdot P_{tri} + \lambda_2 \cdot P_{bi} + \lambda_1 \cdot P_{uni}$$

với $\lambda_1 + \lambda_2 + \lambda_3 = 1$.

Weights $\lambda$ được learn trên held-out data.

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_3/img-001.png]]

### 5.5 Advanced Smoothing Techniques

- **Good-Turing:** Ước lượng xác suất của unseen N-grams dựa trên frequency of frequencies
- **Kneser-Ney:** State-of-the-art cho N-gram LM, sử dụng **continuation probability** — từ xuất hiện trong bao nhiêu context khác nhau
- **Modified Kneser-Ney:** Kneser-Ney với discount values khác nhau cho các frequency bands

> [!NOTE] Suy luận thêm — Tại sao Kneser-Ney hiệu quả?
> Kneser-Ney dùng **continuation count** thay vì raw count cho lower-order models. Ví dụ: "Francisco" xuất hiện nhiều nhưng gần như luôn sau "San", nên unigram probability của nó không nên cao. Continuation count đo "từ này xuất hiện trong bao nhiêu contexts khác nhau", better reflect actual generality.

---

## 6. Model Combination

Khi N tăng:
- **Expressiveness** tăng (capture longer dependencies)
- **Data sparsity** tăng (khó ước lượng reliable)

**Giải pháp:** Combine multiple models với different N, hoặc different training data, hoặc different smoothing.

**Ensemble methods:**
- Linear interpolation (như trên)
- Log-linear combination
- Neural network combination

---

## 7. N-gram LM Trong Bối Cảnh Hiện Đại

### 7.1 Hạn Chế Của N-gram

1. **Fixed context window:** Không capture long-range dependencies (quan trọng cho discourse, coreference)
2. **Discrete representations:** Không generalize giữa similar words ("cat" và "dog" không share statistics)
3. **Data hungry:** Số lượng N-grams tăng theo hàm mũ với N

### 7.2 Neural Language Models

**RNN/LSTM LMs:** Encode variable-length history trong hidden state
**Transformer LMs (GPT, etc.):** Self-attention capture arbitrary-length dependencies

**Tuy nhiên:** N-gram vẫn relevant cho:
- Fast, lightweight models
- Interpretable baselines
- Combination với neural models (interpolation)
- Resource-constrained environments

---

## 8. Kết Luận

Chapter này thiết lập **N-gram Language Model** như framework cơ bản để:
1. Assign probabilities to sentences
2. Evaluate với Perplexity
3. Handle unseen data với Smoothing

**Key insights:**
- LM là bài toán ước lượng phân phối xác suất trên infinite space → cần strong assumptions (Markov)
- MLE overfits khi data sparse → smoothing là essential
- Trade-off giữa model complexity và data requirements là ubiquitous trong ML

Các chapters tiếp theo sẽ áp dụng probabilistic thinking này cho các tác vụ cụ thể: classification (Chapter 4), sequence labeling, parsing.

---

## TODO

- [ ] Liên kết với [[Neural Language Models]] khi tạo concept note
- [ ] Thêm code example tính perplexity
- [ ] So sánh N-gram với GPT-style models về efficiency vs quality
