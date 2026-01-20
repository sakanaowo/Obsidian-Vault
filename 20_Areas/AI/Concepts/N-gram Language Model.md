---
tags:
  - ai
  - nlp
  - language-model
  - probabilistic-model
aliases:
  - N-gram
  - N-grams
  - Bigram Model
  - Trigram Model
status: evergreen
related:
  - "[[Language Model]]"
  - "[[Markov Chain]]"
  - "[[Perplexity]]"
  - "[[Smoothing (NLP)]]"
---

# N-gram Language Model

> [!NOTE] ELI5
> Khi bạn đoán từ tiếp theo trong câu "Tôi muốn ăn...", bạn dựa vào **vài từ gần nhất** — không cần nhớ cả ngày hôm qua. **N-gram model** làm giống vậy: dùng $N-1$ từ cuối để đoán từ thứ $N$. "Bi-gram" (N=2) dùng 1 từ, "Tri-gram" (N=3) dùng 2 từ. Đơn giản nhưng hiệu quả đáng ngạc nhiên!

---

## 1. Định Nghĩa

### 1.1 N-gram là gì?

Một **N-gram** là một chuỗi gồm $N$ items liên tiếp (thường là từ hoặc characters):
- **Unigram (1-gram):** Từ đơn lẻ: "the", "cat", "sat"
- **Bigram (2-gram):** Cặp từ liên tiếp: "the cat", "cat sat"
- **Trigram (3-gram):** Bộ ba từ: "the cat sat"
- **4-gram, 5-gram, ...**

### 1.2 N-gram Language Model

**N-gram LM** sử dụng N-grams để ước lượng xác suất của câu, dựa trên **Markov assumption**:

$$P(w_n | w_1, ..., w_{n-1}) \approx P(w_n | w_{n-N+1}, ..., w_{n-1})$$

**Xác suất của câu:**
$$P(W) = \prod_{i=1}^{n} P(w_i | w_{i-N+1}, ..., w_{i-1})$$

---

## 2. Ước Lượng Tham Số (MLE)

Xác suất N-gram được ước lượng bằng **đếm** từ corpus:

$$P(w_n | w_{n-N+1}, ..., w_{n-1}) = \frac{C(w_{n-N+1}, ..., w_n)}{C(w_{n-N+1}, ..., w_{n-1})}$$

**Ví dụ Bigram:**
$$P(\text{food} | \text{chinese}) = \frac{C(\text{chinese food})}{C(\text{chinese})}$$

Nếu "chinese" xuất hiện 100 lần, "chinese food" xuất hiện 52 lần:
$$P(\text{food} | \text{chinese}) = \frac{52}{100} = 0.52$$

---

## 3. Trade-offs

| Aspect | N nhỏ (Unigram, Bigram) | N lớn (4-gram, 5-gram) |
|--------|-------------------------|------------------------|
| Context captured | Ít | Nhiều |
| Linguistic accuracy | Thấp | Cao |
| Data sparsity | Ít | Nhiều |
| Robustness | Cao | Thấp |
| Memory/Storage | Ít | Nhiều |

**Sweet spot:** Trigram (N=3) thường là balance tốt cho nhiều ứng dụng.

---

## 4. Vấn Đề và Giải Pháp

### 4.1 Zero Probability Problem

N-gram không xuất hiện trong training → $P = 0$ → câu có $P = 0$.

**Giải pháp:** [[Smoothing (NLP)]] — Add-one, Add-k, Backoff, Interpolation, Kneser-Ney

### 4.2 Out-of-Vocabulary (OOV)

Từ không có trong vocabulary → không thể tính xác suất.

**Giải pháp:** 
- Replace rare words với `<UNK>` token
- Character-level hoặc subword N-grams

### 4.3 Long-range Dependencies

N-gram chỉ capture local context, bỏ qua dependencies xa.

**Giải pháp:** Neural LMs (RNN, Transformer)

---

## 5. Đánh Giá

Sử dụng [[Perplexity]]:
$$PP(W) = P(W)^{-1/N} = \sqrt[N]{\frac{1}{P(W)}}$$

Lower perplexity = better model.

---

## 6. Ứng Dụng

- **Speech Recognition:** P(transcription | acoustic)
- **Machine Translation:** P(target sentence)
- **Spelling Correction:** P(corrected) vs P(original)
- **Text Generation:** Sample từ P(next word | context)
- **Keyboard Prediction:** Gợi ý từ tiếp theo

---

## 7. N-gram vs Neural LMs

| Aspect | N-gram | Neural LM |
|--------|--------|-----------|
| Context | Fixed N-1 words | Variable/unlimited |
| Word representations | Discrete | Continuous (embeddings) |
| Generalization | Poor (no similarity) | Good (embedding space) |
| Training | Fast (counting) | Slow (gradient descent) |
| Interpretability | High | Low |
| Resource requirements | Low | High |

---

## TODO

- [ ] Code example với NLTK/spaCy
- [ ] Thêm visualization của N-gram statistics
- [ ] So sánh chi tiết các smoothing methods
