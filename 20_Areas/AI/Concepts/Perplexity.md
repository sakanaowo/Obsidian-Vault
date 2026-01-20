---
tags:
  - ai
  - nlp
  - evaluation
  - language-model
  - information-theory
aliases:
  - PP
  - Model Perplexity
status: evergreen
related:
  - "[[N-gram Language Model]]"
  - "[[Entropy (Information Theory)]]"
  - "[[Cross-Entropy]]"
  - "[[Language Model]]"
---

# Perplexity

> [!NOTE] ELI5
> Tưởng tượng bạn đang chơi đố chữ: mỗi lượt bạn phải đoán chữ cái tiếp theo. Nếu bạn giỏi tiếng Anh, bạn biết sau "qu" thường là "i" hoặc "e" — bạn chỉ cần đoán trong **2-3 lựa chọn**. Nếu bạn không biết gì, bạn phải đoán trong **26 chữ cái**. **Perplexity** đo trung bình bạn phải chọn giữa bao nhiêu options — số càng nhỏ nghĩa là model càng giỏi đoán.

---

## 1. Định Nghĩa

### 1.1 Công Thức

**Perplexity** của Language Model trên test corpus $W = w_1, w_2, ..., w_N$:

$$PP(W) = P(w_1, w_2, ..., w_N)^{-\frac{1}{N}}$$

Tương đương:
$$PP(W) = \sqrt[N]{\frac{1}{P(W)}} = \sqrt[N]{\prod_{i=1}^{N} \frac{1}{P(w_i | w_1, ..., w_{i-1})}}$$

### 1.2 Diễn Giải

**Perplexity = Weighted Average Branching Factor**

- Trung bình, model phải "lựa chọn" giữa bao nhiêu từ tại mỗi vị trí
- $PP = 100$ nghĩa là trung bình model "confused" như thể phải chọn trong 100 từ
- **Lower is better**: Model tự tin hơn → assigns higher probability → lower perplexity

### 1.3 Liên Hệ Với Entropy

$$PP = 2^{H}$$

Trong đó $H$ là **cross-entropy** (bits per word):

$$H = -\frac{1}{N} \sum_{i=1}^{N} \log_2 P(w_i | w_1, ..., w_{i-1})$$

---

## 2. Tính Toán Thực Tế

### 2.1 Log-space để tránh Underflow

Vì $P(W)$ là tích của nhiều số nhỏ, tính trong log-space:

$$\log PP(W) = -\frac{1}{N} \sum_{i=1}^{N} \log P(w_i | context)$$

$$PP(W) = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \log P(w_i | context)\right)$$

### 2.2 Handling Unknown Words

Khi test corpus có OOV (out-of-vocabulary) words:
- Replace với `<UNK>` token (đã có trong training)
- Hoặc exclude OOV từ calculation (report OOV rate riêng)

---

## 3. Ví Dụ Thực Nghiệm

**WSJ Corpus** (38M training words, 1.5M test words, 19,979 vocab):

| Model | Perplexity |
|-------|------------|
| Unigram | 962 |
| Bigram | 170 |
| Trigram | 109 |

**Interpretation:**
- Unigram: Trung bình chọn trong ~962 từ (gần như random trong vocab)
- Trigram: Trung bình chọn trong ~109 từ (context giúp nhiều)

---

## 4. Perplexity Tốt và Xấu

### 4.1 Lower Bound

**Perfect model:** Nếu model biết chính xác từ tiếp theo:
$$P(w_i | context) = 1 \Rightarrow PP = 1$$

**Trong thực tế:** Ngôn ngữ inherently uncertain, nên $PP > 1$ luôn.

### 4.2 Upper Bound (Uniform)

Nếu model gán uniform probability cho tất cả $V$ từ:
$$P(w_i | context) = \frac{1}{V} \Rightarrow PP = V$$

Với $V = 20,000$: $PP = 20,000$ (random guess)

### 4.3 Typical Range

| Domain | Perplexity |
|--------|------------|
| Well-trained trigram on news | 50-200 |
| Neural LM on news | 20-60 |
| GPT-2 on WebText | ~20 |
| Cross-domain (train news, test medical) | 300-1000+ |

---

## 5. Caveats và Limitations

### 5.1 Không Comparable Across Vocabularies

Perplexity phụ thuộc vào **vocabulary size**:
- Vocab 10,000 vs 100,000 → không so sánh trực tiếp được
- Cần cùng vocab và cùng test set

### 5.2 Không Đo Tất Cả Quality Aspects

Perplexity đo **fluency/likelihood**, không đo:
- Semantic correctness
- Factual accuracy
- Coherence across long contexts

**Low perplexity ≠ Good text generation**

### 5.3 Infinite Perplexity

Nếu model gán $P = 0$ cho bất kỳ từ nào trong test:
$$PP = \infty$$

→ Cần smoothing để tránh!

---

## 6. Perplexity vs Other Metrics

| Metric | Measures | When to Use |
|--------|----------|-------------|
| Perplexity | Model fit to data | LM comparison |
| BLEU | N-gram overlap | MT, summarization |
| Accuracy | Correct predictions | Classification |
| F1 | Precision-Recall balance | NER, classification |

---

## 7. Code Example

```python
import math

def perplexity(model, test_sentences):
    """
    Calculate perplexity of a language model on test data.
    model: function that returns P(word | context)
    test_sentences: list of tokenized sentences
    """
    total_log_prob = 0
    total_words = 0
    
    for sentence in test_sentences:
        for i, word in enumerate(sentence):
            context = sentence[:i]
            prob = model(word, context)
            if prob > 0:
                total_log_prob += math.log(prob)
            else:
                return float('inf')  # Zero probability
            total_words += 1
    
    avg_log_prob = total_log_prob / total_words
    perplexity = math.exp(-avg_log_prob)
    return perplexity
```

---

## TODO

- [ ] Thêm visualization của perplexity distribution
- [ ] So sánh perplexity của các model SOTA
- [ ] Liên kết với [[Bits Per Character (BPC)]] cho character-level models
