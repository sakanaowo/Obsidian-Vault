---
tags:
  - ai
  - nlp
  - statistics
  - power-law
  - language-model
aliases:
  - Zipf Distribution
  - Zipfian Distribution
  - Power Law Distribution
status: in_progress
related:
  - "[[N-gram Language Model]]"
  - "[[Tokenization]]"
  - "[[Perplexity]]"
  - "[[Stop Words]]"
---

# Zipf's Law

> [!NOTE] ELI5
> Trong bất kỳ cuốn sách nào, **rất ít từ** xuất hiện **rất nhiều** ("the", "is", "a"), còn **rất nhiều từ** chỉ xuất hiện **1-2 lần** ("magnificence", "chrysanthemum"). Nếu xếp hạng từ theo tần suất thì từ hạng 2 chỉ bằng ~1/2 từ hạng 1, từ hạng 3 bằng ~1/3, v.v. Quy luật này đúng cho **mọi ngôn ngữ** và được gọi là **Zipf's Law**.

---

## 1. Định nghĩa

### 1.1 Công thức

**Zipf's Law** phát biểu: tần suất $n_i$ của phần tử có hạng $i$ (khi sắp xếp giảm dần theo tần suất) tuân theo:

$$n_i \propto \frac{1}{i^{\alpha}}$$

Tương đương (lấy log hai vế):

$$\log n_i = -\alpha \log i + c$$

trong đó:
- $i$ = hạng (rank) — 1 là phổ biến nhất
- $\alpha \approx 1$ = hệ số mũ (exponent)
- $c$ = hằng số

### 1.2 Ý nghĩa trên Log-log plot

Trên đồ thị **log-log** (cả 2 trục dùng scale logarithmic), phân phối Zipf tạo ra **đường thẳng** với slope $= -\alpha$. Đây là cách phát hiện power law distribution trong thực tế.

---

## 2. Ứng dụng trong NLP

### 2.1 Word frequency

Top 10 từ chiếm ~25% tổng số từ trong corpus, nhưng chỉ là 0.2% vocabulary. Phần lớn vocabulary nằm ở "đuôi dài" (long tail) — hiếm gặp nhưng mang nhiều thông tin ngữ nghĩa.

### 2.2 Hệ quả cho N-gram models

Zipf's Law giải thích tại sao N-gram models gặp **data sparsity**:
- Bigrams, trigrams cũng tuân theo Zipf (với $\alpha$ nhỏ hơn)
- Phần lớn N-gram chỉ xuất hiện 0-1 lần → không đủ estimate probability
- → Cần **smoothing** hoặc **neural language models**

---

## 3. Mở rộng ngoài NLP

Zipf's Law xuất hiện trong nhiều lĩnh vực:
- **Dân số thành phố:** Ít thành phố lớn, nhiều thị trấn nhỏ
- **Thu nhập:** Ít tỷ phú, rất nhiều người thu nhập trung bình
- **Truy cập website:** Ít site có hàng tỷ views, rất nhiều site ít người xem

---

## TODO

- [ ] Thêm visualization chi tiết cho unigram, bigram, trigram
- [ ] Giải thích cơ chế tại sao ngôn ngữ tuân theo Zipf (Principle of Least Effort)
- [ ] So sánh $\alpha$ giữa các ngôn ngữ khác nhau
