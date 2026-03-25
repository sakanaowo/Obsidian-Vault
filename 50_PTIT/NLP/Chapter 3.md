---
tags:
  - nlp
  - ptit
  - source-note
status: completed
source:
  - assets/Library/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang.pdf
chapter: 3
aliases:
  - Chapter 3 NLP
  - Statistical Language Models
  - N-gram
---

# Chapter 3 — Statistical Language Models

> [!NOTE] Source
> Tài liệu gốc: `assets/Library/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang.pdf` (Chapter 3, slides 1–21). Nội dung dưới đây bám sát slide-by-slide, kèm diễn giải sâu với ví dụ số cụ thể.

---

## 1. Language Model (LM) là gì? ⭐

> [!NOTE] ELI5
> Language Model (LM) giống như **bộ não đoán từ** trên bàn phím điện thoại. Khi bạn gõ "Tôi muốn ăn", LM đoán từ tiếp theo có thể là "cơm" (xác suất cao) chứ không phải "ghế" (xác suất thấp). Nó làm việc này bằng cách **gán xác suất** cho mỗi chuỗi từ — chuỗi nào nghe "tự nhiên" hơn thì xác suất cao hơn.

[[Language Model]] là mô hình gán **xác suất** cho mỗi chuỗi từ trong một ngôn ngữ.

**Bản chất:** Một LM tốt phải thỏa mãn:
- $P(\text{"I want to eat rice"}) \gg P(\text{"I want to eat chair"})$
- Tổng xác suất của **tất cả** các câu có thể = 1 (phân phối xác suất hợp lệ)

**Tại sao cần LM?**
- **Speech recognition:** phân biệt "recognize speech" vs "wreck a nice beach" (phát âm giống nhau)
- **Machine translation:** chọn bản dịch tự nhiên nhất
- **Spelling correction:** "probablity" → "probability"
- **Text generation:** sinh câu tiếp theo tự nhiên

**So sánh với formal grammar:**

| Tiêu chí       | Formal Grammar                | Language Model                |
| -------------- | ----------------------------- | ----------------------------- |
| Đầu ra         | Binary: hợp lệ / không hợp lệ | Xác suất liên tục [0, 1]      |
| Tính thực tế   | Quá cứng nhắc                 | Linh hoạt, thực tế hơn        |
| Xử lý "câu lạ" | Reject hoàn toàn              | Gán xác suất thấp (không = 0) |

---

## 2. N-gram ⭐

> [!NOTE] ELI5
> N-gram là cách "cắt câu thành từng mảnh nhỏ gồm N từ liên tiếp". Giống như bạn đọc câu bằng cách nhìn qua "cửa sổ trượt" rộng N từ.

[[N-gram]] là chuỗi **N từ liên tiếp** trong văn bản.

**Ví dụ:** Câu "Please turn your homework in"

| Loại | N | Các N-gram |
|------|---|-----------|
| Unigram | 1 | "Please", "turn", "your", "homework", "in" |
| Bigram | 2 | "Please turn", "turn your", "your homework", "homework in" |
| Trigram | 3 | "Please turn your", "turn your homework", "your homework in" |

**Tại sao dùng N-gram cho LM?**

Vấn đề: để tính $P(\text{phone} \mid \text{Please turn off your cell})$ cần biết xác suất có điều kiện dựa trên **toàn bộ ngữ cảnh** trước đó — quá nhiều tham số!

Giải pháp: **N-gram model** chỉ nhìn **N-1 từ trước** (Markov assumption bậc N-1):

| Model | Nhìn | Ví dụ |
|-------|------|-------|
| Unigram | 0 từ trước | $P(\text{phone})$ |
| Bigram | 1 từ trước | $P(\text{phone} \mid \text{cell})$ |
| Trigram | 2 từ trước | $P(\text{phone} \mid \text{your cell})$ |

---

## 3. Chain Rule of Probability ⭐

> [!NOTE] ELI5
> Chain rule giống như cách bạn kể lại câu chuyện từng phần: thay vì phải biết "xác suất cả câu dài" (rất khó), bạn chia nhỏ thành "xác suất từng từ biết tất cả từ trước đó" (dễ hơn từng bước).

**Chain rule** cho phép phân rã xác suất chuỗi thành tích các xác suất có điều kiện:

$$P(w_1^n) = P(w_1) \cdot P(w_2 \mid w_1) \cdot P(w_3 \mid w_1^2) \cdots P(w_n \mid w_1^{n-1}) = \prod_{k=1}^{n} P(w_k \mid w_1^{k-1})$$

**Giải thích từng ký hiệu:**
- $w_1^n$ = chuỗi từ $w_1, w_2, \ldots, w_n$ (ký hiệu viết tắt)
- $P(w_k \mid w_1^{k-1})$ = xác suất từ thứ $k$ xuất hiện, **biết tất cả các từ trước nó**
- $\prod$ = tích (nhân tất cả lại)

**Ví dụ cụ thể:**

$P(\text{"I want to eat"})$
$= P(\text{I}) \cdot P(\text{want} \mid \text{I}) \cdot P(\text{to} \mid \text{I want}) \cdot P(\text{eat} \mid \text{I want to})$

**Vấn đề:** Càng về sau, ngữ cảnh $w_1^{k-1}$ càng dài → cần bảng tra quá lớn, hầu hết tổ hợp chưa bao giờ gặp trong dữ liệu huấn luyện → **không ước lượng được!**

---

## 4. Bigram & N-gram Approximation ⭐

> [!NOTE] ELI5
> Thay vì nhớ cả câu dài phía trước để đoán từ tiếp theo, ta chỉ nhớ **1-2 từ trước** (giống học sinh quay cóp chỉ nhìn được 1 dòng phía trước). Không hoàn hảo, nhưng tốt đáng ngạc nhiên!

Áp dụng **Markov assumption** (từ Chapter 2) vào LM:

**Bigram approximation** (bậc 1):

$$P(w_1^n) \approx \prod_{k=1}^{n} P(w_k \mid w_{k-1})$$

Mỗi từ chỉ phụ thuộc vào **1 từ ngay trước** nó.

**N-gram approximation tổng quát** (bậc N-1):

$$P(w_1^n) \approx \prod_{k=1}^{n} P(w_k \mid w_{k-N+1}^{k-1})$$

Mỗi từ phụ thuộc vào **N-1 từ trước** nó.

**Ví dụ tính xác suất bigram:**

$$ P(\text{"<s\> i want english food \</s\>"})\;$$
$$= P(\text{i} \mid \text{\<s\>}) \times P(\text{want} \mid \text{i}) \times P(\text{english} \mid \text{want}) \times P(\text{food} \mid \text{english}) \times P(\text{\</s\>} \mid \text{food})$$
$= 0.25 \times 0.33 \times 0.0011 \times 0.5 \times 0.68 = 0.000031$

So sánh: **"i want chinese food"**
$= 0.25 \times 0.33 \times 0.0065 \times 0.52 \times 0.68 = 0.00019$

→ Mô hình cho "chinese food" xác suất **cao hơn ~6 lần** "english food" — hợp lý vì trong corpus (nhà hàng), đồ ăn Trung Quốc phổ biến hơn!

---

## 5. Ước lượng xác suất N-gram (MLE) ⭐

> [!NOTE] ELI5
> MLE cho N-gram = **đếm rồi chia** (giống Chapter 2). Muốn biết xác suất "food" đi sau "chinese"? Đếm trong corpus: "chinese food" xuất hiện bao nhiêu lần, chia cho "chinese" xuất hiện bao nhiêu lần.

**Công thức MLE:**

**Bigram:**
$$P(w_n \mid w_{n-1}) = \frac{C(w_{n-1}, w_n)}{C(w_{n-1})}$$

**N-gram tổng quát:**
$$P(w_n \mid w_{n-N+1}^{n-1}) = \frac{C(w_{n-N+1}^{n-1}, w_n)}{C(w_{n-N+1}^{n-1})}$$

**Giải thích:**
- $C(w_{n-1}, w_n)$ = số lần cặp bigram $(w_{n-1}, w_n)$ xuất hiện trong corpus
- $C(w_{n-1})$ = tổng số lần từ $w_{n-1}$ xuất hiện

**Ví dụ cụ thể:**

Corpus: "I want to eat chinese food. I want to eat lunch."

| Bigram | Count | $C(w_{n-1})$ | $P(w_n \mid w_{n-1})$ |
|--------|-------|-------------|----------------------|
| (I, want) | 2 | C(I) = 2 | 2/2 = 1.0 |
| (want, to) | 2 | C(want) = 2 | 2/2 = 1.0 |
| (to, eat) | 2 | C(to) = 2 | 2/2 = 1.0 |
| (eat, chinese) | 1 | C(eat) = 2 | 1/2 = 0.5 |
| (eat, lunch) | 1 | C(eat) = 2 | 1/2 = 0.5 |
| (chinese, food) | 1 | C(chinese) = 1 | 1/1 = 1.0 |

### 5.1 Ký hiệu `<s>` và `</s>` ⭐

Để xử lý **từ đầu câu** và **kết thúc câu**, ta thêm:
- `<s>` = ký hiệu **bắt đầu** câu (start symbol)
- `</s>` = ký hiệu **kết thúc** câu (end symbol)

**Tại sao cần?**
- Để tính $P(w_1)$ — từ đầu tiên: viết thành $P(w_1 \mid \text{\<s\>})$
- Để mô hình biết **khi nào câu kết thúc**: $P(\text{\</s\>} \mid w_n)$

Ví dụ: `<s> I want to eat </s>` — bây giờ mọi từ đều có "từ trước" (kể cả từ đầu tiên).

### 5.2 Generative Model ⭐

N-gram LM cũng có thể được hiểu như **automaton sinh câu**:
1. Bắt đầu với N-1 ký hiệu `<s>`
2. Lặp: chọn ngẫu nhiên từ tiếp theo theo phân phối $P(w \mid \text{N-1 từ trước})$
3. Dừng khi sinh ra `</s>`

→ Đây là cơ chế nền tảng của **text generation**, và là tiền thân tư duy dẫn đến GPT!

---

## 6. Evaluating Language Models ⭐

### 6.1 Extrinsic vs Intrinsic Evaluation ⭐

> [!NOTE] ELI5
> Extrinsic = đánh giá "quanh co": đặt LM vào ứng dụng thật (dịch máy, nhận giọng nói) rồi xem kết quả cuối cùng có tốt không. Intrinsic = đánh giá "trực tiếp": cho LM xem dữ liệu test mới, đo xem nó "hiểu" ngôn ngữ tốt đến mức nào.

| Phương pháp | Ý nghĩa | Ưu điểm | Nhược điểm |
|------------|---------|---------|-----------|
| **Extrinsic** (in vivo) | Đánh giá qua ứng dụng cuối | Thực tế | Đắt, chậm |
| **Intrinsic** | Đánh giá trên test corpus | Rẻ, nhanh | Ít thực tế |

> [!TIP] Quy tắc thực hành
> Slide khuyến cáo: dùng intrinsic evaluation hằng ngày để phát triển nhanh, nhưng phải **kiểm chứng ít nhất 1 lần** rằng intrinsic metric tương quan với ứng dụng thực (extrinsic).

### 6.2 Perplexity (PP) ⭐⭐

> [!NOTE] ELI5
> Perplexity đo "sự bối rối" của mô hình. Tưởng tượng mô hình đang đoán từ tiếp theo — nếu perplexity = 100, nghĩa là mô hình "bối rối" như thể phải chọn 1 từ đúng trong 100 từ khả dĩ. Perplexity **càng thấp càng tốt** — mô hình ít bối rối, đoán chính xác hơn.

**Bản chất toán học:**

$$PP(W) = \sqrt[N]{\frac{1}{P(w_1 w_2 \ldots w_N)}}$$

Hay tương đương:

$$PP(W) = P(w_1 w_2 \ldots w_N)^{-\frac{1}{N}}$$

**Giải thích từng ký hiệu:**
- $W = w_1 w_2 \ldots w_N$ = test corpus (tất cả $N$ từ)
- $P(W)$ = xác suất mô hình gán cho test corpus
- Lấy **nghịch đảo** (vì xác suất cao → tốt, nhưng ta muốn metric nhỏ = tốt)
- **Chuẩn hóa** bằng căn bậc $N$ (để so sánh được giữa test corpus dài/ngắn khác nhau)

**Cách hiểu "weighted average branching factor":**
- Perplexity = 100 → tại mỗi bước, mô hình "thấy" khoảng 100 từ khả dĩ tiếp theo (mức độ "không chắc chắn")
- Perplexity = 10 → chỉ "thấy" 10 từ khả dĩ → mô hình tự tin hơn → tốt hơn
- Perplexity = 1 → mô hình luôn biết chính xác từ tiếp theo → hoàn hảo (không thực tế)

**Kết quả thực nghiệm (WSJ corpus, 38M từ):**

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_3/img-001.png]]
*Hình 1: Perplexity giảm khi tăng bậc N-gram: Unigram=962, Bigram=170, Trigram=109.*

| Model | Perplexity | Nhận xét |
|-------|-----------|---------|
| Unigram | 962 | Rất cao — mỗi từ như chọn ngẫu nhiên trong ~962 từ |
| Bigram | 170 | Giảm mạnh — biết 1 từ trước giúp rất nhiều |
| Trigram | 109 | Tiếp tục giảm — 2 từ trước cho thêm thông tin |

> [!IMPORTANT] Quy luật
> Tăng N → giảm perplexity (tốt hơn), NHƯNG cũng tăng vấn đề **sparsity** (thiếu dữ liệu cho N-gram dài). Có một **điểm cân bằng** tối ưu.

---

## 7. Generalization ⭐

> [!NOTE] ELI5
> Generalization = khả năng "thi tốt" của mô hình khi gặp "đề mới". Nếu mô hình chỉ "giỏi" trên dữ liệu đã học (training data) mà "dở" trên dữ liệu mới → nó không tổng quát hóa được → vô dụng!

[[Generalization]] là khả năng mô hình hoạt động tốt trên dữ liệu **chưa từng thấy** (unseen data). Đây là tiêu chí quan trọng nhất để đánh giá một mô hình ML/NLP.

### 7.1 Bias-Variance Tradeoff ⭐

> [!NOTE] ELI5
> Tưởng tượng bạn đang vẽ đường cong đi qua các điểm dữ liệu. **Bias cao** = vẽ đường thẳng quá đơn giản, bỏ lỡ quy luật (underfitting). **Variance cao** = vẽ đường ngoằn ngoèo qua MỌI điểm, kể cả nhiễu (overfitting). Cần cân bằng: đủ phức tạp để nắm quy luật, đủ đơn giản để không "nhớ" nhiễu.

| Vấn đề | Nguyên nhân | Biểu hiện | Ví dụ trong LM |
|--------|-----------|----------|----------------|
| **Underfitting** (Bias cao) | Mô hình quá đơn giản | Tệ trên cả train và test | Unigram LM — bỏ qua mọi ngữ cảnh |
| **Overfitting** (Variance cao) | Mô hình quá phức tạp | Tốt trên train, tệ trên test | 5-gram trên corpus nhỏ — "nhớ" mọi câu trong train |

**Giải pháp slide nêu:**
1. **Cross-validation** — chia dữ liệu thành train/validation/test, tune trên validation
2. **Regularization** — Smoothing (xem phần dưới)
3. **Transfer learning** — dùng mô hình pretrained (BERT, GPT...)

---

## 8. Smoothing — Vấn đề và Giải pháp ⭐⭐

> [!NOTE] ELI5
> Tưởng tượng bạn đang gieo xúc xắc 6 mặt 10 lần, nhưng chưa bao giờ gieo ra mặt 3. Có phải xác suất mặt 3 = 0? Không! Chỉ là bạn gieo chưa đủ nhiều. **Smoothing** (làm mịn) giải quyết cùng vấn đề: trong corpus, nhiều N-gram **hợp lệ nhưng chưa bao giờ xuất hiện** → MLE cho xác suất = 0 → cả câu bị xác suất = 0 → perplexity = ∞. Smoothing "lấy bớt" xác suất từ N-gram hay gặp, "cho thêm" vào N-gram chưa gặp.

**Bài toán:**
- Với corpus hữu hạn, hầu hết N-gram (đặc biệt khi N ≥ 3) sẽ có count = 0
- MLE: $P = C/C_{\text{total}} = 0/C = 0$ → xác suất = 0
- Hậu quả: nếu **1 bigram** trong câu test có $P = 0$ → toàn bộ câu có $P = 0$ → $PP = \infty$

**Nguyên tắc smoothing:**
- "Lấy" xác suất từ sự kiện **đã thấy** (discounting)
- "Cho" vào sự kiện **chưa thấy**
- Vẫn đảm bảo tổng xác suất = 1 (phân phối hợp lệ)

### 8.1 Laplace Smoothing (Add-One) ⭐⭐

> [!NOTE] ELI5
> Laplace = "giả vờ" rằng mỗi N-gram đã xuất hiện thêm **1 lần** nữa. Dù ban đầu count = 0, sau khi cộng 1 thì count = 1 → xác suất > 0. Đơn giản nhất, dễ hiểu nhất, nhưng hơi "thô" vì phân bổ quá nhiều cho sự kiện chưa gặp.

**Công thức:**

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_3/img-002.png]]
*Hình 2: Công thức Laplace smoothing cho bigram và N-gram.*

$$P_{\text{Laplace}}(w_n \mid w_{n-1}) = \frac{C(w_{n-1}, w_n) + 1}{C(w_{n-1}) + V}$$

**Giải thích từng thành phần:**
- $C(w_{n-1}, w_n) + 1$ : cộng thêm 1 vào count của bigram (kể cả khi count = 0)
- $C(w_{n-1}) + V$ : tổng count của $w_{n-1}$ cộng thêm $V$ (kích thước từ vựng) để tổng vẫn = 1
- $V$ = tổng số từ khác nhau trong từ vựng (vocabulary size)

**Ví dụ tính tay:**

Corpus: "I like cats. I like dogs." → $V = 6$ (I, like, cats, dogs, `<s>`, `</s>`)

| Bigram | Count | MLE | Laplace ($V=6$) |
|--------|-------|-----|----------------|
| (like, cats) | 1 | 1/2 = 0.5 | (1+1)/(2+6) = 0.25 |
| (like, dogs) | 1 | 1/2 = 0.5 | (1+1)/(2+6) = 0.25 |
| (like, fish) | 0 | 0/2 = **0** ❌ | (0+1)/(2+6) = **0.125** ✅ |

→ "like fish" không còn xác suất = 0 nữa! Nhưng "like cats" giảm từ 0.5 xuống 0.25 — **mất quá nhiều** cho sự kiện đã thấy.

> [!WARNING] Hạn chế
> Laplace smoothing **phân bổ quá nhiều** xác suất cho sự kiện chưa gặp (đặc biệt khi $V$ lớn — tiếng Anh có ~50,000+ từ). Xác suất các bigram đã thấy bị "pha loãng" đáng kể.

### 8.2 Add-k Smoothing ⭐

> [!NOTE] ELI5
> Thay vì cộng 1 (hơi quá), cộng một số nhỏ hơn, ví dụ k = 0.5, 0.05, hay 0.01. Ít "phá" xác suất gốc hơn, nhưng khó chọn k tốt nhất.

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_3/img-003.jpg]]
*Hình 3: Công thức Add-k smoothing.*

$$P_{\text{Add-k}}^{*}(w_n \mid w_{n-1}) = \frac{C(w_{n-1}, w_n) + k}{C(w_{n-1}) + kV}$$

**So sánh Laplace vs Add-k:**

| Tiêu chí | Laplace (k=1) | Add-k (k=0.01) |
|----------|--------------|----------------|
| Lượng xác suất cho unseen | Lớn | Nhỏ (tinh tế hơn) |
| Biến dạng xác suất gốc | Nhiều | Ít |
| Cần tune? | Không | Có — chọn k bằng dev set |
| Áp dụng | Text classification | LM, nhưng vẫn hạn chế |

> [!TIP] Thực hành
> Add-k hoạt động tốt cho **text classification** (Naive Bayes). Nhưng cho **language modeling** vẫn có độ variance không tốt → cần phương pháp mạnh hơn.

### 8.3 Interpolation ⭐⭐

> [!NOTE] ELI5
> Thay vì chỉ dùng 1 mô hình (ví dụ chỉ trigram), kết hợp **cả unigram, bigram, và trigram** lại — lấy "ý kiến" của cả 3, rồi trung bình có trọng số. Nếu trigram thiếu dữ liệu, bigram và unigram vẫn "cứu" được.

[[Interpolation]] kết hợp **tuyến tính** (linear combination) nhiều N-gram model bậc khác nhau:

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_3/img-004.png]]
*Hình 4: Công thức Interpolation cho trigram model.*

$$\hat{P}(w_n \mid w_{n-2}, w_{n-1}) = \lambda_1 \cdot P(w_n \mid w_{n-2}, w_{n-1}) + \lambda_2 \cdot P(w_n \mid w_{n-1}) + \lambda_3 \cdot P(w_n)$$

**Giải thích:**
- $\lambda_1, \lambda_2, \lambda_3$ = trọng số, thỏa $\lambda_1 + \lambda_2 + \lambda_3 = 1$
- $P(w_n \mid w_{n-2}, w_{n-1})$ = trigram probability (chi tiết nhất)
- $P(w_n \mid w_{n-1})$ = bigram probability
- $P(w_n)$ = unigram probability (tổng quát nhất)

**Ví dụ cụ thể:**

Giả sử $\lambda_1 = 0.5, \lambda_2 = 0.3, \lambda_3 = 0.2$ và muốn tính $P(\text{food} \mid \text{chinese})$:

| Model | Xác suất | Trọng số | Đóng góp |
|-------|---------|----------|---------|
| Trigram: P(food \| eat chinese) | 0 (chưa gặp) | 0.5 | 0 |
| Bigram: P(food \| chinese) | 0.52 | 0.3 | 0.156 |
| Unigram: P(food) | 0.01 | 0.2 | 0.002 |
| **Kết quả** | | | **0.158** |

→ Dù trigram = 0, bigram và unigram "cứu" lại, kết quả cuối ≠ 0!

**Cách tìm $\lambda$:** Train trên **development set** (tập dữ liệu riêng, không phải train cũng không phải test) bằng cách tối ưu likelihood.

### 8.4 Backoff ⭐⭐

> [!NOTE] ELI5
> Khác với interpolation (luôn kết hợp tất cả), backoff hoạt động theo kiểu "kế hoạch dự phòng": dùng trigram nếu có dữ liệu, nếu không → "lùi" (back off) xuống bigram, nếu vẫn không → lùi xuống unigram. Chỉ dùng mô hình bậc thấp khi bậc cao **thiếu dữ liệu**.

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_3/img-005.png]]
*Hình 5: Công thức Katz Backoff.*

$$P_{\text{katz}}(w_n \mid w_{n-N+1}^{n-1}) = \begin{cases} P^*(w_n \mid w_{n-N+1}^{n-1}) & \text{nếu } C(w_{n-N+1}^n) > 0 \\ \alpha(w_{n-N+1}^{n-1}) \cdot P_{\text{katz}}(w_n \mid w_{n-N+2}^{n-1}) & \text{otherwise} \end{cases}$$

**Giải thích:**
- $P^*$ = xác suất **đã discount** (giảm bớt so với MLE để dành khối lượng xác suất cho unseen)
- $\alpha$ = hệ số backoff (đảm bảo tổng = 1)
- Nếu N-gram có count > 0 → dùng xác suất discounted
- Nếu count = 0 → đệ quy (recursive) lùi xuống (N-1)-gram

**So sánh Backoff vs Interpolation:**

| Tiêu chí | Interpolation | Backoff |
|----------|--------------|---------|
| Khi nào dùng lower-order? | **Luôn luôn** (có trọng số) | **Chỉ khi** higher-order = 0 |
| Complexity | Đơn giản hơn | Cần discount + backoff weight |
| Hiệu quả | Tốt | Thường tốt hơn cho LM |
| Biến thể nổi tiếng | Simple interpolation | **Kneser-Ney** (state-of-the-art) |

### 8.5 Các phương pháp smoothing nâng cao

Slide đề cập thêm:
- **Good-Turing** — dùng tần suất của tần suất (frequency of frequencies) để ước lượng
- **Kneser-Ney** — backoff + modified discount + continuation probability → **tốt nhất** cho N-gram LM truyền thống
- **Class-based N-grams** — nhóm từ thành cluster, tính xác suất trên cluster

---

## 9. Model Combination ⭐

> [!NOTE] ELI5
> Khi N tăng (bigram → trigram → 4-gram...), mô hình "mạnh hơn" (expressive hơn) nhưng "đói dữ liệu" hơn (cần nhiều data hơn để ước lượng chính xác). Giải pháp: **kết hợp** nhiều mô hình N-gram bậc khác nhau (chính là Interpolation/Backoff ở trên).

| N | Expressive power | Data requirement | Sparsity |
|---|-----------------|-----------------|---------|
| 1 (unigram) | Thấp | Thấp | Không có |
| 2 (bigram) | Trung bình | Trung bình | Ít |
| 3 (trigram) | Khá | Cao | Trung bình |
| 5 (5-gram) | Cao | Rất cao | Nghiêm trọng |

→ **Trade-off:** N lớn = hiểu ngữ cảnh nhiều hơn, nhưng cần corpus cực lớn. Trong thực tế, **trigram + Kneser-Ney** là sweet spot cho N-gram truyền thống.

---

## 10. Long-Distance Dependencies ⭐

> [!NOTE] ELI5
> N-gram chỉ "nhìn" N-1 từ trước. Nhưng trong ngôn ngữ thực, có những phụ thuộc **rất xa**. Ví dụ: "The man next to the large oak tree near the grocery store on the corner **is** tall." — "is" phải khớp với "man" (số ít), dù cách nhau 12 từ! Không có N-gram nào (hợp lý) có thể nắm được điều này.

Slide nêu 2 loại phụ thuộc dài hạn:

**Syntactic (cú pháp):**
- "The **man** next to the large oak tree... **is** tall." (man → is, số ít)
- "The **men** next to the large oak tree... **are** tall." (men → are, số nhiều)

**Semantic (ngữ nghĩa):**
- "The **bird** next to the large oak tree... **flies** rapidly." (bird → flies)
- "The **man** next to the large oak tree... **talks** rapidly." (man → talks)

> [!IMPORTANT] Hạn chế cốt lõi của N-gram
> N-gram **không có cơ chế** xử lý long-distance dependencies. Đây chính là lý do ngành NLP chuyển sang **RNN** (nhớ chuỗi dài hơn, nhưng vẫn hạn chế), rồi **Transformer** (attention trực tiếp đến bất kỳ vị trí nào, bất kể khoảng cách).

---

## Kết luận

Chương 3 xây dựng nền tảng **xác suất cho ngôn ngữ**:
1. **LM** gán xác suất cho chuỗi từ — cốt lõi của mọi hệ NLP
2. **N-gram** đơn giản hóa bằng Markov assumption, ước lượng qua MLE
3. **Perplexity** đo chất lượng LM — lower = better
4. **Smoothing** giải quyết zero-probability từ data sparsity
5. **Interpolation & Backoff** kết hợp mô hình bậc khác nhau
6. **Long-distance dependencies** là giới hạn không thể vượt qua của N-gram → mở đường cho neural LM

---

## 📝 Bảng từ điển thuật ngữ

| Thuật ngữ | Tiếng Việt | Giải thích ngắn |
|-----------|-----------|-----------------|
| Language Model | Mô hình ngôn ngữ | Gán xác suất cho chuỗi từ |
| N-gram | N-gram | Chuỗi N từ liên tiếp |
| Chain Rule | Quy tắc chuỗi | Phân rã P(chuỗi) thành tích P(từ\|ngữ cảnh) |
| Bigram | Bigram | 2-gram, chỉ nhìn 1 từ trước |
| Trigram | Trigram | 3-gram, nhìn 2 từ trước |
| MLE | Ước lượng hợp lý cực đại | "Đếm rồi chia" |
| `<s>`, `</s>` | Ký hiệu đầu/cuối câu | Chuẩn hóa biên câu |
| Perplexity | Độ bối rối | Branching factor trung bình (lower = better) |
| Extrinsic eval | Đánh giá ngoại tại | Qua ứng dụng cuối |
| Intrinsic eval | Đánh giá nội tại | Trên test corpus |
| Generalization | Tổng quát hóa | Hoạt động tốt trên dữ liệu mới |
| Bias / Variance | Thiên lệch / Phương sai | Underfitting / Overfitting |
| Smoothing | Làm mịn | Tái phân bổ xác suất cho unseen events |
| Laplace | Smoothing Add-1 | Cộng 1 vào mọi count |
| Add-k | Smoothing Add-k | Cộng k < 1 |
| Interpolation | Nội suy | Kết hợp tuyến tính nhiều bậc N-gram |
| Backoff | Lùi bước | Dùng lower-order khi higher-order thiếu data |
| Kneser-Ney | Kneser-Ney | Backoff + continuation prob (state-of-the-art) |

---

## 🏋️ Bài tập kèm lời giải

### Bài 1: Tính xác suất Bigram

**Đề:** Cho corpus đã thêm `<s>` và `</s>`:
```
<s> I like cats </s>
<s> I like dogs </s>
<s> cats like fish </s>
```

**(a)** Tính $P(\text{like} \mid \text{I})$, $P(\text{cats} \mid \text{like})$, $P(\text{fish} \mid \text{like})$ bằng MLE.
**(b)** Tính xác suất bigram của câu: `<s> I like fish </s>`

> [!TIP]- Lời giải chi tiết
> **(a) Đếm và tính MLE:**
>
> | Bigram | Count | $C(w_{n-1})$ | $P(w_n \mid w_{n-1})$ |
> |--------|-------|-------------|----------------------|
> | (I, like) | 2 | C(I) = 2 | 2/2 = **1.0** |
> | (like, cats) | 1 | C(like) = 3 | 1/3 = **0.333** |
> | (like, fish) | 1 | C(like) = 3 | 1/3 = **0.333** |
>
> **(b) Xác suất câu:**
>
> $P(\text{\<s\> I like fish \</s\>})$
> $= P(\text{I} \mid \text{\<s\>}) \times P(\text{like} \mid \text{I}) \times P(\text{fish} \mid \text{like}) \times P(\text{\</s\>} \mid \text{fish})$
>
> - $P(\text{I} \mid \text{\<s\>}) = C(\text{\<s\>, I}) / C(\text{\<s\>}) = 2/3$
> - $P(\text{like} \mid \text{I}) = 1.0$ (tính ở trên)
> - $P(\text{fish} \mid \text{like}) = 1/3$
> - $P(\text{\</s\>} \mid \text{fish}) = C(\text{fish, \</s\>}) / C(\text{fish}) = 1/1 = 1.0$
>
> $P = \frac{2}{3} \times 1.0 \times \frac{1}{3} \times 1.0 = \frac{2}{9} \approx 0.222$

---

### Bài 2: Perplexity

**Đề:** Mô hình bigram gán xác suất $P = 0.001$ cho một test corpus gồm $N = 100$ từ. Tính perplexity.

> [!TIP]- Lời giải
> $$PP = P(W)^{-1/N} = (0.001)^{-1/100} = (10^{-3})^{-0.01} = 10^{0.03} \approx 1.072$$
>
> **Nhận xét:** Perplexity = 1.072 → gần 1 → mô hình rất tự tin (hoặc test corpus quá ngắn/đơn giản).
>
> **So sánh:** Nếu $P = 10^{-300}$ và $N = 100$:
> $$PP = (10^{-300})^{-0.01} = 10^3 = 1000$$
> → Perplexity = 1000, mô hình rất "bối rối".

---

### Bài 3: Laplace Smoothing ⭐

**Đề:** Dùng cùng corpus ở Bài 1 ($V = 7$: I, like, cats, dogs, fish, `<s>`, `</s>`).

**(a)** Tính $P_{\text{Laplace}}(\text{dogs} \mid \text{I})$ (chú ý: bigram "I dogs" chưa bao giờ xuất hiện!)
**(b)** Tính lại $P_{\text{Laplace}}(\text{like} \mid \text{I})$ và so sánh với MLE.

> [!TIP]- Lời giải
> **(a)** $C(\text{I, dogs}) = 0$, $C(\text{I}) = 2$, $V = 7$
>
> $$P_{\text{Laplace}}(\text{dogs} \mid \text{I}) = \frac{0 + 1}{2 + 7} = \frac{1}{9} \approx 0.111$$
>
> → Dù "I dogs" chưa gặp, Laplace vẫn cho xác suất > 0.
>
> **(b)** $C(\text{I, like}) = 2$
>
> $$P_{\text{Laplace}}(\text{like} \mid \text{I}) = \frac{2 + 1}{2 + 7} = \frac{3}{9} = 0.333$$
>
> So sánh: MLE = 1.0 → Laplace = 0.333. **Giảm 67%!** Đây là nhược điểm chính — xác suất bị "cướp" quá nhiều cho unseen events.
>
> | Phương pháp | P(like\|I) | P(dogs\|I) |
> |------------|-----------|-----------|
> | MLE | 1.0 | 0 |
> | Laplace | 0.333 | 0.111 |

---

### Bài 4: Interpolation

**Đề:** Cho $\lambda_1 = 0.5, \lambda_2 = 0.3, \lambda_3 = 0.2$ và:
- Trigram: $P(\text{food} \mid \text{want, chinese}) = 0$
- Bigram: $P(\text{food} \mid \text{chinese}) = 0.6$
- Unigram: $P(\text{food}) = 0.02$

Tính $\hat{P}(\text{food} \mid \text{want, chinese})$ bằng interpolation.

> [!TIP]- Lời giải
> $$\hat{P} = \lambda_1 \cdot P_{\text{tri}} + \lambda_2 \cdot P_{\text{bi}} + \lambda_3 \cdot P_{\text{uni}}$$
> $$= 0.5 \times 0 + 0.3 \times 0.6 + 0.2 \times 0.02$$
> $$= 0 + 0.18 + 0.004 = \mathbf{0.184}$$
>
> **Nhận xét:** Trigram = 0 (chưa gặp "want chinese food"), nhưng interpolation cho ra 0.184 nhờ bigram "chinese food" có xác suất cao (0.6). Đây là ưu điểm chính: luôn có "lưới an toàn" (safety net) từ lower-order models.

---

### Bài 5: So sánh các phương pháp Smoothing ⭐

**Đề:** Cho bảng count bigram (V = 5):

| Bigram | Count |
|--------|-------|
| (the, cat) | 10 |
| (the, dog) | 5 |
| (the, bird) | 0 |
| Σ C(the) | 15 |

Tính $P(\text{bird} \mid \text{the})$ bằng: (a) MLE, (b) Laplace, (c) Add-0.5, (d) nhận xét.

> [!TIP]- Lời giải
> **(a) MLE:**
> $$P = \frac{0}{15} = 0$$
>
> **(b) Laplace (k=1):**
> $$P = \frac{0 + 1}{15 + 5} = \frac{1}{20} = 0.05$$
>
> **(c) Add-0.5:**
> $$P = \frac{0 + 0.5}{15 + 0.5 \times 5} = \frac{0.5}{17.5} \approx 0.0286$$
>
> **(d) Nhận xét:**
>
> | Phương pháp | P(bird\|the) | P(cat\|the) | Tổng ảnh hưởng |
> |------------|-------------|------------|----------------|
> | MLE | 0 ❌ | 10/15=0.667 | Quá cực đoan |
> | Laplace | 0.05 | 11/20=0.55 | "cat" mất 12% → hơi nhiều |
> | Add-0.5 | 0.029 | 10.5/17.5=0.6 | "cat" mất 7% → hợp lý hơn |
>
> → Add-k với k nhỏ **tinh tế hơn** Laplace, nhưng vẫn kém Kneser-Ney cho LM thực tế.
