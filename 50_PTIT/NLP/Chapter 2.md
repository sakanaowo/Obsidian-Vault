---
tags:
  - nlp
  - ptit
  - source-note
  - pos-tagging
  - hmm
status: completed
source:
  - assets/Library/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang.pdf
aliases:
  - POS Tagging Chapter
  - Chapter 2 NLP PTIT
---

# Chapter 2 — Part-of-Speech Tagging

> [!NOTE] Source
> Tài liệu gốc: `assets/Library/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang.pdf` (Chapter 2, slide 1-35). Nội dung dưới đây bám sát 100% cấu trúc gốc với phần ELI5 và giải thích mục đích/ứng dụng bổ sung.

---

## 1. Words and Word Classes

> [!NOTE] ELI5
> Hãy tưởng tượng bạn có một hộp đồ chơi Lego. Mỗi viên Lego có **hình dạng** khác nhau: viên vuông, viên dài, viên cong. Trong ngôn ngữ, mỗi **từ** cũng có một "hình dạng ngữ pháp" — ta gọi đó là **loại từ** (Part-of-Speech). Ví dụ: "chạy" là động từ, "nhanh" là tính từ, "con mèo" là danh từ.

### 1.1 What counts as a word?

**Corpus** (số nhiều: corpora) là tập hợp văn bản hoặc lời nói được lưu trữ trên máy tính để phân tích.

**Ví dụ từ tài liệu gốc:** Brown Corpus là bộ sưu tập một triệu từ từ 500 văn bản tiếng Anh thuộc nhiều thể loại (báo chí, tiểu thuyết, phi hư cấu, học thuật, v.v.).

Xét câu từ Brown Corpus:
> *"He stepped out into the hall, was delighted to encounter a water brother."*

Câu này có bao nhiêu từ?
- **13 từ** nếu không tính dấu câu
- **15 từ** nếu tính dấu phẩy và dấu chấm như token riêng

**Mục đích:** Xác định chính xác "từ" là bước đầu tiên trong mọi xử lý ngôn ngữ — ta cần biết đơn vị cơ bản để phân tích.

**Ứng dụng:** 
- Trong POS tagging và parsing, dấu câu thường được coi là token riêng vì chúng đánh dấu ranh giới cú pháp
- Tokenization là bước tiền xử lý bắt buộc cho mọi pipeline NLP

### 1.2 Types vs Tokens

Trong NLP, ta phân biệt rõ giữa **types** và **tokens**:

| Khái niệm | Định nghĩa | Ví dụ với câu "the cat sat on the mat" |
|-----------|------------|----------------------------------------|
| **Tokens** | Tổng số từ xuất hiện (bao gồm lặp lại) | 6 tokens |
| **Types** | Số từ phân biệt (distinct) | 5 types (vì "the" xuất hiện 2 lần) |

**Mục đích:** Phân biệt này giúp ta đo lường độ phong phú từ vựng (vocabulary richness) của văn bản.

**Ứng dụng:** Type-Token Ratio (TTR) là metric đánh giá độ đa dạng từ vựng — văn bản học thuật thường có TTR cao hơn văn bản hội thoại.

### 1.3 Lemma vs Wordforms

- **Lemma:** Tập hợp các từ có cùng gốc, cùng loại từ và cùng nghĩa
  - Ví dụ: "cats" và "cat" chia sẻ lemma "cat"
- **Wordforms:** Các biến thể hình thái đầy đủ của từ (số nhiều, thì quá khứ, v.v.)

**Mục đích:** Lemmatization giúp giảm độ thưa (sparsity) trong các mô hình thống kê — thay vì học riêng "run", "runs", "running", ta chỉ cần học "run".

### 1.4 Closed Class vs Open Class

Các loại từ được chia thành hai nhóm lớn:

**Closed Classes** (lớp đóng) — tập thành viên tương đối cố định:
- **Prepositions** (giới từ): in, on, at, by, with...
- **Determiners** (mạo từ/từ hạn định): the, a, this, that...
- **Pronouns** (đại từ): I, you, he, she, it...
- **Conjunctions** (liên từ): and, but, or, because...
- **Auxiliary verbs** (trợ động từ): will, can, should, have...

**Open Classes** (lớp mở) — liên tục có thêm từ mới:
- **Nouns** (danh từ): smartphone, selfie, blockchain...
- **Verbs** (động từ): google, tweet, zoom...
- **Adjectives** (tính từ): viral, sustainable...
- **Adverbs** (trạng từ): digitally, remotely...

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-001.png]]
*Bảng phân loại POS tags theo Open Class, Closed Class và Other*

**Mục đích:** Closed class words có phân phối ổn định và dễ học hơn. Open class words đòi hỏi khả năng xử lý **OOV (Out-of-Vocabulary)**.

**Ứng dụng:** Trong thực tế, các kỹ thuật như subword tokenization (BPE, WordPiece) được phát triển để xử lý open class words chưa từng xuất hiện trong tập huấn luyện.

---

## 2. Part-of-Speech Tagging

> [!NOTE] ELI5
> POS Tagging giống như việc đánh dấu mỗi viên Lego trong câu với nhãn: "đây là danh từ", "đây là động từ". Máy tính đọc câu và tự gắn nhãn cho từng từ.

### 2.1 Định nghĩa

[[Part-of-Speech Tagging]] là quá trình gán nhãn loại từ cho mỗi từ trong văn bản:

- **Input:** Một chuỗi từ (đã tokenize) $x_1, x_2, ..., x_n$ và một tagset
- **Output:** Một chuỗi nhãn $y_1, y_2, ..., y_n$ sao cho mỗi $y_i$ tương ứng với $x_i$

**Ví dụ từ tài liệu gốc:**

> "Hanoi is the capital of Vietnam."

| Word    | POS Tag                                 |
| ------- | --------------------------------------- |
| Hanoi   | NN (Noun)                               |
| is      | VBZ (Verb, 3rd person singular present) |
| the     | DT (Determiner)                         |
| capital | NN (Noun)                               |
| of      | IN (Preposition)                        |
| Vietnam | NN (Noun)                               |

### 2.2 Ambiguity — Tại sao POS Tagging là bài toán Disambiguation?

Tagging là **bài toán disambiguation** vì từ có thể **nhập nhằng** — có nhiều hơn một POS tag khả dĩ.

**Ví dụ từ tài liệu gốc:**

| Câu | Từ "book" | POS |
|-----|-----------|-----|
| "Book that fight" | book | **Verb** (đặt/ghi lại) |
| "Hand me that book" | book | **Noun** (cuốn sách) |

| Câu | Từ "that" | POS |
|-----|-----------|-----|
| "Does that fight serve dinner" | that | **Determiner** (trận đấu đó) |
| "I thought that your fight was earlier" | that | **Complementizer** (rằng) |

**Mục đích:** Mục tiêu của POS tagging là giải quyết các trường hợp nhập nhằng này, chọn đúng tag cho ngữ cảnh.

### 2.3 Importance of POS Tagging

POS tagging là **backbone** cho nhiều tác vụ NLP:

1. **Improves Language Understanding:** Hiểu cấu trúc ngữ pháp của câu
2. **Facilitates Syntax Analysis:** Parser cần biết loại từ để xây dựng cây cú pháp
3. **Enhances Search and Information Retrieval:** Lọc theo loại từ cải thiện precision
4. **Enables Machine Translation:** Alignment và reordering phụ thuộc vào cấu trúc ngữ pháp
5. **Assists Sentiment Analysis:** Adjectives và adverbs mang tín hiệu cảm xúc mạnh

### 2.4 How POS Tagging Works — Supervised vs Unsupervised

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-004.png]]
*Sơ đồ phân loại các phương pháp POS Tagging: Supervised (Rule-based, Stochastic, Neural) và Unsupervised*

**Supervised POS tagging:**
- Sử dụng training dataset đã được gán nhãn
- Học mối quan hệ giữa từ và POS tags
- Dùng model đã train để dự đoán tag cho từ mới dựa trên context
- **Độ chính xác cao** (>97% trên WSJ)

**Unsupervised POS tagging:**
- Không sử dụng labeled training data
- Dựa vào phương pháp thống kê để học quan hệ từ-tag
- **Độ chính xác thấp hơn** (~70-80%)
- Hữu ích cho ngôn ngữ ít tài nguyên (low-resource languages)

---

## 3. Penn Treebank Part-of-Speech Tagset

> [!NOTE] ELI5
> Penn Treebank giống như một "bảng màu" chuẩn cho nhãn POS — thay vì mỗi người tự đặt tên nhãn riêng, tất cả dùng chung một bộ 36 nhãn này để hiểu nhau.

[[Penn Treebank]] tagset là hệ thống nhãn POS được sử dụng rộng rãi nhất cho tiếng Anh:

- Phát triển từ dự án Penn Treebank tại University of Pennsylvania
- Nhằm annotate một large corpus tiếng Anh với thông tin cú pháp và cấu trúc
- Bao gồm **36 tags** chính (không kể punctuation)

**Một số tags quan trọng:**

| Tag | Nghĩa | Ví dụ |
|-----|-------|-------|
| NN/NNS | Noun singular/plural | cat, cats |
| NNP/NNPS | Proper noun singular/plural | Hanoi, Americans |
| VB/VBD/VBG/VBN/VBP/VBZ | Verb các dạng | run, ran, running, run, run, runs |
| JJ/JJR/JJS | Adjective/Comparative/Superlative | good, better, best |
| RB/RBR/RBS | Adverb/Comparative/Superlative | quickly, more quickly, most quickly |
| DT | Determiner | the, a, this |
| IN | Preposition or subordinating conjunction | in, on, because |
| MD | Modal auxiliary | can, could, will, would |

**Link:** https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

---

## 4. HMM Part-of-Speech Tagging

> [!NOTE] ELI5
> Tưởng tượng bạn là thám tử đang theo dõi người qua cửa sổ. Bạn **không thể thấy trực tiếp** người đó đang làm gì (nấu ăn? đọc sách? ngủ?), nhưng bạn **có thể thấy ánh đèn** bật tắt trong các phòng. Từ việc quan sát đèn (observable), bạn suy luận ngược lại hoạt động (hidden state). HMM cũng vậy: từ **các từ ta thấy**, ta suy ra **các POS tag ẩn**.

[[Hidden Markov Model|Hidden Markov Model (HMM)]] là phương pháp thống kê để xác định POS tags cho từ trong câu.

**Các thành phần chính:**
1. Hidden Markov Model
2. Training
3. Tagging Process
4. Decoding

### 4.1 Markov Chains

> [!NOTE] ELI5
> Markov chain giống như chơi cờ cá ngựa: vị trí tiếp theo của bạn chỉ phụ thuộc vào vị trí hiện tại + số xúc xắc, không cần biết bạn đi qua những ô nào trước đó.

Năm 1906, Andrey Markov giới thiệu **Markov chains**.

**Nguyên tắc "Memorylessness":** Xác suất chuyển sang trạng thái tiếp theo **chỉ phụ thuộc vào trạng thái hiện tại**, không phụ thuộc vào chuỗi trạng thái trước đó.

$$P(q_i | q_1, q_2, ..., q_{i-1}) = P(q_i | q_{i-1})$$

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-005.jpg]]
*Sơ đồ Markov đơn giản với 2 trạng thái E và A*

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-006.png]]
*Markov chain với xác suất chuyển đổi giữa các trạng thái E và A*

**Ví dụ từ tài liệu gốc:** Để dự đoán thời tiết **ngày mai**, bạn có thể xét thời tiết **hôm nay** nhưng không được phép xem thời tiết **hôm qua**.

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-009.jpg]]
*Ví dụ Markov chain với 3 trạng thái thời tiết: HOT, COLD, WARM và xác suất chuyển đổi*

**Các thành phần của Markov Chain:**

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-010.png]]
*Định nghĩa formal: Q (tập states), A (transition matrix), π (initial distribution)*

**Ứng dụng của Markov Chains:**
- Market share predictions
- Markov text generators
- Financial predictions (stock market)
- Customer journey predictions
- Population genetics
- Algorithmic music composition
- Page ranks (Google search results)

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-011.jpg]]
*Ví dụ ứng dụng PageRank sử dụng Markov Chain*

### 4.2 Hidden Markov Model

**Markov Model** là mô hình stochastic mô tả hệ thống thay đổi ngẫu nhiên, trong đó trạng thái tương lai chỉ phụ thuộc vào trạng thái hiện tại.

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-012.jpg]]
*State transition diagram với 5 states và các transition probabilities*

**Hidden Markov Model** mở rộng Markov Model bằng cách thêm lớp **observations** được sinh ra từ các **hidden states**:

- **Hidden states:** Các POS tags (NN, VB, DT, ...) — ta không "thấy" trực tiếp
- **Observations:** Các từ (the, dog, runs, ...) — ta thấy trong văn bản
- **Transitions:** Xác suất chuyển từ state này sang state khác (ví dụ: DT → NN)
- **Emissions:** Xác suất sinh từ từ một state (ví dụ: NN → "dog")

### 4.3 The Components of an HMM Tagger

HMM có hai thành phần chính: **A probabilities** và **B probabilities**.

**Ma trận A — Transition Probabilities:**
- $P(t_i | t_{i-1})$ = xác suất tag $t_i$ xuất hiện ngay sau tag $t_{i-1}$
- Tính bằng Maximum Likelihood Estimation (MLE):

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-014.png]]
*Công thức tính Transition Probability*

$$P(t_i | t_{i-1}) = \frac{C(t_{i-1}, t_i)}{C(t_{i-1})}$$

**Ví dụ từ WSJ Corpus (tài liệu gốc):**

MD (modal auxiliary verb) xuất hiện 13,124 lần:
- MD được theo sau bởi VB (verb base) 10,471 lần
- Vậy: $P(VB|MD) = \frac{10471}{13124} = 0.7968$

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-015.png]]
*Ví dụ tính P(VB|MD) = 0.80*

**Ma trận B — Emission Probabilities:**
- $P(w_i | t_i)$ = xác suất từ $w_i$ được gán tag $t_i$
- Tính bằng MLE:

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-016.jpg]]
*Công thức tính Emission Probability*

$$P(w_i | t_i) = \frac{C(t_i, w_i)}{C(t_i)}$$

Emission probability:
- Từ "will" được gán tag MD 4,046 lần trong 13,124 lần MD xuất hiện
- Vậy: $P(\text{will}|MD) = \frac{4046}{13124} = 0.3083$

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-017.png]]
*Ví dụ tính P(will|MD) = 0.31*

### 4.4 HMM Tagging as Decoding

**Decoding** là task xác định chuỗi hidden states (tags) tương ứng với chuỗi observations (từ).

**Bài toán:** Cho HMM $\lambda = (A, B)$ và chuỗi quan sát $O = o_1, o_2, ..., o_T$, tìm chuỗi states $Q = q_1, q_2, ..., q_T$ có xác suất cao nhất:

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-019.png]]
*Công thức Decoding: argmax P(tags|words)*

$$\hat{t}_{1:n} = argmax_{t_{1:n}} P(t_{1:n} | w_{1:n}) \tag{1}$$

**Hai giả định đơn giản hóa:**

**1. Output Independence:** Xác suất của từ chỉ phụ thuộc vào tag của chính nó:

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-021.jpg]]
*Output Independence Assumption*

$$P(w_i | w_1, ..., w_n, t_1, ..., t_n) \approx P(w_i | t_i) \tag{2}$$

**2. Bigram Assumption:** Xác suất của tag chỉ phụ thuộc vào tag ngay trước:
$$P(t_i | t_1, ..., t_{i-1}) \approx P(t_i | t_{i-1}) \tag{3}$$

Kết hợp (2) và (3) vào (1):
$$\hat{t}_{1:n} = \argmax_{t_{1:n}} \prod_{i=1}^{n} P(w_i | t_i) \cdot P(t_i | t_{i-1})$$

**Mục đích:** Công thức này kết hợp hai nguồn tri thức:
- **Tri thức từ vựng** qua $P(w|t)$: từ nào thường là danh từ, từ nào thường là động từ
- **Tri thức cú pháp** qua $P(t|t')$: sau mạo từ thường là danh từ, sau modal thường là verb base

### 4.5 The Viterbi Algorithm

> [!NOTE] ELI5
> Tưởng tượng bạn đang chơi game tìm đường trong mê cung. Tại mỗi ngã rẽ, có nhiều lựa chọn. **Viterbi** là cách thông minh để tìm con đường tốt nhất: thay vì thử TẤT CẢ các đường (rất lâu!), bạn chỉ nhớ **đường tốt nhất đến mỗi điểm** rồi dần dần xây lên đường tốt nhất tổng thể.

[[Viterbi Algorithm]] là thuật toán decoding cho HMM, sử dụng **dynamic programming**.

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-023.png]]
*Pseudocode thuật toán Viterbi*

**Vấn đề với Brute Force:**
- Nếu có $N$ tags và câu có $n$ từ, tổng số chuỗi tag khả dĩ là $N^n$
- Với 36 tags và câu 10 từ: $36^{10} \approx 3.6 \times 10^{15}$ — không khả thi!

**Giải pháp Dynamic Programming:**
- Độ phức tạp giảm xuống $O(N^2 \cdot n)$ — một cải tiến khổng lồ!

**Ví dụ từ tài liệu gốc: "Janet will back the bill"**

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-024.jpg]]
*Viterbi trellis cho câu "Janet will back the bill" với các tags khả dĩ*

**Bước 1:** Lattice với các tag khả dĩ:
- Janet: NNP (proper noun)
- will: MD (modal), NN (noun), VB (verb)
- back: VB (verb), JJ (adjective), NN (noun), RB (adverb)
- the: DT (determiner)
- bill: NN (noun), VB (verb)

**Transition probabilities (A)** từ WSJ corpus:

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-025.png]]
*Ma trận Transition Probabilities A*

**Observation likelihoods (B):**

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-026.png]]
*Ma trận Emission Probabilities B*

**Bước 2:** Điền Viterbi lattice

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-027.jpg]]
*Viterbi trellis với các giá trị và backpointers được tính toán*

Mỗi cell giữ:
- Xác suất của best path đến cell đó
- Pointer đến previous cell trên path đó

**Bước 3:** Backtracing từ end state để reconstruct chuỗi tag tối ưu:

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-028.jpg]]
*Kết quả cuối cùng: Janet/NNP will/MD back/VB the/DT bill/NN*

**Kết quả:** Janet/NNP will/MD back/VB the/DT bill/NN

> [!NOTE] Suy luận thêm — Underflow và Log Probabilities
> Trong thực tế, tích của nhiều xác suất nhỏ dẫn đến **numerical underflow**. Giải pháp: làm việc trong không gian log-probability, biến phép nhân thành phép cộng: $\log(ab) = \log(a) + \log(b)$.

### 4.6 Beam Search

> [!NOTE] ELI5
> Viterbi giữ **tất cả** các đường tốt nhất đến mỗi tag tại mỗi vị trí. Beam Search "tiết kiệm" hơn: chỉ giữ **top k đường tốt nhất**. Giống như cuộc thi chạy — thay vì để tất cả vận động viên chạy đến cuối, ta loại bớt người chậm sau mỗi vòng.

[[Beam Search]] là thuật toán tìm kiếm approximate, nhanh hơn Viterbi.

**Thuật toán:**
1. Generate tags cho $w_1$, tìm top N, set $s_{1j}$ tương ứng ($j = 1, 2, ..., N$)
2. For $i = 2$ to $n$ (độ dài câu):
   - For $j = 1$ to $N$:
     - Generate tags cho $w_i$ given $S_{(i-1)j}$ làm previous tag context
     - Append mỗi tag vào $S_{(i-1)j}$ để tạo sequence mới
   - Tìm N sequences có probability cao nhất, set $s_{ij}$ tương ứng
3. Return sequence có probability cao nhất: $S_{n1}$

**Ưu điểm:**
- **Fast:** Beam sizes của 3-5 thường cho kết quả gần tối ưu
- **Easy to implement:** Không cần dynamic programming phức tạp

**Nhược điểm:**
- **Inexact:** Globally best sequence có thể bị loại sớm (fall off the beam)

**Ứng dụng:** Beam Search không chỉ dùng cho HMM mà còn là thuật toán decoding tiêu chuẩn cho machine translation, text generation với LLM (GPT, etc.).

---

## 5. Kết Luận

Chapter này đã thiết lập **POS tagging** như một bài toán **sequence labeling** cơ bản trong NLP:

1. **Words and Word Classes:** Phân biệt types/tokens, lemma/wordforms, closed/open class — ảnh hưởng đến cách xây dựng mô hình

2. **Ambiguity:** Thách thức trung tâm của POS tagging — cùng một wordform có thể có nhiều POS tags

3. **Penn Treebank:** Tagset chuẩn với 36 tags cho tiếng Anh

4. **HMM Tagger:** Framework xác suất kết hợp:
   - Tri thức từ vựng: $P(w|t)$
   - Tri thức cú pháp: $P(t|t')$

5. **Viterbi Algorithm:** Dynamic programming tìm chuỗi tag tối ưu trong $O(N^2n)$

6. **Beam Search:** Trade-off giữa accuracy và speed

---

## TODO

- [ ] Liên kết với [[Conditional Random Fields (CRF)]]
- [ ] So sánh HMM tagger với neural tagger (BiLSTM-CRF, BERT-based)
- [ ] Thêm ví dụ POS tagging cho tiếng Việt
