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
> Tài liệu gốc: `assets/Library/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang.pdf` (Chapter 2). Nội dung dưới đây là **dịch + diễn giải có phê bình** dựa trên slide; các đoạn được đánh dấu "Suy luận thêm" là phần mở rộng từ kiến thức nền.

---

## 1. Từ và Lớp Từ (Words and Word Classes)

> [!NOTE] ELI5
> Hãy tưởng tượng bạn có một hộp đồ chơi Lego. Mỗi viên Lego có **hình dạng** khác nhau: có viên vuông, viên dài, viên cong. Trong ngôn ngữ, mỗi **từ** cũng có một "hình dạng ngữ pháp" — ta gọi đó là **loại từ** (Part-of-Speech). Ví dụ: "chạy" là động từ, "nhanh" là tính từ, "con mèo" là danh từ. Việc xác định loại từ giúp máy tính "hiểu" câu giống như ta hiểu được viên Lego nào lắp vào đâu.

### 1.1 Định nghĩa "Từ" trong NLP

Trong xử lý ngôn ngữ tự nhiên, khái niệm **từ** (word) phức tạp hơn ta tưởng. Xét câu tiếng Anh từ Brown Corpus:

> *"He stepped out into the hall, was delighted to encounter a water brother."*

Câu này có **13 từ** nếu không tính dấu câu, hoặc **15 từ** nếu tính cả dấu phẩy và dấu chấm như token riêng. Quyết định này phụ thuộc vào mục đích phân tích — trong POS tagging và parsing, dấu câu thường được coi là token riêng vì chúng mang thông tin ranh giới cú pháp.

**Phân biệt Type vs Token:** Đây là distinction cốt lõi trong corpus linguistics. **Token** là tổng số từ xuất hiện (bao gồm lặp lại), trong khi **Type** là số từ phân biệt (distinct). Ví dụ, trong câu "the cat sat on the mat", có 6 token nhưng chỉ 5 type (vì "the" xuất hiện 2 lần). Tỷ lệ Type/Token (TTR - Type-Token Ratio) là một metric đo độ phong phú từ vựng của văn bản.

**Lemma vs Wordform:** Một **lemma** - [[Lemmatization]] là tập hợp các từ có cùng gốc, cùng loại từ và cùng nghĩa. Ví dụ, "cats" và "cat" chia sẻ lemma "cat". Các **wordform** là các biến thể hình thái đầy đủ của từ (số nhiều, thì quá khứ, v.v.). Sự phân biệt này quan trọng vì lemmatization giúp giảm độ thưa (sparsity) trong các mô hình thống kê.

### 1.2 Closed Class vs Open Class

Các loại từ được chia thành hai nhóm lớn dựa trên tính "mở" hay "đóng" của tập thành viên:

**Closed Classes** (lớp đóng) có tập thành viên tương đối cố định và hiếm khi thêm từ mới. Bao gồm:
- **Prepositions** (giới từ): in, on, at, by, with...
- **Determiners** (mạo từ/từ hạn định): the, a, this, that...
- **Pronouns** (đại từ): I, you, he, she, it...
- **Conjunctions** (liên từ): and, but, or, because...
- **Auxiliary verbs** (trợ động từ): will, can, should, have...

**Open Classes** (lớp mở) liên tục có thêm từ mới theo sự phát triển của ngôn ngữ:
- **Nouns** (danh từ): có thể tạo từ mới bất cứ lúc nào (smartphone, selfie, blockchain...)
- **Verbs** (động từ): google, tweet, zoom...
- **Adjectives** (tính từ): viral, sustainable...
- **Adverbs** (trạng từ): digitally, remotely...

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-004.png]]

> [!NOTE] Suy luận thêm — Tại sao phân biệt closed/open class quan trọng?
> Trong thực tế xây dựng POS tagger, closed class words thường có phân phối ổn định và dễ học hơn. Open class words đòi hỏi khả năng xử lý **OOV (Out-of-Vocabulary)** — những từ chưa từng xuất hiện trong tập huấn luyện. Đây là lý do các kỹ thuật như subword tokenization (BPE, WordPiece) trở nên quan trọng trong NLP hiện đại.

---

## 2. Part-of-Speech Tagging: Bài Toán và Ý Nghĩa

> [!NOTE] ELI5
> POS Tagging giống như việc đánh dấu mỗi viên Lego trong câu với nhãn: "đây là danh từ", "đây là động từ". Máy tính đọc câu và tự gắn nhãn cho từng từ. Việc này quan trọng vì cùng một từ có thể là nhiều loại khác nhau — "book" vừa là danh từ (cuốn sách) vừa là động từ (đặt chỗ).

### 2.1 Định nghĩa Formal

[[Part-of-Speech Tagging]] (còn gọi là POS tagging, word-class tagging, hoặc grammatical tagging) là quá trình gán nhãn loại từ cho mỗi từ trong văn bản. Formally:

- **Input:** Một chuỗi từ (đã tokenize) $x_1, x_2, ..., x_n$ và một tagset $T$
- **Output:** Một chuỗi nhãn $y_1, y_2, ..., y_n$ sao cho mỗi $y_i \in T$ tương ứng với $x_i$

Ví dụ với câu *"Hanoi is the capital of Vietnam"*:

| Word    | POS Tag                                 |
| ------- | --------------------------------------- |
| Hanoi   | NNP (Proper Noun)                       |
| is      | VBZ (Verb, 3rd person singular present) |
| the     | DT (Determiner)                         |
| capital | NN (Noun, singular)                     |
| of      | IN (Preposition)                        |
| Vietnam | NNP (Proper Noun)                       |

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-009.jpg]]

### 2.2 Tại Sao POS Tagging Là Bài Toán Disambiguation?

Điểm then chốt của POS tagging là **từ có thể nhập nhằng** (ambiguous) về loại từ. Xét các ví dụ:

> *"Book that fight"* → "Book" là **Verb** (hãy đặt/ghi lại trận đấu đó)
> *"Hand me that book"* → "book" là **Noun** (đưa tôi cuốn sách đó)

> *"Does that fight serve dinner"* → "that" là **Determiner** (trận đấu đó)
> *"I thought that your fight was earlier"* → "that" là **Complementizer** (rằng)

Thống kê từ các corpus lớn cho thấy khoảng **40-60% từ trong từ điển tiếng Anh** có nhiều hơn một POS tag khả dĩ. Tuy nhiên, trong ngữ cảnh cụ thể, con người (và các tagger tốt) có thể disambiguate với độ chính xác rất cao (>97%).

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-006.png]]

### 2.3 Penn Treebank Tagset

[[Penn Treebank]] tagset là hệ thống nhãn POS được sử dụng rộng rãi nhất cho tiếng Anh, phát triển từ dự án Penn Treebank tại University of Pennsylvania. Tagset này bao gồm **36 tags** chính (không kể punctuation), được thiết kế để cân bằng giữa độ chi tiết ngữ pháp và khả năng annotation nhất quán.

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-008.png]]

Một số tags quan trọng:
- **NN/NNS/NNP/NNPS:** Noun singular/plural, Proper noun singular/plural
- **VB/VBD/VBG/VBN/VBP/VBZ:** Verb base/past tense/gerund/past participle/non-3rd person present/3rd person singular present
- **JJ/JJR/JJS:** Adjective/Comparative/Superlative
- **RB/RBR/RBS:** Adverb/Comparative/Superlative
- **DT:** Determiner
- **IN:** Preposition or subordinating conjunction
- **MD:** Modal auxiliary (can, could, may, might, will, would...)

> [!NOTE] Suy luận thêm — Universal Dependencies vs Penn Treebank
> Trong nghiên cứu đa ngôn ngữ hiện đại, **Universal Dependencies (UD)** tagset với 17 tags "phổ quát" đang thay thế Penn Treebank. UD được thiết kế để áp dụng cross-linguistically, cho phép so sánh và transfer learning giữa các ngôn ngữ. Tuy nhiên, Penn Treebank vẫn là baseline quan trọng cho tiếng Anh.

### 2.4 Tầm Quan Trọng Của POS Tagging

POS tagging là **backbone** cho nhiều tác vụ NLP downstream:

1. **Syntax Parsing:** Parser cần biết loại từ để xây dựng cây cú pháp đúng
2. **Named Entity Recognition:** Proper nouns (NNP) là tín hiệu quan trọng cho entity
3. **Information Retrieval:** Lọc theo loại từ giúp cải thiện precision
4. **Machine Translation:** Alignment và reordering phụ thuộc vào cấu trúc ngữ pháp
5. **Sentiment Analysis:** Adjectives và adverbs mang tín hiệu cảm xúc mạnh

---

## 3. Hidden Markov Model (HMM) cho POS Tagging

> [!NOTE] ELI5
> Tưởng tượng bạn là thám tử đang theo dõi một người qua cửa sổ. Bạn **không thể thấy trực tiếp** người đó đang làm gì (nấu ăn? đọc sách? ngủ?), nhưng bạn **có thể thấy ánh đèn bật tắt** trong các phòng. Từ việc quan sát đèn (observable), bạn suy luận ngược lại hoạt động (hidden state). HMM cũng vậy: từ **các từ ta thấy** (observable), ta suy ra **các POS tag ẩn** (hidden states).

### 3.1 Markov Chains: Nền Tảng Xác Suất

[[Markov Chain]] là mô hình xác suất mô tả một chuỗi các trạng thái, trong đó xác suất chuyển sang trạng thái tiếp theo **chỉ phụ thuộc vào trạng thái hiện tại**, không phụ thuộc vào lịch sử trước đó. Đây được gọi là **Markov assumption** (giả định Markov) hay tính chất **memoryless**.

Formally, cho chuỗi trạng thái $q_1, q_2, ..., q_i$:

$$P(q_i | q_1, q_2, ..., q_{i-1}) = P(q_i | q_{i-1})$$

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-014.png]]

**Các thành phần của Markov Chain:**
1. **Tập trạng thái** $Q = \{q_1, q_2, ..., q_N\}$: N trạng thái khả dĩ
2. **Ma trận chuyển trạng thái** $A$: với $a_{ij} = P(q_j | q_i)$ là xác suất chuyển từ trạng thái $i$ sang $j$
3. **Phân phối khởi tạo** $\pi$: với $\pi_i = P(q_1 = i)$ là xác suất bắt đầu ở trạng thái $i$

Ràng buộc: $\sum_j a_{ij} = 1$ (xác suất chuyển từ mỗi trạng thái phải tổng bằng 1) và $\sum_i \pi_i = 1$.

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-015.png]]

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-016.jpg]]

**Ứng dụng của Markov Chains:**
- Dự báo thời tiết, thị trường chứng khoán
- Mô hình hóa chuỗi DNA trong genetics
- PageRank của Google (random walk trên đồ thị web)
- Text generation (Markov text generators)

### 3.2 Hidden Markov Model: Khi Trạng Thái Bị Ẩn

[[Hidden Markov Model (HMM)]] mở rộng Markov Chain bằng cách thêm một lớp **quan sát** (observation) được sinh ra từ các trạng thái ẩn. Trong POS tagging:
- **Hidden states:** Các POS tags (NN, VB, DT, ...) — ta không "thấy" trực tiếp
- **Observations:** Các từ (the, dog, runs, ...) — ta thấy trong văn bản

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-019.png]]

**Các thành phần của HMM:**
1. **Tập trạng thái ẩn** $Q = \{q_1, ..., q_N\}$: N POS tags
2. **Tập quan sát** $O = \{o_1, ..., o_M\}$: M từ trong vocabulary
3. **Ma trận chuyển trạng thái** $A$ ($N \times N$): $a_{ij} = P(t_j | t_i)$ — xác suất tag $j$ xuất hiện sau tag $i$
4. **Ma trận phát xạ** $B$ ($N \times M$): $b_i(o_k) = P(w_k | t_i)$ — xác suất từ $k$ được sinh ra từ tag $i$
5. **Phân phối khởi tạo** $\pi$: xác suất bắt đầu câu với mỗi tag

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-021.jpg]]

### 3.3 Ước Lượng Tham Số HMM (Training)

Các tham số $A$ và $B$ được ước lượng từ **corpus đã gán nhãn** (supervised learning) sử dụng **Maximum Likelihood Estimation (MLE)**:

**Transition probabilities (A):**
$$P(t_i | t_{i-1}) = \frac{C(t_{i-1}, t_i)}{C(t_{i-1})}$$

Trong đó $C(t_{i-1}, t_i)$ là số lần tag $t_i$ xuất hiện ngay sau tag $t_{i-1}$ trong corpus.

**Emission probabilities (B):**
$$P(w_i | t_i) = \frac{C(t_i, w_i)}{C(t_i)}$$

Trong đó $C(t_i, w_i)$ là số lần từ $w_i$ được gán tag $t_i$.

**Ví dụ từ WSJ Corpus:**
- MD (modal) xuất hiện 13,124 lần
- MD được theo sau bởi VB (verb base) 10,471 lần
- Vậy: $P(VB|MD) = \frac{10471}{13124} = 0.7968$

Tương tự:
- Trong 13,124 lần MD xuất hiện, nó được gán cho từ "will" 4,046 lần
- Vậy: $P(\text{will}|MD) = \frac{4046}{13124} = 0.3083$

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-024.jpg]]

### 3.4 HMM Tagging as Decoding

**Bài toán Decoding:** Cho HMM $\lambda = (A, B, \pi)$ và chuỗi quan sát (từ) $W = w_1, w_2, ..., w_n$, tìm chuỗi trạng thái ẩn (tags) $T = t_1, t_2, ..., t_n$ **có xác suất cao nhất**.

Sử dụng Bayes' theorem:

$$\hat{T} = \argmax_T P(T|W) = \argmax_T \frac{P(W|T) \cdot P(T)}{P(W)} = \argmax_T P(W|T) \cdot P(T)$$

(Bỏ $P(W)$ vì nó không phụ thuộc vào $T$)

**Hai giả định đơn giản hóa:**

1. **Output independence:** Xác suất của một từ chỉ phụ thuộc vào tag của chính nó:
$$P(W|T) = \prod_{i=1}^{n} P(w_i | t_i)$$

2. **Bigram assumption (Markov assumption):** Xác suất của một tag chỉ phụ thuộc vào tag ngay trước:
$$P(T) = \prod_{i=1}^{n} P(t_i | t_{i-1})$$

Kết hợp lại:
$$\hat{T} = \argmax_T \prod_{i=1}^{n} P(w_i | t_i) \cdot P(t_i | t_{i-1})$$

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-026.png]]

> [!NOTE] Suy luận thêm — Tại sao công thức này hoạt động?
> Công thức này mã hóa hai loại "tri thức ngôn ngữ": (1) **Tri thức từ vựng** qua $P(w|t)$ — từ nào thường là danh từ, từ nào thường là động từ; (2) **Tri thức cú pháp** qua $P(t_i|t_{i-1})$ — sau mạo từ thường là danh từ, sau modal thường là verb base. Sự kết hợp này cho phép mô hình disambiguate dựa trên cả hai nguồn bằng chứng.

---

## 4. Thuật Toán Viterbi

> [!NOTE] ELI5
> Tưởng tượng bạn đang chơi game tìm đường trong mê cung. Tại mỗi ngã rẽ, có nhiều lựa chọn. **Viterbi** là cách thông minh để tìm con đường tốt nhất: thay vì thử TẤT CẢ các đường (rất lâu!), bạn chỉ nhớ **đường tốt nhất đến mỗi điểm** rồi dần dần xây lên đường tốt nhất tổng thể. Giống như bạn không cần nhớ toàn bộ lịch sử đi qua — chỉ cần biết cách tốt nhất để đến đây.

### 4.1 Vấn Đề Với Brute Force

Nếu có $N$ tags và câu có $n$ từ, tổng số chuỗi tag khả dĩ là $N^n$. Với Penn Treebank (36 tags) và câu 10 từ, đó là $36^{10} \approx 3.6 \times 10^{15}$ — không khả thi để duyệt hết!

### 4.2 Dynamic Programming với Viterbi

[[Viterbi Algorithm]] giải quyết vấn đề này bằng **quy hoạch động** (dynamic programming), dựa trên nhận xét: **đường đi tốt nhất đến trạng thái $(i, t)$ (vị trí $i$, tag $t$) phải đi qua đường đi tốt nhất đến một trạng thái nào đó ở vị trí $i-1$**.

**Định nghĩa:**
$$v_t(j) = \max_{t_1,...,t_{j-1}} P(t_1,...,t_{j-1}, w_1,...,w_j, t_j = t)$$

$v_t(j)$ là xác suất cao nhất của bất kỳ chuỗi tag nào kết thúc ở tag $t$ tại vị trí $j$.

**Công thức đệ quy:**
$$v_t(j) = \max_{t' \in Tags} \left[ v_{t'}(j-1) \cdot a_{t',t} \cdot b_t(w_j) \right]$$

Trong đó:
- $v_{t'}(j-1)$: xác suất tốt nhất đến vị trí $j-1$ với tag $t'$
- $a_{t',t} = P(t|t')$: xác suất chuyển từ $t'$ sang $t$
- $b_t(w_j) = P(w_j|t)$: xác suất phát xạ từ $w_j$ từ tag $t$

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-027.jpg]]

### 4.3 Thuật Toán Viterbi Chi Tiết

**Khởi tạo (j = 1):**
$$v_t(1) = \pi_t \cdot b_t(w_1)$$

**Đệ quy (j = 2 đến n):**
$$v_t(j) = \max_{t'} \left[ v_{t'}(j-1) \cdot a_{t',t} \right] \cdot b_t(w_j)$$
$$bt_t(j) = \argmax_{t'} \left[ v_{t'}(j-1) \cdot a_{t',t} \right]$$

Trong đó $bt_t(j)$ lưu **backpointer** — tag trước đó tốt nhất để đến $(j, t)$.

**Kết thúc:**
$$\hat{t}_n = \argmax_t v_t(n)$$

**Backtracing:** Từ $\hat{t}_n$, đi ngược theo backpointers để lấy toàn bộ chuỗi tag.

**Độ phức tạp:** $O(N^2 \cdot n)$ thay vì $O(N^n)$ — một cải tiến khổng lồ!

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-028.jpg]]

### 4.4 Ví Dụ Minh Họa

Xét câu: *"Janet will back the bill"*

**Bước 1:** Xây dựng lattice với các tag khả dĩ cho mỗi từ:
- Janet: NNP (proper noun)
- will: MD (modal), NN (noun), VB (verb)
- back: VB (verb), JJ (adjective), NN (noun), RB (adverb)
- the: DT (determiner)
- bill: NN (noun), VB (verb)

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-028.jpg]]

**Bước 2:** Tính $v_t(j)$ cho mỗi ô trong lattice:

Ví dụ, tại vị trí "will" với tag MD:
$$v_{MD}(2) = v_{NNP}(1) \cdot P(MD|NNP) \cdot P(\text{will}|MD)$$

**Bước 3:** Tại mỗi ô, lưu backpointer đến ô tốt nhất ở cột trước.

**Bước 4:** Sau khi điền xong, backtrace từ ô có xác suất cao nhất ở cột cuối.

**Kết quả:** Janet/NNP will/MD back/VB the/DT bill/NN

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-000.jpg]]

> [!NOTE] Suy luận thêm — Underflow và Log Probabilities
> Trong thực tế, tích của nhiều xác suất nhỏ dẫn đến **numerical underflow**. Giải pháp: làm việc trong không gian log-probability, biến phép nhân thành phép cộng: $\log(ab) = \log(a) + \log(b)$. Điều này không thay đổi argmax.

---

## 5. Beam Search: Khi Cần Tốc Độ Hơn Nữa

> [!NOTE] ELI5
> Viterbi giữ **tất cả** các đường tốt nhất đến mỗi tag tại mỗi vị trí. Beam Search "tiết kiệm" hơn: chỉ giữ **top k đường tốt nhất** (gọi là "beam width"). Giống như cuộc thi chạy — thay vì để tất cả vận động viên chạy đến cuối rồi mới xếp hạng, ta loại bớt người chậm sau mỗi vòng.

### 5.1 Thuật Toán Beam Search

[[Beam Search]] là một **approximate search** algorithm giữ $k$ (beam width) chuỗi tốt nhất tại mỗi bước:

1. **Khởi tạo:** Tạo tags cho $w_1$, giữ top $k$ chuỗi, gọi là $S_{1,1}, ..., S_{1,k}$
2. **Lặp:** Với $i = 2$ đến $n$:
   - Với mỗi chuỗi $S_{i-1,j}$ trong beam hiện tại:
     - Sinh tất cả tags khả dĩ cho $w_i$
     - Nối mỗi tag vào $S_{i-1,j}$ để tạo chuỗi mới
   - Từ tất cả chuỗi mới sinh ra, giữ top $k$ có xác suất cao nhất
3. **Kết thúc:** Trả về chuỗi có xác suất cao nhất trong beam cuối

### 5.2 Ưu và Nhược Điểm

**Ưu điểm:**
- **Nhanh:** Độ phức tạp $O(k \cdot N \cdot n)$ — tuyến tính theo độ dài câu
- **Đơn giản:** Không cần cấu trúc dữ liệu phức tạp của DP
- **Thực tiễn:** Beam width 3-5 thường cho kết quả gần tối ưu

**Nhược điểm:**
- **Inexact:** Chuỗi tốt nhất toàn cục có thể bị loại sớm (fall off the beam)
- **Không có guarantee:** Không đảm bảo tìm được optimal solution

> [!NOTE] Suy luận thêm — Beam Search trong Deep Learning
> Beam Search không chỉ dùng cho HMM mà còn là thuật toán decoding tiêu chuẩn cho các mô hình sequence-to-sequence (machine translation, text generation với LLM). Trong các hệ thống như GPT, beam search được dùng để sinh văn bản có chất lượng cao hơn greedy decoding.

---

## 6. Supervised vs Unsupervised POS Tagging

### 6.1 Supervised Methods

Các phương pháp **supervised** yêu cầu corpus đã gán nhãn (labeled training data):

- **HMM Tagger:** Như đã mô tả ở trên
- **Maximum Entropy (MaxEnt) Tagger:** Discriminative model sử dụng nhiều features
- **Conditional Random Fields (CRF):** Sequence labeling với global normalization
- **Neural Taggers:** LSTM/Transformer-based models (state-of-the-art hiện tại)

Các phương pháp supervised thường đạt accuracy > 97% trên tiếng Anh benchmark (WSJ).

### 6.2 Unsupervised Methods

Khi không có labeled data (nhiều ngôn ngữ ít tài nguyên), các phương pháp **unsupervised** học cấu trúc từ raw text:

- **Clustering-based:** Nhóm từ theo distributional similarity
- **Bayesian HMM:** Học tham số HMM mà không có labels (EM algorithm)
- **Neural approaches:** Sử dụng cross-lingual transfer từ ngôn ngữ giàu tài nguyên

Unsupervised taggers thường có accuracy thấp hơn đáng kể (~70-80%), nhưng là lựa chọn duy nhất cho nhiều ngôn ngữ.

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-011.jpg]]

---

## 7. Kết Luận và Định Hướng Học

Chapter này đã thiết lập POS tagging như một **bài toán sequence labeling** cơ bản trong NLP. Các điểm chính:

1. **Từ và lớp từ** là đơn vị phân tích nền tảng; sự phân biệt type/token, lemma/wordform, closed/open class ảnh hưởng đến cách xây dựng mô hình.

2. **Ambiguity** là thách thức trung tâm — cùng một wordform có thể có nhiều POS tags, và việc disambiguation phụ thuộc vào ngữ cảnh.

3. **HMM** cung cấp framework xác suất để kết hợp tri thức từ vựng ($P(w|t)$) và tri thức cú pháp ($P(t|t')$) một cách coherent.

4. **Viterbi Algorithm** cho phép tìm chuỗi tag tối ưu trong thời gian đa thức, là ví dụ kinh điển của dynamic programming trong NLP.

5. **Beam Search** cung cấp trade-off giữa độ chính xác và tốc độ, quan trọng cho ứng dụng thực tế.

Các chapter tiếp theo sẽ xây dựng trên POS tagging để giải quyết các bài toán phức tạp hơn: parsing (phân tích cú pháp), NER (nhận dạng thực thể), và semantic analysis.

---

## TODO

- [ ] Liên kết sâu hơn với [[Conditional Random Fields (CRF)]] khi tạo concept note
- [ ] So sánh HMM tagger với neural tagger hiện đại (BiLSTM-CRF, BERT-based)
- [ ] Thêm ví dụ POS tagging cho tiếng Việt (đặc thù tokenization)
