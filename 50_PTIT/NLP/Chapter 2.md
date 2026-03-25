---
tags:
  - nlp
  - ptit
  - source-note
status: completed
source:
  - assets/Library/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang.pdf
chapter: 2
aliases:
  - Chapter 2 NLP
  - POS Tagging
---

# Chapter 2 — Part-of-Speech Tagging

> [!NOTE] Source
> Tài liệu gốc: `assets/Library/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang.pdf` (Chapter 2, slides 1–35). Nội dung dưới đây bám sát slide-by-slide, kèm diễn giải sâu. Phần đánh dấu "Suy luận thêm" là mở rộng từ kiến thức nền.

---

## 1. Words — Từ trong NLP

### 1.1 What counts as a "word"?

**Corpus** (số nhiều: corpora) là tập hợp văn bản/tiếng nói được lưu trữ dưới dạng máy tính có thể đọc. Ví dụ: **Brown corpus** — 1 triệu từ lấy mẫu từ 500 văn bản tiếng Anh thuộc nhiều thể loại (báo chí, tiểu thuyết, học thuật...).

Slide đặt câu hỏi: *Câu "He stepped out into the hall, was delighted to encounter a water brother." có bao nhiêu từ?*
- 13 từ (nếu không tính dấu câu)
- 15 từ (nếu tính dấu câu `,` và `.` là từ riêng)

### 1.2 Types vs Tokens

| Khái niệm | Ý nghĩa | Ví dụ (câu: "the cat sat on the mat") |
|------------|---------|---------------------------------------|
| **Token** | Tổng số từ (kể cả lặp) | 6 tokens: the, cat, sat, on, the, mat |
| **Type** | Số từ phân biệt | 5 types: the, cat, sat, on, mat |

### 1.3 Lemma vs Wordform

- **Lemma** = nhóm các từ có cùng gốc + cùng POS + cùng nghĩa. Ví dụ: "cats", "cat" → lemma `cat`
- **Wordform** = dạng biến đổi đầy đủ (inflected/derived). Ví dụ: "cats", "cat's", "catlike" đều là wordform khác nhau

---

## 2. Word Classes — Lớp từ

### 2.1 Open Class vs Closed Class

| Loại | Đặc điểm | Ví dụ |
|------|----------|-------|
| **Open class** | Liên tục có từ mới được thêm vào | Noun, Verb, Adjective, Adverb |
| **Closed class** | Số lượng cố định, hiếm khi thêm mới | Preposition, Determiner, Conjunction, Pronoun |

Tại sao phân biệt? Vì **closed class** ít nhập nhằng (ít từ → dễ gán nhãn), còn **open class** rất dễ nhập nhằng (từ mới liên tục, nhiều nghĩa).

### 2.2 Universal Dependencies Tagset

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-001.png]]
*Hình 1: Bảng 17 POS tags trong Universal Dependencies tagset — chia rõ Open Class, Closed Class, và Other.*

---

## 3. Penn Treebank POS Tagset

Ngoài Universal Dependencies (17 tags), slide giới thiệu hệ thống chi tiết hơn: **Penn Treebank** (~45 tags), được sử dụng rộng rãi trong nghiên cứu NLP tiếng Anh.

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-003.png]]
*Hình 2: Bảng tags Penn Treebank — phân biệt chi tiết hơn: NN (noun singular), NNS (noun plural), VB (verb base), VBZ (verb 3sg present), v.v.*

**So sánh hai hệ thống:**

| Tiêu chí | Universal Dependencies | Penn Treebank |
|----------|----------------------|---------------|
| Số lượng tags | ~17 | ~45 |
| Phạm vi | Đa ngôn ngữ | Chủ yếu tiếng Anh |
| Mức chi tiết | Thô (NOUN chung) | Mịn (NN, NNS, NNP, NNPS) |
| Sử dụng phổ biến | spaCy, UD treebanks | WSJ corpus, NLTK |

---

## 4. POS Tagging là gì? ⭐

> [!NOTE] ELI5
> POS tagging giống như cô giáo tiếng Việt gạch chân từng từ trong câu rồi ghi phía dưới: "danh từ", "động từ", "tính từ"... Máy tính phải làm điều tương tự — nhưng tự động, trên hàng triệu câu, và phải xử lý những trường hợp nhập nhằng mà con người thường dựa vào cảm giác để phán đoán.

**POS Tagging** (Part-of-Speech Tagging) là quá trình gán nhãn **loại từ** cho mỗi từ (token) trong câu.

- **Input:** chuỗi từ đã tokenize $x_1, x_2, \ldots, x_n$ + tập nhãn (tagset)
- **Output:** chuỗi tag $y_1, y_2, \ldots, y_n$, mỗi $y_i$ tương ứng chính xác 1 $x_i$

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-002.png]]
*Hình 3: Minh họa input/output của POS tagger — "Janet will back the bill" → NOUN AUX VERB DET NOUN.*

**Ví dụ cụ thể:**

| Token | POS Tag (Penn Treebank) | Giải thích |
|-------|------------------------|-----------|
| Hanoi | NNP (proper noun, singular) | Tên riêng |
| is | VBZ (verb, 3sg present) | Động từ ngôi 3 số ít |
| the | DT (determiner) | Mạo từ |
| capital | NN (noun, singular) | Danh từ |
| of | IN (preposition) | Giới từ |
| Vietnam | NNP | Tên riêng |

### 4.1 POS Tagging = Disambiguation (Giải nhập nhằng) ⭐

> [!NOTE] ELI5
> Trong tiếng Anh, rất nhiều từ "đa nhân cách" — cùng một từ nhưng đóng vai trò khác nhau tùy câu. POS tagger phải "nhìn xung quanh" (ngữ cảnh) để chọn đúng vai. Giống như bạn thấy từ "book" — nếu trước nó có "please" thì nó là VERB ("hãy đặt"), nếu trước nó có "that" thì nó là NOUN ("cuốn sách đó").

Slide minh họa với từ **"book"**, **"that"**, và **"back"**:

| Câu | Từ nhập nhằng | Tag đúng | Lý do |
|-----|--------------|----------|-------|
| "**Book** that flight" | book | **VB** (verb) | Đặt chỗ chuyến bay |
| "Hand me that **book**" | book | **NN** (noun) | Cuốn sách |
| "Does **that** flight serve dinner?" | that | **DT** (determiner) | Chỉ định "chuyến bay đó" |
| "I thought **that** your flight was earlier" | that | **IN** (complementizer) | Liên từ "rằng" |
| "Janet will **back** the bill" | back | **VB** (verb) | Ủng hộ |
| "Get **back** to work" | back | **RB** (adverb) | Trở lại |
| "I have a sore **back**" | back | **NN** (noun) | Lưng |

> [!IMPORTANT] Thống kê thú vị
> Khoảng 55-67% các types trong tiếng Anh là **nhập nhằng** (có nhiều POS tag khả dĩ). Tuy nhiên, hầu hết tokens trong thực tế đều có 1 tag **trội hơn hẳn** (ví dụ "the" gần như luôn là DT). Baseline đơn giản nhất — gán tag phổ biến nhất cho mỗi từ — đã đạt ~90% accuracy!

### 4.2 Tầm quan trọng của POS Tagging

Slide liệt kê:
1. **Improves Language Understanding** — hiểu cấu trúc câu
2. **Facilitates Syntax Analysis** — nền tảng cho parsing
3. **Enhances Search & IR** — phân biệt "book a flight" vs "read a book"
4. **Enables Machine Translation** — biết loại từ giúp dịch chính xác
5. **Assists Sentiment Analysis** — adjective/adverb mang tín hiệu cảm xúc mạnh

### 4.3 Các phương pháp POS Tagging

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-004.png]]
*Hình 4: Cây phân loại các phương pháp POS Tagging — Supervised vs Unsupervised, mỗi nhánh có Rule-based, Stochastic, Neural.*

**Supervised** (có nhãn huấn luyện):
- **Rule-based:** Viết luật tay ("nếu từ kết thúc bằng -ly → ADV")
- **Stochastic:** Dùng xác suất — **HMM** (trọng tâm Chapter 2), N-gram
- **Neural:** Dùng mạng nơ-ron (BiLSTM-CRF, Transformer — hiện đại)

**Unsupervised** (không có nhãn):
- **Stochastic:** Baum-Welch algorithm (EM cho HMM)
- Ít chính xác hơn supervised, nhưng dùng được khi không có tập dữ liệu gán nhãn

---

## 5. Markov Chains — Chuỗi Markov ⭐

> [!NOTE] ELI5
> Tưởng tượng bạn đang dự đoán **thời tiết ngày mai**. Nếu bạn chỉ được phép nhìn **thời tiết hôm nay** (không được nhìn hôm qua, hôm kia...) để đoán ngày mai — đó là Markov chain! Nó nói rằng: "Tương lai chỉ phụ thuộc vào hiện tại, quá khứ không quan trọng." Nghe có vẻ đơn giản (và đôi khi sai), nhưng giả định này giúp bài toán giảm từ "cực kỳ phức tạp" xuống "tính toán được".

### 5.1 Định nghĩa

[[Markov Chain]] (do Andrey Markov giới thiệu năm 1906) là **mô hình ngẫu nhiên** mô tả một hệ thống chuyển đổi giữa các **trạng thái** (states) theo xác suất, với tính chất đặc biệt gọi là **memorylessness** (không nhớ quá khứ).

**Ví dụ thời tiết trong slide:**

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-007.jpg]]
*Hình 5: (a) Markov chain cho thời tiết gồm 3 trạng thái: HOT, COLD, WARM. (b) Markov chain cho chuỗi từ: "uniformly", "are", "charming".*

Đọc hình (a):
- Nếu hôm nay **HOT** → xác suất 60% ngày mai cũng HOT, 30% sang WARM, 10% sang COLD
- Nếu hôm nay **COLD** → 80% tiếp tục COLD, 10% sang HOT, 10% sang WARM
- Tổng xác suất từ mỗi trạng thái = 1.0 (tất yếu)

### 5.2 Markov Assumption — Giả định Markov ⭐

> [!NOTE] ELI5
> Giả định Markov giống quy tắc "chỉ nhìn 1 bước lùi": để đoán ngày mai, bạn chỉ cần biết hôm nay là gì, không cần nhớ cả tuần qua. Trong toán học:

$$P(q_i = a \mid q_1 \ldots q_{i-1}) = P(q_i = a \mid q_{i-1})$$

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-008.png]]
*Hình 6: Công thức Markov Assumption.*

**Giải thích từng ký hiệu:**
- $q_i$ = trạng thái tại thời điểm $i$
- $q_1 \ldots q_{i-1}$ = toàn bộ lịch sử trạng thái từ đầu đến hiện tại
- Vế trái: xác suất trạng thái tiếp theo **biết toàn bộ lịch sử** (impossible to compute khi lịch sử dài)
- Vế phải: xác suất trạng thái tiếp theo **chỉ biết trạng thái trước đó** (tính toán được!)

**Ví dụ cụ thể:**
- Thay vì: $P(\text{WARM} \mid \text{HOT}, \text{COLD}, \text{COLD}, \text{HOT})$ → cần bảng khổng lồ
- Ta chỉ cần: $P(\text{WARM} \mid \text{HOT}) = 0.3$ → tra bảng chuyển trạng thái

> [!WARNING] Giới hạn
> Markov assumption rõ ràng là **đơn giản hóa quá mức** (oversimplification). Thời tiết thực tế phụ thuộc vào mùa, xu hướng dài hạn, v.v. Tuy nhiên, giả định này giúp bài toán **khả thi về mặt tính toán** và thực tế hoạt động tốt đáng ngạc nhiên cho nhiều bài NLP.

### 5.3 Các thành phần của Markov Chain

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-010.png]]
*Hình 7: Các thành phần hình thức của Markov chain.*

| Thành phần | Ký hiệu | Ý nghĩa |
|------------|---------|---------|
| **States** | $Q = \{q_1, q_2, \ldots, q_N\}$ | Tập hợp $N$ trạng thái có thể |
| **Transition matrix** | $A = [a_{ij}]$ | Ma trận $N \times N$; $a_{ij}$ = xác suất chuyển từ trạng thái $i$ sang $j$ |
| **Initial distribution** | $\pi = [\pi_1, \pi_2, \ldots, \pi_N]$ | Xác suất bắt đầu ở mỗi trạng thái; $\sum \pi_i = 1$ |

**Ràng buộc quan trọng:** Mỗi hàng của ma trận A phải **tổng bằng 1**: $\sum_{j=1}^{N} a_{ij} = 1 \quad \forall i$. Trực giác: từ bất kỳ trạng thái nào, bạn **chắc chắn** sẽ chuyển sang một trạng thái nào đó (kể cả chính nó).

### 5.4 Ứng dụng của Markov Chains

Slide liệt kê: market share predictions, text generators, asset pricing, customer journey predictions, population genetics, algorithmic music composition, **PageRank** (Google).

---

## 6. Hidden Markov Model (HMM) ⭐⭐

> [!NOTE] ELI5
> Markov chain bình thường: bạn **nhìn thấy** trạng thái (ví dụ: HOT, COLD, WARM). HMM: bạn **KHÔNG nhìn thấy** trạng thái — bạn chỉ thấy **tín hiệu** phát ra từ trạng thái đó. Giống như bạn ở trong phòng không có cửa sổ, chỉ thấy mọi người mặc áo ấm hay áo mát → đoán thời tiết bên ngoài. Trong POS tagging: bạn **thấy các từ** (observations), nhưng **không thấy POS tags** (hidden states) — phải suy ngược lại!

[[Hidden Markov Model]] là mô hình xác suất sinh (generative model) cho chuỗi, trong đó:
- Có tập **trạng thái ẩn** (hidden states) — ví dụ: POS tags (NOUN, VERB, DET...)
- Tại mỗi bước, hệ thống **chuyển trạng thái** theo xác suất (transition probability)
- Tại mỗi trạng thái, hệ thống **phát ra** một quan sát theo xác suất (emission probability)

### 6.1 Tại sao gọi là "Hidden"?

```
Chuỗi quan sát (THẤY được):     Janet    will     back    the    bill
                                   ↑        ↑        ↑      ↑       ↑
Trạng thái ẩn (KHÔNG thấy):     NNP      MD       VB     DT      NN
```

Ta **chỉ thấy** các từ. Nhiệm vụ: **suy ra** chuỗi trạng thái ẩn tốt nhất.

### 6.2 Các thành phần của HMM

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-013.png]]
*Hình 8: Các thành phần hình thức của HMM — so với Markov chain, thêm O (observations) và B (emission probabilities).*

So với Markov chain, HMM thêm 2 thành phần:

| Thành phần | Ký hiệu | Ý nghĩa |
|------------|---------|---------|
| States | $Q = \{q_1, \ldots, q_N\}$ | Tập trạng thái ẩn (POS tags) |
| **Transition matrix** | $A = [a_{ij}]$ | $P(\text{tag}_j \mid \text{tag}_i)$ — xác suất chuyển tag |
| **Observations** | $O = o_1, o_2, \ldots, o_T$ | Chuỗi quan sát (các từ) |
| **Emission probs** | $B = [b_i(o_t)]$ | $P(\text{word} \mid \text{tag})$ — xác suất phát từ |
| Initial distribution | $\pi$ | Xác suất bắt đầu ở mỗi tag |

### 6.3 HMM Tagger: Ma trận A và B ⭐

> [!NOTE] ELI5
> Ma trận A trả lời: "Nếu từ trước là NOUN, từ tiếp theo có khả năng là loại từ gì?" — ví dụ VERB rất hay đi sau NOUN.
> Ma trận B trả lời: "Nếu tag là VERB, từ nào hay xuất hiện với tag đó?" — ví dụ "run", "eat", "play" đều là VERB phổ biến.

**Ma trận A — Transition Probabilities:**

$P(t_i \mid t_{i-1})$ = xác suất tag $t_i$ xuất hiện **ngay sau** tag $t_{i-1}$

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-025.png]]
*Hình 9: Ma trận A — transition probabilities (WSJ corpus). Hàng là tag trước, cột là tag sau. Ví dụ: P(VB|MD) = 0.7968.*

**Đọc bảng — Ví dụ cụ thể:**
- $P(\text{VB} \mid \text{MD}) = 0.7968$ → sau modal verb (can, will, should...) thì gần 80% là **verb base** (run, eat, go...)
- $P(\text{NNP} \mid \text{NNP}) = 0.3777$ → danh từ riêng hay đi liền nhau ("New York", "Los Angeles")
- $P(\text{DT} \mid \langle s \rangle) = 0.2026$ → khoảng 20% câu bắt đầu bằng determiner ("The", "A", "This")

**Ma trận B — Emission Probabilities:**

$P(w_i \mid t_i)$ = xác suất từ $w_i$ **được phát ra** từ tag $t_i$

Ví dụ trong slide (WSJ corpus):
$$P(\text{will} \mid \text{MD}) = \frac{C(\text{MD}, \text{will})}{C(\text{MD})} = \frac{4046}{13124} \approx 0.308$$

Giải thích: trong 13124 lần tag MD xuất hiện, 4046 lần nó đi với từ "will" → xác suất ≈ 30.8%.

### 6.4 Maximum Likelihood Estimation (MLE) ⭐

> [!NOTE] ELI5
> MLE = "đếm rồi chia". Muốn biết xác suất VB đi sau MD? Đếm có bao nhiêu lần MD→VB, chia cho tổng số lần MD xuất hiện. Đơn giản đến bất ngờ, nhưng rất hiệu quả khi có đủ dữ liệu.

**Công thức MLE cho transition probability:**

$$P(t_i \mid t_{i-1}) = \frac{C(t_{i-1}, t_i)}{C(t_{i-1})}$$

- $C(t_{i-1}, t_i)$ = số lần tag $t_{i-1}$ ngay trước tag $t_i$ trong corpus
- $C(t_{i-1})$ = tổng số lần tag $t_{i-1}$ xuất hiện

**Ví dụ:** Trong WSJ corpus, MD xuất hiện 13124 lần, trong đó 10471 lần theo sau bởi VB:
$$P(\text{VB} \mid \text{MD}) = \frac{10471}{13124} \approx 0.7968$$

**Công thức MLE cho emission probability:**

$$P(w_i \mid t_i) = \frac{C(t_i, w_i)}{C(t_i)}$$

**Bản chất:** MLE giả định rằng **tần suất trong corpus ≈ xác suất thật**. Điều này chỉ đúng khi corpus đủ lớn. Với từ/cặp tag hiếm → MLE cho ra xác suất rất nhỏ hoặc 0 (vấn đề **sparsity** — sẽ giải quyết bằng smoothing ở Chapter 3).

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-018.jpg]]
*Hình 10: HMM tagger với 3 trạng thái (VB, MD, NN) — mũi tên liền = transition probabilities (A), mũi tên đứt = emission probabilities (B).*

---

## 7. HMM Decoding — Giải mã HMM ⭐⭐

> [!NOTE] ELI5
> **Decoding** là bài toán ngược: đã biết các từ trong câu, tìm ra chuỗi POS tags **tốt nhất**. Giống như bác sĩ nghe mô tả triệu chứng (observations) → đoán bệnh (hidden states). Nhưng phải đoán **cả chuỗi bệnh** cùng lúc, không phải từng triệu chứng riêng lẻ.

**Bài toán hình thức:**

Cho HMM $\lambda = (A, B)$ và chuỗi quan sát $O = o_1, o_2, \ldots, o_T$ (các từ), tìm chuỗi trạng thái $\hat{Q} = \hat{q}_1, \hat{q}_2, \ldots, \hat{q}_T$ (các tag) sao cho:

$$\hat{t}_1^n = \underset{t_1^n}{\arg\max} \; P(t_1^n \mid w_1^n) \quad \text{...(1)}$$

### 7.1 Hai giả định đơn giản hóa ⭐

Để tính được công thức (1), HMM tagger làm **2 giả định**:

**Giả định 1 — Output Independence:** Xác suất từ chỉ phụ thuộc vào tag **của chính nó**, không phụ thuộc tag/từ xung quanh:
$$P(w_i \mid t_1^n, w_1^n) \approx P(w_i \mid t_i) \quad \text{...(2)}$$

*Ví dụ:* P("will" | MD) không phụ thuộc vào từ trước "will" là gì — chỉ phụ thuộc tag MD.

**Giả định 2 — Bigram Assumption:** Xác suất tag chỉ phụ thuộc vào tag **ngay trước đó**:
$$P(t_i \mid t_1^{i-1}) \approx P(t_i \mid t_{i-1}) \quad \text{...(3)}$$

*Ví dụ:* P(VB | ..., NNP, MD) ≈ P(VB | MD) = 0.7968.

**Ghép lại**, công thức decoding trở thành:

$$\hat{t}_1^n = \underset{t_1^n}{\arg\max} \; \prod_{i=1}^{n} \underbrace{P(w_i \mid t_i)}_{\text{emission B}} \cdot \underbrace{P(t_i \mid t_{i-1})}_{\text{transition A}}$$

> [!TIP] Trực giác
> Chuỗi tag tốt nhất = chuỗi cân bằng giữa: (a) mỗi tag "hợp" với từ tương ứng (emission cao), VÀ (b) các tag liền kề "hợp" với nhau (transition cao). Không phải cứ chọn tag tốt nhất cho từng từ riêng lẻ — phải tối ưu **toàn chuỗi**.

### 7.2 Tại sao không brute-force?

Nếu có $N$ tag và câu dài $T$ từ → có $N^T$ chuỗi tag khả dĩ. Với 45 tags, câu 20 từ: $45^{20} \approx 10^{33}$ chuỗi — **không thể duyệt hết**!

→ Cần thuật toán thông minh hơn: **Viterbi Algorithm**.

---

## 8. Viterbi Algorithm ⭐⭐⭐

> [!NOTE] ELI5
> Tưởng tượng bạn đang tìm đường đi ngắn nhất trong mê cung. Tại mỗi ngã rẽ, bạn **chỉ giữ lại đường tốt nhất dẫn đến đó** và bỏ các đường kém hơn. Khi đến cuối mê cung, bạn lần ngược lại (backtrace) để tìm toàn bộ đường đi tốt nhất. Viterbi làm đúng như vậy — thay vì thử tất cả các chuỗi tag (mê cung), nó chỉ giữ "đường tốt nhất dẫn đến mỗi trạng thái" tại mỗi bước.

[[Viterbi Algorithm]] là thuật toán **quy hoạch động** (dynamic programming) giải bài toán decoding một cách hiệu quả.

### 8.1 Ý tưởng cốt lõi

**Thay vì:** thử tất cả $N^T$ chuỗi → chọn chuỗi tốt nhất
**Viterbi:** tại mỗi bước thời gian $t$ và mỗi trạng thái $s$, chỉ lưu **đường đi có xác suất cao nhất dẫn đến (t, s)**

$\Rightarrow$ Độ phức tạp: $O(N^2 \cdot T)$ thay vì $O(N^T)$

### 8.2 Pseudocode

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-023.png]]
*Hình 11: Pseudocode thuật toán Viterbi.*

**Giải thích từng bước:**

**Bước 1 — Initialization (khởi tạo):**
Với mỗi trạng thái $s$:
$$\text{viterbi}[s, 1] = \pi_s \cdot b_s(o_1)$$
Nghĩa: xác suất bắt đầu ở trạng thái $s$ VÀ phát ra từ đầu tiên $o_1$.

**Bước 2 — Recursion (đệ quy):**
Với mỗi bước thời gian $t$ từ 2 đến $T$, với mỗi trạng thái $s$:
$$\text{viterbi}[s, t] = \max_{s'=1}^{N} \left[ \text{viterbi}[s', t-1] \cdot a_{s',s} \cdot b_s(o_t) \right]$$
$$\text{backpointer}[s, t] = \underset{s'}{\arg\max} \left[ \text{viterbi}[s', t-1] \cdot a_{s',s} \right]$$

Nghĩa: đường tốt nhất đến $(s, t)$ = đường tốt nhất đến một trạng thái $s'$ ở bước trước × transition × emission. Lưu lại $s'$ tốt nhất (backpointer) để truy vết.

**Bước 3 — Termination (kết thúc):**
Chọn trạng thái cuối cùng có viterbi lớn nhất, rồi **lần ngược backpointer** để khôi phục toàn bộ chuỗi tag.

### 8.3 Ví dụ chi tiết: "Janet will back the bill" ⭐⭐

Slide trình bày ví dụ đầy đủ. Sử dụng 7 trạng thái: NNP, MD, VB, JJ, NN, RB, DT.

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-024.jpg]]
*Hình 12: Viterbi lattice — theo chiều ngang là các từ (Janet → will → back → the → bill), theo chiều dọc là các tag khả dĩ. Tag bôi xanh/nét đậm = có xác suất > 0.*

**Bước 1 — Janet (t=1):**

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-028.jpg]]
*Hình 13: Chi tiết tính toán Viterbi cho cột Janet và will.*

Tính $\text{viterbi}[s, 1] = \pi_s \cdot b_s(\text{"Janet"})$ cho mỗi tag:

| Tag $s$ | $\pi_s$ (từ bảng A, hàng $\langle s \rangle$) | $b_s$("Janet") | viterbi[s,1] |
|---------|-------|---------|------------|
| NNP | 0.2767 | 0.000032 | **0.000009** |
| MD | 0.0006 | 0 | 0 |
| VB | 0.0031 | 0 | 0 |
| JJ | 0.0453 | 0 | 0 |
| NN | 0.0449 | 0 | 0 |
| RB | 0.0510 | 0 | 0 |
| DT | 0.2026 | 0 | 0 |

→ Chỉ NNP có xác suất > 0 (Janet là proper noun, chỉ tag NNP phát ra từ này)

**Bước 2 — will (t=2):**

Chỉ cần xét trạng thái trước NNP (vì các trạng thái khác = 0):
$$\text{viterbi}[\text{MD}, 2] = \text{viterbi}[\text{NNP}, 1] \times a_{\text{NNP},\text{MD}} \times b_{\text{MD}}(\text{"will"})$$
$$= 0.000009 \times 0.0110 \times 0.308 = 2.772 \times 10^{-8}$$

Tag MD cho "will" thắng vì: (a) từ "will" là modal verb kinh điển, (b) MD hay đi sau NNP.

**Sau khi hoàn tất tất cả các cột → backtrace:**

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_2/img-027.jpg]]
*Hình 14: Kết quả backtrace — đường đi tốt nhất (nét đậm): NNP → MD → VB → DT → NN.*

**Kết quả:** Janet/NNP will/MD back/VB the/DT bill/NN ✅

> [!TIP] Key insight
> Viterbi **không thử mọi tổ hợp**. Tại mỗi bước, mỗi trạng thái chỉ giữ **1 đường đi tốt nhất** dẫn đến nó. 7 tags × 5 từ = chỉ cần tính 35 ô, thay vì $7^5 = 16807$ chuỗi!

---

## 9. Beam Search ⭐

> [!NOTE] ELI5
> Viterbi = tìm **chính xác** đường tốt nhất, nhưng với tagset lớn hoặc mô hình phức tạp thì chậm. Beam Search = giữ **top-k đường tốt nhất** tại mỗi bước, bỏ qua phần còn lại. Nhanh hơn nhiều, nhưng có thể bỏ sót đường tốt nhất (inexact).

[[Beam Search]] là thuật toán tìm kiếm **gần đúng** (approximate), hoạt động như sau:

1. Tại vị trí đầu tiên ($w_1$): sinh tất cả tag khả dĩ, giữ **top-N** chuỗi
2. Tại mỗi vị trí tiếp theo ($w_i$):
   - Mỗi chuỗi trong top-N mở rộng bằng mọi tag khả dĩ
   - Tính xác suất cho tất cả chuỗi mở rộng
   - Chỉ giữ lại **top-N** có xác suất cao nhất
3. Trả về chuỗi có xác suất cao nhất

**So sánh Viterbi vs Beam Search:**

| Tiêu chí | Viterbi | Beam Search |
|----------|---------|-------------|
| Chính xác? | ✅ Exact | ❌ Approximate |
| Phức tạp | $O(N^2 T)$ | $O(kNT)$, k = beam width |
| Dynamic programming? | ✅ Có | ❌ Không cần |
| Beam width | N/A | k = 3-5 thường đủ tốt |
| Rủi ro | Không mất nghiệm | Nghiệm tốt có thể "rơi khỏi beam" |

> [!TIP] Khi nào dùng gì?
> - **Viterbi**: khi tagset nhỏ (< 50 tags), cần kết quả chính xác
> - **Beam Search**: khi không gian trạng thái quá lớn (translation, generation), chấp nhận gần đúng

---

## Kết luận

Chương 2 tập trung vào:
1. **POS Tagging** — gán nhãn loại từ cho mỗi token, bản chất là bài toán **giải nhập nhằng**
2. **HMM** — mô hình sinh dùng trạng thái ẩn (tags) và quan sát (words), với ma trận A (transition) và B (emission)
3. **Viterbi** — thuật toán quy hoạch động tìm chuỗi tag tốt nhất trong $O(N^2 T)$
4. **Beam Search** — phương pháp gần đúng nhanh hơn cho không gian tìm kiếm lớn

---

## 📝 Bảng từ điển thuật ngữ

| Thuật ngữ | Tiếng Việt | Giải thích ngắn |
|-----------|-----------|-----------------|
| POS Tagging | Gán nhãn từ loại | Gán noun/verb/adj... cho mỗi token |
| Type | Từ phân biệt | Số từ khác nhau (không tính lặp) |
| Token | Đơn vị từ | Tổng số từ (kể cả lặp) |
| Lemma | Dạng từ điển | Gốc từ hợp lệ (play, cat...) |
| Open/Closed class | Lớp mở/đóng | Lớp mở thêm từ mới, lớp đóng cố định |
| Penn Treebank | Hệ thống tag ~45 nhãn | Tagset chi tiết cho tiếng Anh |
| Markov Chain | Chuỗi Markov | Mô hình ngẫu nhiên memoryless |
| Markov Assumption | Giả định Markov | Tương lai chỉ phụ thuộc hiện tại |
| HMM | Mô hình Markov ẩn | Có trạng thái ẩn + quan sát |
| Transition prob (A) | Xác suất chuyển tiếp | P(tag tiếp | tag trước) |
| Emission prob (B) | Xác suất phát ra | P(từ | tag) |
| MLE | Ước lượng hợp lý cực đại | "Đếm rồi chia" |
| Decoding | Giải mã | Tìm chuỗi tag tốt nhất |
| Viterbi | Thuật toán Viterbi | DP tìm chuỗi tối ưu |
| Beam Search | Tìm kiếm chùm | Giữ top-k, gần đúng nhưng nhanh |
| Backpointer | Con trỏ ngược | Lưu vết để truy ngược |

---

## 🏋️ Bài tập kèm lời giải

### Bài 1: Types vs Tokens

**Đề:** Đếm số types và tokens trong câu: *"The cat sat on the mat and the dog sat on the rug."*

> [!TIP]- Lời giải
> **Tokens (tổng số từ):** 14 (tính cả dấu ".")
> 
> Liệt kê: The, cat, sat, on, the, mat, and, the, dog, sat, on, the, rug, .
>
> **Types (từ phân biệt):** 9
>
> {the, cat, sat, on, mat, and, dog, rug, .}
>
> Lưu ý: "The" (viết hoa) và "the" (viết thường) — nếu chuẩn hóa lowercase thì chỉ còn **8 types** → tùy quy ước.

---

### Bài 2: Tính Transition & Emission Probability

**Đề:** Cho corpus nhỏ gồm 3 câu đã tag sẵn:
```
The/DT cat/NN sat/VB .  
A/DT dog/NN ran/VB quickly/RB .  
The/DT big/JJ cat/NN ran/VB .  
```

Tính: (a) $P(\text{NN} \mid \text{DT})$, (b) $P(\text{VB} \mid \text{NN})$, (c) $P(\text{"cat"} \mid \text{NN})$

> [!TIP]- Lời giải
> **Bước 1: Đếm**
>
> Tần suất các cặp tag liên tiếp:
> - DT→NN: "The cat", "A dog" = **2 lần**
> - DT→JJ: "The big" = **1 lần**
> - Tổng DT xuất hiện: 3 lần (The, A, The)
> 
> NN→VB: "cat sat", "dog ran", "cat ran" = **3 lần**
> - Tổng NN xuất hiện: 3 lần (cat, dog, cat)
>
> Tag NN xuất hiện với từ "cat": 2 lần (cat/NN ở câu 1 và 3)
> - Tổng NN xuất hiện: 3 lần
>
> **Bước 2: Tính MLE**
>
> **(a)** $P(\text{NN} \mid \text{DT}) = \frac{C(\text{DT}, \text{NN})}{C(\text{DT})} = \frac{2}{3} \approx 0.667$
>
> **(b)** $P(\text{VB} \mid \text{NN}) = \frac{C(\text{NN}, \text{VB})}{C(\text{NN})} = \frac{3}{3} = 1.0$
>
> **(c)** $P(\text{"cat"} \mid \text{NN}) = \frac{C(\text{NN}, \text{"cat"})}{C(\text{NN})} = \frac{2}{3} \approx 0.667$
>
> **Nhận xét:** P(VB|NN) = 1.0 → trong corpus nhỏ này, NN **luôn** đi trước VB. Đây là vấn đề **data sparsity** — corpus quá nhỏ khiến MLE cho ra xác suất cực đoan.

---

### Bài 3: Viterbi Algorithm (tính tay) ⭐

**Đề:** Cho HMM đơn giản với:
- Tags: {N, V} (N = Noun, V = Verb)
- Từ vựng: {fish, can}
- Initial probs: $\pi_N = 0.6, \; \pi_V = 0.4$
- Transition matrix A:

|     | N   | V   |
|-----|-----|-----|
| N   | 0.3 | 0.7 |
| V   | 0.8 | 0.2 |

- Emission matrix B:

|     | fish | can |
|-----|------|-----|
| N   | 0.8  | 0.2 |
| V   | 0.3  | 0.7 |

Tìm chuỗi tag tốt nhất cho câu: **"fish can"**

> [!TIP]- Lời giải chi tiết
> **Bước 1 — Initialization (t=1, từ "fish"):**
>
> $v_N(1) = \pi_N \cdot b_N(\text{fish}) = 0.6 \times 0.8 = 0.48$
> $v_V(1) = \pi_V \cdot b_V(\text{fish}) = 0.4 \times 0.3 = 0.12$
>
> **Bước 2 — Recursion (t=2, từ "can"):**
>
> $v_N(2) = \max \begin{cases} v_N(1) \cdot a_{NN} \cdot b_N(\text{can}) = 0.48 \times 0.3 \times 0.2 = 0.0288 \\ v_V(1) \cdot a_{VN} \cdot b_N(\text{can}) = 0.12 \times 0.8 \times 0.2 = 0.0192 \end{cases} = 0.0288$ (từ N)
>
> $v_V(2) = \max \begin{cases} v_N(1) \cdot a_{NV} \cdot b_V(\text{can}) = 0.48 \times 0.7 \times 0.7 = 0.2352 \\ v_V(1) \cdot a_{VV} \cdot b_V(\text{can}) = 0.12 \times 0.2 \times 0.7 = 0.0168 \end{cases} = \mathbf{0.2352}$ (từ N)
>
> **Bước 3 — Termination:**
>
> $\max(v_N(2), v_V(2)) = \max(0.0288, 0.2352) = 0.2352$ → tag cuối = **V**
>
> **Backtrace:** V ← N (backpointer của V(2) trỏ về N)
>
> **Kết quả:** fish/**N** can/**V** → "fish" là danh từ (con cá), "can" là động từ (có thể)
>
> **Xác suất đường đi tốt nhất:** 0.2352

---

### Bài 4: So sánh Viterbi vs Brute-Force

**Đề:** Tính số phép tính cần thiết cho câu 10 từ, tagset 45 tags:
(a) Brute-force (duyệt mọi chuỗi tag)
(b) Viterbi

> [!TIP]- Lời giải
> **(a) Brute-force:** $45^{10} = 3.405 \times 10^{16}$ chuỗi → **34 triệu tỷ** phép tính! Với 1 tỷ phép tính/giây → cần ~1 năm.
>
> **(b) Viterbi:** $45^2 \times 10 = 20250$ phép tính → xong trong **< 1 millisecond**!
>
> **Tỷ lệ tăng tốc:** $\frac{45^{10}}{45^2 \times 10} = \frac{45^8}{10} \approx 1.68 \times 10^{12}$ lần (1.68 nghìn tỷ lần nhanh hơn!)

---

### Bài 5: Phân tích POS Ambiguity

**Đề:** Cho từ **"light"**. Tìm ít nhất 3 câu mà "light" mang POS tag khác nhau, xác định tag tương ứng.

> [!TIP]- Lời giải
> | Câu | POS tag | Giải thích |
> |-----|---------|-----------|
> | "Turn on the **light**." | **NN** (noun) | "light" = ánh sáng, ngọn đèn |
> | "**Light** the candle." | **VB** (verb) | "light" = thắp, bật |
> | "This bag is very **light**." | **JJ** (adjective) | "light" = nhẹ |
> | "She stepped **light**ly." → "a **light** step" | **JJ** (adjective) | "light" = nhẹ nhàng |
>
> Từ "light" có **ít nhất 3 POS tags**: NN, VB, JJ — minh họa rõ ràng vấn đề **lexical ambiguity** mà POS tagger phải giải quyết.
