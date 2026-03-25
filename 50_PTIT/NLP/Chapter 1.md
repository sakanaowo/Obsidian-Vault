---
tags:
  - nlp
  - ptit
  - source-note
status: completed
source:
  - assets/Library/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang.pdf
chapter: 1
aliases:
  - Chapter 1 NLP
  - NLP Introduction
---

# Chapter 1 — Introduction to NLP

> [!NOTE] Source
> Tài liệu gốc: `assets/Library/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang.pdf` (Chapter 1, slides 1–24). Nội dung dưới đây là **dịch + diễn giải có phê bình** dựa trên slide; các đoạn được đánh dấu "Suy luận thêm" là phần mở rộng từ kiến thức nền.

---

## 1. NLP là gì?

> [!NOTE] ELI5
> Tưởng tượng bạn nói chuyện với một người nước ngoài. Bạn phải: (1) **nghe hiểu** họ nói gì (Understanding), rồi (2) **nói lại** bằng ngôn ngữ họ hiểu (Generation). NLP dạy máy tính làm đúng 2 việc đó — nhưng khó hơn rất nhiều vì máy không có "trực giác" về ngôn ngữ.

[[Natural Language Processing (NLP)]] là lĩnh vực giao thoa giữa **khoa học máy tính**, **trí tuệ nhân tạo** và **ngôn ngữ học**, nhằm biến tín hiệu ngôn ngữ tự nhiên (văn bản, tiếng nói) thành biểu diễn mà máy có thể suy luận và hành động, rồi khi cần, biến biểu diễn đó trở lại thành ngôn ngữ mà người dùng hiểu.

Slide trình bày NLP qua hai nhiệm vụ nền tảng:

**Natural Language Understanding (NLU):** Đi từ câu nói/câu viết → suy ra **ý định** và/hoặc **nghĩa**. Ví dụ: khi nghe "Hà Nội là thủ đô của Việt Nam", hệ thống NLU phải trích xuất được: *chủ thể* = Hà Nội, *quan hệ* = là thủ đô của, *đối tượng* = Việt Nam.

**Natural Language Generation (NLG):** Đi từ biểu diễn có cấu trúc (ý định, facts, dữ liệu) → tạo ra câu tự nhiên phù hợp. Ví dụ: từ dữ liệu `{city: "Hanoi", role: "capital", country: "Vietnam"}`, NLG sinh ra câu: "Hanoi is the capital of Vietnam."

**Mục tiêu cốt lõi** (slide 3): *"Deep understanding of broad language — not just string processing or keyword matching."* Tức là NLP phải đi sâu hơn việc so khớp chuỗi ký tự, phải hiểu **nghĩa** đằng sau từ ngữ.

---

## 2. Sự đa dạng ngôn ngữ

Slide 4 nhấn mạnh rằng trên thế giới có hàng nghìn ngôn ngữ khác nhau, mỗi ngôn ngữ có đặc trưng riêng về cú pháp, hình thái, và ngữ nghĩa. Điều này tạo ra thách thức lớn cho NLP: một mô hình hoạt động tốt trên tiếng Anh chưa chắc hoạt động được trên tiếng Việt (vì tiếng Việt là ngôn ngữ **đơn lập**, không biến đổi hình thái từ, và ranh giới từ phức tạp hơn).

---

## 3. Ứng dụng của NLP

Slide 5-6 liệt kê các ứng dụng phổ biến của NLP trong thực tế:

| Ứng dụng | Mô tả ngắn |
|-----------|-------------|
| Text Categorization | Phân loại văn bản theo chủ đề, ngôn ngữ, tác giả, spam |
| Sentiment Classification | Phân tích cảm xúc (tích cực/tiêu cực) |
| Spelling & Grammar Correction | Sửa lỗi chính tả, ngữ pháp |
| Speech Recognition | Nhận dạng giọng nói → văn bản |
| Machine Translation | Dịch tự động giữa các ngôn ngữ |
| Information Retrieval | Tìm kiếm tài liệu liên quan đến truy vấn |
| Question Answering | Trả lời câu hỏi từ văn bản |
| Summarization | Tóm tắt văn bản |
| Data Extraction | Chuyển văn bản phi cấu trúc → dữ liệu có cấu trúc |

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_1/img-013.jpg]]
*Hình 1: Các ứng dụng NLP trong doanh nghiệp — Smart Assistants, Search, Translation, Analytics, v.v.*

Cách nhìn "đúng" về các ứng dụng này: tất cả đều yêu cầu chuỗi xử lý **chuẩn hóa → rút trích → suy luận → tạo phản hồi**.

---

## 4. NLP Pipeline

> [!NOTE] ELI5
> Pipeline giống như dây chuyền lắp ráp: bạn không thể gắn bánh xe trước khi có khung xe. Tương tự, bạn không thể hiểu nghĩa câu nếu chưa biết đâu là từng "từ" riêng biệt. Pipeline NLP chia nhỏ bài toán phức tạp thành các bước tuần tự, mỗi bước cho ra "bán thành phẩm" để bước sau sử dụng.

[[NLP Pipeline]] là cách tổ chức bài toán NLP thành các bước tiền xử lý/phân tích tuần tự. Lý do: ngôn ngữ là hệ thống nhiều tầng (ký tự → từ → cấu trúc → nghĩa → ngữ cảnh), mỗi tầng tạo ra tín hiệu hữu ích cho tầng sau.

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_1/img-014.jpg]]
*Hình 2: NLP Pipeline 7 bước — từ Sentence Segmentation đến POS Tagging.*

Slide minh họa pipeline bằng ví dụ câu tiếng Anh về Hà Nội:

> *"Hanoi is the capital and second-most populous city of Vietnam. It covers an area of 3,359.82 km2, and consists of 12 urban districts, one district-leveled town and 17 rural districts. It is located within the Red River Delta of Northern Vietnam."*

### 4.1 Step 1: Sentence Segmentation (Tách câu)

[[Sentence Segmentation]] chia đoạn văn thành các câu riêng biệt. Bản chất đây là bài toán **ước lượng ranh giới câu** dựa trên dấu câu và ngữ cảnh (ví dụ dấu "." có thể là kết câu hoặc viết tắt "Dr.", "U.S.").

Ví dụ trên được tách thành 3 câu:
1. "Hanoi is the capital and second-most populous city of Vietnam."
2. "It covers an area of 3,359.82 km2, and consists of 12 urban districts..."
3. "It is located within the Red River Delta of Northern Vietnam."

### 4.2 Step 2: Word Tokenization (Tách từ/token)

[[Tokenization]] tách câu thành các **token** — đơn vị xử lý nhỏ nhất. Với tiếng Anh, tokenization thường đơn giản (cắt theo khoảng trắng + xử lý dấu câu). Nhưng với tiếng Việt, vấn đề phức tạp hơn nhiều vì "từ" có thể gồm nhiều âm tiết: "Hà Nội" là một từ nhưng gồm 2 âm tiết cách nhau bằng khoảng trắng.

Ví dụ: `"Hanoi is the capital..."` → `["Hanoi", "is", "the", "capital", "and", "second-most", "populous", "city", "of", "Vietnam"]`

### 4.3 Step 3: Stemming (Cắt gốc từ)

[[Stemming]] cắt/biến đổi từ theo quy tắc hình thức để lấy **gốc** (stem) — gốc này **không nhất thiết là từ hợp lệ**.

Ví dụ trong slide: `intelligently`, `intelligence`, `intelligent` → tất cả đều có gốc `intelligen` (không phải từ thực).

Slide 10-11 cũng minh họa việc gán **Part of Speech** cho từng token trong quá trình stemming:

| Token | POS |
|-------|-----|
| Hanoi | Proper Noun |
| is | Verb |
| the | Determiner |
| capital | Noun |
| and | Conjunction |
| second-most | Adverb |
| populous | Adjective |
| city | Noun |
| of | Preposition |
| Vietnam | Proper Noun |

### 4.4 Step 4: Lemmatization (Chuẩn hóa về lemma)

[[Lemmatization]] đưa từ về dạng từ điển (**lemma**) — khác stemming ở chỗ lemma luôn là **từ hợp lệ**.

| Phương pháp | Input | Output | Hợp lệ? |
|-------------|-------|--------|----------|
| Stemming | intelligently | intelligen | ❌ Không phải từ |
| Lemmatization | playing, plays | play | ✅ Từ thật |

### 4.5 Step 5: Stop Word Analysis (Lọc từ dừng)

[[Stop Words]] là các từ xuất hiện rất thường xuyên nhưng mang ít thông tin phân biệt trong một số mô hình cổ điển (BoW, TF-IDF): "is", "the", "and", "a", "of", v.v.

Ví dụ: `"Hanoi is the capital and second-most populous city of Vietnam"` → sau khi lọc: `"Hanoi capital second-most populous city Vietnam"`

Slide 14 minh họa code spaCy để lọc stopwords:

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_1/img-016.jpg]]
*Hình 3: Code spaCy lọc stopwords.*

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_1/img-017.png]]
*Hình 4: Kết quả sau khi lọc stopwords.*

> [!WARNING] Lưu ý quan trọng
> Với mô hình hiện đại (Transformer, BERT...), **không nên** lọc stopwords vì chúng mang tín hiệu cú pháp quan trọng. Chỉ nên lọc trong các baseline BoW/TF-IDF.

### 4.6 Step 6: Dependency Parsing (Phân tích quan hệ phụ thuộc) ⭐

> [!NOTE] ELI5
> Hãy tưởng tượng một câu như một gia đình. Mỗi từ là một thành viên, và **dependency parsing** tìm ra ai là "cha mẹ" của ai. Động từ chính là "trưởng gia" (ROOT), còn các từ khác đều phụ thuộc vào ai đó. Ví dụ: "Hanoi" phụ thuộc vào "is" vì "is" là hành động chính, và "Hanoi" là chủ ngữ thực hiện hành động.

[[Dependency Parsing]] tìm **quan hệ phụ thuộc** giữa các từ trong câu, biểu diễn dưới dạng **cây** (dependency tree) với **động từ chính làm gốc** (ROOT).

**Tại sao cần?** Vì cùng một nhóm từ, nếu thay đổi cấu trúc phụ thuộc, nghĩa có thể khác hoàn toàn:
- "A đánh B" (A là chủ ngữ, "đánh" là ROOT, B là tân ngữ)
- "B đánh A" (B là chủ ngữ, cùng từ nhưng nghĩa ngược lại)

**Cách đọc dependency tree:** Mỗi cạnh (edge) mang một **nhãn quan hệ** (dependency relation) giải thích vai trò của từ con đối với từ cha:

| Nhãn | Ý nghĩa | Ví dụ |
|------|---------|-------|
| `nsubj` | Chủ ngữ (nominal subject) | "Hanoi" → `nsubj` → "is" |
| `ROOT` | Gốc — động từ/trạng thái chính | "is" |
| `det` | Mạo từ/từ hạn định | "the" → `det` → "capital" |
| `attr` | Thuộc tính bổ nghĩa | "capital" → `attr` → "is" |
| `amod` | Tính từ bổ nghĩa | "populous" → `amod` → "city" |
| `prep` | Giới từ | "of" → `prep` → "city" |
| `pobj` | Tân ngữ giới từ | "Vietnam" → `pobj` → "of" |
| `conj` | Liên kết ngang hàng | "city" → `conj` → "capital" |
| `cc` | Liên từ | "and" → `cc` → "capital" |
| `advmod` | Trạng từ bổ nghĩa | "second" → `advmod` → "populous" |

Slide minh họa code spaCy cho dependency parsing:

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_1/img-018.jpg]]
*Hình 5: Code spaCy in dependency tree.*

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_1/img-019.jpg]]
*Hình 6: Kết quả dependency parsing — mỗi dòng cho biết từ X phụ thuộc vào từ Y qua quan hệ gì.*

> [!TIP] Cách đọc output
> `Hanoi --nsubj--> is` nghĩa là: "Hanoi" đóng vai trò **chủ ngữ** (nsubj) của "is".
> `capital --attr--> is` nghĩa là: "capital" là **thuộc tính** (attr) của "is" — tức "is" nối chủ ngữ "Hanoi" với thuộc tính "capital".

### 4.7 Step 7: Part-of-Speech (POS) Tagging ⭐

> [!NOTE] ELI5
> POS tagging giống như gán "vai trò" cho mỗi diễn viên trong một vở kịch. Mỗi từ trong câu được gắn nhãn: nó là **danh từ** (noun), **động từ** (verb), **tính từ** (adjective)... Biết vai trò giúp ta hiểu câu đang nói gì, ai làm gì.

[[Part-of-Speech Tagging]] gán nhãn **loại từ** (danh từ, động từ, tính từ, trạng từ...) cho mỗi token trong câu. Đây là bước cầu nối quan trọng từ "text thô" sang "cấu trúc ngữ pháp".

**Tại sao quan trọng?** Vì cùng một từ có thể thuộc nhiều loại khác nhau tùy ngữ cảnh:
- "**book** that flight" → VERB (đặt chỗ)
- "hand me that **book**" → NOUN (cuốn sách)
- "**that** flight" → DETERMINER (mạo từ)
- "I thought **that**..." → COMPLEMENTIZER (liên từ)

POS tagging phải **giải quyết nhập nhằng** này — chọn đúng tag cho đúng ngữ cảnh. Đây chính là bài toán **disambiguation** (xem mục 5).

Slide minh họa code spaCy:

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_1/img-020.png]]
*Hình 7: Code spaCy gán POS tag.*

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_1/img-021.png]]
*Hình 8: Kết quả POS tagging — PROPN (proper noun), AUX (auxiliary), DET, NOUN, ADV, ADJ, ADP (adposition), PUNCT.*

**Bảng POS tags phổ biến** (Universal Dependencies tagset):

| Tag | Tên đầy đủ | Ví dụ |
|-----|------------|-------|
| NOUN | Danh từ | capital, city |
| PROPN | Danh từ riêng | Hanoi, Vietnam |
| VERB | Động từ | run, eat |
| AUX | Trợ động từ | is, was, will |
| ADJ | Tính từ | populous, beautiful |
| ADV | Trạng từ | second, very |
| DET | Mạo từ/từ hạn định | the, a, this |
| ADP | Giới từ | of, in, at |
| CCONJ | Liên từ đẳng lập | and, or, but |
| SCONJ | Liên từ phụ thuộc | because, although |
| PUNCT | Dấu câu | . , ? ! |

---

## 5. Ambiguity — Vấn đề trung tâm của NLP ⭐

> [!NOTE] ELI5
> Ambiguity = "nói một ý, hiểu nhiều nghĩa". Giống như khi bạn nói "Em thích ăn gà" — có thể là bạn thích **ăn thịt gà** (gà = thức ăn), hoặc bạn thích nhìn **con gà đang ăn** (gà = chủ ngữ). Máy tính gặp vấn đề này ở MỌI cấp độ của ngôn ngữ.

[[Ambiguity (NLP)]] là sự thật nền tảng: một biểu thức ngôn ngữ có thể có **nhiều cách diễn giải hợp lệ**. Đây không phải lỗi — đây là **bản chất** của ngôn ngữ tự nhiên, vì ngôn ngữ nén thông tin bằng cách dựa vào tri thức chung và ngữ cảnh.

Slide nêu rõ: ambiguity xảy ra ở **mọi tầng** ngôn ngữ. Cụ thể:

### 5.1 Lexical Ambiguity (Nhập nhằng từ vựng) ⭐

Xảy ra khi **một từ có nhiều nghĩa** khác nhau:

| Từ | Nghĩa 1 | Nghĩa 2 |
|----|---------|---------|
| bank | ngân hàng | bờ sông |
| bat | con dơi | cây gậy bóng chày |
| single | độc thân | duy nhất |
| light | ánh sáng / nhẹ | bật (đèn) |

Ví dụ kinh điển trong slide: **"There was not a single man at the party"**
- Nghĩa 1: Không có **đàn ông độc thân** nào (single = unmarried)
- Nghĩa 2: Không có **một người đàn ông** nào cả (single = not even one)

**Tại sao khó?** Vì chỉ dựa vào từ "single" thì máy không biết chọn nghĩa nào. Cần kết hợp **ngữ cảnh** (context) xung quanh: topic bữa tiệc, các câu trước/sau, v.v.

### 5.2 Syntactic Ambiguity (Nhập nhằng cấu trúc) ⭐

Xảy ra khi **cùng một câu có thể phân tích cú pháp theo nhiều cách**:

Ví dụ: **"I saw the man with a telescope"**
- Cấu trúc 1: Tôi [dùng kính viễn vọng] nhìn thấy người đàn ông → `with a telescope` bổ nghĩa cho "saw"
- Cấu trúc 2: Tôi nhìn thấy người đàn ông [đang cầm kính viễn vọng] → `with a telescope` bổ nghĩa cho "the man"

Hai cấu trúc khác nhau hoàn toàn nhưng dùng cùng một chuỗi từ. Đây là vấn đề mà **Dependency Parsing** (bước 6 ở pipeline) phải giải quyết.

### 5.3 Semantic Ambiguity (Nhập nhằng ngữ nghĩa) ⭐

Xảy ra khi **ý nghĩa logic** của câu không rõ ràng, dù cú pháp đã xác định:

Ví dụ kinh điển trong slide: **"The chicken is ready to eat"**
- Nghĩa 1: Con gà (= động vật) **sẵn sàng ăn** (thức ăn) → gà là **chủ ngữ** của "eat"
- Nghĩa 2: (Món) gà (= thức ăn) **sẵn sàng để bị ăn** → gà là **đối tượng** của "eat"

Cú pháp hai câu giống nhau, nhưng vai trò ngữ nghĩa (semantic role) của "chicken" khác nhau hoàn toàn.

### 5.4 Pragmatic Ambiguity (Nhập nhằng ngữ dụng) ⭐

Xảy ra khi **mục đích giao tiếp** (communicative intent) khác với **nghĩa bề mặt** của câu:

Ví dụ slide: **"Can you tell me the time?"**
- Nghĩa bề mặt: hỏi về **khả năng** → "Bạn có khả năng nói cho tôi biết giờ không?"
- Ý định thực sự: **yêu cầu** → "Mấy giờ rồi?"

Pragmatic ambiguity đòi hỏi hiểu biết về **quy ước giao tiếp** (conversational conventions), **ngữ cảnh hội thoại**, và **quan hệ xã hội** giữa người nói/nghe.

### 5.5 Discourse Ambiguity (Nhập nhằng liên câu) ⭐

Xảy ra khi các câu liên tiếp có thể liên kết với nhau theo nhiều cách:

Ví dụ: "John saw Mary. **He** waved." — "He" có thể chỉ John hoặc ai đó khác. Đây là bài toán **coreference resolution** (giải quyết đồng tham chiếu).

> [!IMPORTANT] Tổng kết
> Ambiguity không phải "lỗi" cần sửa mà là **đặc trưng bản chất** của ngôn ngữ. Hệ NLP hiện đại giải quyết bằng cách kết hợp: (1) mô hình thống kê/học sâu ước lượng nghĩa hợp lý nhất theo dữ liệu, (2) tri thức nền để loại bỏ diễn giải vô lý, (3) ngữ cảnh hội thoại/tài liệu để "khóa" nghĩa.

---

## 6. Lược sử phát triển NLP ⭐

> [!NOTE] ELI5
> Lịch sử NLP giống như lịch sử nấu ăn: ban đầu người ta nấu theo **công thức cứng** (symbolic — viết luật tay), sau đó chuyển sang **nếm thử và điều chỉnh** (statistical — học từ dữ liệu), và cuối cùng dùng **AI tự học nấu** (neural — mạng nơ-ron tự trích xuất đặc trưng).

Slide trình bày lịch sử NLP qua 2 timeline:

### Timeline 1: Từ 1949 đến 1990s (Symbolic → Statistical)

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_1/img-022.jpg]]
*Hình 9: Lịch sử NLP giai đoạn 1949–1990s.*

| Thập kỷ | Sự kiện | Ý nghĩa |
|---------|---------|---------|
| **1949** | Weaver's memorandum | Warren Weaver đề xuất dùng máy tính để dịch ngôn ngữ — đánh dấu khởi đầu NLP |
| **1960s** | Grammar Theories | Noam Chomsky phát triển lý thuyết ngữ pháp hình thức (formal grammar). Tiếp cận **rule-based**: viết luật ngữ pháp bằng tay |
| **1970s** | Conceptual Ontologies | Xây dựng hệ thống biểu diễn tri thức bằng ontology — cố "dạy" máy hiểu khái niệm |
| **1980s** | Symbolic Models | Hệ chuyên gia (expert systems) áp dụng vào NLP, dùng luật logic + tri thức mã hóa tay |
| **1990s** | Statistical Models | **BƯỚC NGOẶT**: chuyển từ viết luật sang **học từ dữ liệu**. HMM, N-gram, Naive Bayes trở thành công cụ chủ lực |

### Timeline 2: Từ 2003 đến 2018+ (Neural → Pretrained)

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_1/img-023.jpg]]
*Hình 10: Lịch sử NLP giai đoạn 2003–2018.*

| Năm | Sự kiện | Ý nghĩa |
|-----|---------|---------|
| **2003** | Neural Language Models | Bengio et al. đề xuất dùng mạng nơ-ron cho language model — thay thế N-gram |
| **2008** | Multi-task Learning | Collobert & Weston: một mạng nơ-ron giải nhiều tác vụ NLP cùng lúc |
| **2013** | Word Embeddings | Word2Vec (Mikolov) — biểu diễn từ bằng **vector dense**, cách mạng hóa biểu diễn ngôn ngữ |
| **2013** | NLP Neural Nets | RNN/LSTM bắt đầu vượt mặt phương pháp thống kê truyền thống |
| **2014** | Seq-to-seq Learning | Sutskever et al.: kiến trúc encoder-decoder cho dịch máy |
| **2015** | Attention | Bahdanau et al.: cơ chế attention cho phép mô hình "tập trung" vào phần quan trọng |
| **2018** | Pretrained Models | **BƯỚC NGOẶT LỚN**: BERT, GPT — mô hình tiền huấn luyện trên dữ liệu khổng lồ, fine-tune cho nhiều tác vụ. Đây là nền tảng của ChatGPT, Gemini, v.v. |

> [!TIP] Quy luật phát triển
> Mỗi bước nhảy lớn trong NLP đều gắn với **3 yếu tố**: (1) dữ liệu lớn hơn (web-scale), (2) compute mạnh hơn (GPU/TPU), (3) thuật toán mới (attention, transformer). Càng về sau, NLP càng "ít luật tay, nhiều dữ liệu hơn".

---

## 7. NLP Challenges: Thách thức cốt lõi ⭐

Slide 21-24 liệt kê các thách thức mà NLP phải đối mặt. Có thể chia thành các nhóm:

### 7.1 Vấn đề "Who did what to whom" (Ai làm gì cho ai)

Đây là câu hỏi nền tảng mà mọi hệ NLP phải trả lời, nhưng để trả lời cần:
- **Nhiều biến ẩn** (hidden variables): nghĩa từ, cấu trúc câu, vai trò ngữ nghĩa
- **Tri thức về thế giới** (world knowledge): "the blue pen ate the ice-cream" sai về mặt logic thế giới thực
- **Tri thức ngữ cảnh** (context knowledge): ai đang nói, nói với ai, trong hoàn cảnh nào
- **Tri thức giao tiếp** (communication knowledge): pragmatics — "Can you tell me the time?" thực ra là yêu cầu

### 7.2 Problem of Scale (Vấn đề quy mô)

Không gian từ vựng / nghĩa / ngữ cảnh là **vô hạn** (lý thuyết), trong khi dữ liệu quan sát luôn **hữu hạn**. Ví dụ: có thể tạo ra vô số câu hợp lệ mà không ai từng nói/viết trước đó.

### 7.3 Problem of Sparsity (Vấn đề dữ liệu thưa) ⭐

> [!NOTE] ELI5
> Tưởng tượng bạn đang xây từ điển bằng cách đọc sách. Dù đọc 1 triệu cuốn, vẫn sẽ có từ mới bạn chưa gặp. Đây là "sparsity" — dữ liệu không đủ để phủ hết mọi trường hợp. Với NLP, vấn đề này nghiêm trọng vì ngôn ngữ có quá nhiều tổ hợp từ mới.

**Sparsity** (thưa dữ liệu) nghĩa là: hầu hết các tổ hợp từ/cụm từ/câu sẽ **không bao giờ xuất hiện** trong tập huấn luyện, dù tập đó rất lớn. Mô hình thống kê phải **ước lượng** xác suất cho những sự kiện chưa từng thấy — đây là động lực cho các kỹ thuật **smoothing** (sẽ học ở Chapter 3).

### 7.4 Long-range Correlations (Phụ thuộc dài hạn) ⭐

> [!NOTE] ELI5
> Đọc câu: "Người đàn ông mà tôi gặp hôm qua ở quán cà phê gần trường, người mà vợ **anh ấy** đã gọi điện cho tôi..." — từ "anh ấy" ở rất xa "người đàn ông" nhưng chúng liên quan đến nhau. Máy tính phải "nhớ" thông tin qua khoảng cách rất dài — đây là thách thức kinh điển.

**Long-range correlations** = các từ cách xa nhau trong câu/đoạn vẫn có quan hệ ngữ nghĩa với nhau. Ví dụ:
- Đại từ "he/she/it" có thể tham chiếu đến thực thể được nhắc ở câu trước rất xa
- Động từ phụ thuộc vào chủ ngữ dù ở giữa có nhiều mệnh đề chèn

Đây là lý do **RNN** gặp khó khăn (vanishing gradient), và là động lực để phát triển **LSTM**, **Attention**, và **Transformer**.

### 7.5 Các thách thức khác (Slide 23-24)

| Thách thức | Giải thích | Ví dụ |
|------------|-----------|-------|
| Multiple parsing | Cùng câu, phân tích cú pháp khác nhau | "I saw the man with a telescope" |
| Word category ambiguity | Từ thuộc nhiều loại POS | "book" = noun/verb |
| Word sense ambiguity | Từ có nhiều nghĩa | "bank" = ngân hàng/bờ sông |
| Compositionality | Nghĩa tổ hợp khác nghĩa từng phần | "The Times of India" ≠ time + India |
| World knowledge | Cần kiến thức thế giới thực | "the blue pen ate the ice-cream" — vô nghĩa |
| Fictitious worlds | Kuala thực/giả lẫn lộn | "people on Mars can fly" |
| Scope ambiguity | Phạm vi suy diễn | "people like ice-cream" = tất cả mọi người? |
| Language evolution | Ngôn ngữ thay đổi | slang, từ mới, nghĩa mới |

---

## 8. Kết luận

Chương 1 đặt nền tảng tư duy:
1. NLP = hiểu + sinh ngôn ngữ, đi xa hơn keyword matching
2. Pipeline 7 bước tạo ra cấu trúc trung gian để mô hình hóa nghĩa
3. **Ambiguity** là trạng thái mặc định — xảy ra ở mọi cấp độ
4. NLP phát triển từ symbolic → statistical → neural → pretrained
5. Thách thức lớn nhất: sparsity, long-range dependencies, world knowledge

Các chương tiếp theo sẽ đi sâu vào: POS Tagging (Ch2), Language Models (Ch3), Sentiment Classification (Ch4).

---

## 📝 Bảng từ điển thuật ngữ

| Thuật ngữ | Tiếng Việt | Giải thích ngắn |
|-----------|-----------|-----------------|
| NLU | Hiểu ngôn ngữ tự nhiên | Từ câu nói → suy ra ý định/nghĩa |
| NLG | Sinh ngôn ngữ tự nhiên | Từ dữ liệu → tạo câu tự nhiên |
| Tokenization | Tách token | Chia câu thành đơn vị xử lý nhỏ nhất |
| Stemming | Cắt gốc từ | Cắt biến thể → gốc (có thể không hợp lệ) |
| Lemmatization | Chuẩn hóa lemma | Đưa về dạng từ điển (luôn hợp lệ) |
| Stop Words | Từ dừng | Từ phổ biến, ít mang thông tin (the, is, a) |
| Dependency Parsing | Phân tích phụ thuộc | Tìm quan hệ cha-con giữa các từ |
| POS Tagging | Gán nhãn từ loại | Gắn danh từ/động từ/tính từ cho mỗi từ |
| Ambiguity | Nhập nhằng | Một biểu thức — nhiều cách hiểu |
| Sparsity | Dữ liệu thưa | Hầu hết tổ hợp từ chưa từng gặp |
| Long-range correlation | Phụ thuộc dài hạn | Từ xa nhau vẫn liên quan ngữ nghĩa |
| Semantic role | Vai trò ngữ nghĩa | Ai là chủ, ai bị tác động, bằng gì |
| Pragmatics | Ngữ dụng học | Nghĩa phụ thuộc mục đích giao tiếp |

---

## 🏋️ Bài tập kèm lời giải

### Bài 1: Phân tích Pipeline

**Đề:** Cho câu: *"Students study NLP at PTIT."* Thực hiện 7 bước pipeline NLP:

> [!TIP]- Lời giải chi tiết
> **Step 1 — Sentence Segmentation:** Chỉ có 1 câu → giữ nguyên.
>
> **Step 2 — Tokenization:** `["Students", "study", "NLP", "at", "PTIT", "."]`
>
> **Step 3 — Stemming:** Students → student, study → studi, NLP → nlp, at → at, PTIT → ptit
>
> **Step 4 — Lemmatization:** Students → student, study → study, NLP → NLP, at → at, PTIT → PTIT
>
> **Step 5 — Stop word removal:** Loại "at" → `["Students", "study", "NLP", "PTIT"]`
>
> **Step 6 — Dependency Parsing:**
> ```
> Students --nsubj--> study
> study --ROOT--> study
> NLP --dobj--> study
> at --prep--> study
> PTIT --pobj--> at
> . --punct--> study
> ```
>
> **Step 7 — POS Tagging:**
>
> | Token | POS |
> |-------|-----|
> | Students | NOUN |
> | study | VERB |
> | NLP | PROPN |
> | at | ADP |
> | PTIT | PROPN |
> | . | PUNCT |

---

### Bài 2: Phân tích Ambiguity

**Đề:** Xác định loại ambiguity trong các câu sau và giải thích 2 cách hiểu:

**(a)** "Flying planes can be dangerous."
**(b)** "I need to get money from the bank."
**(c)** "Could you open the window?"

> [!TIP]- Lời giải chi tiết
> **(a) Syntactic ambiguity:**
> - Cách hiểu 1: "Việc lái máy bay" (flying planes = gerund phrase) có thể nguy hiểm → "Flying" là **động danh từ**, "planes" là tân ngữ
> - Cách hiểu 2: "Những chiếc máy bay đang bay" (flying planes = participial phrase) có thể nguy hiểm → "Flying" là **tính từ** bổ nghĩa cho "planes"
>
> **(b) Lexical ambiguity:**
> - Cách hiểu 1: bank = **ngân hàng** → lấy tiền từ ngân hàng
> - Cách hiểu 2: bank = **bờ sông** → lấy tiền (có thể giấu) ở bờ sông
>
> **(c) Pragmatic ambiguity:**
> - Nghĩa bề mặt: hỏi về **khả năng** → "Bạn có thể mở cửa sổ không?" (hỏi yes/no)
> - Ý định thực sự: **yêu cầu** → "Làm ơn mở cửa sổ giúp tôi" (indirect speech act)

---

### Bài 3: Lịch sử NLP

**Đề:** Nối mỗi mốc thời gian với sự kiện tương ứng:

| Năm | Sự kiện |
|-----|---------|
| 1949 | (?) |
| 1990s | (?) |
| 2013 | (?) |
| 2018 | (?) |

Danh sách sự kiện: `[Word Embeddings (Word2Vec), Statistical Models, Weaver's memorandum, Pretrained Models (BERT/GPT)]`

> [!TIP]- Lời giải
> | Năm | Sự kiện |
> |-----|---------|
> | 1949 | Weaver's memorandum |
> | 1990s | Statistical Models |
> | 2013 | Word Embeddings (Word2Vec) |
> | 2018 | Pretrained Models (BERT/GPT) |

---

### Bài 4: Dependency Parsing thực hành

**Đề:** Vẽ dependency tree cho câu: *"The cat sat on the mat."*

Gợi ý: xác định ROOT trước, sau đó tìm chủ ngữ (nsubj), mạo từ (det), giới từ (prep), tân ngữ giới từ (pobj).

> [!TIP]- Lời giải chi tiết
> ```
> The  --det-->  cat
> cat  --nsubj-->  sat
> sat  --ROOT-->  sat
> on   --prep-->  sat
> the  --det-->  mat
> mat  --pobj-->  on
> .    --punct-->  sat
> ```
>
> **Giải thích:**
> - "sat" là **ROOT** (động từ chính)
> - "cat" là **chủ ngữ** (nsubj) của "sat"
> - "The" (đầu) là **mạo từ** (det) của "cat"
> - "on" là **giới từ** (prep) bổ nghĩa cho "sat" — cho biết "ngồi ở đâu"
> - "mat" là **tân ngữ giới từ** (pobj) của "on"
> - "the" (sau) là **mạo từ** (det) của "mat"
>
> ```mermaid
> graph TD
>     sat["sat (ROOT)"]
>     cat["cat (nsubj)"]
>     the1["The (det)"]
>     on["on (prep)"]
>     mat["mat (pobj)"]
>     the2["the (det)"]
>     punct[". (punct)"]
>     
>     sat --> cat
>     cat --> the1
>     sat --> on
>     on --> mat
>     mat --> the2
>     sat --> punct
> ```

---

### Bài 5: So sánh Stemming và Lemmatization

**Đề:** Cho các từ: `"running"`, `"better"`, `"studies"`, `"geese"`. Áp dụng stemming (Porter) và lemmatization, ghi kết quả vào bảng.

> [!TIP]- Lời giải
> | Từ gốc | Stemming (Porter) | Lemmatization |
> |--------|-------------------|---------------|
> | running | run | run (verb) / running (noun/adj) |
> | better | better | good (adjective) |
> | studies | studi | study |
> | geese | gees | goose |
>
> **Nhận xét:**
> - Stemming cho `studi` và `gees` — **không phải từ hợp lệ**
> - Lemmatization biết `better` → `good` (dạng so sánh), `geese` → `goose` (số nhiều bất quy tắc) — cần **từ điển** và **phân tích hình thái**
> - Đây là lý do lemmatization chính xác hơn nhưng chậm hơn và phụ thuộc ngôn ngữ
