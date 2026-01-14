---
tags:
  - nlp
  - ptit
  - source-note
status: in_progress
source:
  - assets/Library/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang.pdf
---

# Chapter 1 — Introduction (NLP)

> [!NOTE] Source
> Tài liệu gốc: `assets/Library/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang.pdf` (Chapter 1, PDF pages 1–24). Nội dung dưới đây là **dịch + diễn giải có phê bình** dựa trên slide; các đoạn được đánh dấu “Suy luận thêm” là phần mình mở rộng từ kiến thức nền.

## 1. NLP là gì (từ First Principles)

[[Natural Language Processing (NLP)]] không chỉ là “xử lý chuỗi ký tự”, mà là nỗ lực biến **tín hiệu ngôn ngữ tự nhiên** (văn bản/tiếng nói) thành một dạng **biểu diễn** để máy có thể suy luận và hành động, rồi (khi cần) biến biểu diễn đó trở lại thành ngôn ngữ người dùng hiểu. Điểm then chốt ở đây là: **ngôn ngữ tự nhiên không mã hóa trực tiếp ý nghĩa**; nó chỉ là “bề mặt” của ý nghĩa, trong khi ý nghĩa phụ thuộc vào tri thức nền, ngữ cảnh, mục đích giao tiếp và các quy ước xã hội. Vì vậy, một hệ NLP tốt phải xử lý cả “mặt chữ” và “hàm ý”.

Trong slide, NLP được trình bày qua hai nhiệm vụ nền tảng:

**Natural Language Understanding (NLU)**: đi từ câu nói/câu viết → suy ra **ý định** và/hoặc **nghĩa** (ai, làm gì, cho ai, khi nào, ở đâu, vì sao…). Bản chất của NLU là một bài toán **suy luận dưới bất định**: cùng một chuỗi từ có thể tương ứng nhiều nghĩa, và ta phải chọn nghĩa “hợp lý nhất” dựa trên bằng chứng.

**Natural Language Generation (NLG)**: đi từ một biểu diễn “có cấu trúc” (ý định, facts, dữ liệu) → tạo ra câu tự nhiên phù hợp với người nghe và tình huống. Bản chất của NLG là tối ưu hóa đồng thời **tính đúng** (faithfulness) và **tính tự nhiên** (fluency) trong điều kiện ràng buộc về phong cách, độ dài, lịch sự, v.v.

> [!NOTE] Suy luận thêm — Vì sao “keyword matching” là không đủ?
> Keyword matching giả định “từ” mang nghĩa ổn định và độc lập. Nhưng trong thực tế, nghĩa của từ phụ thuộc mạnh vào **ngữ cảnh** (word sense), cấu trúc câu (scope), và tri thức nền (world knowledge). Ví dụ: “bác sĩ bắt bệnh nhân” có thể gây hiểu nhầm nếu không xét cấu trúc và vai trò ngữ nghĩa; tương tự trong tiếng Anh, “bank” là ngân hàng hay bờ sông là do ngữ cảnh quyết định. Đây là lý do NLP cần đi xa hơn xử lý chuỗi.

## 2. Ứng dụng của NLP: từ nhu cầu kinh doanh đến hệ thống lõi

NLP thường xuất hiện trong các hệ thống người dùng cuối như: **speech recognition**, **machine translation**, **question answering**, **spelling/grammar correction**, **text categorization**, **information retrieval**, **summarization**, v.v. Cách nhìn “đúng” về các ứng dụng này là: chúng đều yêu cầu một (hoặc nhiều) bước trong chuỗi: **chuẩn hóa dữ liệu ngôn ngữ → rút trích thông tin → suy luận → tạo ra phản hồi**.

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_1/img-013.jpg]]

> [!NOTE] Suy luận thêm — “NLP ứng dụng” thường là bài toán tối ưu trade-off
> Trong thực tế, ta hiếm khi tối ưu “hiểu đúng hoàn toàn”. Thay vào đó, ta tối ưu theo mục tiêu kinh doanh/ sản phẩm: độ chính xác chấp nhận được, tốc độ, chi phí gán nhãn, độ trễ, độ riêng tư, khả năng giải thích. Điều này quyết định nên dùng pipeline heuristic hay mô hình thống kê/ học sâu.

## 3. NLP pipeline: tại sao phải “chia nhỏ” vấn đề?

[[NLP Pipeline]] là cách tổ chức một bài toán NLP thành các bước tiền xử lý/ phân tích tuần tự. Lý do căn bản: ngôn ngữ là một hệ thống phức tạp nhiều tầng (từ ký tự → từ → cấu trúc → nghĩa → ngữ cảnh), và mỗi tầng tạo ra “tín hiệu” hữu ích cho tầng sau. Slide minh họa một pipeline 7 bước (đặt theo ví dụ câu tiếng Anh về Hà Nội):

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_1/img-014.jpg]]

### 3.1 Sentence segmentation (tách câu)

[[Sentence Segmentation]] chia đoạn văn thành các câu, giúp các bước sau làm việc trên đơn vị có biên rõ ràng. Về bản chất, đây là bài toán **ước lượng ranh giới câu** dựa trên dấu câu và các tín hiệu ngữ cảnh (ví dụ “.” có thể là dấu kết câu hoặc viết tắt).

Trong ví dụ, một đoạn mô tả Hà Nội được tách thành 3 câu; việc tách câu đúng giúp tokenization và parsing không “trộn” ngữ nghĩa giữa các mệnh đề không liên quan.

### 3.2 Word tokenization (tách token)

[[Tokenization]] tách câu thành các **token**. Ý nghĩa sâu của tokenization không chỉ là “cắt theo khoảng trắng”, mà là chọn **đơn vị xử lý** phù hợp với mô hình/ ngôn ngữ/ miền dữ liệu. Với tiếng Việt, tokenization đặc biệt quan trọng vì “từ” có thể gồm nhiều âm tiết cách nhau bằng khoảng trắng (Suy luận thêm).

### 3.3 Stemming vs Lemmatization: chuẩn hóa hình thái

[[Stemming]] và [[Lemmatization]] đều nhằm đưa các biến thể bề mặt về một dạng chuẩn để giảm độ thưa (sparsity) và giúp mô hình khái quát. Nhưng cơ chế và hệ quả khác nhau:

**Stemming** cắt/biến đổi theo quy tắc hình thức để lấy “gốc” (stem) — gốc này **không nhất thiết là một từ hợp lệ** (ví dụ “intelligently/intelligence/intelligent → intelligen” trong slide). Đây là một sự đánh đổi: tốc độ nhanh và đơn giản, nhưng dễ làm mất sắc thái nghĩa.

**Lemmatization** dùng thông tin từ điển + phân tích hình thái để đưa về **lemma** là một từ hợp lệ (ví dụ “playing/plays → play”). Lemmatization vì thế thường chính xác hơn về mặt ngôn ngữ học, nhưng đòi hỏi tài nguyên (lexicon, tagger) và phụ thuộc ngôn ngữ.

> [!NOTE] Suy luận thêm — Khi nào stemming “nguy hiểm”?
> Trong các tác vụ đòi hỏi phân biệt sắc thái (ví dụ phân tích cảm xúc/ pháp lý), stemming có thể gộp sai các từ gần nhau về mặt hình thức nhưng khác nghĩa/ chức năng, gây nhiễu cho mô hình. Khi đó, lemmatization hoặc subword tokenization thường an toàn hơn.

### 3.4 Stop word analysis (lọc từ dừng)

[[Stop Words]] là các từ xuất hiện rất thường xuyên (ví dụ “is”, “the”, “and”), thường mang ít thông tin phân biệt trong một số mô hình cổ điển (BoW/TF-IDF). Slide minh họa việc loại stopwords để tập trung vào từ mang nội dung (“Hanoi … capital … city … Vietnam”).

> [!NOTE] Suy luận thêm — Lọc stopwords không phải lúc nào cũng đúng
> Với các mô hình ngữ cảnh (Transformer), stopwords có thể mang tín hiệu cú pháp và quan hệ phụ thuộc; việc loại bỏ thô có thể làm mất cấu trúc câu. Thực hành hiện đại thường chỉ dùng stopwords trong các baseline bag-of-words hoặc truy hồi truyền thống.

### 3.5 Dependency parsing & POS tagging: gắn cấu trúc cho câu

[[Dependency Parsing]] nhằm tìm quan hệ phụ thuộc giữa các từ (ai phụ thuộc ai), thường biểu diễn dưới dạng cây với **động từ chính** làm gốc. [[Part-of-Speech Tagging]] gắn nhãn loại từ (NOUN/VERB/ADJ…) để phục vụ phân tích cú pháp, thực thể, và nhiều tác vụ hạ nguồn.

Slide minh họa code spaCy cho POS và parsing (minh họa công cụ, không phải yêu cầu bắt buộc của pipeline):

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_1/img-020.png]]

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_1/img-021.png]]

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_1/img-018.jpg]]

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_1/img-019.jpg]]

> [!NOTE] Suy luận thêm — Vì sao cấu trúc cú pháp quan trọng?
> Nhiều câu khác nhau bề mặt nhưng cùng “ai làm gì cho ai” có thể chuẩn hóa về cùng cấu trúc phụ thuộc, giúp hệ thống tổng quát. Ngược lại, hai câu có cùng từ khóa nhưng cấu trúc khác có thể mang nghĩa khác (ví dụ “A đánh B” vs “B đánh A”). Đây là lý do parsing/POS tagging là cầu nối từ “text” sang “meaning”.

## 4. Ambiguity: vấn đề trung tâm của NLP

[[Ambiguity (NLP)]] là sự thật nền tảng: một biểu thức có thể có nhiều cách diễn giải. Slide nêu rõ “ambiguity xảy ra ở mọi tầng” và đưa ra các lớp phân tích (lexical/syntactic/semantic/discourse/pragmatic). Điều quan trọng là: ambiguity không phải “lỗi”, mà là hệ quả tất yếu của việc ngôn ngữ nén thông tin bằng cách dựa vào tri thức chung và ngữ cảnh.

Ví dụ kinh điển trong slide: “The chicken is ready to eat” có thể hiểu là con gà sẵn sàng ăn (tác nhân) hoặc món gà sẵn sàng để bị ăn (bị thể). Tương tự, “There was not a single man at the party” có thể hiểu là không có đàn ông nào, hoặc không có người đàn ông “độc thân” nào (sai biệt về nghĩa của “single”).

> [!NOTE] Suy luận thêm — Ambiguity là bài toán xác suất + tri thức
> Nếu chỉ dựa trên chuỗi từ, ta không đủ thông tin để chọn nghĩa. Do đó, hệ NLP hiện đại thường kết hợp: (1) mô hình thống kê/ học sâu để ước lượng nghĩa hợp lý theo phân phối dữ liệu; (2) tri thức nền (knowledge base, rules) để loại bỏ diễn giải vô lý; (3) ngữ cảnh hội thoại/ tài liệu để “khóa” nghĩa qua đồng tham chiếu và chủ đề.

## 5. Lược sử phát triển: từ symbolic đến statistical và neural

Hai sơ đồ lịch sử trong slide cho thấy NLP dịch chuyển theo các “làn sóng” phương pháp: từ các tiếp cận **biểu tượng** (grammar/ontology) sang **thống kê**, rồi sang **mạng nơ-ron** và **mô hình tiền huấn luyện**.

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_1/img-022.jpg]]

![[assets/attachments/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang/chapter_1/img-023.jpg]]

> [!NOTE] Suy luận thêm — “Lịch sử” cũng là câu chuyện về dữ liệu và compute
> Các bước nhảy (statistical → neural → pretrained) gắn chặt với ba yếu tố: dữ liệu lớn (web-scale), compute (GPU/TPU), và thuật toán tối ưu. Đây là lý do kiến trúc hiện đại thường “ít luật hơn, nhiều dữ liệu hơn”, nhưng đổi lại là thách thức về kiểm soát hành vi và giải thích.

## 6. NLP challenges: câu hỏi “who did what to whom” chỉ là khởi đầu

[[NLP Challenges]] trong slide được mô tả như một tập các rào cản cấu trúc: hệ NLP phải trả lời “ai làm gì cho ai” trong điều kiện có nhiều biến ẩn, cần tri thức thế giới, tri thức ngữ cảnh và tri thức giao tiếp (pragmatics). Một ví dụ nhỏ: câu hỏi “Can you tell me the time?” bề mặt là hỏi về khả năng, nhưng mục đích giao tiếp thường là yêu cầu cho biết giờ — tức nghĩa nằm ở **hàm ý** chứ không nằm ở cấu trúc cú pháp.

Slide cũng nêu các vấn đề “scale” và “sparsity”: không gian từ/ nghĩa/ ngữ cảnh là rất lớn (thậm chí coi như vô hạn), trong khi dữ liệu quan sát luôn hữu hạn; vì vậy nhiều từ/khái niệm sẽ “chưa từng gặp”. Thêm vào đó, ngôn ngữ có **phụ thuộc dài hạn** (long-range correlations), và nhiều lớp tri thức tương tác làm độ phức tạp tăng theo hàm mũ khi ta cố suy luận tường minh.

> [!NOTE] Suy luận thêm — Thách thức cốt lõi là “biểu diễn nghĩa”
> Nhiều vấn đề trong danh sách (parsing khác nhau, word sense, compositionality, scope, thế giới giả tưởng, ngôn ngữ thay đổi) đều quy về: ta chọn **biểu diễn** nào để (1) đủ giàu để mang nghĩa; (2) đủ gọn để học được từ dữ liệu; (3) đủ ổn định để suy luận và tổng quát. Đây là điểm giao giữa ngôn ngữ học, học máy và khoa học nhận thức.

## 7. Kết luận tạm thời (định hướng học)

Chương 1 đặt nền tảng tư duy: NLP là bài toán đa tầng, mà **ambiguity** là trạng thái mặc định; vì vậy pipeline và các công cụ phân tích (tokenization, parsing, POS, v.v.) không phải “thủ tục”, mà là cách tạo ra cấu trúc trung gian để mô hình hóa nghĩa. Các chương tiếp theo sẽ đi sâu vào từng tầng và cách mô hình/thuật toán giải quyết trade-off giữa tri thức, dữ liệu và tính toán.
