---
tags:
  - nlp
  - syntax
  - pos-tagging
status: in_progress
created_date: 2026-01-13
source:
  - assets/Library/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang.pdf
---

# Part-of-Speech (POS) Tagging

**Part-of-Speech (POS) tagging** là bài toán gán nhãn loại từ (danh từ, động từ, tính từ, trạng từ, giới từ, liên từ…) cho từng token trong câu. Mục tiêu sâu của POS tagging là cung cấp một lớp “nhãn cú pháp” giúp các bước phía sau (parsing, NER, rút trích quan hệ) giảm mơ hồ và học nhanh hơn, vì mô hình không phải tự phát hiện lại mọi tín hiệu chức năng từ dữ liệu thô.

POS tagging khó vì **một từ có thể mang nhiều loại từ** tùy ngữ cảnh (“record” là danh từ hay động từ), và vì ranh giới từ/tokenization có thể không ổn định theo ngôn ngữ. Do đó, POS tagging thường được xem như một bài toán dự đoán theo chuỗi (sequence labeling), nơi mỗi quyết định phụ thuộc vào ngữ cảnh hai phía.

> [!NOTE] Suy luận thêm — POS tagging trong thời đại Transformer
> Dù nhiều hệ end-to-end không cần POS tagging như một bước riêng, POS tagging vẫn hữu ích như một “tín hiệu kiểm tra” (diagnostic) và như dữ liệu phụ trợ cho các hệ thống cần giải thích hoặc ràng buộc ngôn ngữ học.

## POS tagging hoạt động như thế nào? (ELI5 → Deep)

> [!NOTE] ELI5
> Hãy tưởng tượng mỗi từ trong câu là một “mảnh LEGO”, POS tag là cái nhãn nói mảnh đó dùng để làm gì: mảnh **chỉ người/vật** (noun), mảnh **chỉ hành động** (verb), mảnh **miêu tả** (adjective)… Máy sẽ nhìn từ đó và các từ xung quanh, rồi chọn nhãn hợp lý nhất cho từng từ, giống như mình đọc câu và đoán “từ này đang đóng vai gì”.

Ở tầng cơ chế, POS tagging là bài toán **sequence labeling**: với chuỗi token $$x_{1:n}$$, ta tìm chuỗi nhãn $$y_{1:n}$$ trong một **tagset** (ví dụ Penn Treebank) sao cho “hợp lý nhất” theo một tiêu chí xác suất/điểm số. Cái khó nằm ở chỗ “hợp lý” không chỉ phụ thuộc vào một token đơn lẻ, mà phụ thuộc vào **ngữ cảnh** và **ràng buộc cú pháp mềm**: một **determiner (DT)** thường mở đầu một **noun phrase (NP)**; một **verb** thường cần một **subject**; và nhiều từ “đa nghĩa loại từ” (record, book, like) chỉ được phân giải khi nhìn cả cụm.

### Pipeline tối thiểu

1) **Tokenization**: tách câu thành token. Đây là bước tưởng đơn giản nhưng có thể làm sai lệch nhãn nếu ranh giới token không phù hợp (đặc biệt với ngôn ngữ không phân tách từ bằng khoảng trắng, hoặc với từ ghép/viết tắt).

2) **Scoring theo chuỗi**: mô hình gán điểm/xác suất cho từng nhãn theo ngữ cảnh.
   - **Generative (HMM)**: giả định cấu trúc Markov trên nhãn. Ta tối đa hóa:
     $$
     \arg\max_{y_{1:n}} \; p(y_{1:n}) \, p(x_{1:n}\mid y_{1:n})
     $$
     Trong đó $p(y_{1:n})$ là **transition** (khả năng nhãn này theo sau nhãn kia) và $p(x_i\mid y_i)$ là **emission** (khả năng token xuất hiện dưới nhãn đó). Suy luận thường dùng **Viterbi** để tìm đường đi tốt nhất.
   - **Discriminative (CRF)**: mô hình hóa trực tiếp $p(y\mid x)$, cho phép đưa vào nhiều đặc trưng (prefix/suffix, chữ hoa, hậu tố “-ly”, ngữ cảnh 2 bên…) mà không cần giả định sinh dữ liệu như HMM. CRF cũng hay dùng Viterbi để decode chuỗi nhãn tối ưu.
   - **Neural/Transformer**: encoder (BiLSTM/Transformer) tạo biểu diễn ngữ cảnh cho từng vị trí; head phân loại dự đoán tag. Thực tế hay ghép thêm **CRF layer** để giữ tính “nhất quán theo chuỗi” (ví dụ tránh những cặp nhãn hiếm/không hợp cú pháp).

3) **Decoding**: chọn nhãn cuối cùng. Nếu mỗi vị trí dự đoán độc lập thì chỉ cần argmax từng token; nếu có ràng buộc chuỗi (HMM/CRF) thì decode theo toàn chuỗi để tránh quyết định cục bộ sai.

### Ví dụ “tại sao phải nhìn ngữ cảnh”

Trong câu “The quick brown fox **jumps** … and **disappears**”, việc gán **VBZ** cho “jumps/disappears” không chỉ vì hình thái “-s” (dấu hiệu ngôi 3 số ít), mà còn vì chúng xuất hiện sau một NP “The … fox” và tạo thành vị ngữ hợp lệ. Ngược lại, cùng một từ “record” có thể là **NN** (“a record”) hoặc **VB** (“to record”)—chỉ ngữ cảnh mới “khoá” được vai trò.
