---
type: concept
title: BERT
aliases:
  - Bidirectional Encoder Representations from Transformers
  - Masked Language Model
tags:
  - ai
  - nlp
  - transformers
  - self-supervised-learning
---

**BERT (Bidirectional Encoder Representations from Transformers)** là một phương pháp **pre-training tự giám sát** cho NLP do Devlin et al. (Google, 2018) giới thiệu. Ý tưởng cốt lõi của BERT là **masked language modeling (MLM)**: che (mask) một phần nhỏ token trong câu (thường 15%) và huấn luyện mô hình dự đoán các token bị che dựa trên ngữ cảnh **hai chiều** (bidirectional) — tức nhìn cả token trước và sau. Điều này khác với các mô hình tự hồi quy như GPT chỉ nhìn token trước.

Về mặt kiến trúc, BERT sử dụng **Transformer encoder** (không có decoder). Mô hình nhận một chuỗi token, trong đó một số token được thay bằng token đặc biệt `[MASK]`, và dự đoán token gốc tại các vị trí đó. Ngoài MLM, BERT còn sử dụng một nhiệm vụ phụ gọi là **Next Sentence Prediction (NSP)** để học mối quan hệ giữa các câu (mặc dù các nghiên cứu sau cho thấy NSP không quan trọng bằng MLM).

**Tại sao masking ratio thấp (15%) hoạt động trong NLP mà không trong Vision?**

Ngôn ngữ là tín hiệu **do con người tạo ra**, giàu ngữ nghĩa và nén thông tin. Mỗi từ trong câu mang ý nghĩa đáng kể — che vài từ đã tạo ra bài toán đòi hỏi **hiểu ngữ nghĩa sâu** để dự đoán đúng. Ví dụ, trong câu "The cat sat on the ___", việc dự đoán từ bị che đòi hỏi hiểu về ngữ cảnh, ngữ pháp, và thế giới thực.

Ngược lại, hình ảnh là tín hiệu tự nhiên với **dư thừa không gian cao** — một pixel/patch có thể được suy ra từ các pixel/patch lân cận bằng nội suy đơn giản mà không cần hiểu ngữ nghĩa. Đây là lý do [[Masked Autoencoders (MAE)]] cần masking ratio cao (75%) để tạo bài toán khó trong vision.

BERT đã mở đường cho kỷ nguyên **pre-training tự giám sát quy mô lớn**. Các mô hình như RoBERTa, ALBERT, ELECTRA đều cải tiến trên nền tảng BERT. Trong vision, [[Masked Autoencoders (MAE)]] là nỗ lực "đưa triết lý BERT" (masked prediction) sang hình ảnh, với các điều chỉnh cần thiết cho đặc thù của tín hiệu thị giác.

> [!NOTE] So sánh BERT vs MAE
> | Aspect | BERT (NLP) | MAE (Vision) |
> |--------|-----------|--------------|
> | Masking ratio | 15% | 75% |
> | Mask token in encoder | Có | Không |
> | Reconstruction target | Token (discrete) | Pixel (continuous) |
> | Decoder | Trivial (MLP) | Lightweight Transformer |
