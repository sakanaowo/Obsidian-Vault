---
tags:
  - transformer-architecture
  - llm
  - nlp
  - deep-learning
  - concept
status: permanent
created: 2025-12-10
---

# Transformer Architecture (Kiến trúc Transformer)

## Định nghĩa và Bối cảnh lịch sử

**Transformer Architecture** là một kiến trúc mạng nơ-ron được giới thiệu vào năm 2017 bởi nhóm Google Brain trong bài báo "Attention Is All You Need". Kiến trúc này đã cách mạng hóa lĩnh vực Xử lý Ngôn ngữ Tự nhiên (NLP) và trở thành nền tảng cho sự phát triển của các [[Large Language Models]] (LLMs) hiện đại như GPT-series của OpenAI, BERT của Google, và các mô hình khác.

Trước Transformer, các mô hình xử lý ngôn ngữ truyền thống (như RNN, LSTM) xử lý văn bản một cách tuần tự, điều này hạn chế khả năng hiểu cấu trúc ngôn ngữ trên các khoảng cách dài. Transformer đã giải quyết vấn đề này bằng cách cho phép mô hình xử lý toàn bộ văn bản cùng một lúc và tập trung vào các mối quan hệ giữa các từ không phụ thuộc vào vị trí của chúng.

## Các Thành phần Cốt lõi

Kiến trúc Transformer bao gồm hai thành phần chính: **Encoder** và **Decoder**.

### 1. Encoder (Bộ mã hóa)
*   **Mục đích:** Xử lý đầu vào (ví dụ: một câu) và tạo ra một biểu diễn ngữ cảnh phong phú.
*   **Thành phần:** Gồm nhiều lớp giống hệt nhau, mỗi lớp có hai lớp phụ:
    *   **Multi-Head Self-Attention (Cơ chế Tự chú ý đa đầu):** Đây là trái tim của Transformer. Nó cho phép mô hình "cân" mức độ quan trọng của mỗi từ trong câu đối với tất cả các từ khác. "Đa đầu" có nghĩa là mô hình thực hiện cơ chế chú ý nhiều lần song song với các tập trọng số khác nhau, cho phép nó tập trung vào các loại mối quan hệ khác nhau.
    *   **Feed-Forward Network (Mạng Truyền thẳng):** Một mạng nơ-ron truyền thẳng được áp dụng riêng biệt cho từng vị trí trong chuỗi.
*   **Vector Representations:** Mỗi từ đầu vào được chuyển đổi thành [[Tokenization]] và sau đó là [[Vector Representations]] (embeddings). Các vector này được điều chỉnh bởi vị trí từ (Positional Encoding) để giữ lại thông tin về thứ tự.

### 2. Decoder (Bộ giải mã)
*   **Mục đích:** Tạo ra chuỗi đầu ra (ví dụ: bản dịch, câu trả lời) dựa trên đầu ra của Encoder và các token đã được tạo ra trước đó.
*   **Thành phần:** Tương tự Encoder, nhưng có thêm một lớp phụ:
    *   **Masked Multi-Head Self-Attention (Cơ chế Tự chú ý đa đầu có mặt nạ):** Ngăn không cho Decoder "nhìn" vào các token tương lai trong chuỗi đầu ra, đảm bảo rằng việc dự đoán một token chỉ dựa trên các token đã được tạo ra.
    *   **Multi-Head Attention (Cơ chế Chú ý đa đầu):** Thực hiện cơ chế chú ý giữa đầu ra của Decoder và đầu ra của Encoder, cho phép Decoder tập trung vào các phần liên quan của đầu vào khi tạo ra đầu ra.
    *   **Feed-Forward Network:** Tương tự như trong Encoder.

## Cơ chế Self-Attention (Tự chú ý)

Cơ chế Self-Attention cho phép mô hình gán trọng số khác nhau cho các từ khác nhau trong câu đầu vào khi xử lý từng từ. Ví dụ, trong câu "The animal didn't cross the street because it was too tired", từ "it" có thể ám chỉ "animal" hoặc "street". Self-attention giúp mô hình xác định từ "it" đang ám chỉ từ nào bằng cách tạo ra các điểm số chú ý.

*   **Query (Q), Key (K), Value (V):** Để tính toán chú ý, mỗi token đầu vào được chuyển đổi thành ba vector: Query, Key và Value.
*   **Tính điểm chú ý:** Điểm số chú ý được tính bằng cách lấy tích vô hướng của Query của token hiện tại với Key của tất cả các token khác. Các điểm số này sau đó được chuẩn hóa (ví dụ: bằng softmax) và được sử dụng để lấy tổng trọng số của các vector Value.
*   **Output:** Tổng trọng số này trở thành vector biểu diễn cho token hiện tại, đã được làm giàu ngữ cảnh.

## Lợi ích của Kiến trúc Transformer

*   **Xử lý song song:** Không giống như các mô hình tuần tự, Transformer có thể xử lý tất cả các token đầu vào song song, giúp tăng tốc độ huấn luyện đáng kể.
*   **Xử lý phụ thuộc tầm xa:** Cơ chế Self-Attention cho phép mô hình nắm bắt các mối quan hệ phụ thuộc giữa các từ ở xa nhau trong câu một cách hiệu quả hơn so với các mô hình trước đây.
*   **Tính tổng quát:** Kiến trúc này đã chứng minh tính hiệu quả trên nhiều tác vụ NLP khác nhau, bao gồm dịch máy, tóm tắt, trả lời câu hỏi, và tạo văn bản.
*   **Khả năng mở rộng:** Dễ dàng mở rộng để tạo ra các mô hình rất lớn (LLMs), với hàng tỷ tham số.

Kiến trúc Transformer là một bước đột phá trong AI, cung cấp nền tảng mạnh mẽ cho sự phát triển của các hệ thống AI có khả năng hiểu và tạo ngôn ngữ tự nhiên một cách tinh vi.
