---
tags:
  - tokenization
  - nlp
  - llm
  - data-preprocessing
status: in_progress
created: 2025-12-10
source:
  - Prompt Engineering for Generative AI (James Phoenix, Mike Taylor)
  - RAG roadmap.pdf
---

# Tokenization (Mã hóa Token)

## Định nghĩa

**Tokenization** là quá trình phá vỡ một chuỗi văn bản lớn thành các đơn vị nhỏ hơn, được gọi là **token**. Đây là một bước tiền xử lý quan trọng trong [[Natural Language Processing (NLP)]] và là một bước thiết yếu để chuẩn bị dữ liệu cho các [[Large Language Models]] (LLMs).

Các token có thể đại diện cho các đơn vị ngôn ngữ khác nhau như từ, cụm từ, ký hiệu, hoặc thậm chí là các ký tự con, tùy thuộc vào phương pháp tokenization được sử dụng.

## Tầm quan trọng đối với LLMs

Tokenization là cần thiết vì LLMs hoạt động với các token số học, không phải văn bản thô.

1.  **Đầu vào cho mô hình:** LLMs xử lý văn bản dưới dạng các chuỗi token số. Tokenization chuyển đổi văn bản sang định dạng mà mô hình có thể hiểu và xử lý.
2.  **Quản lý [[Context Window]]:** Mỗi LLM có một giới hạn về số lượng token mà nó có thể xử lý trong một lần. Tokenization giúp ước tính độ dài của văn bản tính bằng token, cho phép tối ưu hóa prompt để tránh vượt quá giới hạn ngữ cảnh, điều này có thể dẫn đến việc cắt bớt output hoặc lỗi.
3.  **Hiểu ngôn ngữ:** Tokenization đóng vai trò quan trọng trong việc giúp mô hình nhận diện và tạo ra các từ hoặc cụm từ, ngay cả khi chúng không phổ biến trong dữ liệu huấn luyện, làm cho mô hình thích ứng và linh hoạt hơn.
4.  **Hiệu quả tài nguyên:** Đếm số lượng token chính xác giúp quản lý tài nguyên hiệu quả, đặc biệt khi sử dụng API của OpenAI hoặc các nhà cung cấp khác, nơi chi phí thường được tính dựa trên số lượng token.

## Các phương pháp Tokenization phổ biến

Có một số phương pháp tokenization, mỗi phương pháp có ưu điểm riêng và phù hợp với các trường hợp sử dụng cụ thể:

1.  **Word-based Tokenization (Token hóa dựa trên từ):** Chia văn bản thành các từ riêng lẻ. Đây là phương pháp đơn giản nhất nhưng có thể gặp vấn đề với các từ không chuẩn, từ ghép hoặc các từ có dấu câu.
2.  **Character-based Tokenization (Token hóa dựa trên ký tự):** Chia văn bản thành các ký tự riêng lẻ. Phương pháp này tránh vấn đề với các từ mới hoặc không chuẩn nhưng dẫn đến chuỗi token rất dài.
3.  **Subword Tokenization (Token hóa dựa trên từ con):** Đây là phương pháp phổ biến nhất trong các LLMs hiện đại, kết hợp ưu điểm của cả hai phương pháp trên. Nó chia các từ thành các đơn vị con (subwords), cho phép xử lý hiệu quả các từ hiếm gặp và các từ mới.
    *   **[[Byte-Pair Encoding (BPE)]]:** Bắt đầu bằng cách coi văn bản như một chuỗi các ký tự riêng lẻ. Sau đó, nó liên tục kết hợp các cặp ký tự hoặc từ con xuất hiện thường xuyên nhất thành một đơn vị mới. Ví dụ: từ "apple" ban đầu có thể là `a, p, p, l, e` và sau đó được kết hợp thành `appl` và `e`.
    *   **WordPiece:** Tương tự như BPE, được sử dụng trong các mô hình như BERT.
    *   **SentencePiece:** Xử lý tất cả các ký tự làm đầu vào, bao gồm cả khoảng trắng, và có thể token hóa các ngôn ngữ không có khoảng trắng rõ ràng.

## Encodings và Tiktoken

*   **Encodings:** Định nghĩa phương pháp chuyển đổi văn bản thành token. Các mô hình khác nhau sử dụng các mã hóa khác nhau.
*   **Tiktoken:** Là một tokenizer BPE nhanh, được thiết kế để sử dụng với các mô hình của OpenAI. Nó cung cấp hiệu suất nhanh hơn so với các tokenizer mã nguồn mở tương đương. Tiktoken hỗ trợ ba mã hóa phổ biến:
    *   `cl100k_base`: Dành cho GPT-4, GPT-3.5-turbo, `text-embedding-ada-002`.
    *   `p50k_base`: Dành cho Codex models, `text-davinci-002`, `text-davinci-003`.
    *   `r50k_base` (hoặc `gpt2`): Dành cho các mô hình GPT-3.

## Ước tính số lượng Token

Việc ước tính số lượng token là rất quan trọng để quản lý chi phí và hiệu suất của các API LLM.

*   Mô hình chat GPT-3.5-turbo và GPT-4 sử dụng token tương tự như các mô hình hoàn thành trước đó. Tuy nhiên, cấu trúc dựa trên tin nhắn làm cho việc đếm token cho các cuộc trò chuyện trở nên khó khăn hơn.
*   [[LangChain]] cung cấp các công cụ hiệu quả để đếm token, giúp bạn kiểm soát tương tác với các mô hình AI tạo sinh.

Tokenization là một bước nền tảng trong việc làm việc với LLMs, giúp biến văn bản con người thành dữ liệu có cấu trúc mà máy tính có thể xử lý, đồng thời quản lý hiệu quả tài nguyên và giới hạn của mô hình.
