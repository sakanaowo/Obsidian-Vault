---
tags:
  - llm
  - generative-ai
  - nlp
  - foundational-models
status: in_progress
created: 2025-12-10
source:
  - Prompt Engineering for Generative AI (James Phoenix, Mike Taylor)
  - RAG roadmap.pdf
---

# Large Language Models (LLMs)

## Định nghĩa và Tổng quan

**Large Language Models (LLMs)** là các mô hình ngôn ngữ lớn, được huấn luyện trên một lượng lớn dữ liệu văn bản và hình ảnh từ internet. Chúng có khả năng xử lý và tạo ra văn bản gần giống con người, vượt trội hơn so với các thế hệ mô hình ngôn ngữ trước đó về khả năng hiểu và phản hồi ngữ cảnh.

LLMs đã được ứng dụng rộng rãi trong nhiều lĩnh vực, từ tạo nội dung, tự động hóa phát triển phần mềm, đến xây dựng các chatbot tương tác theo thời gian thực.

## Các Khái niệm Cốt lõi của LLMs

### 1. Kiến trúc Transformer
Đây là nền tảng của hầu hết các LLMs hiện đại. Kiến trúc [[Transformer Architecture]] sử dụng cơ chế [[Learning DL,ML-deprecated/Transformer/Self-Attention]] để xử lý các từ trong câu. Mỗi từ sẽ "nhìn" vào tất cả các từ khác để hiểu ngữ cảnh tốt hơn. Điều này giúp mô hình nắm bắt các mối quan hệ ngữ pháp và ngữ nghĩa phức tạp trong văn bản, đặc biệt là với các khoảng cách xa.

### 2. Tokenization (Mã hóa Token)
[[Tokenization]] là quá trình chia văn bản thành các đơn vị nhỏ hơn gọi là **token**.
*   **Token** có thể là từ, cụm từ, hoặc thậm chí là các ký tự con.
*   **Token ≠ Word:** Một từ có thể bao gồm nhiều token (ví dụ: "hamburger" có thể là 1 token, nhưng "9.11" có thể là 3 token).
*   **Các phương pháp phổ biến:** Byte-Pair Encoding (BPE), WordPiece, SentencePiece.

### 3. Context Window (Cửa sổ Ngữ cảnh)
*   **Giới hạn:** Mỗi LLM có một giới hạn về số lượng token mà nó có thể xử lý trong một lần. Đây là một ràng buộc quan trọng mà các kỹ thuật như [[Retrieval Augmented Generation (RAG)]] và [[Chunking Text]] được sử dụng để giải quyết.
*   **Ý nghĩa:** Khi một đoạn văn bản quá dài, nó sẽ bị cắt bớt hoặc mô hình không thể "nhớ" toàn bộ thông tin, dẫn đến giảm chất lượng phản hồi.

### 4. Probabilistic Generation (Tạo sinh Dựa trên Xác suất)
*   **Hoạt động:** LLMs dự đoán token tiếp theo dựa trên xác suất, chọn token có khả năng xuất hiện cao nhất.
*   **Không "Biết" sự thật:** Các LLMs không "biết" các sự thật theo cách con người hiểu, mà chúng biết các tương quan thống kê từ dữ liệu huấn luyện. Điều này có nghĩa là chúng có thể tạo ra thông tin không chính xác hoặc bịa đặt, gọi là [[Hallucination]].

### 5. Prompt Engineering
[[Prompt Engineering]] là kỷ luật tập trung vào việc thiết kế các prompt hiệu quả để cải thiện độ tin cậy, hiệu quả và độ chính xác của các mô hình AI. Nó bao gồm các nguyên tắc như [[Give Direction]], [[Specify Format]], [[Provide Examples]] (Few-shot learning), [[Evaluate Quality]] và [[Divide Labor]].

## Các Mô hình LLM Nổi bật

*   **OpenAI GPT Series:** GPT-3.5-turbo, GPT-4, GPT-4V(ision). Nổi bật với khả năng tạo văn bản và suy luận mạnh mẽ.
*   **Google Gemini:** Một đối thủ cạnh tranh mạnh mẽ, tích hợp các khả năng đa phương thức (multimodal).
*   **Meta Llama Series:** Các mô hình mã nguồn mở (open source) như Llama 2, Llama 3, khuyến khích sự phát triển hợp tác trong cộng đồng AI.
*   **Anthropic Claude:** Claude 2, Opus, Haiku. Tập trung vào an toàn và căn chỉnh AI, với cửa sổ ngữ cảnh lớn.
*   **Mistral AI:** Mistral 7B, Mixtral 8x7b. Các mô hình mã nguồn mở, nổi bật về hiệu quả và khả năng.

## Thách thức và Hạn chế

*   **Hallucination:** LLMs có xu hướng tạo ra thông tin không có thật, tự tin nhưng sai lệch.
*   **Biases (Thiên vị):** Dữ liệu huấn luyện khổng lồ có thể chứa các thiên vị xã hội, dẫn đến các phản hồi không mong muốn hoặc không công bằng.
*   **Context Window Limits:** Hạn chế về số lượng token có thể xử lý, ảnh hưởng đến khả năng xử lý tài liệu dài.
*   **Cost & Latency:** Chi phí tính toán và thời gian phản hồi có thể cao, đặc biệt với các mô hình lớn và API.

Việc hiểu rõ các khái niệm này là rất quan trọng để làm việc hiệu quả với các hệ thống AI trong thực tế và để xử lý các thách thức phát sinh.
