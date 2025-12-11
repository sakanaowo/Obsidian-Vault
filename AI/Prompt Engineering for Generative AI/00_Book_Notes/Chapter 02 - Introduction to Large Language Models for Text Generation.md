---
tags:
  - large-language-models
  - generative-ai
  - nlp
  - transformer-architecture
  - model-comparison
status: processed
created: 2025-12-10
source: Prompt Engineering for Generative AI (James Phoenix, Mike Taylor)
---

# Chương 2: Giới thiệu về Mô hình Ngôn ngữ Lớn cho Tạo sinh Văn bản

## Tổng quan về Mô hình Tạo sinh Văn bản

Các mô hình tạo sinh văn bản sử dụng các thuật toán tiên tiến để hiểu ý nghĩa của văn bản và tạo ra đầu ra thường không thể phân biệt được với văn bản do con người viết.

*   **Tokens:** Đơn vị ngôn ngữ cơ bản trong NLP và LLMs. Một token có thể là một từ, một câu, hoặc một phần của từ (subword). Ví dụ, 100 token tương đương khoảng 75 từ.
*   **Tokenization:** Quá trình chia nhỏ văn bản thành các token. Các phương pháp phổ biến bao gồm **Byte-Pair Encoding (BPE)**, WordPiece, và SentencePiece. BPE hiệu quả trong việc xử lý từ vựng đa dạng bằng cách kết hợp các ký tự thường xuyên xuất hiện cùng nhau.

## Biểu diễn Vector: Bản chất Số học của Ngôn ngữ

Trong NLP, từ ngữ được chuyển đổi thành các con số gọi là **vector** (hay **word embeddings**). Đây là các mảng số đa chiều nắm bắt các mối quan hệ ngữ nghĩa và cú pháp.

$$ w \rightarrow \mathbf{v} = [v_1, v_2, ..., v_n] $$

Các từ có ý nghĩa tương tự sẽ nằm gần nhau trong không gian embedding (ví dụ: *virtue* và *moral*). Sự gần gũi về không gian này giúp mô hình hiểu được ngữ cảnh và các mối quan hệ phức tạp của ngôn ngữ.

## Kiến trúc Transformer: Điều phối các Mối quan hệ Ngữ cảnh

Kiến trúc **Transformer** là nền tảng của các LLM hiện đại như BERT và GPT.

*   **Cơ chế hoạt động:** Nó chuyển đổi các vector từ đầu vào và sử dụng các phép toán để hiểu mối quan hệ giữa các từ (cú pháp và ngữ nghĩa).
*   **Self-Attention:** Cho phép mỗi từ trong câu "nhìn" vào tất cả các từ khác để hiểu ngữ cảnh tốt hơn. Điều này giúp mô hình xác định chính xác vai trò và ý nghĩa của từng từ, tạo ra các biểu diễn ngữ cảnh phong phú.

$$ \mathbf{v}'_i = \text{Transformer}(\mathbf{v}_1, \mathbf{v}_2, ..., \mathbf{v}_m) $$

## Tạo sinh Văn bản Dựa trên Xác suất (Probabilistic Text Generation)

Sau khi hiểu ngữ cảnh, mô hình tạo ra văn bản mới dựa trên xác suất. Nó tính toán khả năng xuất hiện của từ tiếp theo dựa trên chuỗi từ hiện tại và chọn từ có khả năng cao nhất.

$$ w_{next} = \text{argmax } P(w | w_1, w_2, ..., w_m) $$

Quá trình này lặp lại để tạo ra một chuỗi văn bản mạch lạc.

## Lịch sử và Sự phát triển của Kiến trúc Transformer

*   **Attention Is All You Need:** Bài báo đột phá của Google Brain giới thiệu kiến trúc Transformer và khái niệm **attention**. Cơ chế này cho phép mô hình liên kết trực tiếp các từ xa nhau mà không cần xử lý tuần tự, cải thiện đáng kể khả năng hiểu và hiệu quả.
*   **Chi phí và Tài nguyên:** Huấn luyện LLM đòi hỏi tài nguyên tính toán khổng lồ (GPU) và lượng dữ liệu lớn. Ví dụ, GPT-4 có khoảng 1.7 nghìn tỷ tham số.
*   **Phần cứng:** Sự bùng nổ của LLM thúc đẩy nhu cầu về GPU hiệu suất cao (như NVIDIA H100) và các phần cứng chuyên dụng như TPU của Google.

## Các Mô hình Nổi bật

### OpenAI GPT Series
*   **GPT-1 & GPT-2:** Các bước đầu tiên chứng minh tiềm năng của Transformer. GPT-2 mạnh mẽ đến mức OpenAI ban đầu ngần ngại phát hành.
*   **GPT-3:** Bước nhảy vọt về khả năng tạo văn bản giống con người.
*   **GPT-3.5-turbo & ChatGPT:** Phiên bản tối ưu hóa, được tinh chỉnh (fine-tuned) trên ngữ cảnh hội thoại bằng Reinforcement Learning from Human Feedback (RLHF), giúp mô hình hữu ích và an toàn hơn.
*   **[[GPT-4]]:** Mô hình tiên tiến với khả năng hiểu các truy vấn phức tạp và tạo văn bản mạch lạc (ví dụ: đạt điểm cao trong kỳ thi luật). Sử dụng phương pháp **mixture-of-experts**.
*   **GPT-4o:** Mô hình đa phương thức (multimodal) xử lý văn bản, âm thanh và hình ảnh trong thời gian thực.

### Google Gemini
*   Ban đầu là Bard (dựa trên LaMDA/PaLM), sau đó được nâng cấp và đổi tên thành **Gemini**.
*   **Gemini 1.5:** Có khả năng xử lý ngữ cảnh cực lớn (lên đến 1 triệu token) và đạt hiệu suất tương đương GPT-4.

### Meta Llama và Mã nguồn mở
*   **[[Meta Llama]] (Llama 2, Llama 3):** Chiến lược mã nguồn mở của Meta nhằm thúc đẩy hệ sinh thái phát triển AI hợp tác.
*   **Lợi ích:** Minh bạch, đổi mới nhanh chóng, cho phép các doanh nghiệp tự vận hành và tinh chỉnh mô hình.
*   **Rủi ro:** Nguy cơ sử dụng sai mục đích do thiếu kiểm soát tập trung.

### Mistral AI
*   **[[Mistral AI]] (Mistral 7B, Mixtral 8x7b):** Các mô hình mã nguồn mở hiệu quả cao từ startup Pháp. Mixtral sử dụng kiến trúc mixture-of-experts tương tự GPT-4.

### Anthropic Claude
*   **[[Anthropic Claude]] (Claude 2, Opus, Haiku):** Tập trung vào an toàn AI với phương pháp **Constitutional AI**.
*   **Đặc điểm:** Cửa sổ ngữ cảnh lớn (100k+ tokens), khả năng xử lý tài liệu dài xuất sắc.

## Kỹ thuật Tối ưu hóa: Quantization và LoRA

Để chạy các mô hình lớn trên phần cứng hạn chế (consumer-grade), các kỹ thuật sau được sử dụng:
*   **[[Quantization]]:** Giảm độ chính xác số học của các tham số mô hình (ví dụ: từ 32-bit xuống 4-bit) để giảm kích thước mà không làm giảm đáng kể hiệu suất.
*   **[[Low-Rank Adaptation (LoRA)]]:** Tối ưu hóa kiến trúc mạng để tinh chỉnh (fine-tuning) hiệu quả hơn.

## So sánh Mô hình

Mặc dù OpenAI đang dẫn đầu với GPT-4, các mô hình mã nguồn mở như Llama và Mistral đang nhanh chóng bắt kịp. Việc lựa chọn mô hình phụ thuộc vào:
*   **Khả năng:** GPT-4 tốt nhất cho các tác vụ phức tạp và tuân thủ hướng dẫn.
*   **Chi phí & Tốc độ:** Các mô hình nhỏ hơn (Haiku, Mistral) nhanh hơn và rẻ hơn.
*   **Quyền riêng tư:** Mô hình mã nguồn mở cho phép chạy cục bộ, bảo vệ dữ liệu nhạy cảm.

---
**Liên kết tham khảo:**
- [[Large Language Models]]
- [[Transformer Architecture]]
- [[Probabilistic Generation]]
- [[Vector Representations]]
- [[Quantization]]
- [[Low-Rank Adaptation (LoRA)]]
- [[GPT-4]]
- [[Meta Llama]]
- [[Anthropic Claude]]
- [[Mistral AI]]
- [[Google Gemini]]
- [[Multimodal Models]]
