---
tags:
  - prompt-engineering
  - generative-ai
  - chatgpt
  - text-generation
  - nlp
  - techniques
status: done
created_date: 2025-12-11
---

# Chapter 3: Standard Practices for Text Generation with ChatGPT

Chương này đi sâu vào các kỹ thuật **Prompt Engineering** tiêu chuẩn để tối ưu hóa việc tạo văn bản với các mô hình LLM như ChatGPT (GPT-3.5 và GPT-4). Trọng tâm chuyển từ các nguyên tắc cơ bản sang các phương pháp thực hành cụ thể để giải quyết các vấn đề thực tế như định dạng dữ liệu, xử lý văn bản dài, và nâng cao khả năng suy luận.

## 1. Structured Data Generation (Tạo Dữ liệu Cấu trúc)

Khả năng chuyển đổi ngôn ngữ tự nhiên thành dữ liệu có cấu trúc là một trong những ứng dụng mạnh mẽ nhất của LLM trong quy trình phát triển phần mềm.

### 1.1. Generating Lists (Tạo Danh sách)
Việc tạo danh sách đơn giản thường gặp các vấn đề như: đánh số không mong muốn, văn bản thừa (preamble/postscript), hoặc độ dài không kiểm soát.
*   **Giải pháp:** Sử dụng **Constraints** (Ràng buộc) rõ ràng trong prompt. Ví dụ: "Return only a bulleted list", "Do not include introductory text".

### 1.2. Hierarchical Lists & Data Structures (Danh sách Phân cấp)
Đối với nội dung phức tạp (ví dụ: dàn ý bài viết), cấu trúc phẳng là không đủ.
*   **Kỹ thuật:** Yêu cầu mô hình tạo cấu trúc lồng nhau (nested structure).
*   **Ứng dụng:** Tạo dàn ý chi tiết cho bài viết dài hoặc kế hoạch dự án.

### 1.3. JSON & YAML Generation
Các hệ thống phần mềm cần định dạng máy có thể đọc được (machine-readable) như JSON hoặc YAML, chứ không phải văn bản thô.
*   **JSON:** Phổ biến cho APIs. Cần chỉ rõ schema mong muốn và yêu cầu "Valid JSON only".
    *   *Lưu ý:* Cần xử lý các lỗi cú pháp (syntax errors) hoặc markdown backticks (```json) mà LLM thường thêm vào.
*   **YAML:** Thường ít tốn token hơn JSON và dễ đọc hơn đối với con người trong các cấu hình phức tạp.
*   **Error Handling:** Sử dụng thư viện validation (như Pydantic trong Python) để đảm bảo đầu ra từ LLM tuân thủ đúng định dạng.

## 2. Style & Context Modification (Điều chỉnh Phong cách & Ngữ cảnh)

### 2.1. Explain It Like I'm 5 (ELI5)
Kỹ thuật này yêu cầu LLM đơn giản hóa các khái niệm phức tạp (ví dụ: vật lý lượng tử, hợp đồng pháp lý) thành ngôn ngữ dễ hiểu.
*   **Cơ chế:** Buộc mô hình thay đổi từ vựng và cấu trúc câu để phù hợp với đối tượng mục tiêu giả định (trẻ 5 tuổi).

### 2.2. Universal Translation (Dịch thuật Đa năng)
LLM hoạt động như một bộ dịch thuật vạn năng, không chỉ giữa các ngôn ngữ (Anh -> Việt) mà còn giữa các *phương thức* (Code Python -> Code JavaScript, Văn bản -> Emoji).

### 2.3. Ask for Context (Yêu cầu Ngữ cảnh)
Thay vì để LLM "ảo giác" (hallucinate) hoặc đưa ra câu trả lời chung chung khi thiếu thông tin, hãy hướng dẫn nó **đặt câu hỏi ngược lại** cho người dùng.
*   **Kỹ thuật:** Thêm chỉ dẫn: "If you need more context to answer accurate, please ask me questions."
*   **Lợi ích:** Biến tương tác thành một cuộc đối thoại hai chiều, tăng độ chính xác của kết quả cuối cùng.

### 2.4. Text Style Unbundling (Bóc tách Phong cách Văn bản)
Thay vì chỉ yêu cầu "Viết như Steve Jobs", kỹ thuật này phân tích một đoạn văn mẫu để trích xuất các đặc trưng (tone, vocabulary, sentence structure).
*   **Quy trình:**
    1.  Cung cấp văn bản mẫu.
    2.  Yêu cầu LLM phân tích và liệt kê các đặc điểm phong cách.
    3.  Sử dụng các đặc điểm đó làm prompt để tạo nội dung mới.

## 3. Handling Long Text: Summarization & Chunking (Xử lý Văn bản Dài)

Giới hạn **Context Window** (Cửa sổ ngữ cảnh) là thách thức lớn khi làm việc với văn bản dài (sách, báo cáo).

### 3.1. Summarization Strategies (Chiến lược Tóm tắt)
*   **Basic Summarization:** Tóm tắt trực tiếp (cho văn bản ngắn).
*   **Constraint Summarization:** Tóm tắt với giới hạn từ hoặc câu cụ thể (dù LLM thường không giỏi đếm chính xác).

### 3.2. Text Chunking (Phân mảnh Văn bản)
Để xử lý văn bản vượt quá context window, cần chia nhỏ văn bản thành các "chunks".
*   **Vấn đề của việc chia cắt ngây thơ (Naive Splitting):** Chia cắt giữa chừng câu hoặc từ làm mất ngữ nghĩa.
*   **Các chiến lược Chunking:**
    *   **By Sentence/Paragraph:** Giữ nguyên vẹn ý nghĩa câu/đoạn.
    *   **Sliding Window (Cửa sổ trượt):** Tạo các chunk có phần chồng lấp (overlap) để đảm bảo không mất ngữ cảnh ở biên.
*   **Tokenization:** Sử dụng các bộ tokenizer (như **Tiktoken** của OpenAI) để đếm token chính xác, thay vì đếm ký tự hoặc từ, giúp tối ưu hóa chi phí và giới hạn input.

## 4. Reasoning & Logic Enhancement (Nâng cao Suy luận)

Để giải quyết các tác vụ phức tạp, prompt đơn giản là không đủ. Cần các kỹ thuật kích hoạt khả năng suy luận của mô hình.

### 4.1. Chain-of-Thought (CoT)
Khuyến khích mô hình "suy nghĩ từng bước" (think step-by-step) trước khi đưa ra câu trả lời cuối cùng.
*   **Hiệu quả:** Tăng đáng kể độ chính xác trong các bài toán logic, toán học và suy luận phức tạp.

### 4.2. Least-to-Most Prompting
Phá vỡ một vấn đề lớn thành các vấn đề nhỏ hơn và giải quyết tuần tự.
*   **Ví dụ:** Để viết một hàm Python, trước hết hãy viết kiến trúc, sau đó viết từng hàm con, và cuối cùng là viết test case.

### 4.3. Inner Monologue (Độc thoại Nội tâm)
Cho phép mô hình "nháp" suy nghĩ của mình ra output, nhưng hướng dẫn để ẩn phần suy nghĩ đó đi hoặc tách biệt nó khỏi câu trả lời cuối cùng gửi cho người dùng. Điều này giúp mô hình tự sửa lỗi (self-correction) trong quá trình suy luận.

## 5. Reliability & Evaluation (Độ tin cậy & Đánh giá)

### 5.1. Preventing Hallucinations (Chống Ảo giác)
*   **Reference Text:** Yêu cầu mô hình chỉ trả lời dựa trên văn bản tham chiếu được cung cấp. Nếu không tìm thấy câu trả lời, hãy nói "I don't know".
*   **Citation:** Yêu cầu mô hình trích dẫn nguồn (câu/đoạn) từ văn bản tham chiếu để chứng minh cho câu trả lời.

### 5.2. Self-Evaluation (Tự đánh giá)
Yêu cầu mô hình tự kiểm tra kết quả của chính mình.
*   **Ví dụ:** Sau khi tạo code, yêu cầu mô hình: "Review the code above for bugs and potential edge cases."

### 5.3. Classification & Sentiment Analysis (Phân loại & Phân tích Cảm xúc)
Sử dụng LLM để dán nhãn dữ liệu (ví dụ: Tích cực/Tiêu cực/Trung lập).
*   **Majority Vote:** Chạy prompt nhiều lần (với temperature > 0) và lấy kết quả xuất hiện nhiều nhất để tăng độ tin cậy.

### 5.4. Meta Prompting
Sử dụng LLM để viết prompt cho chính nó hoặc cho các mô hình khác (ví dụ: dùng ChatGPT để viết prompt chi tiết cho Midjourney).

## 6. Advanced Concepts

### 6.1. Role Prompting
Gán một "persona" (nhân vật) cho AI (ví dụ: "Act as a Senior Python Developer"). Điều này giúp định hình tone, từ vựng và góc nhìn giải quyết vấn đề của mô hình.

### 6.2. Thinking Time
Cho mô hình "thời gian suy nghĩ" bằng cách yêu cầu nó giải thích quy trình hoặc liệt kê các giả định trước khi đưa ra kết luận. Điều này tương tự như CoT nhưng tập trung vào việc mở rộng không gian suy luận.

---
**Key Takeaway:** Chương 3 nhấn mạnh việc chuyển từ các câu lệnh đơn giản sang các quy trình (workflows) có cấu trúc. Việc kết hợp **Chunking**, **Chain-of-Thought**, và **Structured Output** là chìa khóa để xây dựng các ứng dụng AI tin cậy và mạnh mẽ.
