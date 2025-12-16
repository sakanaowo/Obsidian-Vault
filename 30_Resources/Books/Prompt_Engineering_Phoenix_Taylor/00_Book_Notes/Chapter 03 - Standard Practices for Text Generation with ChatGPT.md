---
tags:
  - prompt-engineering
  - generative-ai
  - chatgpt
  - text-generation
  - nlp
  - techniques
  - chapter-notes
status: in_progress
created_date: 2025-12-14
---

# Chapter 3: Standard Practices for Text Generation with ChatGPT (Phiên bản Refactor)

Chương này trình bày các kỹ thuật **Prompt Engineering** tiêu chuẩn được sử dụng để tối ưu hóa đầu ra từ các mô hình ngôn ngữ lớn ([[Large Language Models|LLMs]]), đặc biệt là ChatGPT. Trọng tâm là chuyển từ các nguyên tắc cơ bản sang các phương pháp thực hành cụ thể nhằm giải quyết các thách thức thực tế trong việc tạo văn bản, bao gồm định dạng dữ liệu, xử lý văn bản dài và nâng cao khả năng suy luận.

## 1. Tạo Dữ liệu có cấu trúc từ LLMs (Generating Structured Data)

Khả năng chuyển đổi ngôn ngữ tự nhiên thành các định dạng dữ liệu có cấu trúc là một trong những ứng dụng mạnh mẽ nhất của LLMs trong quy trình phần mềm.

### 1.1. Tạo Danh sách (Generating Lists)

Việc tự động tạo danh sách là một tác vụ cơ bản nhưng thường gặp phải các vấn đề như định dạng không nhất quán, sự xuất hiện của văn bản dẫn nhập hoặc kết luận không mong muốn. Để đảm bảo đầu ra danh sách có cấu trúc và sạch sẽ, cần áp dụng các **ràng buộc (constraints)** rõ ràng trong prompt. Ví dụ, chỉ dẫn như "Return only a bulleted list of 5 items, with no introductory or concluding remarks" và cung cấp một ví dụ mẫu (few-shot example) có thể cải thiện đáng kể độ tin cậy của đầu ra.

### 1.2. Tạo Danh sách Phân cấp (Hierarchical List Generation)

Đối với các cấu trúc phức tạp hơn như dàn ý bài viết hoặc các kế hoạch đa bước, danh sách phẳng là không đủ. Kỹ thuật này yêu cầu LLM tạo ra dữ liệu lồng nhau (nested data structures). Các từ khóa như "Hierarchical" và "Incredibly detailed" trong prompt, kết hợp với một ví dụ về cấu trúc mong muốn, sẽ hướng dẫn mô hình tạo ra đầu ra phân cấp.

Việc phân tích cú pháp (parsing) các đầu ra phân cấp này có thể được thực hiện bằng Regular Expressions (Regex). Tuy nhiên, phương pháp này dễ gặp lỗi nếu định dạng đầu ra của LLM thay đổi dù chỉ một chút. Điều này dẫn đến sự cần thiết của các định dạng dữ liệu tự mô tả hơn.

### 1.3. Dữ liệu có cấu trúc: JSON và YAML

Để tích hợp liền mạch đầu ra của LLM vào các hệ thống phần mềm, việc yêu cầu định dạng máy đọc được như JSON hoặc YAML là rất quan trọng.

*   **JSON (JavaScript Object Notation):** Là định dạng dữ liệu phổ biến cho các API. Khi yêu cầu JSON, prompt cần chỉ rõ "Return only valid JSON" và cung cấp một lược đồ (schema) mẫu. Một thách thức phổ biến là LLM có thể bọc JSON trong markdown backticks (```json), yêu cầu xử lý hậu kỳ để loại bỏ. Các thư viện validation như Pydantic trong Python rất hữu ích để xác thực đầu ra.
*   **YAML (YAML Ain't Markup Language):** Cung cấp một định dạng gọn gàng hơn JSON, dễ đọc hơn đối với con người và thường tốn ít token hơn. YAML đặc biệt hữu ích cho các file cấu hình phức tạp. Một ứng dụng thú vị là sử dụng LLM như một công cụ suy luận để lọc dữ liệu dựa trên schema YAML được cung cấp.

   ![[assets/attachments/Prompt Engineering for Generative AI Future-Proof Inputs for Reliable Al Outputs (James Phoenix, Mike Taylor) (Z-Library)/fig_3-1_Using_an_LLM_to_determine_the_control_flow_of_an_a.png]]
   *Figure 3-1: Sơ đồ luồng điều khiển minh họa cách LLM có thể hoạt động như một bộ định tuyến thông minh để xác định tính hợp lệ của truy vấn người dùng so với schema YAML, trả về kết quả đã lọc hoặc thông báo "No items". Điều này giảm sự phụ thuộc vào code cứng và cho phép LLM đảm nhận vai trò trong logic ứng dụng.*

## 2. Tạo Định dạng Đa dạng (Diverse Format Generation)

LLM không chỉ giới hạn trong việc tạo văn bản thuần túy mà còn có thể tạo ra các định dạng khác nhau:

*   **Mermaid Diagrams:** LLM có thể tạo ra cú pháp Mermaid, cho phép người dùng vẽ biểu đồ luồng, sơ đồ hoặc các hình ảnh trực quan khác.
    ![[assets/attachments/Prompt Engineering for Generative AI Future-Proof Inputs for Reliable Al Outputs (James Phoenix, Mike Taylor) (Z-Library)/fig_3-2_A_streamlined_flow_diagram_created_using_mermaid_s.png]]
    *Figure 3-2: Một sơ đồ luồng được tạo ra bằng cú pháp Mermaid thông qua LLM. Biểu đồ này minh họa một quy trình đặt hàng thực phẩm đơn giản, từ "Chọn món ăn" đến "Thanh toán bữa ăn", với các bước trung gian như "Thêm vào giỏ hàng" và "Xác nhận giỏ hàng".*

*   **Mock CSV Data:** Tạo dữ liệu CSV mẫu để kiểm thử hoặc minh họa.

## 3. Điều chỉnh Phong cách và Ngữ cảnh (Style & Context Modification)

Khả năng thích ứng với các phong cách và ngữ cảnh khác nhau là một sức mạnh cốt lõi của LLMs.

### 3.1. Giải thích như cho Trẻ 5 tuổi (Explain It Like I'm 5 - ELI5)

Kỹ thuật này yêu cầu LLM đơn giản hóa các khái niệm phức tạp (ví dụ: vật lý lượng tử, các điều khoản pháp lý) thành ngôn ngữ dễ hiểu đối với một đứa trẻ 5 tuổi. Điều này buộc mô hình phải thay đổi từ vựng, cấu trúc câu và mức độ chi tiết, làm cho thông tin dễ tiếp cận hơn với nhiều đối tượng.

### 3.2. Dịch thuật Đa năng (Universal Translation)

LLM không chỉ là bộ dịch thuật giữa các ngôn ngữ tự nhiên (ví dụ: tiếng Anh sang tiếng Việt) mà còn hoạt động như một "bộ dịch thuật vạn năng" giữa các hình thức thông tin khác nhau (ví dụ: code Python sang code JavaScript, văn bản sang emoji).

### 3.3. Yêu cầu Ngữ cảnh (Ask for Context)

Để tránh `[[Hallucination]]` hoặc phản hồi chung chung khi thiếu thông tin, một chiến thuật hiệu quả là hướng dẫn LLM **đặt câu hỏi ngược lại** cho người dùng để có thêm ngữ cảnh khi cần.

   ![[assets/attachments/Prompt Engineering for Generative AI Future-Proof Inputs for Reliable Al Outputs (James Phoenix, Mike Taylor) (Z-Library)/fig_3-3_The_decision_process_of_an_LLM_while_asking_for_co.png]]
   *Figure 3-3: Sơ đồ này minh họa quy trình ra quyết định của LLM khi được phép yêu cầu thêm ngữ cảnh. Nếu ngữ cảnh ban đầu không đủ, LLM sẽ chủ động hỏi người dùng để có thêm thông tin, thay vì cố gắng đưa ra câu trả lời không chắc chắn. Điều này cho phép mô hình tạo ra một phản hồi chính xác và phù hợp hơn sau khi có đủ thông tin.*

### 3.4. Bóc tách Phong cách Văn bản (Text Style Unbundling)

Thay vì chỉ yêu cầu mô hình "viết như Steve Jobs", kỹ thuật này yêu cầu LLM phân tích một văn bản mẫu để trích xuất các đặc trưng phong cách cốt lõi (như giọng văn, từ vựng, cấu trúc câu, độ dài). Sau đó, các đặc trưng này có thể được sử dụng làm prompt để tạo nội dung mới, đảm bảo tính nhất quán về phong cách và giọng điệu.

## 4. Tóm tắt (Summarization)

Tóm tắt là một ứng dụng phổ biến khác, giúp chắt lọc thông tin quan trọng từ văn bản dài.

### 4.1. Giới hạn Context Window (Context Window Limitations)

Một thách thức lớn trong việc xử lý văn bản dài là giới hạn của [[Context Window]]. Khi một tài liệu quá lớn so với giới hạn token của LLM, cần áp dụng các chiến lược đặc biệt.

### 4.2. Quy trình Tóm tắt (Summarization Pipeline)

Để tóm tắt các tài liệu vượt quá giới hạn Context Window, một quy trình phổ biến là: chia nhỏ tài liệu thành các phần (chunking), tóm tắt từng phần, sau đó gộp và tóm tắt lại các bản tóm tắt đó (Map-Reduce).

   ![[assets/attachments/Prompt Engineering for Generative AI Future-Proof Inputs for Reliable Al Outputs (James Phoenix, Mike Taylor) (Z-Library)/fig_3-4_A_summarization_pipeline_that_uses_text_splitting_.png]]
   *Figure 3-4: Một pipeline tóm tắt sử dụng kỹ thuật chia nhỏ văn bản (text splitting) và nhiều bước tóm tắt. Tài liệu lớn được chia thành các chunk nhỏ hơn, mỗi chunk được tóm tắt riêng, sau đó các bản tóm tắt này được tổng hợp lại để tạo ra một bản tóm tắt cuối cùng của toàn bộ tài liệu.*

## 5. Phân mảnh Văn bản (Text Chunking)

[[Text Chunking]] là một kỹ thuật thiết yếu để chia nhỏ văn bản dài thành các đơn vị nhỏ hơn, dễ quản lý hơn, nhằm phù hợp với [[Context Window]] của LLM và tối ưu hóa quá trình xử lý.

### 5.1. Lợi ích của Chunking

*   **Phù hợp với giới hạn Context Window:** Đảm bảo toàn bộ văn bản có thể được xử lý mà không bị cắt bớt.
*   **Giảm chi phí và độ trễ:** Xử lý ít token hơn giúp tiết kiệm chi phí API và tăng tốc độ phản hồi.
*   **Cải thiện hiệu suất:** Giảm tải xử lý cho LLM, cho phép phản hồi nhanh hơn.

   ![[assets/attachments/Prompt Engineering for Generative AI Future-Proof Inputs for Reliable Al Outputs (James Phoenix, Mike Taylor) (Z-Library)/fig_3-5_Topic_extraction_with_an_LLM_after_chunking_text_B.png]]
   *Figure 3-5: Minh họa cách trích xuất chủ đề từ một văn bản lớn sau khi được chia thành các chunk nhỏ hơn. Mỗi chunk được xử lý độc lập để trích xuất chủ đề, sau đó các chủ đề này được tổng hợp lại để có cái nhìn tổng thể về nội dung.*

### 5.2. Các chiến lược Chunking

*   **By Sentence/Paragraph:** Tách văn bản dựa trên ranh giới câu hoặc đoạn văn, giúp bảo toàn ngữ cảnh. Các thư viện NLP như SpaCy có thể được sử dụng cho việc này.
*   **By Token:** Sử dụng các bộ mã hóa token (tokenizer) như [[Tiktoken]] để chia văn bản thành các token, đảm bảo tính chính xác về số lượng token cho mỗi chunk.
*   **Sliding Window:** Kỹ thuật này tạo ra các chunk chồng lấp (overlap) với nhau. Đây là một phương pháp quan trọng để bảo toàn ngữ cảnh ở các điểm nối giữa các chunk, giảm nguy cơ mất thông tin quan trọng.
    ![[assets/attachments/Prompt Engineering for Generative AI Future-Proof Inputs for Reliable Al Outputs (James Phoenix, Mike Taylor) (Z-Library)/fig_3-6_A_sliding_window,_with_a_window_size_of_5_and_a_st.png]]
    *Figure 3-6: Sơ đồ mô tả cơ chế của sliding window với kích thước cửa sổ là 5 và bước nhảy (step size) là 1. Các chunk chồng lấp nhau đảm bảo ngữ cảnh liên tục được duy trì giữa các đoạn văn bản.*

## 6. Phân tích Cảm xúc (Sentiment Analysis)

[[Sentiment Analysis]] là một kỹ thuật NLP giúp LLMs xác định sắc thái cảm xúc (tích cực, tiêu cực, trung tính) trong văn bản. Để cải thiện độ chính xác, việc tiền xử lý văn bản (loại bỏ ký tự đặc biệt, chuyển chữ thường, sửa lỗi chính tả) và cung cấp các ví dụ mẫu (few-shot examples) trong prompt là cần thiết. LLMs có thể được sử dụng để phân loại cảm xúc, nhưng các thách thức như nhận diện mỉa mai hoặc cảm xúc phụ thuộc ngữ cảnh vẫn còn tồn tại.

## 7. Các Kỹ thuật Suy luận và Nâng cao (Advanced Reasoning Techniques)

Để giải quyết các tác vụ phức tạp, các prompt đơn giản là không đủ. Cần các kỹ thuật kích hoạt khả năng suy luận mạnh mẽ của LLM.

### 7.1. Suy luận theo Chuỗi suy nghĩ (Chain of Thought - CoT)

[[Chain of Thought (CoT)]] là một kỹ thuật khuyến khích mô hình "suy nghĩ từng bước" hoặc trình bày các bước suy luận trung gian trước khi đưa ra câu trả lời cuối cùng. Điều này cải thiện đáng kể độ chính xác trong các bài toán logic, toán học và suy luận đa bước bằng cách cho phép mô hình tự kiểm tra và điều chỉnh.

### 7.2. Least-to-Most Prompting

[[Divide Labor]] được áp dụng trong kỹ thuật **Least-to-Most Prompting**, nơi một vấn đề phức tạp được chia thành một chuỗi các vấn đề nhỏ hơn và được giải quyết tuần tự. Đầu ra của mỗi bước trở thành đầu vào cho bước tiếp theo, xây dựng kiến thức dần dần. Ví dụ, để phát triển một ứng dụng Flask, quy trình có thể bao gồm các bước: lập kế hoạch kiến trúc, viết từng hàm riêng lẻ, và sau đó viết các trường hợp kiểm thử.

### 7.3. Các chiến thuật Prompting của GPT (GPT Prompting Tactics)

Các chiến thuật này tập trung vào việc tối đa hóa hiệu quả và độ tin cậy của các phản hồi LLM:

*   **Văn bản Tham chiếu (Reference Text):** Để chống lại `[[Hallucination]]`, LLM được hướng dẫn chỉ trả lời dựa trên văn bản tham chiếu được cung cấp. Nếu không tìm thấy thông tin, mô hình phải trả lời "I don't know".
*   **Thời gian Suy nghĩ / Chuỗi suy nghĩ (Thinking Time / Chain of Thought):** Yêu cầu mô hình trình bày các bước suy luận của nó ("Let's think step by step") trước khi đưa ra kết luận.
*   **Độc thoại Nội tâm (Inner Monologue):** Cho phép mô hình tạo ra các "suy nghĩ nháp" được ẩn khỏi người dùng cuối, giúp mô hình tự sửa lỗi mà không làm lộ quá trình suy luận.
*   **Tự đánh giá (Self-Evaluation):** Yêu cầu mô hình tự kiểm tra và phê bình đầu ra của chính nó, tìm kiếm lỗi hoặc điểm yếu.

## 8. Phân loại (Classification)

LLMs là công cụ mạnh mẽ cho các tác vụ phân loại, có thể phân loại văn bản thành các danh mục được xác định trước.

*   **Zero-shot Learning:** Phân loại dữ liệu mà không cần bất kỳ ví dụ cụ thể nào.
*   **Few-shot Learning:** Cung cấp một số ít ví dụ mẫu để hướng dẫn LLM trong việc phân loại và định dạng đầu ra.
*   **Majority Vote:** Để tăng độ tin cậy, đặc biệt là với các tác vụ chủ quan, có thể chạy prompt nhiều lần và chọn kết quả phân loại xuất hiện nhiều nhất.

## 9. Siêu Prompt (Meta Prompting)

[[Meta Prompting]] là một kỹ thuật tiên tiến sử dụng LLM để tạo ra các prompt cho chính nó hoặc cho các mô hình AI khác (ví dụ: một LLM tạo prompt cho một mô hình sinh ảnh như Midjourney hoặc DALL-E).

   ![[assets/attachments/Prompt Engineering for Generative AI Future-Proof Inputs for Reliable Al Outputs (James Phoenix, Mike Taylor) (Z-Library)/fig_3-8_Utilizing_an_LLM_to_generate_image_prompts_for_Mid.png]]
   *Figure 3-8: Quy trình Meta Prompting trong việc tạo sách thiếu nhi có minh họa. Một LLM tạo ra cốt truyện, sau đó tạo ra các prompt hình ảnh từ cốt truyện đó, và các prompt này được sử dụng để sinh ảnh bằng các mô hình như Midjourney.*

Quy trình này cho phép tự động hóa việc tạo nội dung đa phương tiện và cải thiện chất lượng của các prompt.

*   **Prompt Rewriting / Refining:** Sử dụng LLM để cải thiện prompt đầu vào của người dùng (ví dụ: để làm cho prompt hiệu quả hơn hoặc phù hợp với phong cách cụ thể).
    ![[assets/attachments/Prompt Engineering for Generative AI Future-Proof Inputs for Reliable Al Outputs (James Phoenix, Mike Taylor) (Z-Library)/fig_3-9_ChatGPT_refining_a_meta_prompt_by_two_URL_web_page.png]]
    *Figure 3-9: ChatGPT tinh chỉnh một meta prompt dựa trên thông tin từ hai trang web URL. ChatGPT phân tích các bài blog được cung cấp và tạo ra một hướng dẫn kiểu viết được cải tiến, có thể dùng để tạo các bài viết blog khác theo cùng phong cách.*

---
**Tổng kết:** Chương 3 này đã mở rộng các kỹ thuật **Prompt Engineering** từ các nguyên tắc cơ bản lên các ứng dụng thực tế phức tạp. Việc hiểu sâu về cách xử lý input ([[Text Chunking]], [[Tokenization]]), định hướng suy luận ([[Chain of Thought]], [[Divide Labor]], [[Least-to-Most Prompting]]) và kiểm soát định dạng output (JSON, YAML) là nền tảng để xây dựng các hệ thống AI đáng tin cậy và mạnh mẽ. Khả năng **Meta Prompting** cho phép tự động hóa việc tạo prompt, mở ra cánh cửa cho các quy trình làm việc AI phức tạp hơn.
