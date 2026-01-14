---
tags:
  - Resources/BookNote
  - AI/LangChain
  - AI/PromptEngineering
  - AI/Agents
created: 2026-01-04
source: [[Prompt Engineering for Generative AI Future-Proof Inputs for Reliable Al Outputs (James Phoenix, Mike Taylor) (Z-Library).pdf]]
author: James Phoenix, Mike Taylor
---

# Chapter 04: Advanced Techniques for Text Generation with LangChain

## 1. Giới thiệu: Tại sao cần LangChain?

Trong khi các kỹ thuật Prompt Engineering cơ bản tập trung vào việc tối ưu hóa câu chữ cho một lần gọi model, việc xây dựng một ứng dụng AI thực tế đòi hỏi nhiều hơn thế. **LangChain** ra đời như một framework để giải quyết các vấn đề cốt lõi của việc "productizing" (thương mại hóa) LLM:

1.  **Thiếu sự chuẩn hóa (Standardization Gap):** Mỗi nhà cung cấp mô hình (OpenAI, Anthropic, Cohere) đều có API và định dạng input/output riêng. LangChain cung cấp một lớp trừu tượng hóa (abstraction layer) giúp lập trình viên dễ dàng chuyển đổi giữa các model mà không phải viết lại code.
2.  **Kết nối dữ liệu (Data Awareness):** LLM mặc định bị giới hạn trong dữ liệu huấn luyện. LangChain cung cấp các công cụ (Loaders, Splitters, Retrievers) để kết nối model với dữ liệu riêng (PDF, Database, Web) - nền tảng của kỹ thuật **RAG**.
3.  **Khả năng hành động (Agency):** Thay vì chỉ trả lời câu hỏi, ứng dụng cần thực hiện hành động (gửi email, query SQL). LangChain tích hợp sâu với **Function Calling** và khái niệm **Agents**.

![[fig_4-1_The_major_modules_of_the_LangChain_LLM_framework_E.png]]
*Figure 4-1: Kiến trúc modular của LangChain: Từ Model I/O, Retrieval đến Chains và Agents.*

## 2. Model I/O: Nền tảng của giao tiếp

### Chat Models & Message Types
LangChain phân biệt rõ ràng giữa *LLM* (text-in, text-out) và *Chat Models* (message-in, message-out). Trong Chat Models (như GPT-4), input không phải là một chuỗi string đơn thuần mà là một danh sách các message có vai trò cụ thể:
-   **SystemMessage:** Chỉ thị cấp cao nhất, định hình hành vi và persona của model ("Bạn là một chuyên gia lập trình...").
-   **HumanMessage:** Input từ người dùng.
-   **AIMessage:** Phản hồi từ model.

> [!NOTE] Tại sao cần phân loại Message?
> Việc phân loại giúp model hiểu rõ ngữ cảnh hội thoại, tránh việc model bị nhầm lẫn giữa chỉ thị hệ thống và câu nói của người dùng (prompt injection defense), đồng thời tận dụng khả năng instruction-following tốt hơn của các model đời mới.

### Prompt Templates
Thay vì cộng chuỗi thủ công (`"Dịch câu này: " + user_input`), LangChain sử dụng **Prompt Templates**.
-   **Lợi ích:** Tái sử dụng, validate input, và quản lý các biến động (dynamic variables) dễ dàng.
-   **Few-Shot Prompting:** LangChain cung cấp `FewShotChatMessagePromptTemplate` và `LengthBasedExampleSelector` để tự động chọn và đưa các ví dụ (shots) phù hợp nhất vào prompt dựa trên độ dài context window, giúp tối ưu chi phí token mà vẫn đảm bảo hiệu quả.

### Output Parsers: Cấu trúc hóa câu trả lời
Đây là thành phần quan trọng để biến LLM thành một component trong phần mềm. Model trả về text, nhưng ứng dụng cần object (JSON, List, Date).
-   **PydanticOutputParser:** Sử dụng thư viện Pydantic của Python để định nghĩa schema dữ liệu mong muốn.
-   **Cơ chế:** Parser sẽ tự động sinh ra một đoạn "format instructions" (hướng dẫn định dạng) chèn vào prompt, yêu cầu model trả về JSON đúng chuẩn. Sau đó, nó validate và parse JSON string từ model thành Python object.

```python
# Ví dụ concept Pydantic Parser
class Person(BaseModel):
    name: str = Field(description="Tên người")
    age: int = Field(description="Tuổi")

parser = PydanticOutputParser(pydantic_object=Person)
# Prompt sẽ tự động được thêm: "The output should be formatted as a JSON instance that conforms to the JSON schema below..."
```

## 3. LangChain Expression Language (LCEL)

LCEL là một cú pháp khai báo (declarative syntax) mạnh mẽ, lấy cảm hứng từ Unix pipe (`|`). Nó cho phép kết nối các thành phần thành một chuỗi xử lý (chain) mượt mà.

```python
# Dữ liệu chảy từ prompt -> model -> output_parser
chain = prompt | model | output_parser
```

Lợi ích của LCEL:
-   **Tự động song song hóa (Parallelism):** Các bước không phụ thuộc nhau có thể chạy đồng thời.
-   **Streaming:** Hỗ trợ stream token-by-token ngay từ đầu.
-   **Dễ dàng debug:** Quan sát được input/output của từng bước trong chuỗi.

## 4. Function Calling & Agents

### Function Calling (Tool Use)
[[Function Calling]] cho phép model "biết" về các hàm (functions) mà lập trình viên cung cấp.
1.  Người dùng hỏi: "Thời tiết Hà Nội thế nào?"
2.  Model phân tích và trả về JSON đặc biệt: `{"name": "get_weather", "args": {"location": "Hanoi"}}` (thay vì trả lời bằng văn bản).
3.  Hệ thống thực thi hàm `get_weather` và gửi kết quả lại cho model.
4.  Model tổng hợp kết quả thành câu trả lời cuối cùng.

Điều này giải quyết triệt để vấn đề [[Hallucination]] khi model cần truy xuất thông tin thời gian thực hoặc tính toán chính xác.

### Agents
Nếu **Chain** là một chuỗi hành động được định nghĩa cứng (hard-coded sequence), thì **Agent** sử dụng LLM như một bộ não để *tự quyết định* hành động tiếp theo là gì.
Agent hoạt động theo vòng lặp: **Thought -> Act -> Observe -> Thought...** cho đến khi hoàn thành nhiệm vụ.

![[fig_4-3_Task_decomposition_with_LLMs_170.png]]
*Figure 4-3: Tư duy phân rã tác vụ (Task Decomposition) là nền tảng của Agent.*

## 5. Xử lý văn bản dài (Handling Long Context)

Khi nội dung vượt quá giới hạn [[Context Window]] (ví dụ: tóm tắt một cuốn sách), LangChain cung cấp các chiến lược "Divide and Conquer":

### 1. Stuff Documents
-   **Cách làm:** Nhồi tất cả văn bản vào prompt.
-   **Ưu điểm:** Đơn giản, model thấy toàn bộ ngữ cảnh.
-   **Nhược điểm:** Dễ dàng vượt quá token limit, chi phí cao.

### 2. Map-Reduce
![[fig_4-6_Refine_documents_chain_Map_Reduce_The_map_reduce_d.png]]
*Figure 4-6: Mô hình Map-Reduce.*
-   **Cách làm:**
    1.  **Map:** Chia văn bản thành các chunk nhỏ, tóm tắt song song từng chunk.
    2.  **Reduce:** Gom các bản tóm tắt nhỏ lại và tóm tắt một lần nữa thành kết quả cuối.
-   **Ưu điểm:** Xử lý được tài liệu cực lớn, chạy nhanh (nhờ song song).
-   **Nhược điểm:** Mất ngữ cảnh giữa các chunk (thông tin ở chunk 1 có thể không liên kết được với chunk 2 trong bước Map).

### 3. Refine
-   **Cách làm:** Tuần tự. Tóm tắt chunk 1 -> Lấy tóm tắt đó + chunk 2 đưa vào model để cập nhật (refine) bản tóm tắt -> Lặp lại đến hết.
-   **Ưu điểm:** Chất lượng cao nhất, giữ được mạch văn và ngữ cảnh liên tục.
-   **Nhược điểm:** Chậm (không thể song song hóa).

### 4. Map-Rerank
-   **Cách làm:** Chạy prompt cho từng chunk độc lập để tìm câu trả lời, model tự chấm điểm độ tin cậy (score). Chọn câu trả lời có điểm cao nhất.
-   **Ứng dụng:** Tìm kiếm câu trả lời cụ thể trong đống tài liệu hỗn độn.

## 6. Data Connection (RAG Pipeline)

Để "dạy" model kiến thức mới mà không cần training, LangChain xây dựng pipeline **RAG (Retrieval Augmented Generation)**:

1.  **Document Loaders:** Đọc dữ liệu từ nhiều nguồn (PDF, Notion, S3...).
2.  **Text Splitters:** Chia nhỏ văn bản.
    > **Quan trọng:** Cần dùng `RecursiveCharacterTextSplitter` để chia văn bản thông minh (giữ nguyên cấu trúc đoạn văn, câu) thay vì chia cắt thô bạo làm mất ngữ nghĩa.
3.  **Embeddings:** Chuyển văn bản thành vector.
4.  **Vector Stores:** Lưu trữ và tìm kiếm vector tương đồng (Similarity Search).

![[fig_4-2_A_data_connection_to_retrieval_pipeline_Data_Conne.png]]
*Figure 4-2: Pipeline kết nối dữ liệu.*

## 7. Đánh giá (Evaluation)

Đánh giá LLM là một bài toán khó vì output không xác định (non-deterministic). LangChain đề xuất:
-   **String Evaluators:** So sánh string cơ bản (Exact match, Regex).
-   **LLM-as-a-Judge:** Dùng GPT-4 để chấm điểm output của các model nhỏ hơn dựa trên tiêu chí (độ hữu ích, độ an toàn) hoặc so sánh cặp (Pairwise Comparison). Đây là phương pháp hiện đại và hiệu quả nhất để đánh giá các tác vụ sáng tạo.

---
**Tổng kết:** Chương 4 không chỉ hướng dẫn cách dùng công cụ, mà còn định hình tư duy thiết kế hệ thống AI: từ việc quản lý prompt, cấu trúc dữ liệu, đến việc thiết kế các luồng xử lý dữ liệu lớn và đánh giá hiệu quả hệ thống.