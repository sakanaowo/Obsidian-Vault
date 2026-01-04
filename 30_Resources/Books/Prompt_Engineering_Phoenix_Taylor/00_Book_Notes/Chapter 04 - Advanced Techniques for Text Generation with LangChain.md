---
tags:
  - Resources/BookNote
  - AI/LangChain
  - AI/PromptEngineering
created: 2026-01-04
source: [[Prompt Engineering for Generative AI Future-Proof Inputs for Reliable Al Outputs (James Phoenix, Mike Taylor) (Z-Library).pdf]]
author: James Phoenix, Mike Taylor
---

# Chapter 04: Advanced Techniques for Text Generation with LangChain

## 1. Tổng quan về LangChain

Trong khi Prompt Engineering cơ bản tập trung vào việc tối ưu hóa câu chữ đầu vào, **[[LangChain]]** nâng tầm việc phát triển ứng dụng AI bằng cách cung cấp một framework để kết nối [[Large Language Models]] với dữ liệu và môi trường bên ngoài.

Chương này đi sâu vào cách sử dụng LangChain để giải quyết các hạn chế của LLM thô (raw LLM):
- **Context Awareness:** Kết nối với dữ liệu riêng (RAG).
- **Agency:** Khả năng thực hiện hành động thông qua Tools.
- **Structure:** Đảm bảo output tuân theo định dạng chuẩn (JSON, List).

![[fig_4-1_The_major_modules_of_the_LangChain_LLM_framework_E.png]]
*Figure 4-1: Các module chính trong hệ sinh thái LangChain.*

## 2. Các thành phần cốt lõi (Core Modules)

### Prompt Templates & Output Parsers
Thay vì hard-code chuỗi văn bản, LangChain sử dụng **Prompt Templates** để tạo ra các prompt linh hoạt, có thể tái sử dụng và quản lý các biến động (dynamic variables).

Kết hợp với **Output Parsers**, lập trình viên có thể kiểm soát định dạng đầu ra của model, chuyển đổi từ văn bản thô sang các cấu trúc dữ liệu lập trình (như Python Object, JSON).

```python
# Ví dụ LCEL (LangChain Expression Language)
chain = prompt | model | output_parser
```

> [!NOTE] Pydantic Parser
> LangChain tận dụng thư viện **Pydantic** của Python để định nghĩa schema dữ liệu mong muốn, giúp tự động sinh ra hướng dẫn (format instructions) cho model và validate dữ liệu trả về.

### Function Calling (Tool Use)
[[Function Calling]] là bước nhảy vọt giúp LLM không chỉ là một công cụ tạo văn bản ("Chatbot") mà trở thành một **Agent** có khả năng hành động. Model có thể quyết định gọi hàm nào dựa trên input của người dùng.

Ví dụ: Thay vì tự bịa ra thông tin thời tiết ([[Hallucination]]), model sẽ gọi hàm `get_weather(location="Hanoi")` để lấy dữ liệu thực.

## 3. Các kỹ thuật Chaining nâng cao

Để giải quyết các tác vụ phức tạp, chương này giới thiệu kỹ thuật [[Prompt Chaining]] (xâu chuỗi prompt), chia nhỏ vấn đề lớn thành các bước nhỏ hơn.

![[fig_4-4_A_sequential_story_creation_process_Prompt_Chainin.png]]
*Figure 4-4: Quy trình tạo câu chuyện tuần tự (Sequential Chain).*

### Sequential Chain
Đầu ra của bước 1 (ví dụ: Tóm tắt văn bản) trở thành đầu vào của bước 2 (ví dụ: Dịch sang tiếng Pháp). Giúp kiểm soát logic và dễ dàng debug từng bước.

### Map-Reduce & Refine Documents Chain
Khi xử lý tài liệu dài vượt quá [[Context Window]], LangChain cung cấp các chiến lược:

1.  **Map-Reduce:** Chia nhỏ tài liệu thành các chunk, tóm tắt song song từng chunk (Map), sau đó tổng hợp các tóm tắt lại (Reduce).
    ![[fig_4-6_Refine_documents_chain_Map_Reduce_The_map_reduce_d.png]]
    *Figure 4-6: Mô hình Map-Reduce để xử lý văn bản dài.*

2.  **Refine:** Tóm tắt chunk đầu tiên, sau đó đưa tóm tắt đó cùng với chunk tiếp theo vào model để "tinh chỉnh" (update) bản tóm tắt. Tuần tự nhưng bảo toàn ngữ cảnh tốt hơn.

## 4. Evaluation & Data Connection

### LLM Evaluation
Việc đánh giá output của LLM là thách thức lớn do tính chất không xác định (non-deterministic). LangChain đề xuất sử dụng **LLM-as-a-Judge**: dùng một model mạnh (như GPT-4) để chấm điểm output của model khác dựa trên các tiêu chí (Criteria) hoặc so sánh cặp (Pairwise comparison). Xem chi tiết tại [[LLM Evaluation]].

### Data Connection (RAG Pipeline)
LangChain cung cấp các công cụ mạnh mẽ để:
- **Load:** Đọc dữ liệu từ PDF, CSV, Word.
- **Transform:** Chia nhỏ văn bản ([[Text Chunking]]) bằng `RecursiveCharacterTextSplitter`.
- **Embed & Store:** Lưu trữ vào [[Vector Databases]] để tìm kiếm ngữ nghĩa.

![[fig_4-2_A_data_connection_to_retrieval_pipeline_Data_Conne.png]]
*Figure 4-2: Pipeline kết nối dữ liệu cho RAG.*

## 5. Kỹ thuật Few-Shot Prompting
Chương này cũng nhấn mạnh tầm quan trọng của [[Few-Shot Prompting]]. LangChain hỗ trợ `FewShotChatMessagePromptTemplate` và `LengthBasedExampleSelector` để tự động chọn các ví dụ phù hợp nhất đưa vào prompt mà không làm tràn bộ nhớ context.

---
**Liên kết:** [[Chapter 03 - Standard Practices for Text Generation with ChatGPT]] | [[Chapter 05 - Vector Databases with FAISS and Pinecone]]
