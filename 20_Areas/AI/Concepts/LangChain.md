---
tags:
  - AI/Framework
  - AI/Tools
  - Concept
aliases:
  - LangChain Framework
created: 2026-01-04
---

### Định nghĩa

**LangChain** là một open-source framework được thiết kế để đơn giản hóa việc phát triển các ứng dụng sử dụng [[Large Language Models]] (LLMs). Nguyên lý cốt lõi của LangChain là các ứng dụng AI mạnh mẽ không chỉ dựa vào việc gọi API của model mà còn phải có khả năng:
1.  **Data Awareness:** Kết nối model với các nguồn dữ liệu bên ngoài (Data connection).
2.  **Agentic:** Cho phép model tương tác với môi trường xung quanh (Agents/Tools).

### Kiến trúc Modular

LangChain cung cấp các module trừu tượng hóa (abstractions) để làm việc với LLMs một cách linh hoạt:

*   **Model I/O:** Quản lý input (Prompts) và output (Parsers) của model.
*   **Retrieval:** Tích hợp với dữ liệu ngoài thông qua [[Vector Databases]], Text Splitters và Document Loaders (cơ sở cho [[Retrieval Augmented Generation (RAG)]]).
*   **Chains:** Kết hợp nhiều component hoặc nhiều lời gọi model thành một chuỗi xử lý tuần tự (Sequential) hoặc phức tạp.
*   **Agents:** Sử dụng LLM làm "bộ não" để quyết định action nào cần thực hiện và theo trình tự nào (thay vì hard-code).
*   **Memory:** Giúp model "nhớ" trạng thái của hội thoại giữa các lần tương tác.
*   **Callbacks:** Hook vào các giai đoạn của pipeline để log, monitor hoặc stream dữ liệu.

### LangChain Expression Language (LCEL)

LCEL là một cú pháp khai báo (declarative) để kết nối các chain một cách dễ dàng, hỗ trợ streaming, async và parallel execution ngay từ đầu.

```python
chain = prompt | model | output_parser
```

### Tại sao dùng LangChain?

*   **Standardization:** Chuẩn hóa giao diện làm việc với nhiều model provider khác nhau (OpenAI, Anthropic, HuggingFace...).
*   **Flexibility:** Dễ dàng thay đổi model hoặc component trong pipeline mà không cần viết lại nhiều code.
*   **Ecosystem:** Hệ sinh thái integrations khổng lồ với các công cụ và database khác.
