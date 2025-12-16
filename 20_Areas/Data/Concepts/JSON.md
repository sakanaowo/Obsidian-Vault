---
tags:
  - json
  - data-format
  - prompt-engineering
  - llm
status: in_progress
created: 2025-12-10
source:
  - Prompt Engineering for Generative AI (James Phoenix, Mike Taylor)
---

# JSON (JavaScript Object Notation)

## Định nghĩa

**JSON (JavaScript Object Notation)** là một định dạng dữ liệu văn bản nhẹ, độc lập với ngôn ngữ, được sử dụng để lưu trữ và truyền tải dữ liệu. Nó dễ đọc và dễ viết đối với con người, và dễ dàng phân tích, tạo ra đối với máy móc.

## JSON trong Prompt Engineering và LLMs

Trong bối cảnh [[Prompt Engineering]], JSON đóng vai trò quan trọng trong việc chỉ định [[Specify Format]] đầu ra mong muốn từ các mô hình [[Large Language Models]] (LLMs). LLMs có khả năng tạo ra phản hồi dưới nhiều định dạng khác nhau, và việc yêu cầu đầu ra JSON mang lại nhiều lợi ích:

*   **Tính cấu trúc (Structured Output):** Đảm bảo phản hồi của LLM có một cấu trúc nhất quán, giúp dễ dàng phân tích cú pháp và xử lý tự động bởi các ứng dụng phần mềm hoặc script.
*   **Giảm lỗi phân tích cú pháp (Parsing Errors):** Khi tích hợp LLM vào các hệ thống sản xuất, việc đầu ra không nhất quán có thể gây ra lỗi. Yêu cầu JSON giúp giảm thiểu các lỗi này, vì các thư viện JSON tiêu chuẩn có thể kiểm tra định dạng hợp lệ.
*   **Trao đổi dữ liệu:** JSON là định dạng phổ biến cho các API, giúp dữ liệu từ LLM có thể dễ dàng được sử dụng làm đầu vào cho các hệ thống khác hoặc để hiển thị trên giao diện người dùng.

## Cách sử dụng trong Prompt

Để yêu cầu LLM trả về đầu ra dưới định dạng JSON, bạn cần:

1.  **Chỉ dẫn rõ ràng:** Trong prompt, chỉ dẫn rõ ràng rằng bạn muốn đầu ra là JSON.
    *   Ví dụ: "Return the results in JSON format." hoặc "Only return valid JSON."
2.  **Cung cấp ví dụ (tùy chọn nhưng khuyến khích):** Cung cấp một ví dụ về cấu trúc JSON mong muốn. Điều này giúp LLM hiểu rõ hơn về cách tổ chức dữ liệu trong JSON.

### Ví dụ về Prompt yêu cầu JSON:

```
Input:
Compose a detailed article outline on "The benefits of learning code" with a JSON payload structure that highlights key points. Only return valid JSON.

Here is an example of the JSON structure:
{
  "Introduction": [
    "a. Overview of coding and programming languages",
    "b. Importance of coding in today's technology-driven world"
  ],
  "Conclusion": [
    "a. Recap of the benefits of learning code",
    "b. The ongoing importance of coding skills in the modern world"
  ]
}

Output (LLM):
{
  "Introduction": [
    "a. Understanding the fundamentals of programming",
    "b. The role of coding in modern innovation"
  ],
  "Core Concepts": [
    "a. Problem-solving through algorithms",
    "b. Data structures and their applications"
  ],
  "Benefits": [
    "a. Enhanced career opportunities",
    "b. Fostering logical thinking and creativity"
  ],
  "Conclusion": [
    "a. Summary of key takeaways",
    "b. Future outlook for coding proficiency"
  ]
}
```

## Các Lưu ý khi làm việc với JSON và LLM

*   **Đầu ra không hợp lệ:** Đôi khi, LLM có thể trả về JSON không hợp lệ hoặc thêm các văn bản không mong muốn xung quanh JSON (ví dụ: ```json ... ```). Cần có cơ chế xử lý lỗi trong mã nguồn của bạn để đối phó với những trường hợp này.
*   **PydanticOutputParser (LangChain):** Các thư viện như LangChain cung cấp các công cụ như `PydanticOutputParser` cho phép bạn định nghĩa schema JSON mong muốn bằng các mô hình Pydantic trong Python. Điều này giúp tự động hóa việc xác thực và phân tích cú pháp đầu ra của LLM.
*   **Grammars (với Llama Models):** Một số mô hình như Llama cho phép bạn chỉ định ngữ pháp (grammars) để đảm bảo đầu ra tuân thủ một định dạng cụ thể (ví dụ: JSON Schema).
*   **Kích thước JSON:** Đối với JSON lớn, có thể vượt quá [[Context Window]] của LLM, cần cân nhắc các chiến lược [[Chunking Text]] hoặc [[Divide Labor]].

Việc sử dụng JSON giúp cải thiện đáng kể độ tin cậy và khả năng sử dụng của các ứng dụng LLM, biến chúng thành công cụ mạnh mẽ hơn để tự động hóa các tác vụ phức tạp.
