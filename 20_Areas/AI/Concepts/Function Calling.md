---
tags:
  - AI/Technique
  - AI/Capability
  - Concept
alias:
  - Tool Use
created: 2026-01-04
---

### Định nghĩa

**Function Calling** (hoặc Tool Use) là khả năng của [[Large Language Models]] (LLMs) trong việc kết nối với các công cụ và API bên ngoài. Thay vì chỉ trả về văn bản tự do, model được huấn luyện để phát hiện khi nào cần gọi một hàm (function) và tạo ra output là **JSON object** chứa các tham số (arguments) tuân theo schema đã định nghĩa trước.

### Cơ chế hoạt động

1.  **Define:** Lập trình viên mô tả các hàm có sẵn (tên, mô tả, tham số) cho model thông qua prompt hoặc API system parameter (ví dụ: JSON Schema).
2.  **Invoke:** Người dùng gửi query (ví dụ: "Thời tiết Hà Nội thế nào?").
3.  **Detect & Generate:** Model phân tích query, nhận thấy cần dùng hàm `get_weather`, và trả về JSON: `{"tool": "get_weather", "args": {"location": "Hanoi"}}`.
4.  **Execute:** Ứng dụng client nhận JSON, thực thi code thực tế (gọi API thời tiết), và lấy kết quả.
5.  **Response:** Kết quả thực thi được gửi lại cho model để tạo câu trả lời cuối cùng bằng ngôn ngữ tự nhiên.

### Vai trò trong Agent

Function Calling là nền tảng của các **AI Agents**. Nó cho phép LLM thoát khỏi giới hạn của dữ liệu training tĩnh để tương tác với thế giới thực: truy vấn database, gửi email, tính toán phức tạp, v.v.

### Ví dụ (OpenAI)

```json
// Model response
{
  "tool_calls": [
    {
      "type": "function",
      "function": {
        "name": "get_current_weather",
        "arguments": "{\"location\": \"Boston, MA\"}"
      }
    }
  ]
}
```

```