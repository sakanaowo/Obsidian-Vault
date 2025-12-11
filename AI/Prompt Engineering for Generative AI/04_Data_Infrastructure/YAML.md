---
tags:
  - yaml
  - data-format
  - prompt-engineering
  - llm
status: in_progress
created: 2025-12-10
source:
  - Prompt Engineering for Generative AI (James Phoenix, Mike Taylor)
---

# YAML (YAML Ain't Markup Language)

## Định nghĩa

**YAML (YAML Ain't Markup Language)** là một ngôn ngữ tuần tự hóa dữ liệu thân thiện với con người, thường được sử dụng cho các tệp cấu hình và trao đổi dữ liệu giữa các ngôn ngữ. Nó được thiết kế để dễ đọc hơn [[JSON]] và thường được sử dụng trong các tình huống mà sự rõ ràng và khả năng đọc của con người là ưu tiên hàng đầu.

## YAML trong Prompt Engineering và LLMs

Tương tự như [[JSON]], YAML là một định dạng hữu ích trong [[Prompt Engineering]] để chỉ định [[Specify Format]] đầu ra từ các mô hình [[Large Language Models]] (LLMs). YAML mang lại một số lợi ích đặc biệt trong bối cảnh này:

*   **Dễ đọc cho con người (Human-Readability):** Cú pháp dựa trên thụt lề (indentation) của YAML loại bỏ nhu cầu về dấu ngoặc nhọn, dấu ngoặc vuông và dấu phẩy, làm cho nó dễ đọc và chỉnh sửa hơn JSON. Điều này đặc biệt hữu ích khi làm việc với các prompt phức tạp hoặc có cấu trúc lồng nhau.
*   **Hỗ trợ Comment:** Không giống như JSON, YAML cho phép thêm các comment trực tiếp vào file. Điều này rất có giá trị để thêm chú thích hoặc giải thích vào prompt, giúp cải thiện khả năng hiểu và cộng tác, đặc biệt trong các dự án lớn hoặc khi xem lại prompt sau một thời gian.
*   **Cấu trúc rõ ràng:** Cú pháp của YAML tự động tạo ra một cấu trúc phân cấp, giúp đảm bảo đầu ra của LLM có trật tự và dễ phân tích. Điều này đặc biệt hữu ích khi bạn muốn LLM tạo ra các dàn ý, cấu hình, hoặc các cấu trúc dữ liệu phức tạp khác.

## Cách sử dụng trong Prompt

Để yêu cầu LLM trả về đầu ra dưới định dạng YAML, bạn cần:

1.  **Chỉ dẫn rõ ràng:** Trong prompt, chỉ dẫn rõ ràng rằng bạn muốn đầu ra là YAML.
    *   Ví dụ: "Return the results in YAML format." hoặc "Only return valid YAML."
2.  **Cung cấp ví dụ về schema (tùy chọn nhưng khuyến khích):** Cung cấp một ví dụ về cấu trúc YAML mong muốn hoặc một schema YAML. Điều này giúp LLM hiểu rõ hơn về cách tổ chức dữ liệu.

### Ví dụ về Prompt yêu cầu YAML:

```
Input:
Below you'll find the current yaml schema.
You can update the quantities based on a User Query.
Filter the User Query based on the schema below, if it doesn't match and
there are no items left then return "No Items".
If there is a partial match, then return only the items that are
within the schema below:

# schema:
#   item: Apple Slices
#   quantity: 5
#   unit: pieces
#   item: Milk
#   quantity: 1
#   unit: gallon
#   item: Bread
#   quantity: 2
#   unit: loaves
#   item: Eggs
#   quantity: 1
#   unit: dozen

User Query: "5 apple slices, and 2 dozen eggs."

Given the schema below, please return only a valid .yml based on the User
Query. If there's no match, return "No Items". Do not provide any
commentary or explanations.

Output (LLM):
item: Apple Slices
quantity: 5
unit: pieces
item: Eggs
quantity: 2
unit: dozen
```

## Các Lưu ý khi làm việc với YAML và LLM

*   **Xử lý Payload không hợp lệ:** Tương tự như JSON, LLM có thể đôi khi tạo ra YAML không hợp lệ. Cần có cơ chế xử lý lỗi trong code của bạn để xác thực đầu ra.
*   **Lồng ghép (Nesting):** YAML xuất sắc trong việc biểu diễn các cấu trúc dữ liệu lồng ghép một cách rõ ràng, phù hợp cho việc tạo các dàn ý bài viết hoặc cấu hình phức tạp.
*   **Control Flow (Luồng điều khiển):** Việc sử dụng YAML trong prompt có thể giúp LLM hướng dẫn các phần của luồng điều khiển ứng dụng, cho phép nó "lý luận" và quyết định các hành động dựa trên dữ liệu có cấu trúc.

Việc tận dụng YAML trong Prompt Engineering giúp tạo ra các tương tác LLM hiệu quả hơn, với đầu ra dễ đọc, dễ quản lý và dễ tích hợp vào các quy trình làm việc khác.
