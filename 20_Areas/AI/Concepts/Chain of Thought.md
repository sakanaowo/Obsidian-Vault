---
tags:
  - chain-of-thought
  - cot
  - prompt-engineering
  - llm
  - reasoning
status: in_progress
created: 2025-12-10
source:
  - Prompt Engineering for Generative AI (James Phoenix, Mike Taylor)
  - RAG roadmap.pdf
---

# Chain of Thought (CoT)

## Định nghĩa

**Chain of Thought (CoT)** là một kỹ thuật [[Prompt Engineering]] nhằm hướng dẫn các [[Large Language Models]] (LLMs) suy luận thông qua một loạt các bước trung gian hoặc các kết nối logic để đi đến kết luận hoặc giải quyết vấn đề. Thay vì chỉ cung cấp một câu trả lời trực tiếp, mô hình được khuyến khích hiển thị quá trình "suy nghĩ" của nó.

## Cơ chế hoạt động (Mechanism)

Thay vì yêu cầu mô hình đi thẳng từ `Input` -> `Output`, CoT khuyến khích mô hình đi theo quy trình:
`Input` -> `Reasoning Step 1` -> `Reasoning Step 2` -> ... -> `Output`.

Điều này giúp kích hoạt các kiến thức liên quan cần thiết ở từng bước nhỏ, thay vì bắt mô hình phải "nhảy cóc" đến kết quả ngay lập tức.

## Tầm quan trọng và Lợi ích

Kỹ thuật CoT đặc biệt hữu ích cho các tác vụ đòi hỏi sự hiểu biết sâu sắc về ngữ cảnh hoặc xem xét nhiều yếu tố.

*   **Cải thiện Khả năng Suy luận:** Bằng cách khuyến khích mô hình "suy nghĩ từng bước", CoT giúp LLM thực hiện các tác vụ suy luận phức tạp hơn, chẳng hạn như các vấn đề toán học, logic, hoặc lập kế hoạch đa bước.
*   **Giảm [[Hallucination]]:** Khi mô hình phải giải thích các bước suy luận của mình, khả năng tạo ra thông tin không chính xác hoặc bịa đặt sẽ giảm xuống vì nó buộc mô hình phải tự kiểm tra tính nhất quán trong quá trình.
*   **Tăng tính Minh bạch và Giải thích được:** Quá trình suy nghĩ của mô hình trở nên minh bạch hơn, cho phép người dùng hoặc nhà phát triển hiểu được cách mô hình đi đến một câu trả lời cụ thể. Điều này rất quan trọng cho việc gỡ lỗi và kiểm tra chất lượng.
*   **Xử lý các Vấn đề phức tạp:** CoT cho phép LLM chia nhỏ các vấn đề phức tạp thành các thành phần nhỏ hơn, dễ quản lý hơn, sau đó giải quyết từng phần một. Điều này tương tự như nguyên tắc [[Divide Labor]] trong kỹ thuật phần mềm.
*   **Hiệu suất nhất quán:** Khi có quá trình suy nghĩ rõ ràng, các phản hồi của LLM thường trở nên nhất quán và đáng tin cậy hơn, đặc biệt là trong các tình huống nhạy cảm hoặc quan trọng.

## Cách sử dụng Chain of Thought

Kỹ thuật CoT thường được kích hoạt bằng cách thêm một cụm từ đơn giản vào prompt, yêu cầu mô hình giải thích các bước suy nghĩ của nó.

### Ví dụ đơn giản:

```
Input:
Jane có 5 quả táo. John lấy 2 quả táo từ Mary. Mary ban đầu có 15 quả táo. Sau đó John đưa táo của anh ấy cho Jane.
Hỏi Jane có bao nhiêu quả táo?

Proposed solution:
1. Jane ban đầu có 5 quả táo.
2. John lấy 2 quả táo từ Mary, nhưng điều này không ảnh hưởng đến số lượng táo của Jane tại thời điểm này.
3. John đưa táo của anh ấy cho Jane.
Vậy, số táo cuối cùng của Jane là 5 quả táo ban đầu cộng thêm 2 quả táo mà John đưa cho, tổng cộng là 7 quả táo.

Output (LLM):
Hãy cùng phân tích vấn đề:
1. Jane ban đầu có 5 quả táo.
2. John lấy 2 quả táo từ Mary. Tuy nhiên, hành động này không ảnh hưởng trực tiếp đến số lượng táo mà Jane có tại thời điểm này.
3. John đưa táo của anh ấy cho Jane.

Theo các bước này, chúng ta cần cộng 2 quả táo mà John đưa cho Jane vào số táo ban đầu của Jane. Do đó:
5 quả táo (số táo ban đầu của Jane) + 2 quả táo (do John đưa cho) = 7 quả táo.
Vậy, Jane hiện có 7 quả táo. Giải pháp của tôi khớp với giải pháp đề xuất.
```

### Các chiến lược CoT nâng cao:

*   **"Let's think step by step":** Một cụm từ đơn giản nhưng hiệu quả để kích hoạt suy luận theo chuỗi suy nghĩ.
*   **Inner Monologue Tactic (Độc thoại nội tâm):** Hướng dẫn mô hình tạo ra các phần của output mà người dùng không nhìn thấy, ví dụ: để mô hình tự giải quyết vấn đề, so sánh giải pháp của nó với giải pháp của người dùng, và chuẩn bị gợi ý nếu cần.
*   **ReAct (Reason and Act):** Một framework nâng cao kết hợp suy luận CoT với khả năng thực hiện các hành động thông qua các công cụ. Mô hình tạo ra một suy nghĩ, sau đó quyết định một hành động, thực hiện hành động đó và quan sát kết quả, sau đó lặp lại.
*   **Tree of Thoughts (ToT):** Cho phép mô hình khám phá nhiều con đường suy luận khác nhau và tự đánh giá các quyết định ở mỗi bước, đặc biệt hữu ích cho các tác vụ đòi hỏi lập kế hoạch phức tạp hoặc tìm kiếm.

## Thách thức

*   **Chi phí và Độ trễ:** Việc tạo ra các chuỗi suy nghĩ dài có thể làm tăng chi phí token và độ trễ của phản hồi.
*   **Chất lượng Suy luận:** Mặc dù CoT cải thiện suy luận, chất lượng của chuỗi suy nghĩ vẫn phụ thuộc vào chất lượng của prompt và khả năng của mô hình cơ sở.
*   **Over-specification:** Quá nhiều chỉ dẫn hoặc quá trình suy nghĩ có thể hạn chế sự sáng tạo của mô hình.

Việc tích hợp CoT vào Prompt Engineering giúp nâng cao đáng kể khả năng của LLM trong việc giải quyết các vấn đề phức tạp, mang lại kết quả đáng tin cậy và giải thích được.
