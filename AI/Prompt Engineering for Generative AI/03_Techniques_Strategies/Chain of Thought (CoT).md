---
tags:
  - prompt-engineering
  - reasoning
  - technique
  - logic
status: done
created_date: 2025-12-11
---

# Chain of Thought (CoT)

**Chain of Thought (CoT)** là một kỹ thuật trong [[Prompt Engineering]] nhằm cải thiện khả năng suy luận của các mô hình ngôn ngữ lớn ([[Large Language Models]]) bằng cách yêu cầu chúng tạo ra một chuỗi các bước suy nghĩ trung gian trước khi đưa ra câu trả lời cuối cùng.

## Cơ chế hoạt động
Thay vì yêu cầu mô hình đi thẳng từ `Input` -> `Output`, CoT khuyến khích mô hình đi theo quy trình:
`Input` -> `Reasoning Step 1` -> `Reasoning Step 2` -> ... -> `Output`.

## Tại sao CoT hiệu quả?
1.  **Phân rã vấn đề:** Nó buộc mô hình chia nhỏ các bài toán phức tạp (đặc biệt là toán học hoặc logic) thành các bước đơn giản hơn.
2.  **Debug:** Giúp người dùng hiểu được logic sai lầm của mô hình nếu kết quả đầu ra không chính xác.
3.  **Kích hoạt tri thức:** Các bước trung gian giúp mô hình truy xuất các kiến thức liên quan cần thiết để giải quyết vấn đề.

## Ví dụ
*   **Standard Prompt:**
    *   Q: Roger có 5 quả bóng tennis. Anh ấy mua thêm 2 hộp nữa. Mỗi hộp có 3 quả. Hỏi anh ấy có bao nhiêu quả bóng?
    *   A: 11.
*   **CoT Prompt:**
    *   Q: Roger có 5 quả bóng tennis. Anh ấy mua thêm 2 hộp nữa. Mỗi hộp có 3 quả. Hỏi anh ấy có bao nhiêu quả bóng? Hãy suy nghĩ từng bước.
    *   A: Roger bắt đầu với 5 quả bóng. 2 hộp bóng, mỗi hộp 3 quả nghĩa là 2 * 3 = 6 quả. Tổng cộng là 5 + 6 = 11 quả. Đáp án là 11.

## Biến thể
*   **Zero-shot CoT:** Chỉ cần thêm cụm từ *"Let's think step by step"* (Hãy suy nghĩ từng bước) vào cuối prompt.
*   **Few-shot CoT:** Cung cấp các ví dụ mẫu (few-shot) bao gồm cả câu hỏi và quá trình suy luận chi tiết.
