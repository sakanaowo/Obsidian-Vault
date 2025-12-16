---
tags:
  - prompt-engineering
  - technique
  - automation
status: done
created_date: 2025-12-11
---

# Meta Prompting

**Meta Prompting** là kỹ thuật sử dụng một mô hình ngôn ngữ lớn ([[Large Language Models]]) để tạo ra, tối ưu hóa, hoặc viết lại các câu lệnh (prompts) cho chính nó hoặc cho các mô hình AI khác. Nói cách khác, đây là việc dùng AI để điều khiển AI.

## Các ứng dụng chính

### 1. Prompt Generation (Tạo Prompt)
Người dùng cung cấp một yêu cầu sơ sài, và LLM sẽ viết lại thành một prompt chi tiết, đầy đủ cấu trúc và tối ưu hơn.
*   *Ví dụ:* Người dùng nhập "Vẽ con mèo", Meta Prompt sẽ biến nó thành "A highly detailed oil painting of a fluffy cat sitting on a windowsill, golden hour lighting, 4k resolution..." cho [[Midjourney]].

### 2. Prompt Optimization (Tối ưu Prompt)
Sử dụng LLM để tinh chỉnh một prompt hiện có nhằm đạt kết quả tốt hơn.
*   *Quy trình:* Đưa prompt cũ và kết quả (chưa tốt) cho LLM -> Yêu cầu LLM phân tích lý do và viết lại prompt mới.

### 3. Creating Personas/Systems
Yêu cầu LLM đóng vai một "Chuyên gia Prompt Engineering" để thiết kế các hệ thống prompt phức tạp cho các tác vụ cụ thể (như viết code, dịch thuật).

## Lợi ích
*   **Tiết kiệm thời gian:** Giảm thiểu quy trình thử-sai (trial and error) thủ công.
*   **Chất lượng cao:** Tận dụng kiến thức của LLM về cách nó "muốn" được ra lệnh để tạo ra các chỉ dẫn hiệu quả nhất.
*   **Tự động hóa:** Cho phép xây dựng các hệ thống AI tự hành có khả năng tự điều chỉnh hành vi.
