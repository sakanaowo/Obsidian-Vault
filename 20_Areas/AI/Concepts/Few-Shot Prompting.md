---
tags:
  - AI/PromptEngineering
  - AI/Technique
  - Concept
aliases:
  - In-Context Learning
created: 2026-01-04
---

### Định nghĩa

**Few-Shot Prompting** là kỹ thuật cung cấp cho model một vài ví dụ minh họa (shots) về input và output mong muốn ngay trong prompt để hướng dẫn model thực hiện tác vụ. Kỹ thuật này tận dụng khả năng **In-Context Learning** của LLM, giúp model hiểu rõ context, format và style mà không cần fine-tuning lại trọng số.

*   **Zero-shot:** Không cung cấp ví dụ nào.
*   **One-shot:** Cung cấp 1 ví dụ.
*   **Few-shot:** Cung cấp n ví dụ (thường từ 2-5).

### Cấu trúc

Một Few-Shot Prompt thường bao gồm:
1.  **Task Description:** Mô tả nhiệm vụ.
2.  **Examples:** Các cặp Input/Output mẫu.
3.  **Prompt:** Input mới cần xử lý.

### Ví dụ

```text
Nhiệm vụ: Phân loại cảm xúc (Tích cực/Tiêu cực).

Input: Món ăn này thật tuyệt vời!
Output: Tích cực

Input: Dịch vụ quá tệ, tôi sẽ không quay lại.
Output: Tiêu cực

Input: Đồ ăn bình thường, không có gì đặc sắc.
Output:
```

### Lưu ý quan trọng

*   **Lựa chọn ví dụ:** Chất lượng và sự đa dạng của ví dụ quan trọng hơn số lượng. Ví dụ cần bao quát các edge cases.
*   **Dynamic Selection:** Với context window giới hạn, có thể sử dụng semantic search (như `LengthBasedExampleSelector` trong [[LangChain]]) để chọn động các ví dụ phù hợp nhất với input hiện tại.
*   **Overfitting:** Cẩn thận model có thể copy y nguyên format của ví dụ mà bỏ qua logic thực tế nếu ví dụ quá áp đặt.
