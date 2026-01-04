---
tags:
  - AI/Evaluation
  - AI/Technique
  - Concept
aliases:
  - Model Eval
  - LLM-as-a-Judge
created: 2026-01-04
---

### Định nghĩa

**LLM Evaluation** (Đánh giá LLM) là quá trình đo lường hiệu suất và chất lượng đầu ra của [[Large Language Models]]. Khác với phần mềm truyền thống có đầu ra xác định (đúng/sai rõ ràng), LLM sinh ra ngôn ngữ tự nhiên nên việc đánh giá thường mang tính chủ quan và khó khăn hơn (ví dụ: đánh giá tính sáng tạo, độ trôi chảy).

### Các phương pháp đánh giá

1.  **Deterministic Metrics (Định lượng xác định):**
    *   Sử dụng cho các task có câu trả lời chính xác (ví dụ: extraction).
    *   Regex match, Exact match, JSON validation.
    *   Code execution (chạy code do model viết để xem có lỗi không).

2.  **Distance-based Metrics:**
    *   So sánh độ tương đồng giữa output và reference answer (ground truth).
    *   **Levenshtein Distance:** Đếm số ký tự khác biệt.
    *   **Embedding Similarity (Cosine Similarity):** So sánh ý nghĩa ngữ nghĩa (semantic) thông qua vector space.

3.  **Human Evaluation:**
    *   Con người trực tiếp chấm điểm. Chính xác nhất nhưng tốn kém, chậm và khó mở rộng (scale).

4.  **Model-based Evaluation (LLM-as-a-Judge):**
    *   Sử dụng một model mạnh hơn (như GPT-4) để chấm điểm output của model khác.
    *   **Pairwise Comparison:** Đưa 2 output cho model giám khảo và hỏi cái nào tốt hơn + lý do.
    *   **Criteria-based:** Chấm điểm theo thang điểm cụ thể (ví dụ: 1-5 về độ hữu ích).

### Framework hỗ trợ

Các framework như [[LangChain]] (LangSmith) hay Ragas cung cấp các công cụ tự động hóa quy trình đánh giá này, giúp xây dựng các bộ test dataset và theo dõi hiệu suất model theo thời gian.
