---
tags:
  - AI/Modeling
  - AI/Concept
  - Concept
aliases:
  - Generative Models
  - Discriminative Models
created: 2026-01-04
---

### Định nghĩa & Sự khác biệt cốt lõi

Trong Machine Learning (đặc biệt là bài toán phân loại văn bản), có hai triết lý chính để xây dựng mô hình:

#### 1. Discriminative Models (Mô hình Phân biệt)
*   **Mục tiêu:** Học trực tiếp ranh giới quyết định (decision boundary) giữa các lớp.
*   **Toán học:** Mô hình hóa xác suất có điều kiện **$P(y|x)$**.
    *   *Hỏi:* "Cho đầu vào $x$ này, xác suất nó thuộc nhãn $y$ là bao nhiêu?"
*   **Ví dụ:** Logistic Regression, SVM, Neural Networks (standard), Conditional Random Fields (CRF).
*   **Analogy:** Giống như một giáo viên chấm bài thi. Giáo viên chỉ quan tâm bài làm này đúng hay sai (nhãn), chứ không cần biết học sinh đã học như thế nào để viết ra bài đó.

#### 2. Generative Models (Mô hình Sinh)
*   **Mục tiêu:** Học cấu trúc và cách dữ liệu được sinh ra.
*   **Toán học:** Mô hình hóa xác suất đồng thời **$P(x, y)$** (hoặc $P(x|y) \cdot P(y)$ theo định lý Bayes).
    *   *Hỏi:* "Xác suất để cặp dữ liệu $(x, y)$ này cùng xuất hiện là bao nhiêu?"
*   **Ví dụ:** Naive Bayes, Hidden Markov Models (HMM), GANs, Diffusion Models, Large Language Models (GPT).
*   **Analogy:** Giống như việc học ngôn ngữ. Bạn phải hiểu cách ngữ pháp và từ vựng kết hợp với nhau để tạo ra (sinh ra) một câu hoàn chỉnh.

### So sánh chi tiết

| Đặc điểm | Generative ($P(x, y)$) | Discriminative ($P(y\|x)$) |
| :--- | :--- | :--- |
| **Trọng tâm** | Hiểu dữ liệu ($x$). | Phân biệt nhãn ($y$). |
| **Dữ liệu** | Cần nhiều dữ liệu hơn để học phân phối. | Thường cần ít dữ liệu hơn để tìm biên. |
| **Outliers** | Xử lý tốt hơn (vì học được phân phối tổng thể). | Dễ bị ảnh hưởng bởi outliers gần biên. |
| **Khả năng** | Có thể sinh ra dữ liệu mới (Generation). | Chỉ có thể phân loại/dự đoán. |
| **Điển hình NLP**| Naive Bayes, n-gram, LLMs. | Logistic Regression, BERT (khi finetune). |

### Tại sao lại quan trọng trong NLP?

*   **Discriminative** thường tốt hơn cho các bài toán phân loại (Text Classification, Sentiment Analysis) vì chúng tập trung tối ưu hóa độ chính xác dự đoán.
*   **Generative** là nền tảng của các mô hình ngôn ngữ hiện đại (GPT, Claude, Gemini) vì khả năng sinh văn bản (Text Generation) và hiểu ngữ cảnh sâu sắc hơn.
