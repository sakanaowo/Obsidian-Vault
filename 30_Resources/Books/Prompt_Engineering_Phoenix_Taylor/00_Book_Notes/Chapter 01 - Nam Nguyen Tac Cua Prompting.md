---
tags:
  - prompt-engineering
  - generative-ai
  - methodology
  - llm
  - best-practices
status: processed
created: 2025-12-10
source: Prompt Engineering for Generative AI (James Phoenix, Mike Taylor)
---

# Chương 1: Năm Nguyên Tắc Của Prompting

## Tổng quan về Prompt Engineering

**Prompt Engineering** là quy trình mang tính hệ thống nhằm thiết kế các đầu vào (inputs) để nhận được kết quả (outputs) mong muốn và đáng tin cậy từ các mô hình AI. Một **Prompt** đóng vai trò là tập hợp các chỉ dẫn mà mô hình sử dụng để dự đoán phản hồi tiếp theo, dù là văn bản từ [[Large Language Models]] (LLMs) như ChatGPT hay hình ảnh từ [[Diffusion Models]] như Midjourney.

Các mô hình ngôn ngữ hoạt động dựa trên xác suất (probabilistic nature). Việc thay đổi nhỏ trong prompt có thể dẫn đến sự thay đổi lớn trong xác suất của token tiếp theo, từ đó thay đổi hoàn toàn kết quả đầu ra. Do đó, mục tiêu của prompt engineering là tối ưu hóa prompt để giảm thiểu sự ngẫu nhiên không mong muốn và tăng cường độ chính xác.

## Năm Nguyên Tắc Cốt Lõi (The Five Principles)

Để chuyển đổi từ các tương tác ngây thơ (naive prompting) sang các hệ thống AI sẵn sàng cho sản xuất (production-ready), cần tuân thủ 5 nguyên tắc sau:

### 1. Give Direction (Đưa ra Định hướng)
Mô hình AI cần bối cảnh rõ ràng để hiểu ý định của người dùng. Nếu thiếu định hướng, mô hình sẽ dựa vào dữ liệu huấn luyện trung bình, thường dẫn đến các câu trả lời chung chung.

*   **Role-Playing (Đóng vai):** Yêu cầu AI đóng vai một chuyên gia hoặc một nhân vật cụ thể (ví dụ: Steve Jobs, Elon Musk) giúp định hình văn phong và quan điểm của câu trả lời.
*   **Pre-warming/Internal Retrieval:** Yêu cầu AI liệt kê các thực hành tốt nhất (best practices) về một chủ đề trước khi thực hiện tác vụ chính. Điều này giúp mô hình "truy xuất nội bộ" các kiến thức liên quan trước khi xử lý.
*   **Context Injection:** Cung cấp thông tin bổ sung, quy tắc ngành, hoặc dữ liệu cụ thể vào prompt để hướng dẫn mô hình.

### 2. Specify Format (Quy định Định dạng)
AI là các "biên dịch viên vạn năng" (universal translators), có thể chuyển đổi giữa ngôn ngữ tự nhiên và các cấu trúc dữ liệu. Việc quy định rõ định dạng đầu ra giúp tích hợp AI vào các hệ thống phần mềm dễ dàng hơn.

*   **Các định dạng phổ biến:** [[JSON]], [[YAML]], Markdown, danh sách phân cách bằng dấu phẩy (CSV).
*   **Kỹ thuật:** Cung cấp ví dụ về cấu trúc mong muốn hoặc chỉ thị rõ ràng (ví dụ: "Return only JSON"). Điều này giúp tránh việc phải xử lý hậu kỳ (post-processing) phức tạp hoặc gặp lỗi khi parse dữ liệu.

### 3. Provide Examples (Cung cấp Ví dụ)
Đây là kỹ thuật **Few-Shot Learning**. Việc cung cấp các ví dụ (input-output pairs) giúp mô hình hiểu rõ hơn về tác vụ cần thực hiện, đặc biệt là khi các chỉ dẫn bằng lời nói (zero-shot) không đủ để mô tả sự phức tạp.

*   **Zero-shot:** Không cung cấp ví dụ, chỉ có chỉ dẫn.
*   **One-shot:** Cung cấp một ví dụ duy nhất.
*   **Few-shot:** Cung cấp nhiều ví dụ (thường từ 3-5).
*   **Lưu ý:** Có sự đánh đổi giữa độ tin cậy và tính sáng tạo. Cung cấp quá nhiều ví dụ tương tự nhau có thể làm giảm tính đa dạng của đầu ra (overfitting vào ví dụ).

### 4. Evaluate Quality (Đánh giá Chất lượng)
Để tối ưu hóa prompt, cần có cơ chế đánh giá đầu ra một cách định lượng hoặc định tính. Không thể cải thiện những gì không thể đo lường.

*   **Phương pháp thủ công:** Sử dụng hệ thống đánh giá đơn giản (Thumbs up/down) hoặc thang điểm.
*   **Phương pháp tự động:** Sử dụng các mô hình mạnh hơn (như GPT-4) để chấm điểm các mô hình nhỏ hơn, hoặc so sánh với "ground truth" (kết quả chuẩn) trong các bài toán phân loại/trích xuất.
*   **Mục tiêu:** Đánh giá các yếu tố như độ chính xác, độ trôi chảy (fluency), và khả năng tuân thủ chỉ dẫn.

### 5. Divide Labor (Phân chia Công việc)
Thay vì cố gắng giải quyết một vấn đề phức tạp bằng một prompt duy nhất (dễ dẫn đến [[Hallucination]] hoặc suy luận kém), hãy chia nhỏ tác vụ thành nhiều bước hoặc chuỗi các prompt (Prompt Chaining).

*   **Chain of Thought (CoT):** Yêu cầu mô hình "suy nghĩ từng bước" (Let's think step by step). Kỹ thuật này kích hoạt khả năng suy luận logic của mô hình, giúp giải quyết các bài toán toán học hoặc logic phức tạp.
*   **Task Decomposition:** Chia quy trình thành các công đoạn: Nghiên cứu -> Lập dàn ý -> Viết nháp -> Đánh giá -> Chỉnh sửa.
*   **Self-Correction:** Yêu cầu mô hình tự đánh giá kết quả của chính nó trước khi đưa ra câu trả lời cuối cùng.

## Kết luận
Năm nguyên tắc này không phải là các thủ thuật ngắn hạn (hacks) mà là các quy tắc nền tảng, có thể áp dụng cho hầu hết các mô hình Generative AI hiện tại và tương lai (bao gồm cả Text và Image generation). Việc áp dụng chúng giúp kiểm soát tính ngẫu nhiên của mô hình, giảm chi phí token, và tăng độ tin cậy cho hệ thống.

---
**Liên kết tham khảo:**
- [[Large Language Models]]
- [[Chain of Thought]]
- [[Few-Shot Learning]]
- [[Hallucination]]
- [[Diffusion Models]]
