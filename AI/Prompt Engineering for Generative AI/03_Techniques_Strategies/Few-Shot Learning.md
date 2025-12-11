---
tags:
  - few-shot-learning
  - prompt-engineering
  - llm
  - machine-learning
status: in_progress
created: 2025-12-10
source:
  - Prompt Engineering for Generative AI (James Phoenix, Mike Taylor)
  - RAG roadmap.pdf
---

# Few-Shot Learning (Học với vài ví dụ)

## Định nghĩa

**Few-Shot Learning** là một kỹ thuật trong [[Prompt Engineering]] và học máy, nơi các mô hình [[Large Language Models]] (LLMs) được cung cấp một số ít ví dụ (thường là 1 đến 5 ví dụ) về tác vụ mong muốn để hướng dẫn quá trình tạo ra phản hồi. Kỹ thuật này giúp mô hình hiểu rõ hơn về định dạng, phong cách, hoặc các quy tắc cụ thể cần tuân theo, đặc biệt khi các hướng dẫn bằng lời nói (zero-shot) không đủ chi tiết.

## Phân loại Prompting dựa trên số lượng ví dụ

*   **Zero-Shot Learning (Không ví dụ):** Mô hình được yêu cầu thực hiện một tác vụ mà không được cung cấp bất kỳ ví dụ nào. Nó dựa hoàn toàn vào kiến thức đã học được trong quá trình huấn luyện.
*   **One-Shot Learning (Một ví dụ):** Mô hình được cung cấp một ví dụ duy nhất về tác vụ cần thực hiện. Ngay cả một ví dụ cũng có thể cải thiện đáng kể độ chính xác và khả năng tuân thủ hướng dẫn.
*   **Few-Shot Learning (Vài ví dụ):** Mô hình được cung cấp một vài ví dụ (thường từ 3-5 ví dụ) về tác vụ. Đây là cách tiếp cận phổ biến và hiệu quả để tối ưu hóa hành vi của mô hình, dẫn đến kết quả mong muốn hơn.

## Lợi ích của Few-Shot Learning

*   **Tăng độ chính xác và độ tin cậy:** Cung cấp ví dụ giúp mô hình hiểu rõ hơn về ý định của người dùng và các ràng buộc của tác vụ, từ đó cải thiện chất lượng và độ chính xác của đầu ra.
*   **Kiểm soát định dạng và phong cách:** Bằng cách trình bày các ví dụ về định dạng hoặc phong cách mong muốn, bạn có thể hướng dẫn mô hình tạo ra phản hồi phù hợp hơn.
*   **Giảm [[Hallucination]]:** Khi LLM được cung cấp các ví dụ cụ thể, khả năng tạo ra thông tin bịa đặt sẽ giảm đi, đặc biệt khi các ví dụ này liên quan trực tiếp đến dữ liệu nguồn.
*   **Phù hợp với tác vụ cụ thể:** Few-Shot Learning đặc biệt hữu ích khi LLM cần thực hiện các tác vụ với các yêu cầu rất cụ thể hoặc trong các miền (domain) hẹp mà dữ liệu huấn luyện chung của mô hình có thể chưa bao gồm đầy đủ.
*   **Giảm thiểu chi phí API (trong một số trường hợp):** Mặc dù việc thêm ví dụ vào prompt làm tăng độ dài prompt (và do đó có thể tăng chi phí token), nhưng nếu các ví dụ này giúp mô hình đưa ra câu trả lời chính xác và đáng tin cậy hơn ngay từ lần đầu, nó có thể tiết kiệm chi phí so với việc phải chạy lại prompt nhiều lần hoặc chỉnh sửa thủ công.

## Hạn chế và Thách thức

*   **Giới hạn về Token (Context Window Limits):** Số lượng ví dụ bạn có thể cung cấp bị giới hạn bởi [[Context Window]] của LLM. Có một sự đánh đổi giữa số lượng ví dụ và độ dài của phản hồi mong muốn.
*   **Nguy cơ Overfitting:** Các mô hình được huấn luyện trước (như GPT-4) đôi khi có thể "overfit" vào các ví dụ few-shot, khiến chúng ưu tiên các ví dụ hơn là prompt thực tế.
*   **Thiết kế ví dụ:** Việc chọn lựa các ví dụ đa dạng và mang tính hướng dẫn là rất quan trọng. Các ví dụ không phù hợp có thể dẫn đến kết quả kém.
*   **Chi phí:** Việc chạy nhiều ví dụ có thể làm tăng chi phí token, đặc biệt là trong các ứng dụng sản xuất quy mô lớn.

## Các chiến lược để tối ưu hóa Few-Shot Learning

*   **Ví dụ đa dạng:** Cung cấp một tập hợp các ví dụ đa dạng, bao gồm cả các trường hợp biên (edge cases) hoặc các kịch bản bất thường, để giúp mô hình tổng quát hóa tốt hơn.
*   **Hướng dẫn rõ ràng:** Kết hợp Few-Shot Learning với các chỉ dẫn rõ ràng để hướng dẫn mô hình cách sử dụng các ví dụ.
*   **Experimentation:** Thử nghiệm với số lượng và loại ví dụ khác nhau để tìm ra cấu hình tối ưu cho tác vụ cụ thể của bạn.
*   **Fine-tuning (Tinh chỉnh):** Đối với các tác vụ phức tạp hoặc khi Few-Shot Learning không mang lại kết quả mong muốn, việc tinh chỉnh mô hình có thể cung cấp sự hiểu biết sâu sắc hơn về tác vụ.

Few-Shot Learning là một công cụ mạnh mẽ trong hộp công cụ của prompt engineer, giúp cải thiện đáng kể hiệu suất của LLMs trong nhiều ứng dụng khác nhau.
