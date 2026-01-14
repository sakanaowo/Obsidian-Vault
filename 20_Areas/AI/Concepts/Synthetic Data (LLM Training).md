---
title: Synthetic Data (LLM Training)
aliases:
  - Synthetic data
  - Generated data
type: concept-note
tags:
  - ai
  - llm
  - data
---

# Synthetic data là gì trong huấn luyện LLM?

**Synthetic data** là dữ liệu được sinh ra (một phần hoặc toàn phần) bởi mô hình khác, thay vì thu thập trực tiếp từ thế giới (web, sách, log người dùng). Trong LLM, synthetic data thường xuất hiện dưới ba dạng chính: (i) “textbook/lesson” có tính giải thích cao; (ii) “bài tập + lời giải” để tăng tín hiệu kỹ năng; (iii) “instruction–response” để căn chỉnh hành vi. [[Textbooks Are All You Need]] là ví dụ điển hình: họ dùng GPT-3.5 để tạo giáo trình Python và tập bài tập CodeExercises, nhằm tăng mật độ tín hiệu sư phạm và giảm nhiễu so với dữ liệu repo code.

# Lợi ích thật sự: tăng mật độ tín hiệu

Điểm mạnh của synthetic data không phải là “tạo thêm token”, mà là tạo token đúng cấu trúc. Một prompt được thiết kế tốt có thể ép mô hình sinh tạo ra văn bản giải thích + code minh họa theo một chương trình sư phạm: từ khái niệm → ví dụ → biến thể → lỗi thường gặp. Khi đó, mỗi token có xu hướng mang nhiều thông tin về quan hệ nhân–quả và cấu trúc lập luận hơn token trong dữ liệu thô. Nếu coi huấn luyện là tối ưu hóa loss trên token, thì synthetic data là cách ta “định hình gradient” để nó phục vụ kỹ năng ta muốn.

# Rủi ro: đồng dạng hóa và suy giảm đa dạng

Synthetic data dễ bị lặp và đồng dạng vì mô hình sinh thường đi theo đường xác suất cao nhất. Nếu không có cơ chế tiêm đa dạng (ràng buộc ngẫu nhiên, ép phủ chủ đề, thay đổi đối tượng người học), tập dữ liệu sẽ có nhiều mẫu na ná nhau; mô hình học rất nhanh những khuôn mẫu bề mặt đó và trở nên “giòn” khi gặp trường hợp khác. Một rủi ro sâu hơn là hiện tượng “model collapse/dementia” (mô hình học trên dữ liệu do mô hình khác sinh quá nhiều): tri thức có thể co lại quanh các mẫu phổ biến, làm giảm khả năng tổng quát và làm nghèo phân phối đầu ra.

> [!NOTE] Quy tắc thực hành
> Synthetic data chỉ thực sự hữu ích khi bạn kiểm soát được đa dạng và chất lượng: cần có chiến lược curation, de-dup, và kiểm tra contamination nếu benchmark có nguy cơ rò rỉ.

