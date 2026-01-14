---
title: Data Curation
aliases:
  - Data selection
  - Data filtering
type: concept-note
tags:
  - ai
  - llm
  - data
---

# Data Curation là gì (định nghĩa theo cơ chế)

**Data Curation** trong huấn luyện LLM không đơn giản là “làm sạch dữ liệu” theo nghĩa bỏ rác/loại trùng. Về cơ chế, nó là quá trình *thiết kế phân phối dữ liệu huấn luyện* sao cho mỗi token cung cấp tín hiệu đúng với kỹ năng ta muốn mô hình học, trong khi giảm tín hiệu nhiễu gây lệch hành vi. Nếu pretraining là bài toán tối đa hóa xác suất chuỗi ký tự, thì curation là cách ta chọn “thế giới văn bản” để mô hình sống trong đó: thế giới này có thể thúc đẩy mô hình học lập luận rõ ràng, hoặc ngược lại làm mô hình quen với boilerplate, mảnh code thiếu ngữ cảnh, và phong cách viết vô ích.

Điểm then chốt của curation là: mô hình không “biết” đâu là ví dụ sư phạm; nó chỉ tối ưu loss. Vì vậy, curation là cách ta can thiệp vào gradient bằng cách thay đổi dữ liệu đầu vào. Khi dữ liệu có chất lượng “giáo trình” (rõ ràng, tự-contained, cân bằng chủ đề), ta thường có thể giảm mạnh quy mô token mà vẫn đạt được tiến bộ lớn, như lập luận trong [[Textbooks Are All You Need]].

# Vì sao curation có thể mạnh hơn “thêm dữ liệu”?

Một cách nhìn theo first principles là coi huấn luyện như việc ước lượng một hàm ánh xạ $f$: từ ngữ cảnh sang token kế tiếp. Nếu dữ liệu chứa nhiều mẫu “không dạy gì” (ví dụ cấu hình, constant, scaffolding), mô hình sẽ học tốt những quy luật rẻ tiền đó vì chúng xuất hiện dày đặc, trong khi tín hiệu về lập luận/thuật toán lại hiếm và bị pha loãng. Khi đó, thêm token nhiều khả năng chỉ tăng độ chắc chắn của các quy luật rẻ tiền. Ngược lại, curation tăng tỷ lệ mẫu “đắt giá” nên mỗi bước cập nhật có nhiều thông tin hơn; vì thế đường cong hiệu năng có thể dịch lên dù token ít hơn.

# Kỹ thuật curation điển hình (khung thao tác)

Trong thực tế, curation thường kết hợp nhiều tầng bộ lọc. Một tầng thô có thể dựa trên heuristic (độ dài, tỷ lệ ký tự lạ, ngôn ngữ, license, dedup). Một tầng “giá trị” hơn là dựa trên mô hình: gán nhãn hoặc chấm điểm “giá trị giáo dục”, “tính hữu ích”, “tính an toàn”, rồi học một classifier/ranker để lọc. [[Textbooks Are All You Need]] là ví dụ điển hình: họ dùng GPT-4 để gán nhãn một tập nhỏ, sau đó huấn luyện bộ phân loại để lọc The Stack/StackOverflow thành một tập nhỏ nhưng “đáng học”.

> [!NOTE] Bẫy thường gặp
> Nếu tiêu chí lọc quá hẹp, curation có thể tạo ra phân phối “đẹp nhưng nghèo”: mô hình học rất tốt những gì bạn đo/định nghĩa, nhưng mất độ bao phủ, và trở nên giòn khi gặp ngoài phân phối. Vì vậy, curation luôn phải đi cùng câu hỏi “mình đang tối ưu cho kỹ năng nào, và mình chấp nhận đánh đổi gì?”.

