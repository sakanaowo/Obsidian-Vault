---
title: Superficial Alignment Hypothesis
aliases:
  - Superficial alignment
type: concept-note
tags:
  - ai
  - llm
  - alignment
---

# Giả thuyết “alignment bề mặt” là gì?

**Superficial Alignment Hypothesis** (được nêu trong [[LIMA - Less Is More for Alignment]]) phát biểu rằng phần lớn **kiến thức** và **năng lực** của LLM được học trong giai đoạn pretraining, còn alignment (instruction tuning, preference tuning, RLHF) chủ yếu dạy mô hình *chọn đúng “kênh hành vi”* khi tương tác với người dùng: giọng điệu, cấu trúc, mức độ chi tiết, và các chuẩn mực phản hồi. Nói cách khác, alignment không nhất thiết tạo ra năng lực mới tương ứng với tri thức hoàn toàn mới; nó điều khiển việc “bộc lộ” năng lực đã có bằng cách thay đổi phân phối đầu ra.

Giả thuyết này hấp dẫn vì nó đưa alignment về đúng dạng vấn đề: không phải “nhồi thêm kiến thức”, mà là “thiết kế giao diện hành vi” cho một mô hình đã biết nhiều. Khi nhìn như vậy, dữ liệu alignment chất lượng cao sẽ giống như một bộ ví dụ về “định dạng đầu ra đúng”: nếu mô hình đã biết cách lập kế hoạch du lịch, thứ nó cần là biết rằng với prompt dạng này, nên trả lời theo dạng gạch mục, có cảnh báo, có giả định… chứ không phải học lại địa lý.

> [!NOTE] Suy luận thêm (cần kiểm chứng thêm)
> Giả thuyết này có thể đúng hơn với mô hình nền rất mạnh và đúng phân phối, và yếu hơn khi mô hình nền thiếu năng lực hoặc khi domain quá mới. Trong trường hợp đó, alignment có thể đóng vai trò như “đào sâu” kỹ năng (skill learning) chứ không chỉ là format learning.

