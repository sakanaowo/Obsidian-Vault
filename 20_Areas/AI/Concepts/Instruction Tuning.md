---
title: Instruction Tuning
aliases:
  - Instruction fine-tuning
type: concept-note
tags:
  - ai
  - llm
  - alignment
---

# Instruction Tuning là gì và nó “thay đổi mô hình” ở đâu?

**Instruction Tuning** là huấn luyện giám sát trên các cặp *instruction/prompt → response* để buộc LLM hành xử như một trợ lý: hiểu yêu cầu, chọn giọng điệu phù hợp, trình bày có cấu trúc, và ưu tiên “giải quyết vấn đề cho người dùng” hơn là tiếp tục văn bản theo kiểu ngẫu nhiên. Về mặt tối ưu, nó vẫn là next-token prediction (hoặc cross-entropy) nhưng được áp lên các chuỗi đối thoại đã được “đóng vai” sẵn, nên gradient hướng mô hình vào một vùng tham số mà ở đó token tiếp theo tương ứng với hành vi trợ lý (giải thích, lập kế hoạch, hỏi lại…).

Điều quan trọng là: instruction tuning không nhất thiết “bơm” thêm kiến thức thế giới. Nó thường hoạt động như một cơ chế *định tuyến hành vi*: cùng một năng lực tiềm ẩn từ pretraining, nhưng mô hình học cách kích hoạt đúng chế độ trả lời khi gặp tín hiệu “bạn là trợ lý”. Đây là ý mà [[LIMA - Less Is More for Alignment]] khai thác mạnh: chỉ với 1.000 ví dụ chất lượng cao, mô hình nền đủ mạnh có thể học format/role để bộc lộ tri thức đã có.

# Ví dụ tối giản để hiểu cơ chế

Giả sử pretraining đã giúp mô hình có thể viết một đoạn giải thích toán học, nhưng khi gặp câu hỏi “giải thích định lý Bayes”, mô hình có thể chọn trả lời ngắn, hoặc lạc sang văn phong blog, hoặc đi kể chuyện. Instruction tuning đưa vào nhiều ví dụ mà ở đó câu trả lời “đúng chuẩn trợ lý”: có định nghĩa, có ví dụ, có cảnh báo nhầm lẫn. Khi tối ưu loss, mô hình học rằng chuỗi token mang cấu trúc đó là “phần tiếp theo hợp lý” trong bối cảnh instruction.

> [!NOTE] Hệ quả thực hành
> Nếu dữ liệu instruction có phong cách không nhất quán (lúc thì chat, lúc thì diễn đàn, lúc thì trích dẫn link), mô hình sẽ học một “giọng nói lai” và khó kiểm soát. Vì vậy, tính nhất quán về style thường quan trọng không kém số lượng ví dụ.

