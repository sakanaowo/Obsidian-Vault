---
title: Reinforcement Learning from Human Feedback (RLHF)
aliases:
  - RLHF
type: concept-note
tags:
  - ai
  - llm
  - alignment
---

# RLHF là gì (một pipeline tối ưu hành vi)

**RLHF** là một cách căn chỉnh LLM bằng cách biến “sở thích của con người” thành tín hiệu thưởng/phạt, rồi dùng học tăng cường để tối ưu chính sách sinh văn bản. Cấu trúc kinh điển có ba bước. Bước một là [[Supervised Fine-Tuning (SFT)]] để mô hình biết trả lời theo instruction ở mức cơ bản. Bước hai là huấn luyện **reward model**: cho con người (hoặc annotator) so sánh hai đáp án, học một hàm $r(x, y)$ ước lượng mức “được thích” của câu trả lời $y$ cho prompt $x$. Bước ba là chạy RL (thường là PPO) để tăng kỳ vọng reward, tức tối ưu mô hình sao cho nó sinh câu trả lời mà reward model chấm cao.

Điểm khác biệt bản chất so với SFT là: SFT bắt mô hình bắt chước đáp án mẫu, còn RLHF cho phép mô hình *tự tìm* đáp án khác đáp án mẫu nhưng vẫn được chấm cao theo sở thích. Vì vậy RLHF thường được dùng để tinh chỉnh các thuộc tính khó “đóng gói” thành một đáp án chuẩn, như lịch sự, an toàn, mức độ chi tiết, hay tránh toxic.

# Vì sao RLHF vừa mạnh vừa nguy hiểm?

RLHF mạnh vì nó biến một mục tiêu mơ hồ (“hãy hữu ích”) thành một hàm mục tiêu có thể tối ưu; nhưng nó nguy hiểm vì reward model chỉ là một mô hình xấp xỉ. Khi tối ưu mạnh, LLM có thể học “lách” reward (reward hacking): tạo ra văn phong thuyết phục, dài dòng, hoặc né tránh rủi ro theo cách làm giảm tính đúng. Điều này giải thích vì sao RLHF có thể cải thiện cảm giác “hữu ích” nhưng đôi khi làm giảm trung thực hoặc làm tăng mức tự tin sai.

> [!NOTE] Liên hệ với LIMA
> [[LIMA - Less Is More for Alignment]] cho thấy trong nhiều tình huống “helpfulness”, SFT chất lượng cao có thể đã đủ cạnh tranh, gợi ý rằng RLHF không phải luôn là yêu cầu tối thiểu để có câu trả lời tốt. Tuy nhiên, safety và các ràng buộc giá trị thường là nơi RLHF (hoặc các biến thể như Constitutional AI, DPO/RLAIF) vẫn đóng vai trò quan trọng.

