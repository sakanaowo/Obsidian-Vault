---
title: Supervised Fine-Tuning (SFT)
aliases:
  - SFT
type: concept-note
tags:
  - ai
  - llm
  - alignment
---

# SFT là gì (và vì sao nó thường là bước “đủ dùng”)

**Supervised Fine-Tuning (SFT)** là fine-tune LLM bằng loss giám sát (thường là cross-entropy/next-token) trên dữ liệu cặp *input → output*. Trong bối cảnh alignment, input thường là prompt hoặc lịch sử hội thoại với role user/assistant; output là câu trả lời mẫu theo chuẩn mong muốn. SFT đơn giản nhưng cực kỳ thực dụng vì nó cung cấp tín hiệu trực tiếp, ổn định và dễ kiểm soát: bạn có thể quyết định phong cách, mức độ chi tiết, cách từ chối, và khuôn trình bày chỉ bằng cách thay đổi dữ liệu.

Điểm “first principles” của SFT là: nó không cần một hàm thưởng hay vòng lặp RL, nên giảm rủi ro reward hacking và giảm chi phí thu thập preference. Khi mô hình nền đã mạnh, SFT thường đóng vai trò “mở khóa” hành vi trợ lý bằng cách dạy mô hình cách phản hồi nhất quán, như minh họa trong [[LIMA - Less Is More for Alignment]].

> [!NOTE] Giới hạn
> SFT buộc mô hình bắt chước đáp án mẫu; nếu dữ liệu không bao phủ đủ trường hợp ranh giới (ví dụ prompt nguy hiểm, yêu cầu mơ hồ), mô hình dễ suy diễn sai. Ngoài ra, SFT khó tối ưu các thuộc tính “mềm” (như lịch sự vừa đủ, an toàn nhưng không né tránh quá mức) nếu đáp án mẫu không nhất quán.

