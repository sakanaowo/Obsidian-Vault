---
tags:
  - nlp
  - challenges
status: in_progress
created_date: 2026-01-13
source:
  - assets/Library/Natural-Language-Processing-PTIT-Nguyen-Thi-Mai-Trang.pdf
---

# NLP Challenges

Các “thách thức” của NLP không chỉ là danh sách khó khăn kỹ thuật; chúng phản ánh việc ngôn ngữ tự nhiên **ẩn** nhiều biến quan trọng mà người giao tiếp ngầm hiểu nhưng không nói ra. Slide tóm tắt bằng câu hỏi: “who did what to whom” — nhưng thực tế còn thêm “khi nào/ở đâu/vì sao/điều kiện nào” và đặc biệt là “người nói muốn gì”.

**Tri thức thế giới** và **ngữ cảnh** là bắt buộc vì ngôn ngữ chỉ mã hóa bề mặt. Câu “Can you tell me the time?” minh họa sự khác biệt giữa nghĩa đen (hỏi khả năng) và hành động giao tiếp (yêu cầu cung cấp giờ), cho thấy NLP phải xử lý cả lớp ngữ dụng.

Về mặt học máy, NLP đối diện các giới hạn cấu trúc:

- **Scale**: không gian từ/ý nghĩa/ngữ cảnh rất lớn, nên biểu diễn và suy luận trở thành bài toán tối ưu hóa dưới ràng buộc compute.
- **Sparsity**: dữ liệu quan sát hữu hạn, nhiều từ/khái niệm “chưa từng gặp”; cần cơ chế khái quát (subword, embeddings, pretrained).
- **Long-range correlations**: thông tin quyết định có thể nằm rất xa trong câu/tài liệu.
- **Tương tác tri thức**: nhiều lớp tri thức (từ vựng, cú pháp, ngữ nghĩa, ngữ cảnh) tương tác khiến suy luận tường minh dễ bùng nổ độ phức tạp.

> [!NOTE] Suy luận thêm — Điểm nghẽn thiết kế: biểu diễn nghĩa
> Nếu biểu diễn quá nghèo, mô hình không thể phân biệt các diễn giải quan trọng; nếu biểu diễn quá giàu, mô hình không học nổi hoặc suy luận quá đắt. Vì vậy, thiết kế biểu diễn (embedding, cấu trúc, memory, retrieval) thường là “đòn bẩy” lớn nhất trong hệ NLP hiện đại.

