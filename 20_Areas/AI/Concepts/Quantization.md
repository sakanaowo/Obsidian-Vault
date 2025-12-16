---
tags:
  - quantization
  - optimization
  - llm
  - machine-learning
status: in_progress
created: 2025-12-10
source: Prompt Engineering for Generative AI (James Phoenix, Mike Taylor)
---

# Quantization (Lượng tử hóa)

## Định nghĩa

**Quantization** là một kỹ thuật tối ưu hóa trong học máy và [[Large Language Models]] (LLMs) giúp giảm kích thước bộ nhớ và yêu cầu tính toán của mô hình bằng cách giảm độ chính xác của các tham số (trọng số và bias).

## Cơ chế hoạt động

Thông thường, các mô hình LLM được huấn luyện và lưu trữ với độ chính xác dấu phẩy động 32-bit (FP32) hoặc 16-bit (FP16). Quantization chuyển đổi các giá trị này sang các định dạng có độ chính xác thấp hơn, chẳng hạn như số nguyên 8-bit (INT8) hoặc thậm chí 4-bit (INT4).

*   **Giảm kích thước:** Giảm đáng kể dung lượng RAM cần thiết để tải mô hình. Ví dụ, một mô hình 7 tỷ tham số ở FP16 cần khoảng 14GB VRAM, nhưng ở 4-bit chỉ cần khoảng 4GB.
*   **Tăng tốc độ:** Các phép toán trên số nguyên (integer) thường nhanh hơn so với số thực dấu phẩy động (floating point) trên phần cứng hiện đại.

## Lợi ích và Đánh đổi

*   **Lợi ích:** Cho phép chạy các mô hình LLM mạnh mẽ trên phần cứng tiêu dùng (consumer-grade hardware) như GPU của máy tính cá nhân hoặc thậm chí CPU, thay vì phải sử dụng các trung tâm dữ liệu đắt tiền.
*   **Đánh đổi:** Việc giảm độ chính xác có thể dẫn đến giảm nhẹ hiệu suất hoặc độ chính xác của mô hình. Tuy nhiên, các kỹ thuật quantization hiện đại (như QLoRA) đã giảm thiểu đáng kể sự sụt giảm này.

## Ứng dụng

Quantization là chìa khóa để dân chủ hóa quyền truy cập vào LLM, cho phép cộng đồng mã nguồn mở tinh chỉnh và triển khai các mô hình như [[Meta Llama]] trên các thiết bị cục bộ.
