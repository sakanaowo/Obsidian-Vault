---
tags:
  - lora
  - fine-tuning
  - llm
  - optimization
  - peft
status: in_progress
created: 2025-12-10
source: Prompt Engineering for Generative AI (James Phoenix, Mike Taylor)
---

# Low-Rank Adaptation (LoRA)

## Định nghĩa

**Low-Rank Adaptation (LoRA)** là một kỹ thuật hiệu quả để tinh chỉnh (fine-tuning) các [[Large Language Models]] (LLMs). Thay vì cập nhật toàn bộ hàng tỷ tham số của mô hình gốc (full fine-tuning) - một quá trình tốn kém và chậm chạp - LoRA đóng băng các trọng số của mô hình gốc và chỉ huấn luyện một số lượng nhỏ các tham số mới được thêm vào.

## Cơ chế hoạt động

LoRA hoạt động dựa trên giả thuyết rằng sự thay đổi trọng số trong quá trình thích ứng với tác vụ mới có "hạng thấp" (low intrinsic rank).

*   **Ma trận Hạng thấp:** LoRA chèn các cặp ma trận phân rã hạng thấp (low-rank decomposition matrices) vào các lớp của mạng nơ-ron transformer.
*   **Huấn luyện:** Trong quá trình huấn luyện, chỉ các ma trận nhỏ này được cập nhật.
*   **Hợp nhất:** Sau khi huấn luyện, các trọng số LoRA có thể được hợp nhất lại với mô hình gốc hoặc giữ riêng để tải động (plug-and-play).

## Lợi ích

*   **Hiệu quả tài nguyên:** Giảm đáng kể yêu cầu về VRAM và thời gian huấn luyện.
*   **Linh hoạt:** Cho phép tạo ra nhiều phiên bản tinh chỉnh nhỏ (adapter) cho các tác vụ khác nhau (ví dụ: một adapter cho viết code, một cho sáng tác thơ) mà không cần lưu trữ nhiều bản sao của mô hình gốc khổng lồ.
*   **Kết hợp với Quantization:** Khi kết hợp với [[Quantization]] (QLoRA), nó cho phép tinh chỉnh các mô hình khổng lồ trên một GPU đơn lẻ.

LoRA là một công nghệ cốt lõi trong sự bùng nổ của các mô hình LLM mã nguồn mở và tinh chỉnh cá nhân hóa.
