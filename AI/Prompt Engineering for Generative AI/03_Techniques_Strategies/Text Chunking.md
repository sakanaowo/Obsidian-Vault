---
tags:
  - prompt-engineering
  - nlp
  - data-processing
  - rag
status: done
created_date: 2025-12-11
---

# Text Chunking

**Text Chunking** (Phân mảnh văn bản) là quá trình chia nhỏ các văn bản dài thành các đoạn nhỏ hơn, dễ quản lý hơn để phù hợp với giới hạn [[Context Window]] của các mô hình ngôn ngữ lớn ([[Large Language Models]]). Đây là bước quan trọng trong các hệ thống **RAG** (Retrieval Augmented Generation).

## Tại sao cần Chunking?
1.  **Giới hạn Token:** LLM có giới hạn cứng về số lượng token đầu vào (ví dụ: 4k, 8k, 32k).
2.  **Mất mát thông tin:** Khi văn bản quá dài, mô hình có xu hướng "quên" thông tin ở giữa (hiện tượng "Lost in the Middle").
3.  **Chi phí:** Xử lý ít token hơn giúp tiết kiệm chi phí API và giảm độ trễ.

## Các chiến lược Chunking

### 1. By Character/Length (Theo ký tự/độ dài)
Chia văn bản dựa trên số lượng ký tự cố định.
*   *Ưu điểm:* Đơn giản, dễ cài đặt.
*   *Nhược điểm:* Dễ cắt ngang từ hoặc câu, làm mất ngữ nghĩa.

### 2. By Sentence/Paragraph (Theo câu/đoạn)
Sử dụng các thư viện NLP (như SpaCy) để nhận diện ranh giới câu.
*   *Ưu điểm:* Giữ nguyên vẹn ý nghĩa của câu.
*   *Nhược điểm:* Khó kiểm soát độ dài chính xác của mỗi chunk.

### 3. Sliding Window (Cửa sổ trượt)
Tạo các chunk có phần chồng lấp (overlap) với nhau (ví dụ: Chunk 1 từ 0-1000, Chunk 2 từ 800-1800).
*   *Ưu điểm:* Bảo toàn ngữ cảnh ở các điểm nối, tránh việc thông tin quan trọng bị cắt đôi.

### 4. Token-based (Dựa trên Token)
Sử dụng bộ đếm token của chính mô hình (ví dụ: **Tiktoken** cho GPT) để chia cắt.
*   *Ưu điểm:* Tối ưu hóa tuyệt đối cho context window của mô hình, tránh lãng phí.

## Best Practice
Kết hợp **Token-based chunking** với **Sliding Window** (overlap khoảng 10-20%) thường mang lại kết quả tốt nhất cho việc truy xuất thông tin.
