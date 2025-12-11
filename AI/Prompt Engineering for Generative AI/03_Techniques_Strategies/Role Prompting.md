---
tags:
  - prompt-engineering
  - technique
  - persona
status: done
created_date: 2025-12-11
---

# Role Prompting

**Role Prompting** (còn gọi là Persona Prompting) là kỹ thuật gán một vai trò, chức danh, hoặc nhân cách cụ thể cho mô hình AI trong câu lệnh đầu vào. Kỹ thuật này giúp định hình giọng điệu (tone), phong cách ngôn ngữ, và góc nhìn giải quyết vấn đề của mô hình.

## Cấu trúc cơ bản
Thường bắt đầu bằng các cụm từ như:
*   "Act as a..." (Hãy đóng vai là...)
*   "You are a..." (Bạn là một...)
*   "Pretend to be..." (Hãy giả vờ là...)

## Tại sao nó hiệu quả?
Các [[Large Language Models]] được huấn luyện trên dữ liệu khổng lồ chứa nhiều phong cách và chuyên môn khác nhau. Việc gán vai trò giúp "kích hoạt" vùng kiến thức cụ thể liên quan đến vai trò đó trong không gian tiềm ẩn (latent space) của mô hình.

## Ví dụ
*   **Không có Role:** "Làm thế nào để giảm cân?" -> *Câu trả lời chung chung về calo và tập thể dục.*
*   **Có Role:** "Hãy đóng vai một chuyên gia dinh dưỡng thể thao cho vận động viên Olympic. Làm thế nào để giảm cân mà không mất cơ bắp?" -> *Câu trả lời chuyên sâu về macro, protein, và chu kỳ tập luyện.*

## Các rủi ro
*   **Stereotyping (Định kiến):** Gán các vai trò nhất định có thể khiến mô hình bộc lộ các định kiến xã hội có trong dữ liệu huấn luyện.
*   **Hallucination:** Nếu gán một vai trò quá cụ thể hoặc giả tưởng, mô hình có thể bịa đặt thông tin để "nhập vai" tốt hơn.
