---
tags:
  - anthropic-claude
  - llm
  - ai-safety
  - constitutional-ai
  - anthropic
status: in_progress
created: 2025-12-10
source: Prompt Engineering for Generative AI (James Phoenix, Mike Taylor)
---

# Anthropic Claude

## Tổng quan

**Claude** là dòng mô hình ngôn ngữ lớn được phát triển bởi **Anthropic**, một công ty AI tập trung vào an toàn và đạo đức, được thành lập bởi các cựu nhân viên của OpenAI. Claude được định vị là một đối thủ cạnh tranh an toàn hơn, ít độc hại hơn so với các mô hình khác.

## Phương pháp tiếp cận: Constitutional AI

Điểm đặc biệt của Claude là việc sử dụng phương pháp **Constitutional AI** (AI Hiến pháp). Thay vì chỉ dựa vào phản hồi của con người (RLHF), Claude được huấn luyện để tuân theo một bộ quy tắc hoặc nguyên tắc đạo đức (gọi là "hiến pháp") để tự điều chỉnh hành vi của mình, giảm thiểu các đầu ra có hại, phân biệt đối xử hoặc bất hợp pháp.

## Các phiên bản

*   **Claude 2:** Nổi bật với cửa sổ ngữ cảnh rất lớn (100k token), cho phép người dùng tải lên toàn bộ cuốn sách hoặc tài liệu pháp lý để tóm tắt và phân tích.
*   **Claude 3 (Haiku, Sonnet, Opus):** Dòng mô hình mới nhất với hiệu suất vượt trội.
    *   **Opus:** Phiên bản thông minh nhất, cạnh tranh ngang ngửa với GPT-4.
    *   **Sonnet:** Cân bằng giữa tốc độ và trí tuệ.
    *   **Haiku:** Phiên bản nhanh nhất và rẻ nhất.

## Đặc điểm

*   **Xử lý văn bản dài:** Khả năng vượt trội trong việc duy trì ngữ cảnh qua các cuộc hội thoại dài và xử lý tài liệu lớn.
*   **An toàn:** Ít có khả năng bị "bẻ khóa" (jailbreak) hoặc tạo ra nội dung độc hại.
