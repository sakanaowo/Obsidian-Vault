---
tags:
  - AI/Agents
  - AI/Protocol
  - Concept
alias:
  - Model Context Protocol Client
created: 2026-01-18
---

> [!NOTE] ELI5
> **MCP client** là người “đứng giữa”: nó nghe bạn nói, hỏi model nên làm gì, rồi đi “đặt hàng” ở MCP server và mang kết quả về cho model. Nó giống như một người trợ lý: biết khi nào cần đọc tài liệu và khi nào cần nhờ người khác làm việc.

## Client làm gì trong vòng đời một agent

Trong **[[Model Context Protocol (MCP)]]**, client thường nằm trong ứng dụng host (IDE, desktop app, backend service). Nó gánh ba trách nhiệm chính:

1) **Orchestration**: duy trì vòng lặp agent (nhận mục tiêu → lựa chọn action → thực thi → phản hồi). Đây là phần gần với [[Autonomous Agents]] và framework như [[LangChain]].

2) **Context Assembly**: quyết định “đưa gì vào context” từ **[[MCP Resources]]** (chunking, caching, citation, trimming). Claim: agent ổn định hơn khi context assembly được kiểm soát bởi code, không để model “nhồi” bừa bãi; điều này đúng vì context window và attention đều là tài nguyên hữu hạn.

3) **Tool Execution Gateway**: gọi **[[MCP Tools]]** và áp policy (rate limit, user consent, sandbox). Đây là nơi “AI decision” chạm vào “side effects”, nên client thường là điểm phù hợp để yêu cầu xác nhận người dùng hoặc gắn guardrails.

## Nuance: client không phải chỉ là “SDK gọi API”

Một sai lầm phổ biến là nghĩ client chỉ là lớp transport. Trên thực tế, client quyết định **khi nào** nên:

- đọc resource (và đọc *phần nào*),
- gọi tool (và có cần xác nhận không),
- dừng vòng lặp (termination),
- ghi log/trace để debug prompt injection và tool misuse.

Vì vậy, thiết kế client tốt là thiết kế “policy + observability”, không chỉ “wiring”.

