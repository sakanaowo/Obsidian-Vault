---
tags:
  - Resources/BookNote
  - AI/Agents
  - AI/LangChain
  - AI/ReAct
created: 2026-01-04
source: [[Prompt Engineering for Generative AI Future-Proof Inputs for Reliable Al Outputs (James Phoenix, Mike Taylor) (Z-Library).pdf]]
author: James Phoenix, Mike Taylor
---

# Chapter 06: Autonomous Agents with Memory and Tools

## 1. Sự trỗi dậy của Autonomous Agents

Chương này đánh dấu sự chuyển dịch quan trọng từ các mô hình AI thụ động (chỉ trả lời khi được hỏi) sang **[[Autonomous Agents]]** (Tác nhân tự hành) - các hệ thống có khả năng tự suy nghĩ, lập kế hoạch và hành động để đạt được mục tiêu.

### Agent là gì?
Agent là một hệ thống sử dụng **[[Large Language Models]] (LLM)** làm bộ não trung tâm để:
1.  **Perceive:** Nhận thức môi trường (Inputs).
2.  **Reason:** Suy luận về việc cần làm ([[Chain of Thought]]).
3.  **Act:** Thực hiện hành động thông qua các công cụ ([[Function Calling]] / Tools).

## 2. Framework ReAct: Reason + Act

Cốt lõi của hầu hết các Agent hiện đại là framework **[[ReAct Framework]]**.

![[fig_6-1_The_ReAct_framework_Agents.png]]
*Figure 6-1: Khung làm việc ReAct - Vòng lặp giữa Suy nghĩ (Thought), Hành động (Act) và Quan sát (Observation).*

*   **Thought:** Agent phân tích yêu cầu. ("Người dùng muốn biết thời tiết. Mình cần dùng tool Weather.")
*   **Act:** Agent gọi tool. (`Weather(city="London")`)
*   **Observation:** Agent nhận kết quả từ tool. ("Trời đang mưa, 15 độ.")
*   **Thought:** Phân tích kết quả và quyết định bước tiếp theo hoặc trả lời. ("Đã có thông tin. Trả lời người dùng.")

> [!NOTE] So sánh với OpenAI Functions
> [[Function Calling]] của OpenAI là một cách triển khai kỹ thuật để hỗ trợ ReAct. Nó giúp model trả về cấu trúc JSON chính xác để gọi hàm, thay vì model phải tự "bịa" ra cú pháp gọi hàm trong văn bản thuần túy.

## 3. Công cụ (Tools) và Toolkits

Agent chỉ mạnh khi nó có công cụ mạnh. LangChain cung cấp khái niệm **[[Agent Toolkits]]**:
*   **Tools:** Các hàm đơn lẻ (ví dụ: `GoogleSearch`, `PythonREPL`, `Calculator`).
*   **Toolkits:** Bộ công cụ theo chủ đề (ví dụ: `SQLDatabaseToolkit` để tương tác với SQL, `Office365Toolkit` để gửi mail/lịch).

### Ví dụ Code: Tạo Agent với Tools

```python
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)
tools = load_tools(["serpapi", "llm-math"], llm=llm) # Load Search và Calculator

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent.run("Ai là bạn gái của Leonardo DiCaprio hiện tại? Cô ấy bao nhiêu tuổi mũ 0.43?")
```
*Trong ví dụ này, Agent sẽ tự động Search để tìm tên bạn gái, sau đó dùng Calculator để tính toán số tuổi mũ 0.43.*

## 4. Trí nhớ (Memory)

Để Agent hoạt động hiệu quả qua nhiều bước, nó cần **[[Agent Memory]]**.

![[fig_6-3_Memory_within_LangChain_In_every_operation_of_the_.png]]
*Figure 6-3: Cơ chế Memory trong LangChain.*

*   **ConversationBufferMemory:** Nhớ nguyên văn hội thoại (tốn token).
*   **ConversationSummaryMemory:** Dùng một LLM khác để tóm tắt hội thoại cũ, tiết kiệm token.
*   **VectorStoreMemory:** Sử dụng [[Vector Databases]] để lưu trữ ký ức dài hạn, cho phép Agent truy xuất thông tin từ quá khứ xa.

## 5. Các kiến trúc Agent nâng cao

Chương này cũng giới thiệu các biến thể Agent phức tạp hơn:

### Plan-and-Execute (ví dụ: [[BabyAGI]])
Thay vì nghĩ từng bước (như ReAct), Agent này:
1.  **Plan:** Lập ra toàn bộ danh sách việc cần làm (Task List) ngay từ đầu.
2.  **Execute:** Lần lượt thực hiện từng việc.
3.  **Prioritize:** Sắp xếp lại thứ tự ưu tiên dựa trên kết quả mới nhất.

![[fig_6-4_BabyAGI’s_agent_architecture_The_plan-and-execute_.png]]
*Figure 6-4: Kiến trúc của BabyAGI - Tự động tạo và quản lý Task.*

### Tree of Thoughts (ToT)
Cho phép Agent khám phá nhiều "nhánh" suy nghĩ khác nhau, tự đánh giá và quay lui (backtrack) nếu một nhánh dẫn đến ngõ cụt.

---
**Liên kết:** [[Chapter 05 - Vector Databases with FAISS and Pinecone]] | [[Chapter 07 - Introduction to Diffusion Models for Image Generation]]
