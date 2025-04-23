---
aliases:
  - Self-Attention
tags:
  - Self-Attention
  - seq2seq
Reference:
  - "[[Transformer]]"
---
# Self-attention là gì ?
Như chúng ta đã biết, **Word Embedding** là vector đại diện cho ngữ nghĩa của một từ trong câu. Trong bước tiền xử lý, chúng ta đã tạo ra một **không gian vector** chứa các vector embedding của các từ. Những từ có nghĩa tương tự nhau sẽ có vector ở gần nhau trong không gian đó và ngược lại. Tuy nhiên, ý nghĩa của các từ riêng lẻ trong một câu không thể đại diện cho toàn bộ ý nghĩa của câu đó. Chẳng hạn, trong câu "The apple on the table", từ "apple" trong ngữ cảnh này được hiểu là quả "táo" . Nhưng nếu đặt trong một ngữ cảnh khác, như trong câu "The Apple keynote was interesting", từ "Apple" có thể ám chỉ đến công ty công nghệ.
Cơ chế Self-Attention được đề xuất trong bài báo [Attention Is All You Need](https://arxiv.org/abs/1706.03762v7) có thể giải quyết tốt vấn đề này. Ý tưởng của nó là so sánh các từ với nhau đôi một, bao gồm cả chính nó (self), để tìm ra mức độ quan trọng của mỗi từ mà mô hình nên chú ý tới (thể hiện qua trọng số). Điều này giúp mô hình hiểu đúng ý nghĩa của từ trong ngữ cảnh cụ thể, thay vì chỉ dựa vào ý nghĩa tổng quát của từ đó khi đứng riêng lẻ.

# Cơ chế hoạt động
## Attention trong seq2seq
Nếu chúng ta chỉ dịch từng từ ở ngôn ngữ này sang ngôn ngữ khác bằng cách ánh xạ word-by-word , thì đó quả là một phương pháp tệ, thiếu hiệu quả do không học được thông tin từ những từ xung quanh.
![](https://i.imgur.com/JgBHj7P.png)


## Chi tiết về #Self-Attention 


