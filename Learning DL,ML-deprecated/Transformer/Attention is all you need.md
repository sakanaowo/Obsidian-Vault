---
aliases:
  - Attention
tags:
  - transformer
  - CNN
  - RNN
  - vanishing-gradient
  - Multi-head
Reference:
  - "[[Transformer]]"
  - "[[Convolutional Neural Network|CNN]]"
  - "[[Vanishing Gradient|vanishing gradient]]"
---
---
# 1. Intro
### 🔍 Vấn đề cần giải quyết

- Các mô hình dịch máy truyền thống như RNN và CNN có nhiều hạn chế:
    - **RNN xử lý tuần tự**, gây khó khăn trong việc tận dụng GPU song song.        
    - **CNN có phạm vi nhìn nhận cục bộ**, không thể học được mối quan hệ xa trong câu.
    - Attention đã được dùng để cải thiện RNN/CNN, nhưng vẫn chưa tận dụng hết tiềm năng của nó.

### ✅ Transformer mang đến điều gì mới?
- Hoàn toàn dựa trên **Self-Attention**, loại bỏ hoàn toàn RNN/CNN.
- Mô hình có thể **xử lý song song toàn bộ câu**, giúp huấn luyện nhanh hơn.
- Hiệu suất dịch ngôn ngữ tốt hơn các mô hình cũ.
# 2. Background (Bối cảnh)
### 🔍 Vấn đề trong xử lý chuỗi

- **RNN:**
    
    - Học mối quan hệ giữa các từ theo thứ tự, nhưng khó học các quan hệ xa.
        
    - Dễ gặp vấn đề **vanishing gradient**, khiến việc học các phụ thuộc dài trở nên kém hiệu quả.
        
- **CNN:**
    
    - Có thể xử lý song song, nhưng chỉ tập trung vào các vùng cục bộ, không thể nhìn toàn bộ câu một lúc.
        

### ✅ Self-Attention khắc phục ra sao?

- **Self-Attention có thể nhìn toàn bộ câu cùng một lúc**.
    
- Dễ dàng mở rộng mô hình nhờ khả năng xử lý song song.
# 3. Model Architecture (Kiến trúc)
![](https://i.imgur.com/7ICFlKY.png)

### 3.1 **Encoder and Decoder Stacks**

🔍 Vấn đề:
- Các mô hình cũ phải dùng cơ chế Attention để liên kết Encoder và Decoder nhưng vẫn dựa trên RNN/CNN.  

✅ Giải pháp:
- Transformer dùng hoàn toàn **Self-Attention** và **Feed-Forward Networks (FFN)** trong cả Encoder và Decoder.

### 3.2 **Attention**
![](https://i.imgur.com/Zyyp40c.png)


🔍 Vấn đề:
- Trong một câu, không phải từ nào cũng quan trọng như nhau.  

✅ Giải pháp:
- **Multi-Head Self-Attention** giúp mô hình có thể nhìn vào nhiều phần khác nhau của câu đồng thời.  

### 3.3 **Position-wise Feed-Forward Networks**

🔍 Vấn đề:
- Các mô hình dựa trên Attention cần thêm một lớp phi tuyến để tăng độ phức tạp.  

✅ Giải pháp:
- Mỗi từ sau khi được xử lý bởi Self-Attention sẽ đi qua một mạng **Feed-Forward riêng biệt**.
    

### 3.4 **Embeddings and Softmax**

🔍 Vấn đề:

- Cần ánh xạ từ ngữ thành vector số để mô hình có thể xử lý.  
    ✅ Giải pháp:
    
- Dùng **Word Embeddings** và **Softmax** để chuyển đổi dữ liệu đầu vào và đầu ra.
    

### 3.5 **Positional Encoding**

🔍 Vấn đề:

- Transformer không có cơ chế tuần tự như RNN, nên không biết thứ tự từ trong câu.  
    ✅ Giải pháp:
    
- **Positional Encoding** giúp mô hình biết vị trí của từng từ trong câu.
# 4. Why Self-Attention (Tại sao sử dụng Self-Attention)

# 5. Training

# 6. Result

# 7. Conclusion
