---
aliases:
  - vanishing gradient
tags:
  - vanishing-gradient
  - neural-network
  - loss-function
  - backpropagation
  - sigmoid
  - ReLU
Reference:
  - "[[ReLU|ReLU]]"
  - "[[Neural Network|Neural network]]"
---
---
**Vanishing gradient** (hay **gradient biến mất**) là một vấn đề phổ biến trong việc huấn luyện các mô hình học sâu (deep learning), đặc biệt là khi sử dụng mạng nơ-ron có nhiều lớp (**deep neural networks**). 
Vấn đề này xảy ra khi các gradient (đạo hàm) của hàm mất mát (**loss function**) đối với các tham số của mô hình trở nên rất nhỏ trong quá trình lan truyền ngược (**backpropagation**), dẫn đến việc các tham số ở các lớp gần đầu vào không được cập nhật hiệu quả.

Cụ thể, khi huấn luyện mạng nơ-ron, trong quá trình lan truyền ngược, các gradient được tính toán và lan truyền từ lớp đầu ra ngược về lớp đầu vào. Trong một số trường hợp, **các gradient này có thể bị giảm dần (vanish) khi qua các lớp sâu, đặc biệt khi sử dụng các hàm kích hoạt như sigmoid hoặc tanh**. Kết quả là, các **trọng số của các lớp gần đầu vào không thay đổi nhiều**, làm cho **quá trình học trở nên chậm hoặc không hiệu quả**.

### Nguyên nhân:

- **Hàm kích hoạt sigmoid hoặc tanh**: Khi sử dụng các hàm kích hoạt như sigmoid hoặc tanh, đạo hàm của chúng có giá trị nhỏ trong phạm vi lớn của đầu vào. Do đó, gradient có thể giảm xuống rất nhanh khi truyền qua các lớp nhiều tầng.
    
- **Lớp sâu (deep layers)**: Khi mạng có nhiều lớp, gradient sẽ được nhân với các đạo hàm của các hàm kích hoạt trong mỗi lớp. Nếu các đạo hàm này nhỏ, gradient sẽ giảm dần qua các lớp và không còn tác dụng với các lớp gần đầu vào.
    

### Hậu quả:

- Quá trình học trở nên rất chậm, hoặc trong một số trường hợp, mô hình không thể học được gì ở các lớp đầu vào, chỉ có các lớp gần đầu ra có thể học được.
    

### Giải pháp:

- **ReLU (Rectified Linear Unit)**: Một giải pháp phổ biến là sử dụng hàm kích hoạt ReLU thay cho sigmoid hoặc tanh. ReLU có đạo hàm là một giá trị cố định (1 hoặc 0) đối với các giá trị dương của đầu vào, điều này giúp giảm thiểu vấn đề vanishing gradient.
    
- **Batch Normalization**: Giúp chuẩn hóa đầu ra của các lớp để giữ cho gradient không bị quá nhỏ.
    
- **LSTM (Long Short-Term Memory) hoặc GRU (Gated Recurrent Units)**: Các mô hình này được thiết kế đặc biệt để giải quyết vấn đề vanishing gradient trong học sâu liên quan đến chuỗi thời gian.