---
aliases:
  - ReLU
tags:
  - ReLU
  - activation-function
  - vanishing-gradient
Reference:
  - "[[Vanishing Gradient|vanishing gradient]]"
---
---
**ReLU** (Rectified Linear Unit) là một hàm kích hoạt rất phổ biến trong học sâu (deep learning), đặc biệt là trong các mạng nơ-ron. ReLU giúp tăng tốc quá trình huấn luyện và giảm thiểu vấn đề _vanishing gradient_ mà các hàm kích hoạt như sigmoid hoặc tanh thường gặp phải.

### Định nghĩa:

Hàm ReLU có dạng toán học đơn giản:

$$f(x) = \max(0, x)$$

Điều này có nghĩa là:

- Nếu giá trị đầu vào xx lớn hơn 0, hàm ReLU sẽ trả về giá trị của xx.
    
- Nếu giá trị đầu vào xx nhỏ hơn hoặc bằng 0, hàm ReLU sẽ trả về giá trị 0.
    

### Ví dụ:

- ReLU(3) = 3
    
- ReLU(-2) = 0
    
- ReLU(0) = 0
    

### Ưu điểm của ReLU:

1. **Giảm vanishing gradient**: ReLU không gặp vấn đề về vanishing gradient như các hàm sigmoid hoặc tanh, vì đạo hàm của ReLU là 1 (đối với x>0x > 0), giúp gradient không bị mất đi khi lan truyền ngược.
    
2. **Dễ dàng tính toán**: ReLU rất đơn giản và nhanh chóng để tính toán, giúp tăng tốc quá trình huấn luyện.
    
3. **Tạo tính không tuyến tính**: Mặc dù hàm ReLU có vẻ đơn giản, nhưng nó vẫn tạo ra tính không tuyến tính cho mạng nơ-ron, cho phép mô hình học được các mối quan hệ phức tạp.
    

### Nhược điểm:

1. **Dead neurons**: Một vấn đề có thể xảy ra với ReLU là một số "neuron chết" (dead neurons). Nếu một trọng số nào đó bị điều chỉnh sao cho giá trị đầu vào luôn nhỏ hơn hoặc bằng 0, thì các neuron này sẽ luôn trả về giá trị 0 và không học được gì nữa. Vấn đề này có thể được giải quyết bằng cách sử dụng các biến thể của ReLU như **Leaky ReLU** hoặc **Parametric ReLU**.
    

### Các biến thể của ReLU:

1. **Leaky ReLU**: Hàm này cho phép một lượng nhỏ giá trị âm (thường là 0.01), thay vì trả về hoàn toàn 0 khi giá trị đầu vào âm. Cụ thể, Leaky ReLU có dạng:
    
    f(x)={xneˆˊu x>0αxneˆˊu x≤0f(x) = \begin{cases} x & \text{nếu } x > 0 \\ \alpha x & \text{nếu } x \leq 0 \end{cases}
    
    Trong đó, α\alpha là một hằng số nhỏ (ví dụ 0.01).
    
2. **Parametric ReLU (PReLU)**: Đây là một biến thể của Leaky ReLU, trong đó tham số α\alpha không cố định mà được học trong quá trình huấn luyện.
    

### Kết luận:

ReLU là một hàm kích hoạt cực kỳ hiệu quả trong học sâu nhờ tính đơn giản, khả năng giảm thiểu vanishing gradient và khả năng học nhanh. Tuy nhiên, việc sử dụng ReLU cũng cần lưu ý đến các vấn đề như dead neurons, và có thể cần thử nghiệm với các biến thể khác nếu gặp phải những khó khăn này.