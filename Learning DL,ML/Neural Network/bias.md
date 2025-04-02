### **Bias trong Machine Learning & Neural Networks**

**Bias (độ lệch)** là một giá trị thêm vào trong mô hình **học máy** hoặc **mạng nơ-ron** để giúp dịch chuyển kết quả của hàm kích hoạt, giúp mô hình học được tốt hơn.

---

## **1. Bias trong Mô Hình Hồi Quy Tuyến Tính**

Trong **hồi quy tuyến tính**, phương trình của một đường thẳng có dạng:

y=wx+by = w x + b

Trong đó:

- ww là **hệ số (weight)** quyết định độ dốc của đường thẳng.
    
- xx là **đầu vào (input feature)**.
    
- bb là **bias**, giúp điều chỉnh đường thẳng lên/xuống mà không thay đổi độ dốc.
    

💡 **Nếu không có bias (b=0b = 0)**, đường thẳng sẽ luôn đi qua gốc tọa độ (0,0)(0,0), có thể làm giảm khả năng học của mô hình.

---

## **2. Bias trong Mạng Nơ-ron Nhân Tạo (Neural Networks)**

Trong mạng nơ-ron, đầu ra của một neuron được tính theo công thức:

y=σ(w1x1+w2x2+...+wnxn+b)y = \sigma(w_1 x_1 + w_2 x_2 + ... + w_n x_n + b)

Trong đó:

- w1,w2,...,wnw_1, w_2, ..., w_n là trọng số (weights).
    
- x1,x2,...,xnx_1, x_2, ..., x_n là đầu vào (input features).
    
- bb là bias.
    
- σ\sigma là hàm kích hoạt (activation function).
    

💡 **Bias giúp dịch chuyển đường phân tách trong không gian đặc trưng, giúp mô hình học tốt hơn ngay cả khi tất cả trọng số wiw_i bằng 0.**

---

## **3. Ví Dụ Cụ Thể**

### **Ví Dụ 1: Không có bias**

Giả sử ta có một mạng nơ-ron đơn giản với một đầu vào xx và trọng số w=2w = 2:

y=2xy = 2x

Nếu x=0x = 0, ta có:

y=2(0)=0y = 2(0) = 0

🔴 **Mạng này không thể dự đoán giá trị khác 0 khi x=0x = 0, gây hạn chế trong học tập.**

---

### **Ví Dụ 2: Có bias**

Nếu thêm bias b=3b = 3:

y=2x+3y = 2x + 3

Khi x=0x = 0:

y=2(0)+3=3y = 2(0) + 3 = 3

🟢 **Bây giờ, mạng có thể tạo ra đầu ra khác 0 ngay cả khi x=0x = 0, giúp nó học tốt hơn.**

---

## **4. Bias trong Python (NumPy)**

Giả sử ta có một mô hình tuyến tính với bias:

```python
import numpy as np

# Định nghĩa trọng số và bias
w = np.array([2])  # Trọng số w = 2
b = 3  # Bias b = 3

# Tạo dữ liệu đầu vào
x = np.array([0, 1, 2, 3])

# Tính đầu ra y = wx + b
y = w * x + b
print(y)  # Kết quả: [3, 5, 7, 9]
```

---

## **5. Ký Hiệu Bias trong LaTeX**

Bạn có thể viết bias trong LaTeX như sau:

- Công thức hồi quy tuyến tính:
    
    ```latex
    y = wx + b
    ```
    
    Hiển thị:
    
    y=wx+by = wx + b
- Công thức tổng quát trong mạng nơ-ron:
    
    ```latex
    y = \sigma\left(\sum_{i=1}^{n} w_i x_i + b\right)
    ```
    
    Hiển thị:
    
    y=σ(∑i=1nwixi+b)y = \sigma\left(\sum_{i=1}^{n} w_i x_i + b\right)

---

## **6. Tóm Tắt**

📌 **Bias** là một tham số thêm vào để giúp dịch chuyển đầu ra của mô hình.  
📌 **Không có bias**, mô hình có thể bị giới hạn trong việc học.  
📌 **Bias giúp mạng nơ-ron học được các mẫu dữ liệu phức tạp hơn.**

Bạn cần giải thích thêm về phần nào không? 😊