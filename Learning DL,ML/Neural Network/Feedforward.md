---
aliases:
  - Feedforward
  - lan truyền tiến
tags:
  - bias
---
### **Feedforward trong Mạng Nơ-ron Nhân tạo**

**Feedforward** là quá trình **lan truyền tiến** của dữ liệu qua các lớp của mạng nơ-ron để tính toán đầu ra. Đây là bước quan trọng trong **Forward Propagation (Lan truyền xuôi)** của một mạng nơ-ron.

---

## **1. Cách Hoạt Động Của Feedforward**

Giả sử ta có một mạng nơ-ron nhiều lớp (Multilayer Perceptron - MLP) với:

- **L layers**
    
- $X$ là đầu vào (input)
    
- **$W^{(k)}$ là ma trận trọng số tại lớp kk
    
- $b^{(k)}$ **là bias tại lớp `k`**
    
- σ **là hàm kích hoạt (activation function)**
    

Ta tính toán lan truyền tiến như sau:

1️⃣ **Tính tổng trọng số đầu vào tại lớp kk**:

$z^{(k)} = W^{(k)} a^{(k-1)} + b^{(k)}$

với $a^{(k-1)}$ là đầu ra của lớp trước.

2️⃣ **Áp dụng hàm kích hoạt**:

a(k)=σ(z(k))a^{(k)} = \sigma(z^{(k)})

3️⃣ **Tiếp tục quá trình này cho đến lớp cuối cùng** để tính toán đầu ra.

---

## **2. Ví Dụ Minh Họa**

Giả sử ta có mạng nơ-ron với:

- **2 neurons đầu vào**
    
- **1 lớp ẩn có 3 neurons**
    
- **1 neuron đầu ra**
    

Ta có các trọng số:

W(1)=[w11w12w21w22w31w32]W^{(1)} = \begin{bmatrix} w_{11} & w_{12} \\ w_{21} & w_{22} \\ w_{31} & w_{32} \end{bmatrix}

### **Lan truyền tiến qua từng lớp**

- **Lớp ẩn**:
$$z^{(1)} = W^{(1)} X + b^{(1)} $$$$a^{(1)} = \sigma(z^{(1)})$$

- **Lớp đầu ra**:

$$z^{(2)}=W^{(2)}a^{(1)}+b^{(2)}$$$$\hat{y} = \sigma(z^{(2)})$$

---

## **3. Vai Trò Của Feedforward**

✅ Dùng để **tính đầu ra của mạng** từ dữ liệu đầu vào.  
✅ Là bước đầu tiên trước khi tính **hàm mất mát (Loss Function)**.  
✅ Sau khi feedforward, ta thực hiện **backpropagation** để cập nhật trọng số.

---

## **4. Tóm Tắt**

📌 **Feedforward** là quá trình **lan truyền dữ liệu từ input đến output** qua các lớp của mạng nơ-ron.  
📌 Quá trình này gồm **nhân trọng số, cộng bias, áp dụng hàm kích hoạt**.  
📌 Là bước tính toán trước khi thực hiện **huấn luyện mô hình bằng Backpropagation**.