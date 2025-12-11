# Quy tắc chuỗi

### **Quy tắc chuỗi (Chain Rule) là gì?**

**Quy tắc chuỗi** (**Chain Rule**) là một quy tắc quan trọng trong **đạo hàm** dùng để tính **đạo hàm của hàm hợp (composition function)**.

---

## **1. Công thức quy tắc chuỗi**

Cho một hàm hợp **y=f(g(x))y = f(g(x))**, đạo hàm của yy theo xx được tính bằng:

ddxf(g(x))=f′(g(x))⋅g′(x)\frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x)

📌 **Ý nghĩa:** Đạo hàm của hàm ngoài **f(x)f(x)** nhân với đạo hàm của hàm trong **g(x)g(x)**.

---

## **2. Ví dụ đơn giản**

### **Ví dụ 1: Đạo hàm của (2x+3)5(2x + 3)^5**

Cho:

y=(2x+3)^5 = (2x + 3)^5

Gọi u=2x+3u = 2x + 3, nên y=u5y = u^5.

Sử dụng quy tắc chuỗi:

$$
dydx=5u4⋅dudx=5(2x+3)4⋅2\frac{dy}{dx} = 5u^4 \cdot \frac{du}{dx} = 5(2x+3)^4 \cdot 2
$$

=10(2x+3)4= 10(2x + 3)^4

📌 **Nhận xét:** Ta đạo hàm **lũy thừa trước**, rồi nhân với đạo hàm của phần bên trong.

---

### **Ví dụ 2: Đạo hàm của Sigmoid**

Hàm Sigmoid:

$$
S(x)=11+e−xS(x) = \frac{1}{1 + e^{-x}}
$$

Đạo hàm của nó (sử dụng quy tắc chuỗi):

S′(x)=S(x)⋅(1−S(x))S'(x) = S(x) \cdot (1 - S(x))

📌 **Ứng dụng:** Quy tắc chuỗi rất quan trọng trong **Deep Learning**, vì hàm Sigmoid, ReLU, và Softmax đều cần đạo hàm khi tối ưu hóa mạng nơ-ron.

---

## **3. Mở rộng quy tắc chuỗi cho nhiều biến**

Nếu z=f(x,y)z = f(x, y) với x=g(t)x = g(t) và y=h(t)y = h(t), thì:

dzdt=∂f∂xdxdt+∂f∂ydydt\frac{dz}{dt} = \frac{\partial f}{\partial x} \frac{dx}{dt} + \frac{\partial f}{\partial y} \frac{dy}{dt}

📌 **Ứng dụng:** Trong **Machine Learning**, quy tắc chuỗi mở rộng giúp tính toán **đạo hàm ngược (Backpropagation)** trong mạng nơ-ron.

---