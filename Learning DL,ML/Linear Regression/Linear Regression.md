---
aliases:
  - Linear Regression
tags:
  - element-wise
  - loss-function
  - gradient-descent
  - dahamard
  - kronecker
---

# Linear Regression

# Mục tiêu:

- Giải quyết các bài toán có đầu ra thực
- Dự đoán giá cả,…

# Bài toán:

- Cho dữ liệu về diện tích và giá nhà, dự đoán giá nhà?
- Trên thực tế, giá nhà phụ thuộc vào rất nhiều yếu tố nhưng ở bài toán này thì giá nhà chỉ phụ thuộc vào diện tích

Dữ liệu:

| Diện tích(m2) | Giá bán (triệu VNĐ) |
| --- | --- |
| 30 | 448.524 |
| 32.4138 | 509.248 |
| 34.8276 | 535.104 |
| 37.2414 | 551.432 |
| 39.6552 | 623.418 |
| … | … |

![image.png](Linear%20Regression/Linear%20Regression%201c7bd9fab3c380938f85e3c9e261d277/image.png)

# Thiết lập công thức:

## Model

![image.png](Linear%20Regression/Linear%20Regression%201c7bd9fab3c380938f85e3c9e261d277/image%201.png)

Phương trình có dạng:

$$
y=w_1*x+w_0
$$

Từ bảng trên → các tập dữ liệu: $(x_1,y_1)=(30,448.524)$, …

→ Tòa nhà có diện tích $x_i$  thì có giá $y_i$

## Loss function

Việc tìm $(w_0,w_1)$ có thể dễ tính bằng tay nhưng máy tính thì không → giá trị ban đầu của chúng được đặt bằng $(0,1)$ sau đó được chỉnh dần 

![Sự khác nhau tại điểm x = 42 của model đường thẳng y = x và giá trị thực tế ở bảng 1](Linear%20Regression/Linear%20Regression%201c7bd9fab3c380938f85e3c9e261d277/image%202.png)

Sự khác nhau tại điểm x = 42 của model đường thẳng y = x và giá trị thực tế ở bảng 1

→ Cần 1 hàm đánh giá đường thẳng $y=x$ với $(w_1,w_0)=(1,0)$ có tốt không bằng cách so sánh giá dự đoán và giá thực qua công thức:

$$
\frac{(\hat{y_i}-y_i)^2}{2}
$$

→ Độ chênh lệch trên toàn bộ dữ liệu: 

$$
J=\frac{\sum_{i=1}^{N}(\hat y_i-y_i)^2}{2*N}
$$

Nhận thấy:

- J không âm
- J nhỏ dần khi đường $y$ tiến tới tập dữ liệu
- $J=0$ khi $y$ đi qua tất cả các điểm dữ liệu

⇒ Bài toán thay đổi → tìm $(w_0,w_1)$ sao cho J nhỏ nhất 

Để tìm được $(w_0,w_1)$ nhỏ nhất thì cần đến thuật toán để tìm giá trị đó 

# Gradient Descent

- Thuật toán để tìm giá trị nhỏ nhất của hàm $f(x)$ dựa trên đạo hàm
- Thuật toán:
    1. Khởi tạo giá trị $x=x_0$ tùy ý
    2. Gán giá trị $x=x-learning\_ rate*f'(x)$
    3. Tính lại $f(x)$
        - Nếu $f(x)$ đủ nhỏ → dừng
        - Ngược lại, quay lại bước 2

Ví dụ:

Tìm giá trị nhỏ nhất của $y=x^2$

![image.png](Linear%20Regression/Linear%20Regression%201c7bd9fab3c380938f85e3c9e261d277/image%203.png)

- B1: $x=-2$
- B2:
    - $f'(-2)=-4<0$
    - $x=x-learning \_rate*f'(-2)$ lớn hơn $x_0$ → Đồ thị tại điểm C
- B3: Tiếp tục → hàm dần tiến tới $x=0$

Nhận xét:

- Thuật toán hoạt động tốt trong trường hợp không thể tìm được giá trị nhỏ nhất bằng đại số tuyến tính
- Việc chọn $learning \_rate$ rất quan trọng:
    - Nếu nhỏ → hàm mất nhiều thời gian để thực hiện B2
    - Nếu quá lớn → gây ra overshoot và không đạt được giá trị nhỏ nhất
- Cách kiểm tra $learning \_rate$ là kiểm tra hàm $f(x)$ sau mỗi lần thực hiện B2 bằng đồ thị

![image.png](Linear%20Regression/Linear%20Regression%201c7bd9fab3c380938f85e3c9e261d277/image%204.png)

## Áp dụng vào bài toán

Cần tìm min của hàm:

$$
J(w_0,w_1)=\frac{1}{2}*(\sum_{i=1}^{N}(\hat y_i-y_i)^2)=\frac{1}{2}*(\sum_{i=1}^{N}(w_0+w_1*x_i-y_i)^2)
$$

Có:

$$
h(x)=f(g(x))=>h'(x)=f'(g)*g'(x)
$$

Áp dụng vào $J$ tại $(x_i,y_i)$:

$$
f(w_0,w_1)=\frac{1}{2}∗(w_0+w_1∗x_i−y_i)^2
$$

$$
\frac{df}{dw_0}=w_0+w_1*x_i-y_i =>\frac{dJ}{dw_0}=\sum_{i=1}^{N} (w_0+w_1*x_i-y_i)
$$

$$
\frac{df}{dw_1}=x_i*(w_0+w_1*x_i-y_i)=>\sum_{i=1}^{N}x_i*(w_0+w_1*x_i-y_i)
$$

# Matrix

### Phép nhân Hadamard - Element-wise multiplication

- Cho:
    
    $$
    A=\begin{bmatrix} a_{11}&a_{12}\\a_{21}&a_{22} \end{bmatrix} ,B=\begin{bmatrix}b_{11}&b_{12}\\b_{21}&b_{22}\end{bmatrix}
    $$
    

$$
=>A \circ B=\begin{bmatrix}
a_{11}*b_{11}&a_{12}*b_{12}\\
a_{21}*b_{21}&a_{22}*b_{22}\\
\end{bmatrix}
$$

### Phép nhân Kronecker

- Cho:

$$
A=\begin{bmatrix}
a&b\\
a&d
\end{bmatrix},
B=\begin{bmatrix}
e&f\\
g&h
\end{bmatrix}
$$

$$
=>A\otimes B=\begin{bmatrix}aB&bB\\cB&dB\end{bmatrix}=
\begin{bmatrix}ae&af&be&bf\\
ag&ah&bg&bh\\
ce&cf&de&df\\
cg&ch&dg&dh
\end{bmatrix}
$$

## Biểu diễn bài toán

Do với mỗi điểm $x_i,y_i$ ta cần tính $(w_0+w_1*x_i-y_i)$ → sử dụng ma trận biểu diễn:

$$
X=\begin{bmatrix}1&x_1 \\... \ &...\\1&x_n\end{bmatrix}, Y=\begin{bmatrix}y_1\\...\\y_n\end{bmatrix},W=\begin{bmatrix}w_0\\w_1\end{bmatrix}
$$

$$
=>\hat Y=X*W=\begin{bmatrix}w_0+w_1*x_1\\...\\w_0+w_1*x_n\end{bmatrix}
$$

Tạo hàm $Sum(X)$  là hàm tính tổng phần tử trong ma trận $X$ ở trên 

→ $Sum(X[:,1])=x_1+x_2+...+x_n$ với $X[:,1]=\begin{bmatrix}x_1\\...\\x_n\end{bmatrix}$

⇒ $\frac{dJ}{dw_0}=Sum(\hat Y-Y)$, $\frac{dJ}{dw_1}=Sum(X[:,1] \circ(\hat Y -Y))$

# Mở rộng

—Skip

# Code

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data_linear.csv').values
N = data.shape[0]
x = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)
plt.scatter(x, y)
plt.xlabel('mét vuông')
plt.ylabel('giá')

x = np.hstack((np.ones((N, 1)), x))

w = np.array([0., 1.]).reshape(-1, 1)

numOfIteration = 100
cost = np.zeros((numOfIteration, 1))
learning_rate = 0.000001
for i in range(1, numOfIteration):
    r = np.dot(x, w) - y
    cost[i] = 0.5 * np.sum(r * r)
    w[0] -= learning_rate * np.sum(r)
    # correct the shape dimension
    w[1] -= learning_rate * np.sum(np.multiply(r, x[:, 1].reshape(-1, 1)))
    print(cost[i])
predict = np.dot(x, w)
plt.plot((x[0][1], x[N - 1][1]), (predict[0], predict[N - 1]), 'r')
plt.show()

x1 = 50
y1 = w[0] + w[1] * 50
print('Giá nhà cho 50m^2 là : ', y1)
```