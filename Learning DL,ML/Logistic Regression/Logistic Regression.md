---
aliases:
  - Logistic Regression
tags:
---

# Logistic Regression

# Bài toán

- Tìm giải pháp để dự đoán 2000 hồ sơ có đủ khả năng để cho vay hay không
- Sau khi phân tích → rút ra được 2 yêu tố quyết định được việc cho vay là mức lương và thời gian công tác, có dạng:
    

| Lương | Thời gian làm việc | Cho vay |
| ----- | ------------------ | ------- |
| 10    | 1                  | 1       |
| 5     | 2                  | 1       |
| …     | …                  | 1       |
| 8     | 0.1                | 0       |
| 7     | 0.15               | 0       |
| …     | …                  | 0       |
    
![Đồ thị giữa mức lương, năm kn và kết quả cho vay ](Logistic%20Regression/Logistic%20Regression%201c7bd9fab3c380ffb9b9c9e9d14dd96c/image.png)
    
Đồ thị giữa mức lương, năm kn và kết quả cho vay 
    

![Đường phân chia và dự đoán điểm dữ liệu mới](Logistic%20Regression/Logistic%20Regression%201c7bd9fab3c380ffb9b9c9e9d14dd96c/image%201.png)

Đường phân chia và dự đoán điểm dữ liệu mới

- Dự đoán cho hồ sơ của người có mức lương 6 triệu và 1 năm kinh nghiệm là không cho vay.
- Chỉ những hồ sơ nào chắc chắn trên 80% mới được vay

→ Tìm xác suất nên cho hồ sơ vay là bao nhiêu

# Hàm Sigmoid

![Đồ thị hàm sigmoid](Logistic%20Regression/Logistic%20Regression%201c7bd9fab3c380ffb9b9c9e9d14dd96c/image%202.png)

Đồ thị hàm sigmoid

Nhận Xét:

- Hàm liên tục, nhận giá trị trong khoảng $(0,1)$
- Có đạo hàm tại mọi điểm → gradient descent

# Thiết lập bài toán

## Các bước thiết lập bài toán:

- Thiết lập model
- Thiết lập loss function
- Tìm tham số bằng tối ưu loss function
- Dự đoán dữ liệu mới bằng model

## Model

Gọi lương là $x_1^{(i)}$, thời gian làm việc là $x_2^{(i)}$

→ $p(x^{(i)}=1)=\hat y_i$ là xác suất model dự đoán hồ sơ $i$ được cho vay

→ $p(x^{(i)}=0)=1-\hat y_i$ là xác suất model dự đoán hồ sơ $i$ không được cho vay

$$
=> p(x^i=1)+p(x^i=0)=1
$$

Hàm sigmoid:

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

Công thức logistic regression:

$$
\hat y_i=\sigma(w_0+w_1*x_1^{(i)}+w_2*x_2^{(1)})=\frac{1}{1+e^{-w_0+w_1*x_1^{(i)}+w_2*x_2^{(1)}}}
$$

## Loss function

- Nếu hồ sơ thứ $i$ được vay $<=> y_i=1$ thì $\hat y_i$ càng gần 1 càng tốt
- Và ngược lại

Với mỗi điểm $(x^i,y_i)$, gọi hàm loss:

$$
L=-(y_i*log(\hat y_i))-(1-y_i)*log(1-\hat y_i)
$$

### Nếu $y_i=1=> L=-log(\hat y_i)$

![image.png](Logistic%20Regression/Logistic%20Regression%201c7bd9fab3c380ffb9b9c9e9d14dd96c/image%203.png)

Nhận Xét:

- L giảm dần khi $\hat y_i$  từ 0 → 1
- Khi model dự đoán $\hat y_i$ gần 1 hay giá trị dự đoán gần với giá trị thật $y_i$ thì L ~ 0
- Khi model dự đoán $\hat y_i$ gần 0 hay giá trị dự đoán ngược với giá trị thật $y_i$ thì L ~ $+\infty$

### Nếu $y_i=0=>L=-log(1-\hat y_i)$

![image.png](Logistic%20Regression/Logistic%20Regression%201c7bd9fab3c380ffb9b9c9e9d14dd96c/image%204.png)

Nhận Xét:

- Hàm L tăng khi $\hat y_i$ từ 0→1
- Khi model dự đoán $\hat y_i$ gần 0 hay giá trị dự đoán gần với giá trị thật $y_i$ thì L ~ 0
- Khi model dự đoán $\hat y_i$ gần 1 hay giá trị dự đoán ngược với giá trị thật $y_i$ thì L ~ $+\infty$

⇒ L càng nhỏ thì model càng đoán đúng → bài toán tìm min

**Hàm loss trên toàn bộ dữ liệu:** 

$$
J=-\sum_{i=1}^{N}(y_i*log(\hat y_i)+(1-y_i)*log(1-\hat y_i))
$$

# Chain Rule

[Quy tắc chuỗi ](Quy%20tắc%20chuỗi.md)

## Áp dụng gradient descent

Với mỗi điểm $(x^i,y_i)$ gọi hàm loss:

$$
L=-y_i*log(\hat y_i)-(1-y_i)*log(1-\hat y_i) \ (*)
$$

trong đó:

$$
\sigma(w_0+w_1*x_1^{i}+w_2*x_2^{i})=\hat y_i
$$

$\hat y_i$ là giá trị mà model dự đoán, còn $y_i$ là giá trị thật của dữ liệu

Tính được:

$$
\frac{d\hat y_i}{dw_0}=\frac{\sigma(w_0+w_1*x_1^{i}+w_2*x_2^{i})}{dw_0}=\hat y_i*(1-\hat y_i)
$$

$$
\frac{d\hat y_i}{dw_1}=\frac{\sigma(w_0+w_1*x_1^{i}+w_2*x_2^{i})}{dw_1}=x_1^{i}*\hat y_i*(1-\hat y_i)
$$

$$
\frac{d\hat y_i}{dw_2}=\frac{\sigma(w_0+w_1*x_1^{i}+w_2*x_2^{i})}{dw_2}=x_2^{i}*\hat y_i*(1-\hat y_i)
$$

Do đó:

$$
\frac{dL}{dw_0}=\frac{dL}{d\hat y_i}*\frac{d\hat y_i}{dw_0}=-(\frac{y_i}{\hat y_i}-\frac{1-y_i}{1-\hat y_i})*\hat y_i*(1-\hat y_i)=\hat y_i-y_i
$$

$$
\frac{dL}{dw_1}=x_1^{i}*(\hat y_i-y_i)
$$

$$
\frac{dL}{dw_2}=x_2^{i}*(\hat y_i-y_i)
$$

Suy ra, trên toàn bộ dữ liệu:

$$
\frac{dL}{dw_0}=\sum_{i=1}^{N}(\hat y_i-y_i)
$$

$$
\frac{dL}{dw_1}=\sum_{i=1}^{N}x_1^{i}(\hat y_i-y_i)
$$

$$
\frac{dL}{dw_2}=\sum_{i=1}^{N}x_2^{i}(\hat y_i-y_i)
$$

### Biểu diễn bài toán:

Khởi tạo:

$$
X=\begin{bmatrix}
1&x_1^{1}&x_2^{1}\\
1&x_1^{2}&x_2^{2}\\
1&...&...\\
1&x_1^{n}&x_2^{n}
\end{bmatrix}, y=\begin{bmatrix}y_1\\y_2\\...\\y_n\end{bmatrix},w=\begin{bmatrix}w_0\\w_1\\w_2\end{bmatrix}
$$

$$
=>\hat y=\sigma(Xw)
$$

Binary-cross Entropy

$$
J=-sum(y\circ log(\hat y)+(1-y)\circ log(1-\hat y))
$$

$$
\frac{dJ}{dw}=X^{T}*(\hat y-y), X^T=\begin{bmatrix}1&1&...&1\\x_1^1&x_1^2&...&x_1^n\\x_2^1&x_2^2&...&x_2^n\end{bmatrix}
$$

Thực hiện Gradient Descent → thu được $w_0,w_1,w_2$ 

Với mỗi hồ sơ $x^t$ mới ta sẽ tính được % cho vay $\hat y_t=\sigma(w_0+w_1*x_1^t+w_2*x_2^t)$, sau đó so sánh với ngưỡng cho vay $t=0.8$, nếu $y_t\geq t$

# Quan hệ giữa xác suất và đường thẳng

Xét $y=ax+b$ và $f=y-(ax+b)$, miền giá trị âm và dương của $f$ 

# Ứng dụng

- Spam detection
- Credit card fraud
- Health
- Banking
- Investment

# Code

```jsx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

data = pd.read_csv('dataset.csv').values
N, d = data.shape
x = data[:, 0:d - 1].reshape(-1, d - 1)
y = data[:, 2].reshape(-1, 1)

x_cho_vay = x[y[:, 0] == 1]
x_tu_choi = x[y[:, 0] == 0]

plt.scatter(x_cho_vay[:, 0], x_cho_vay[:, 1], c='red', edgecolors='none', s=30, label='cho vay')
plt.scatter(x_tu_choi[:, 0], x_tu_choi[:, 1], c='blue', edgecolors='none', s=30, label='từ chối')
plt.legend(loc=1)
plt.xlabel('mức lương (triệu)')
plt.ylabel('kinh nghiệm (năm)')

# add col 1 into data
x = np.hstack((np.ones((N, 1)), x))

w = np.array([0., 0.1, 0.1]).reshape(-1, 1)

numOfIteration = 1000
cost = np.zeros((numOfIteration, 1))
learning_rate = 0.01

for i in range(1, numOfIteration):
    y_predict = sigmoid(np.dot(x, w))
    cost[i] = -np.sum(np.multiply(y, np.log(y_predict)) + np.multiply(1 - y, np.log(1 - y_predict)))
    # gradient descent
    w -= learning_rate * np.dot(x.T, y_predict - y)
    print('cost:', cost[i])
t = 0.5
plt.plot((4, 10), (-(w[0] + 4 * w[1] + np.log(1 / t - 1)) / w[2], -(w[0] + 10 * w[1] + np.log(1 / t - 1)) / w[2]), 'g')
plt.show()

```