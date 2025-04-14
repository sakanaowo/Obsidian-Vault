---
aliases:
  - Neural network
  - mạng nơ ron
tags:
  - sigmoid
  - layer
  - activation-function
  - bias
  - neural-network
Requirement:
  - "[[Linear Regression]]"
  - "[[Logistic Regression]]"
  - "[[Feedforward|Feedforward]]"
  - "[[bias]]"
---


# Neural Network là gì
## Hoạt động của neuron
# Mô hình của Neural Network
## Logistic Regression
- Là mô hình neural network đơn giản nhất chỉ với input layer và output layer
- Xem thêm ở [[Linear Regression]]
- Có 2 bước:
	- Tổng linear: $z=1*w_0+x_1*w_1+x_2*w_2$ 
	- Áp dụng tính sigmoid function: $\hat y=\sigma(z)$ 
	- Mô hình logistic
		![](https://i.imgur.com/gQNCLz0.png)
- Hệ số $w_0$ được gọi là bias  ~ hệ số tự do 
- Hàm `sigmoid` được gọi là `activation function`
## Mô hình tổng quát
![](https://i.imgur.com/coL6Gdr.png)

Mô hình neural network:
- `Input Layer`: 
- Các `hidden layer`:
- `Output layer`:
Mỗi `node` trong `hidden layer` và `output layer`:
- Liên kết với tất cả các node ở layer trước đó với hệ số $w$ riêng
- Mỗi node có 1 hệ số bias `b` riêng
- Quy trình diễn ra 2 bước: 
	- Tính tổng `linear regression`
	- Áp dụng `activation function` ~ hàm `sigmoid`
## Kí hiệu

> [!NOTE]
> Số node trong layer thứ $i$ là $l^{i}$

- Ma trận $W^k$ có kích thước $l^{k-1}*l^k$ là mà trận hệ số giữa layer k-1 và layer k, trong đó $w_{ij}^{k}$ là hệ số kết nối từ node thứ i của layer k-1 đến node thứ j của layer k
- Vector $b^k$ kích thước $l^{k}*1$ là hệ số bias của các node trong layer k, trong đó $b_{i}^{k}$ là bias của node thứ i trong layer k
- Với node thứ i trong layer $l$ có trong bias $b_{i}^{l}$ , thực hiện 2 bước:
	- Tính tổng `Linear`: $$z_{i}^l=\sum_{j=1}^{l^{l-1}}a_j^{l-1}*w_{ji}^l+b_i^l$$
	- Áp dụng `activation function`: $a_i^l=\sigma(z_i^l)$
- Vector $z^k$ kích thước $l^k*1$ là giá trị các node trong layer k sau bước tính tổng linear
- Vector $a^k$ kích thước $l^k*1$ là giá trị các node trong layer k sau khi áp dụng hàm kích hoạt  
![](https://i.imgur.com/t82TKVb.png)
![](https://i.imgur.com/yJU2VcW.png)

# Feedforward
[[Feedforward]]

## Biểu diễn ma trận

Khi làm việc với dữ liệu thì cần dự đoán cho nhiều dữ liệu một lúc -> Gọi $X$ là ma trận $n*d$
Trong đó:
- `n`: số dữ liệu
- `d`: số trường trong mỗi dữ liệu

> [!NOTE] 
> $x_j^{[i]}$ là giá trị trường dữ liệu thứ `j` của dữ liệu thứ `i` 

### Ví dụ: dataset trong [[Logistic Regression]] 

| Lương | Thời gian làm việc |
| ----- | ------------------ |
| 10    | 1                  |
| 5     | 2                  |
| 6     | 1.8                |
| 7     | 1                  |
Thì $n=4,d=2,x_1^{[1]}=10,x_2^{[1]}=1,...$  

Do $x^{[1]}$ là vector kích thước $d*1$, và dữ liệu ở ma trận $X$ cần được viết theo hàng -> cần transpose $x^{[1]}$ thành kích thước $1*d$ , kí hiệu: $$-(x^{[1]})^T-$$
-> Biểu diễn được ma trận $X$:$$X=\begin{bmatrix}x_1^{[1]}&x_2^{[1]}&...&x_d^{[1]}\\x_1^{[2]}&x_2^{[2]}&...&x_d^{[2]}\\...&...&...&...\\x_1^{[d]}&x_2^{[d]}&...&x_d^{[d]}\end{bmatrix}=\begin{bmatrix}-(x^{[1]})^T-\\-(x^{[2]})^T-\\...\\-(x^{[d]})^T-\end{bmatrix}$$
Gọi ma trận $Z^{(i)}$ kích thước $N*l^{(i)}$
Trong đó:
- $z_j^{(i)[k]}$ là giá trị thứ j trong layer i sau bước tính tổng linear của dữ liệu thứ k trong dataset 
Tương tự, ma trận $A^{(i)}$ kích thước $N*l^{(i)}$ trong đó $a_j^{(i)[k]}$ là giá trị thứ j trong layer i sau khi áp dụng hàm kích hoạt của dữ liệu thứ k trong datase
# Logistic regression với XOR

## NOT

## AND

## OR

## XOR

