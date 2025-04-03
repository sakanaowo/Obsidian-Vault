---
aliases:
  - Xử lý ảnh
tags:
  - convolution
  - gray-scale
  - tensor
  - kernel
  - padding
  - stride
Requirement:
---
---
# Ảnh trong máy tính
## Hệ màu RGB
![](https://i.imgur.com/5PX0qa0.png)

## Ảnh màu

![](https://i.imgur.com/UBUxKZi.png)

- Ảnh màu kích thước 800x600 pixel có thể biểu diễn dưới dạng ma trận 600x800 
	$$\begin{bmatrix}w_{1,1}&w_{1,2}&...&w_{1,800}\\ w_{2,1}&w_{2,2}&...&w_{2,800} \\ ...&...&...&... \\ w_{600,1}&w_{600,2}&...&w_{600,800} \end{bmatrix}$$
- Trong đó, mỗi $w_{ij}$ là một pixel: $$w_{ij}=(r_{ij},g_{ij},b_{ij})$$
- Tổng quát: $$
\begin{bmatrix}
w_{1,1} & w_{1,2} & \dots & w_{1,800} \\
w_{2,1} & w_{2,2} & \dots & w_{2,800} \\
\vdots & \vdots & \ddots & \vdots \\
w_{600,1} & w_{600,2} & \dots & w_{600,800}
\end{bmatrix}
=
\begin{bmatrix}
(r_{1,1},g_{1,1},b_{1,1}) & (r_{1,2},g_{1,2},b_{1,2}) & \dots & (r_{1,800},g_{1,800},b_{1,800}) \\
(r_{2,1},g_{2,1},b_{2,1}) & (r_{2,2},g_{2,2},b_{2,2}) & \dots & (r_{2,800},g_{2,800},b_{2,800}) \\
\vdots & \vdots & \ddots & \vdots \\
(r_{600,1},g_{600,1},b_{600,1}) & (r_{600,2},g_{600,2},b_{600,2}) & \dots & (r_{600,800},g_{600,800},b_{600,800})
\end{bmatrix}



\Rightarrow
\begin{bmatrix} R \end{bmatrix}, 
\begin{bmatrix} G \end{bmatrix}, 
\begin{bmatrix} B \end{bmatrix}

$$
$$=>\begin{bmatrix}R\end{bmatrix},\begin{bmatrix}G\end{bmatrix},\begin{bmatrix}B\end{bmatrix}$$
## Tensor là gì

## Ảnh xám
- Ảnh xám cũng là ảnh với ma trận 600x800
- Mỗi pixel là 1 giá trị nguyên trong khoảng $[0,255]$ 
## Chuyển hệ màu ảnh
- Chuyển ảnh màu -> xám:
	$$X=r*0.299+g*0.587+b*0.114$$
- Với X là giá trị pixel của ảnh xám 

# Phép tích chập - Convolution

## Convolution
Kí hiệu: $\otimes$ 
![](https://i.imgur.com/FH6Oo5L.png)

- Công thức: $$Y(i,j)=\sum_{p=0}^{k-1}\sum_{q=0}^{k-1}X(i+p,j+q) \cdot W(p,q)  $$
	- Trong đó:
		- Ma trận $X$ kích thước $m*n$ 
		- Ma trận $W$ kích thước $k*k$ 
		- Matrix $Y$ có kích thước: $(m-k+1)*(n-k+1)$ 
## Paddding
- Mỗi lần thực hiện phép tính #convolution thì kích thước ma trận Y đều nhỏ hơn X 
- Nếu muốn ma trận Y thu được có kích thước bằng ma trận X -> tìm cách giải quyết các phần tử ở viền -> thêm giá trị `0` ở ngoài viền ma trận $X$ 
![](https://i.imgur.com/HYFa78X.png)

- Phép này gọi là `padding=1` 
- `padding=k`: thêm k vector vào mỗi phía của matrix 
## Stride

- `Stride`: bước nhảy của phép #convolution 
- VD: `stride=2` Kernel di chuyển từng bước 2, bỏ qua 1 phần tử -> giảm kích thước đầu ra 
- Công thức tính kích thước đầu ra:$$O=\frac{I-K}{S}+1$$
	- Trong đó:
		- $I$ là kích thước đầu vào, Input
		- $K$ kernel 
		- $S$ : stride
## Ý nghĩa 
- Mục đích của convolution trên ảnh:
	- Làm mờ
	- Làm nét
	- Xác định cạnh
- Mỗi kernel khác nhau thì phép convolution sẽ có ý nghĩa khác nhau 
![](https://i.imgur.com/3gG0SmZ.png)
