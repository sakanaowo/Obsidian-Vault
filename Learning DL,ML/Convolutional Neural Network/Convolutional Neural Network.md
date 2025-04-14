---
aliases:
  - Mạng tích chập
  - CNN
tags:
  - CNN
  - convolution
  - kernel
  - layer
  - neural-network
  - convolutional-layer
  - pooling-layer
Requirement:
  - "[[Neural Network]]"
  - "[[Xử lí ảnh|Xử lý ảnh]]"
---
---
# Thiết lập bài toán
Bài toán: Input ảnh màu 64x64, output ảnh có chứa mặt người hay không
# Convolutional Neural Network 
## Convolutional Layer
![](https://i.imgur.com/oCuMQOv.png)

- Mỗi **hidden layer** được gọi là **fully connected layer**
- Cả model được gọi là **FCN** - **Fully Connnected Neural Network**
### Vấn đề của FCN với xử lý ảnh
- Ảnh màu 64x64 được biểu diễn dưới dạng #tensor 
-> để biểu thị hết nội dung ảnh -> truyền vào input layer tất cả $64*64*3=12288$ pixel -> input layer chứa 12288 nodes 
- Giả sử số lượng node trong layer là `1000` 
	-> lượng trọng số $W$ giữa `input layer` và `hidden layer` là $12288*1000=12288000$ 
	-> Số lượng #bias là 1000 
	-> Tổng tham số ~ `parameter` là 12289000
	-> Nếu kích thước ảnh tăng -> số lượng parameter tăng -> cần giải pháp tối ưu 
### Nhận xét:
- Các pixel cạnh nhau thường liên kết với nhau hơn là những pixel cách xa nhau 
- Trong phép #convolution trong ảnh, chỉ 1 #kernel được dùng trên toàn bộ ảnh -> Các pixel ảnh **chia sẻ hệ số** với nhau
=> Áp dụng phép #convolution vào layer trong #neural-network -> có thể giải quyết được vấn đề về số lượng lớn parameter
### Convolutional layer đầu tiền 
- Với ảnh có 3 kênh màu -> định nghĩa #kernel là #tensor bậc 3 với kích thước $k*k*3$ 
![](https://i.imgur.com/N3PcM04.png)

- Định nghĩa #kernel có cùng độ sâu (_depth_) với biểu diễn ảnh -> thực hiện chuyển khối #kernel tương tự khi thực hiện trên #gray-scale  
![](https://i.imgur.com/jFReTe8.png)

![](https://i.imgur.com/VgwxU7k.png)

- Với mỗi #kernel khác nhau -> học được những đặc trưng khác nhau của ảnh
-> trong mỗi #convolutional-layer, ta sẽ dùng nhiều #kernel để học được **nhiều thuộc tính** của ảnh 
- Mỗi kernel cho ra output là 1 matrix -> k kernel cho ra k matrix -> tensor bậc 3 
![](https://i.imgur.com/y6BQ9be.png)

### Convolutional Layer tổng quát 
- Cho:
	- Input của 1 #convolutional-layer là #tensor có kích thước $H*W*D$ 
	- Kernel có kích thước $F*F*D$
		- $F$ : số lẻ
		- $S$: stride
		- $P$: padding
-> ouput của layer là tensor bậc 3 có kích thước: $$(\frac{H-F+2P}{S}+1)*(\frac{W-F+2P}{S}+1)*K$$

![](https://i.imgur.com/aERYpaa.png)

- Output của #convolutional-layer sẽ qua hàm #activation-function trước khi thành input của convolutional layer tiếp theo 
- **Tổng số parameter của layer**:
	- Mỗi kernel có kích thước $F*F*D$ và có 1 hệ số #bias 
	-> **Tổng parameter của kernel** là $F*F*D+1$ 
	- Convolutional layer áp dụng K kernel 
	=> **Tổng số Parameter trong layer**: $K*(F*F*D+1)$  

## Pooling Layer
- Thường được dùng **giữa các Convolution layer** để giảm kích thước nhưng vẫn giữ thuộc tính quan trọng của dữ liệu  -> giảm tính toán trong model 
- Ở dưới là ví dụ pooling size $K*K$ 
![](https://i.imgur.com/WqljRjf.png)

- Với mỗi ma trận trên vùng kích thước $K*K$ thì tìm **maximum** hoặc **AVG**
- Hầu hết khi dùng #pooling-layer  thì sẽ dùng size = 2x2, stride=2, padding=0 -> output width và height của dữ liệu sẽ giảm đi 1 nửa, depth giữ nguyên 
![](https://i.imgur.com/2lb1rSd.png)

## Fully Connected Layer
## Visualize CNN
Mô hình CNN:
Input image -> Convolutional Layer + Pooling Layer -> FC -> Output 
# Mạng VGG 16 
- Kiến trúc mạng CNN được đề suất bởi K. Simonyan và A. Zisserman từ ĐH Oxford 
- Model được train bởi mạng này có độ chính xác đến 92.7% 
- Kiến trúc:
	![](https://i.imgur.com/f5pC754.png)

- Phân tích:
	- Convolutional Layer: 
		- size: 3x3
		- padding = 1
		- stride = 1
	- Pool/2: 
		- Max pooling layer với size 2x2
	- 3x3 conv, 64:
		- 64 kernel được áp dụng ~ depth của output của layer đó
	- Các conv layer càng về sau thì kích cỡ WxH giảm nhưng depth tăng  