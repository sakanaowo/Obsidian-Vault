---
aliases:
  - Resnet
tags:
  - deep-neural-network
  - neural-network
  - overfitting
  - slow-convergence
  - vanishing-gradient
  - exploding-gradient
  - convolution
  - batch-normalization
  - ReLU
Reference:
  - "[[Neural Network|Neural network]]"
  - "[[Vanishing Gradient|vanishing gradient]]"
  - "[[Convolutional Neural Network|CNN]]"
---

> [!NOTE] 1 trong Backbone của CNN
> Resnet ~ Residual Network,  kiến trúc #neural-network được giới thiệu bởi `Kaiming He` vào 2015. Resnet có độ chính xác vượt trội, khắc phục được 1 số vấn đề lớn của #deep-neural-network : suy giảm độ chính xác khi mạng càng sâu

# Ý tưởng chính
- Khi mạng #neural-network  càng sâu -> khó huấn luyện, do:
	- #vanishing-gradient 
	- #exploding-gradient
	- #overfitting hoặc #slow-convergence
- Để giải quyết điều này -> ResNet giới thiệu khối `residual` với ý tưởng học phần hiệu chỉnh (residual) giữa đầu vào và ra thay vì học toàn bộ hàm ánh xạ
- Công thức
	- $$F(x)=H(x)-x$$
- -> Đầu ra của khối residual là:
  $$y=F(x)+x$$
# Kiến trúc Residual Block

![](https://i.imgur.com/n6I5xEj.png)

![](https://i.imgur.com/5VN4lGs.png)


```lua
Input → [Conv → BN → ReLU → Conv → BN] + Input → ReLU → Output
```

- Hai lớp #convolution  đi kèm #batch-normalization và #ReLU 
- Sau đó cộng đầu vào gốc `x` với output nhánh chính 
# 3. Các phiên bản phổ biến

| Kiến trúc | Số lớp | Ứng dụng                              |
| --------- | :----: | ------------------------------------- |
| Resnet-18 |   18   | Nhẹ -> dùng cho edge/IoT              |
|           |   34   | Phù hợp cho tác vụ tầm trung          |
|           |   50   | Dùng nhiều trong nghiên cứu           |
|           |  101   | Tăng độ chính xác                     |
|           |  152   | Cấu trúc sâu -> dùng cho bài toán khó |

# Ưu điểm 
- Dễ dàng huấn luyện mạng sâu lên tới hàng trăm lớp.
    
- Giảm overfitting nhờ residual connection.
    
- Trở thành **kiến trúc nền tảng** cho nhiều mô hình khác: DenseNet, EfficientNet, Mask R-CNN, v.v.
# Ứng dụng
- Nhận dạng ảnh (ImageNet, CIFAR)
    
- Nhận diện khuôn mặt
    
- Phân đoạn ảnh y tế
    
- Phát hiện vật thể (object detection)

