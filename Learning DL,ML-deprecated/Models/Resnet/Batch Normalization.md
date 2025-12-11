---
aliases:
  - Chuẩn hóa Batch
  - BatchNorm
tags:
  - batch-normalization
  - CNN
  - deep-neural-network
  - slow-convergence
  - vanishing-gradient
  - exploding-gradient
Reference:
  - "[[Linear Regression]]"
  - "[[Convolutional Neural Network|CNN]]"
  - "[[Phương sai]]"
---
# Tổng quan
#batch-normalization là kĩ thuật đẩy nhanh huấn luyện #deep-neural-network  và làm cho mô hình ổn định hơn bằng cách chuẩn hóa đầu ra 
# Vấn đề cần giải quyết

Khi huấn luyện #deep-neural-network , phân phối của đầu ra từ các lớp trung gian thay đổi liên tục -> #slow-convergence -> khó học

`BatchNorm ` giải quyết bằng cách:
- Giữ phấn phối đầu ra của mỗi lớp có kì vọng (mean) gần 0 và phương sai (variance) gần 1
# BatchNorm hoạt động như thế nào?

Giả sử có 1 batch đầu vào $x\in R^{B*D}$ với:
- B: số mẫu trong batch
- D: số đặc trưng
Quá trình chuẩn hóa:
1. Tính trung bình ~ mean:
	$$\mu =\frac{1}{B}\sum_{i=1}^{B}x_i$$
2. Tính phương sai ~ #variance :
	$$\sigma^2=\frac{1}{B}\sum_{i=1}^{B}(x_i^2-\mu)^2$$
3. Chuẩn hóa:
	$$\hat x_i=\frac{x_i-\mu}{\sqrt{\sigma^{2}+\epsilon}}$$
4. Áp dụng scale&shift:
	$$y_{i}=\gamma \hat x_i+\beta$$
Trong đó:
- $\gamma$ : hệ số scale (học được)
- $\beta$: Hệ số dịch chuyển (học được)
- $\epsilon$: Hằng số nhỏ để tránh chia cho 0
# Lợi ích của BatchNorm

- Tăng tốc huấn luyện
- Giảm #exploding-gradient , #vanishing-gradient 
- Tối ưu ổn định
- Giảm nhu cầu regularization
- Hỗ trợ learning rate cao 

# Vị trí 

Thông thường
```lua
Linear/Conv → BatchNorm → ReLU
```

Trong Resnet
```lua
Conv → BN → ReLU → Conv → BN → Add
```
