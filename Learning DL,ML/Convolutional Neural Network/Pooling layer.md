---
aliases:
  - Pooling
  - pooling
  - Pooling Layer
tags:
  - pooling-layer
  - CNN
Reference:
  - "[[Resnet Overview|Resnet]]"
  - "[[Convolutional Neural Network|CNN]]"
  - "[[Neural Network|Neural network]]"
---

Pooling layer (lớp pooling) là một **thành phần quan trọng trong mạng nơ-ron tích chập (CNN)**, giúp giảm kích thước không gian (chiều rộng và chiều cao) của đặc trưng đầu vào trong khi vẫn giữ lại thông tin quan trọng nhất.
# 🎯 **Mục đích của Pooling Layer**

1. **Giảm chiều dữ liệu** → giúp giảm số lượng tham số và tính toán.
    
2. **Ngăn overfitting** bằng cách tạo ra một biểu diễn trừu tượng hơn.
    
3. **Tăng tính bất biến đối với dịch chuyển nhỏ** (translation invariance).
    

---

# 🧩 **Các loại Pooling phổ biến**

| Loại                | Cách hoạt động                                                    |
| ------------------- | ----------------------------------------------------------------- |
| **Max Pooling**     | Lấy giá trị **lớn nhất** trong mỗi vùng nhỏ (ví dụ 2x2).          |
| **Average Pooling** | Tính **trung bình** các giá trị trong vùng nhỏ.                   |
| **Global Pooling**  | Áp dụng pooling toàn bộ bản đồ đặc trưng → ra 1 giá trị duy nhất. |

---

# 📐 **Ví dụ: Max Pooling 2x2**

Giả sử bạn có ma trận 4x4:

```lua
[[1, 3, 2, 4],
 [5, 6, 1, 2],
 [7, 8, 9, 0],
 [4, 3, 2, 1]]
```

Sau khi áp dụng **Max Pooling 2x2**, stride = 2, bạn được:

```lua 
[[6, 4],
 [8, 9]]
```

> Mỗi ô trong ma trận kết quả là giá trị lớn nhất từ một vùng 2x2 tương ứng.

---

# 🧠 **Khi nào dùng Pooling?**

- Sau các lớp **convolution**, thường sẽ có một lớp pooling.
    
- Trong các kiến trúc CNN hiện đại như **ResNet**, pooling được dùng ở đầu và gần cuối mạng (Global Average Pooling).
    

---

# ⚠️ Lưu ý

- Pooling không học tham số như convolution, mà chỉ **áp dụng toán học** cố định.
    
- Một số mô hình mới thay pooling bằng **stride convolution** hoặc bỏ luôn để giữ thông tin chi tiết hơn (ví dụ trong segmentation).
    
