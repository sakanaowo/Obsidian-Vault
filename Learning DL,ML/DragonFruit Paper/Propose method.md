### **2.1. Định nghĩa bài toán**

**Mô hình học kết nối ngữ nghĩa (Semantic Connection Learning – SCL)** là một mô hình học liên tục cho bài toán **phân loại**, nhằm tạo ra một **bộ phân loại thống nhất**. Giả sử có một **chuỗi các lớp** khác nhau $L$ trong tập huấn luyện $D_{training}$​, được ký hiệu:  
**L = L₁, L₂, ..., Lₙ**, trong đó $Li​=(X,yi​), i \in \mathbb{N}$, là tập huấn luyện thứ i với nhãn $y_i$

- $X$: các **đặc trưng dữ liệu** thuộc lớp $Y_i$​
    
- $y_i \in Y$, với $Y$ là **không gian nhãn** của bài toán **nhận diện bệnh trên cây thanh long**
    

**Mục tiêu** là đảm bảo rằng khi thêm các lớp mới vào quá trình huấn luyện, **hiệu suất mô hình không bị ảnh hưởng** và vẫn có thể **phân loại chính xác** các loại bệnh.

> **keyword:** `học liên tục`, `bộ phân loại thống nhất`, `dữ liệu đặc trưng`, `không gian nhãn`, `phân loại bệnh`, `thanh long`

---

### **2.2. Kiến trúc mô hình**

![Hình 1](https://i.ibb.co/5X9hY7p7/image.png)


Như minh họa , chúng tôi đề xuất một **mô hình học kết nối ngữ nghĩa (SCL)** để tăng cường khả năng học cho cả **nhãn cũ và mới**.

Ban đầu, mô hình được **huấn luyện với tập dữ liệu đã gán nhãn**. Khi có thêm dữ liệu mới, mô hình mới được **khởi tạo**, tích hợp các **nhãn mới vào không gian làm việc**. Mô hình **SCL được huấn luyện để tạo ra không gian làm việc** không thiên vị giữa nhãn cũ và mới, cho phép mô hình **học song song cả hai**.

Ngoài ra, mô hình **tổng hợp các nhãn chưa gán** để hỗ trợ **học chuyển giao** và **tổng hợp kiến thức**, đảm bảo mô hình có thể tiếp tục **học tích lũy trong tương lai**.

> **Từ khóa:** `SCL`, `không gian làm việc`, `học song song`, `nhãn mới`, `học chuyển giao`, `kiến thức tích lũy`

---

### **2.3. Mục tiêu tiền huấn luyện**

Trong bối cảnh hiện nay, việc **tái sử dụng mô hình đã huấn luyện trước (pre-trained)** rất phổ biến. Thay vì huấn luyện lại từ đầu, chúng ta có thể **kế thừa trọng số quan trọng** hoặc **sao chép các lớp then chốt**, rồi **tùy chỉnh các lớp cuối** để phù hợp với bài toán hiện tại:


```lua
model_old = copy_layer(model_pretrain(Li))     (1)
```

Sau khi có mô hình tiền huấn luyện, để tránh xung đột với lớp cũ, chúng tôi **đóng băng một số lớp**, và **mở rộng mô hình** bằng cách kết nối giữa các mô hình cũ:


```lua
model_new = working_space(keep(model_old1, model_old2, ..., model_oldN), model_old(Li))     (2)
```

Sau khi mô hình mới được tạo, **không gian làm việc sẽ cập nhật thông tin về các nhiệm vụ**, nhằm **dự đoán chính xác trong tương lai** và **tránh ghi đè thông tin nhãn mới lên nhãn cũ**. Điều này giúp **tối ưu hóa chi phí huấn luyện** và **nâng cao khả năng mở rộng trong tương lai**.

> **Từ khóa:** `mô hình tiền huấn luyện`, `sao chép lớp`, `không gian làm việc`, `đóng băng lớp`, `tránh xung đột`, `tối ưu chi phí`