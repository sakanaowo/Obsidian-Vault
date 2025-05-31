---
aliases:
  - Kết quả thực nghiệm và phân tích
tags:
---


## 4. Kết quả thực nghiệm và phân tích

### 4.1. Mô tả tập dữ liệu

Chúng tôi thu thập **11 nhãn tương ứng với các bệnh trên thân cây thanh long** và **1 nhãn cho cây khỏe mạnh**, mỗi nhãn khoảng **500 ảnh**. Bộ dữ liệu này được đặt tên là **D-Dragon**, dùng để **phân tích và đánh giá hiệu quả của mô hình đề xuất** trong việc **phân loại bệnh trên thân cây thanh long**.

Dựa vào các **đặc trưng được trích xuất** từ D-Dragon, chúng tôi tiến hành phân loại ảnh bị sâu bệnh và ảnh không bị. Nếu ảnh bị bệnh, mô hình sẽ xác định **loại bệnh cụ thể**.

Trong quá trình huấn luyện, dữ liệu được chia theo lớp. **5 lớp đầu** được sử dụng để huấn luyện ban đầu, các lớp còn lại chia làm **2 nhóm nhỏ**, thêm dần vào các giai đoạn tiếp theo. Chúng tôi sử dụng **phương pháp xáo trộn ngẫu nhiên thứ tự lớp** để đảm bảo **công bằng trong huấn luyện liên tục**.

![](Pasted%20image%2020250531162911.png)

> **Từ khóa:** `D-Dragon`, `bệnh trên thanh long`, `phân loại`, `tập dữ liệu`, `chia lớp`, `xáo trộn`, `công bằng`

---

### 4.2. Cấu hình thực nghiệm

Các thí nghiệm được thực hiện trên GPU **Nvidia 4090 25GB**, sử dụng **framework Pytorch**. Chúng tôi **tinh chỉnh mô hình tiền huấn luyện ViT** và đánh giá trên 2 tập: **ImageNet-mini** và **D-Dragon**.

Trong mô hình SCL, chúng tôi sử dụng **thuật toán tối ưu SGD**, với **tốc độ học 0.001**, **tham số cân bằng 0.01**, và **quá trình huấn luyện hội tụ sau 38 epoch**.

---

**Nghiên cứu tập trung vào 2 câu hỏi:**

- **RQ1**: Mô hình SCL cải thiện độ chính xác bao nhiêu so với các phương pháp nền trên 2 tập dữ liệu?
    
- **RQ2**: Độ chính xác dự đoán của mô hình SCL gần với giá trị thực tế như thế nào?
    

> **Từ khóa:** `SGD`, `ViT`, `fine-tuning`, `ImageNet-mini`, `câu hỏi nghiên cứu`, `tối ưu`, `tốc độ học`, `cân bằng`

---

### 4.3. So sánh hiệu năng SCL với các mô hình khác (RQ1)

Chúng tôi so sánh mô hình **SCL** với các phương pháp học tăng cường lớp khác trên 2 tập. **Bảng 1** trình bày kết quả so sánh. Kết quả cho thấy **SCL đạt độ chính xác cao nhất**, vượt trội hơn cả mô hình hiện tại tốt nhất **EASE** với mức tăng từ **1-2%**.


| Methods    | Exemplars |   Datasets    |                 |
| ---------- | --------- | :-----------: | :-------------: |
|            |           | ImageNet-mini | D-Dragon (ours) |
| iCaRL      | 5         |     70.56     |      82.64      |
| DER        | 5         |     78.25     |      89.73      |
| MEMO       | 5         |     73.84     |      85.68      |
| EASE       | 5         |     79.67     |      90.24      |
| SCL (ours) | 5         |     80.34     |      91.89      |


Mỗi mô hình đều được huấn luyện ban đầu với **5 lớp**, sau đó thêm dần các lớp mới. SCL **duy trì độ chính xác ổn định** trong suốt quá trình huấn luyện.

> **Từ khóa:** `hiệu năng`, `so sánh`, `học tăng cường lớp`, `bảng so sánh`, `SOTA`, `độ chính xác`, `EASE`, `SCL`

---

### 4.4. Đánh giá định tính (RQ2)

Chúng tôi tiến hành đánh giá định tính trên **5 nhãn bệnh cụ thể**, bao gồm:

- “**Đốm nâu**”
    
- “**Mắt cua**”
    
- “**Nấm cành**”
    
- “**Đốm đen**”
    
- “**Thối đầu trái**”
    

Đây là các loại bệnh phổ biến trên **lá và quả cây thanh long**, ảnh hưởng lớn đến **năng suất của nông dân**. Mô hình **SCL** thực hiện **dự đoán chính xác** và **không bị ảnh hưởng tiêu cực** khi thêm nhãn mới.

Ảnh đầu vào có **kích thước 1024x480**, kết quả hiển thị ở **Hình 3**, minh họa **ảnh gốc và kết quả dự đoán**.

> **Từ khóa:** `đánh giá định tính`, `nhãn bệnh`, `kết quả dự đoán`, `thêm nhãn`, `năng suất`, `kích thước ảnh`, `minh họa`
