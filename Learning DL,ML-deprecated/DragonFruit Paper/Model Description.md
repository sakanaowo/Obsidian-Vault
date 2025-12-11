

## 3. Mô tả mô hình

### 3.1. Học dựa trên Kết nối Ngữ nghĩa

Từ các **mạng CNN nền tảng (backbone)**, chúng tôi thu được các **vector đặc trưng nhúng (embedding)** toàn diện. Trên cơ sở đó, chúng tôi xây dựng các **bộ phân loại (classifier)** dựa trên mô hình **prototype**, nhằm hỗ trợ quá trình **dự đoán**.

Cụ thể, mỗi **không gian làm việc (workspace)** sẽ chứa thông tin embedding để thực hiện các **nhiệm vụ phân loại cụ thể** tương ứng với từng **nhãn**.

Sau khi chuyển đổi từ ảnh sang **vector đặc trưng nhúng**, dữ liệu trở nên **nhẹ hơn đáng kể** và yêu cầu **ít tham số hơn** so với ảnh gốc. Đồng thời, **chi phí lưu trữ các embedding này cũng thấp**.

---

Khi đã có **tập nhúng tổng thể**, chúng tôi tiến hành xây dựng **kết nối giữa không gian embedding và các tập nhãn tương ứng**.

Việc trích xuất prototype từ lớp thứ $i$ trong không gian kết nối được thực hiện như sau:

$$P_i = \frac{1}{N} \sum_{j=1}^{N} I(y_j = i)$$

Trong đó:

- $N$: số lượng mẫu dữ liệu trong lớp $i$
    
- $P_i$: **tập kết nối prototype** của lớp $i$
    
- $I$: **phương pháp trích xuất prototype**
    

---

Khi **thêm một lớp mới**, **tập kết nối** và **không gian làm việc** sẽ thực hiện nhiệm vụ cập nhật. Ví dụ, nếu có không gian làm việc $W_1$ với tập kết nối $P = \{P_1, P_2, ..., P_N\}$, khi thêm một **nhãn mới** PnewP_{new}, chúng tôi sẽ cập nhật W1W_1 và PP bằng cách **tính lại trọng số và tổng hợp prototype của các lớp cũ** trong không gian mới.

Để **tránh xung đột** giữa không gian làm việc cũ và mới, các **tập kết nối (connection sets)** sẽ đảm nhận nhiệm vụ **tìm các kết nối mới có độ tương đồng gần nhất**.

Độ tương đồng giữa prototype mới PiP_i và một prototype cũ PjP_j được tính như sau:

$$sim_{i,j} = \frac{P_i}{\|P_i\|_2} \cdot \frac{P_j^T}{\|P_j\|_2}$$

---

**Giải thích:**

- **Độ tương đồng (similarity)** phản ánh **mối quan hệ cục bộ** giữa các lớp mới và cũ
    
- Đây là cơ chế chia sẻ **không gian làm việc**, giúp **cân bằng giữa lớp cũ và mới**
    
- Sau khi tính toán, các **liên kết giữa các lớp được thiết lập** để xây dựng **không gian làm việc mới**
    

> **Từ khóa:** `mạng CNN`, `embedding`, `bộ phân loại`, `prototype`, `không gian làm việc`, `nhãn mới`, `trích xuất đặc trưng`, `tập kết nối`, `độ tương đồng`, `tổng hợp prototype`, `tránh xung đột`

---

### 3.2. Không gian làm việc (Working Space)

**Bản chất của không gian làm việc (workspace)** là **khu vực trung gian** để chuyển đổi giữa các **tập kết nối (connection sets)** thuộc các lớp khác nhau. **Đầu vào** là các **vector đặc trưng nhúng** từ ảnh.

Không gian làm việc sẽ **tổng hợp nhiều prototype** đã được nhúng từ các **tập kết nối khác nhau**. Các tập kết nối này trích xuất các đặc trưng cụ thể **theo từng nhiệm vụ của từng lớp**. Nhờ đó, các prototype rất phù hợp cho việc thực hiện **nhiệm vụ phân loại** và hỗ trợ trong **lớp dự đoán cuối cùng**.

---

Trong quá trình huấn luyện, chúng tôi tìm kiếm **một không gian làm việc phù hợp cho mỗi nhiệm vụ**, nhằm **mã hóa thông tin cần thiết**. Việc trích xuất các thông tin nhúng từ ảnh đầu vào chính là **quá trình kết nối và tổng hợp prototype từ các lớp trước đó**.

Cuối cùng, không gian làm việc sẽ thực hiện việc **phân loại cân bằng** trên tất cả các lớp để **tạo ra dự đoán chính xác**. Do đó, sau khi huấn luyện xong, **không gian làm việc chỉ lưu trữ các tập kết nối**, không còn cần các vector đặc trưng nhúng nữa.

> **Từ khóa:** `không gian làm việc`, `prototype`, `tập kết nối`, `trích xuất nhúng`, `phân loại cân bằng`, `lưu trữ`, `kết nối đặc trưng`

---

### **3.3. Hàm mất mát (Loss Function)**

**Mục tiêu chính** của mô hình là **phân loại hình ảnh bệnh và sâu trên cây thanh long**. Chúng tôi xây dựng một **hàm mất mát kép (dual loss)** giữa tập nhãn $y$ và các tập kết nối đã được **chuẩn hóa** $y′$ sinh ra từ mô hình đã huấn luyện:

$$L_{ESL}(y, y') = \frac{1}{N} \sum_{j=1}^{N} (y_j - y'_j)^2$$

Để đảm bảo **việc cập nhật hiệu quả không gian làm việc**, chúng tôi xây dựng một **hàm mất mát riêng biệt cho từng tập kết nối được cập nhật**. Nếu một lớp cũ có tập kết nối đạt độ chính xác cao, khi thêm lớp mới, chúng tôi sử dụng **hàm mục tiêu từ mô hình cũ** như một mục tiêu hồi quy. Ngược lại, nếu không đạt, thì **đầu ra từ mô hình có tập kết nối mới sẽ được chọn làm thay thế**.

![](Pasted%20image%2020250531160320.png)

Điều này giúp **đánh giá hội tụ toàn diện** của tập kết nối và mô hình tổng thể ESL:

$$L_{connection-based}(c, c') = \sum_{k} smoothLoss(c_k - c'_k)$$

> **Từ khóa:** `hàm mất mát`, `phân loại ảnh bệnh`, `chuẩn hóa nhãn`, `mục tiêu hồi quy`, `tập kết nối`, `hội tụ mô hình`, `ESL`, `smoothLoss`
