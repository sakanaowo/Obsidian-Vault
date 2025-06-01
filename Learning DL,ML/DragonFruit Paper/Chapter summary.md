

## Chương 1. Giới thiệu

- Nhận diện bệnh trên cây thanh long là một thách thức quan trọng trong nông nghiệp Việt Nam.
    
- Học sâu (Deep Learning) đang dần chứng minh hiệu quả trong việc nhận dạng hình ảnh lá bệnh.
    
- Vấn đề nổi bật: thiếu dữ liệu cho tất cả các loại bệnh → khó huấn luyện mô hình toàn diện.
    
- Giải pháp: phát triển mô hình **Học dựa trên Kết nối Ngữ nghĩa (SCL)** để học nhãn mới mà không làm quên nhãn cũ.
    

---

### Chương 1.1. Các nghiên cứu liên quan

- Học tăng cường lớp (incremental learning) thường gặp vấn đề “quên” thông tin cũ.
    
- Một số giải pháp hiện tại:
    
    - Lưu mẫu cũ
        
    - Mô hình giáo viên-học sinh
        
    - Kết hợp đặc trưng nhúng (embedding) và trích xuất prototype
        
- Các mô hình tiền huấn luyện (pre-trained models) và loss function cải tiến đang được tận dụng.
    

---

### Chương 1.2. Động lực và đóng góp

- Xây dựng mô hình SCL giúp:
    
    - Học nhãn mới hiệu quả
        
    - Không làm ảnh hưởng đến lớp cũ
        
- Đóng góp chính:
    
    1. Thiết kế mô hình SCL tối ưu chi phí
        
    2. Tạo tập dữ liệu bệnh trên thanh long (**D-Dragon**)
        
    3. Đánh giá mô hình trên cả tập mới và so với các mô hình hiện tại
        

---

## Chương 2. Phương pháp đề xuất

- **2.1. Định nghĩa bài toán:**  
    Huấn luyện mô hình học liên tục để phân loại, đảm bảo không làm giảm hiệu suất khi thêm lớp mới.
    
- **2.2. Kiến trúc mô hình:**  
    Mô hình SCL gồm không gian làm việc kết nối nhãn cũ và mới, học song song và tích lũy thông tin.
    
- **2.3. Tiền huấn luyện:**  
    Sử dụng mô hình đã được huấn luyện trước để tiết kiệm chi phí và hỗ trợ huấn luyện tiếp theo.
    

---

### **Chương 3. Mô tả mô hình**

- **3.1. Kết nối ngữ nghĩa:**  
    Vector đặc trưng được nhúng từ ảnh và liên kết với tập prototype để mô hình học được bản chất của từng nhãn.
    
- **3.2. Không gian làm việc:**  
    Là khu vực tổng hợp các prototype từ các nhãn cũ và mới, giúp cân bằng quá trình phân loại.
    
- **3.3. Hàm mất mát:**  
    Hàm mất mát kép (dual loss) giúp mô hình học chính xác hơn và đảm bảo hội tụ khi thêm lớp mới.
    

---

### **Chương 4. Kết quả thực nghiệm và phân tích**

- **4.1. Dữ liệu:**  
    Tập D-Dragon gồm 12 nhãn (11 bệnh + 1 khỏe), mỗi nhãn ~500 ảnh. Dữ liệu được huấn luyện theo từng giai đoạn.
    
- **4.2. Cấu hình:**  
    Dùng GPU 4090, huấn luyện bằng Pytorch + ViT, dùng SGD. Hội tụ sau 38 epochs.
    
- **4.3. So sánh hiệu suất:**  
    SCL vượt trội hơn các mô hình khác như iCaRL, DER, MEMO, EASE. Cao hơn EASE từ 1–2% độ chính xác.
    
- **4.4. Đánh giá định tính:**  
    SCL dự đoán tốt trên các nhãn: Đốm nâu, Mắt cua, Nấm cành, Đốm đen, Thối đầu trái. Việc thêm nhãn không làm giảm hiệu suất.
    

---

### **Chương 5. Kết luận**

- SCL là một hướng tiếp cận hiệu quả để mở rộng phân loại bệnh mà không làm quên thông tin cũ.
    
- Mô hình có khả năng tổng hợp đặc trưng ngữ nghĩa tốt.
    
- Hạn chế: cần nhiều bộ nhớ → hướng tương lai là tối ưu hóa kích thước mô hình và không gian làm việc.
    

---

Bạn có muốn tôi xuất tổng kết này thành một file riêng (PDF/TXT/Markdown) không? Hoặc tích hợp vào tài liệu chính?