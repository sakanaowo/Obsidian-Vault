---
aliases: 
tags:
---


## 5. Kết luận

Trong nghiên cứu này, chúng tôi hướng đến việc **cải tiến phương pháp học tăng cường nhãn mới (incremental learning)** thông qua **học ngữ nghĩa (semantic learning)** bằng cách kết nối giữa **nhãn và đặc trưng**, nhằm tăng cường khả năng học trong môi trường thực tế.

Chúng tôi đã đề xuất phương pháp **Học dựa trên Kết nối Ngữ nghĩa (Semantic Connection-based Learning – SCL)** để mở rộng lớp trên nền các **mô hình đã được huấn luyện trước**, từ đó hỗ trợ **nhận diện bệnh trên cây thanh long**.

---

Cụ thể, chúng tôi đã:

- Xây dựng **không gian làm việc đa dạng** thông qua các **tập kết nối (connection sets)**
    
- Tổng hợp **đặc trưng ngữ nghĩa**, giúp mô hình hiểu được **đặc trưng đại diện (prototype)** của các lớp
    
- Tạo ra các tập kết nối toàn diện, hỗ trợ thêm nhãn mới mà **không làm giảm chất lượng dự đoán** đối với nhãn cũ
    

Ngoài ra, chúng tôi cũng xây dựng một **tập dữ liệu bệnh trên cây thanh long**, một loại cây có giá trị cao trong **nông nghiệp Việt Nam**. Các thực nghiệm đã chứng minh **hiệu quả vượt trội của mô hình SCL**.

---

Tuy nhiên, nhược điểm hiện tại của mô hình là **yêu cầu bộ nhớ lớn** để lưu trữ toàn bộ mô hình. Vì vậy, trong tương lai, chúng tôi sẽ tập trung vào:

- Thiết kế mô hình nhẹ hơn
    
- **Tối ưu hóa không gian kết nối** nhằm **giảm tiêu thụ bộ nhớ**
    

> **Từ khóa:** `kết luận`, `học tăng cường`, `học ngữ nghĩa`, `mô hình tiền huấn luyện`, `bệnh cây thanh long`, `không gian làm việc`, `prototype`, `tối ưu bộ nhớ`, `hướng phát triển`
