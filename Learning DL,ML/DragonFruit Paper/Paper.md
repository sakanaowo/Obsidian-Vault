Dưới đây là bản dịch toàn bộ bài báo **"Semantic Connection-Based Learning for Dragon Fruit Disease Classification"** sang tiếng Việt:



## Học Dựa Trên Kết Nối Ngữ Nghĩa Cho Phân Loại Bệnh Trên Cây Thanh Long

**Tác giả:**

- Dương Mạnh Hà – Văn phòng Bộ Nông nghiệp và Phát triển Nông thôn, Hà Nội, Việt Nam
    
- Trần Hùng – Trung tâm Nghiên cứu và Chuyển giao Công nghệ Thủy lợi khu vực miền núi phía Bắc
    
- Nguyễn Xuân Kiều – Trung tâm Nghiên cứu và Chuyển giao Công nghệ Thủy lợi khu vực miền núi phía Bắc
    
- Nguyễn Gia Vương – Trung tâm Nghiên cứu và Chuyển giao Công nghệ Thủy lợi khu vực miền núi phía Bắc
    
- Quỳnh Đào Thị Thúy – Khoa CNTT, Học viện Công nghệ Bưu chính Viễn thông (PTIT), Phòng thí nghiệm IC4SD, Hà Nội, Việt Nam _(tác giả liên hệ)_
    

---

### Tóm tắt

Nhận dạng lá bị bệnh là một trong những nhiệm vụ quan trọng trong nông nghiệp. Tại Việt Nam, cây thanh long đặc biệt dễ bị mắc nhiều loại bệnh trong quá trình phát triển. Gần đây, các kỹ thuật trí tuệ nhân tạo (AI) đã đạt được nhiều thành tựu đáng kể trong nhận dạng hình ảnh. Do đó, việc áp dụng AI để phát hiện lá bị bệnh, đặc biệt là trên cây thanh long, là rất cần thiết.

Tuy nhiên, một thách thức lớn là cây thanh long có thể bị nhiều loại bệnh khác nhau – có 11 bệnh đã biết nhưng dữ liệu chỉ có cho 5 bệnh. Do đó, mô hình học sâu chỉ hiệu quả với 5 bệnh này, và việc thêm dữ liệu bệnh mới có thể ảnh hưởng đến độ chính xác.

Để giải quyết vấn đề này, chúng tôi đề xuất một mô hình học mới dựa trên **kết nối ngữ nghĩa (Semantic Connection-Based Learning – SCL)**, nhằm tạo ra các không gian phân loại riêng biệt cho từng nhãn. Mô hình có khả năng tích hợp đặc trưng mới từ cả lớp cũ và lớp mới để tăng độ chính xác. Thử nghiệm của chúng tôi đạt **độ chính xác 92%** khi thêm lớp mới.

**Từ khóa:** Nhận dạng lá bị bệnh; Học dựa trên kết nối ngữ nghĩa; Học sâu; Phân loại bệnh trên cây thanh long.

---

# 1. Giới thiệu

Trong những năm gần đây, ứng dụng của học sâu trong nông nghiệp đã đạt nhiều thành tựu. Đặc biệt, trong việc phát hiện sâu bệnh hại trên lá cây, nhiều nhà nghiên cứu đã ứng dụng AI và deep learning.

Việc phát hiện bệnh trên thân cây thanh long là một vấn đề đầy thách thức nhưng thú vị. Mặc dù khó khăn, việc phát triển mô hình deep learning để phát hiện sớm bệnh hại cây thanh long sẽ mang lại lợi ích thiết thực cho nông dân.

Học sâu có hiệu quả cao trong nhận dạng hình ảnh, nhưng thách thức lớn là khi mô hình học thêm dữ liệu mới thì có thể **quên** thông tin cũ (hiện tượng “catastrophic forgetting”). Một số nghiên cứu đã đề xuất mô hình có thể học thêm mà không làm mất đi kiến thức cũ.

Hầu hết mô hình CNN truyền thống học đặc trưng cố định và hoạt động tốt với các lớp được huấn luyện ban đầu. Tuy nhiên, khi học thêm lớp mới, sẽ xảy ra xung đột đặc trưng, làm giảm độ chính xác.

Một số mô hình hiện nay giải quyết vấn đề này bằng cách lưu đặc trưng cũ hoặc dùng kiến trúc mở rộng, nhưng chi phí lưu trữ và tính toán cao. Do đó, chúng tôi đề xuất một mô hình mới **mở rộng học lớp mới mà không quên lớp cũ**, giảm chi phí tính toán bằng cách xây dựng **không gian kết nối ngữ nghĩa** để đại diện cho các tác vụ.

---

## 1.1. Các nghiên cứu liên quan

Một số mô hình học sâu có khả năng học thêm lớp mới mà không ảnh hưởng đến lớp cũ, ví dụ như lưu đặc trưng cũ vào cơ sở dữ liệu hoặc điều chỉnh tham số mô hình. Một hướng tiếp cận khác là mô hình giáo viên – học sinh (teacher-student) để cân bằng khả năng dự đoán giữa lớp cũ và lớp mới.

Các yếu tố ảnh hưởng chính đến việc mở rộng mô hình là: (1) chi phí bộ nhớ để lưu đặc trưng, (2) thời gian tính toán để huấn luyện lớp mới.

---

## 1.2. Động lực và đóng góp

Chúng tôi đề xuất mô hình **Học Dựa Trên Kết Nối Ngữ Nghĩa (SCL)** nhằm:

1. Tạo **không gian đặc trưng** phục vụ cho việc học lớp mới mà không làm giảm độ chính xác lớp cũ.
    
2. Xây dựng **bộ dữ liệu D-Dragon**, gồm ảnh cây thanh long với các loại bệnh thu thập từ thực địa.
    
3. **Đánh giá hiệu quả mô hình SCL** so với các phương pháp hiện tại.
    

---

# 2. Phương pháp đề xuất

## 2.1. Định nghĩa bài toán

Giả sử ta có một tập huấn luyện gồm nhiều lớp: $L=L_1,L_2,...,L_N$, trong đó mỗi lớp $L_i = (X, y_i)$, với $y_i$ là nhãn của lớp $i$. Mục tiêu là đảm bảo mô hình vẫn phân loại chính xác khi thêm lớp mới vào.

## 2.2. Kiến trúc mô hình

Mô hình SCL hoạt động theo các bước:

- Huấn luyện mô hình ban đầu với dữ liệu có nhãn.
    
- Khi có dữ liệu mới, tạo **không gian làm việc** (working space) để tích hợp nhãn mới mà không ảnh hưởng tới nhãn cũ.
    
- Áp dụng học chuyển giao (transfer learning) và kết nối ngữ nghĩa giữa lớp mới – lớp cũ.
    

## 2.3. Mục tiêu tiền huấn luyện

Thay vì huấn luyện lại từ đầu, mô hình sử dụng các mô hình đã huấn luyện trước để tiết kiệm chi phí. Một số lớp được “đóng băng”, chỉ học thêm phần cần thiết.

---

# 3. Mô tả mô hình

## 3.1. Học dựa trên kết nối ngữ nghĩa

- Từ backbone CNN, trích xuất đặc trưng thành **vector embedding**.
    
- Tính trung bình đặc trưng theo lớp để tạo thành **prototype vector**.
    
- Khi thêm lớp mới, tính **độ tương đồng (similarity)** giữa prototype mới và cũ để cập nhật kết nối trong không gian làm việc.
    

## 3.2. Không gian làm việc

- Là nơi kết nối đặc trưng của ảnh đầu vào với các prototype để hỗ trợ dự đoán.
    
- Mỗi không gian làm việc chỉ lưu các prototype (không lưu embedding đầy đủ), giúp giảm chi phí lưu trữ.
    

## 3.3. Hàm mất mát

Gồm hai hàm chính:

- **ESL loss**: Đánh giá độ chính xác dự đoán giữa nhãn thật và nhãn từ prototype.
    
- **Loss dựa trên kết nối**: Giúp đảm bảo khi thêm lớp mới, không làm giảm hiệu quả lớp cũ.
    

---

# 4. Kết quả thực nghiệm và phân tích

## 4.1. Mô tả bộ dữ liệu

- Bộ dữ liệu **D-Dragon** gồm **12 nhãn** (11 loại bệnh + 1 nhãn khỏe mạnh), mỗi nhãn khoảng **500 ảnh**.
    
- Dữ liệu được chia làm nhiều pha huấn luyện: ban đầu huấn luyện 5 lớp, sau đó thêm dần lớp mới.
    

## 4.2. Cài đặt thực nghiệm

- Thực nghiệm trên GPU Nvidia RTX 4090, framework PyTorch.
    
- Sử dụng mô hình ViT (Vision Transformer) để tinh chỉnh.
    
- Tốc độ hội tụ tốt sau 38 epoch, learning rate = 0.001.
    

## 4.3. So sánh hiệu suất mô hình

So sánh mô hình SCL với các phương pháp khác như iCaRL, DER, MEMO, EASE. Kết quả:

| Phương pháp | ImageNet-mini (%) | D-Dragon (%) |
| ----------- | ----------------- | ------------ |
| iCaRL       | 70.56             | 82.64        |
| DER         | 78.25             | 89.73        |
| MEMO        | 73.84             | 85.68        |
| EASE        | 79.67             | 90.24        |
| **SCL**     | **80.34**         | **91.89**    |

SCL vượt trội hơn 1–2% so với các phương pháp tốt nhất hiện tại.

## 4.4. Nghiên cứu định tính

- Mô hình SCL được đánh giá với ảnh thực tế gồm các bệnh phổ biến như: **Đốm nâu**, **Mắt cá**, **Nấm cành**, **Đốm đen**, **Thối đầu trái**.
    
- Kết quả cho thấy mô hình phân loại rất tốt dù thêm lớp mới.
    

---

# 5. Kết luận

Chúng tôi đề xuất mô hình SCL giúp mở rộng khả năng học phân loại bệnh mới trên cây thanh long mà vẫn giữ độ chính xác cao. Mô hình tận dụng kết nối ngữ nghĩa để tạo không gian học mới mà không ảnh hưởng dữ liệu cũ.

Tuy nhiên, hạn chế là chi phí bộ nhớ cao. Trong tương lai, nhóm sẽ nghiên cứu cải tiến mô hình nhẹ hơn, tiết kiệm tài nguyên hơn.

---

### Lời cảm ơn

Nghiên cứu này được tài trợ bởi đề tài khoa học mã số **KC-4.0-11/19-25** của Bộ Khoa học và Công nghệ Việt Nam.
