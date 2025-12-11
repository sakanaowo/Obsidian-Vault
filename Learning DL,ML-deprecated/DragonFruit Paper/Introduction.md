Trong những năm gần đây, các ứng dụng của học sâu (deep learning) trong lĩnh vực nông nghiệp đã đạt được nhiều bước tiến quan trọng. Cụ thể, trong lĩnh vực nhận diện sâu bệnh trên lá cây, nhiều nhà nghiên cứu trên khắp thế giới đã triển khai các công nghệ mới như trí tuệ nhân tạo (AI) và học sâu.

Nhận diện bệnh trên thân cây thanh long là một trong những bài toán vừa thách thức vừa thú vị. Dù tồn tại nhiều khó khăn, việc phát triển một mô hình học sâu giúp phát hiện sớm các bệnh trên thân cây thanh long sẽ là một thành công nhỏ nhưng mang lại lợi ích lớn cho nông dân và ngành nông nghiệp nói chung.

Sự phát triển mạnh mẽ của các công nghệ AI đã đem lại hiệu quả đáng kể cho các ứng dụng thực tế trong lĩnh vực nhận dạng hình ảnh. Khi công nghệ ngày càng hiện đại, lượng dữ liệu cũng tăng theo cấp số nhân, đòi hỏi các hệ thống phải có khả năng xử lý lượng dữ liệu mới này. Thách thức lớn nhất của các mô hình học sâu là việc học các lớp dữ liệu mới có thể ảnh hưởng đến mô hình đã học trước đó, gây ra hiện tượng “quên” những đặc trưng cũ.

Một số nghiên cứu đã đề xuất xây dựng các mô hình học sâu có khả năng tiếp thu lớp dữ liệu mới mà không làm mất đi các đặc trưng cũ, từ đó thể hiện hiệu suất vượt trội và đáp ứng yêu cầu của các ứng dụng thực tiễn hiện nay.

Các mô hình mạng nơ-ron tích chập (CNN) thường tổng quát hóa các đặc trưng và học để tạo ra các trọng số cố định nhằm thực hiện nhiệm vụ phân loại. Vì quá trình huấn luyện thường dựa trên các lớp cố định, nên quá trình dự đoán khá chính xác và đáng tin cậy. Tuy nhiên, việc học thêm đặc trưng mới có thể gây xung đột với các đặc trưng cũ, dẫn đến mô hình dự đoán không tối ưu.

Một số mô hình đã được phát triển nhằm mục tiêu học đặc trưng mới trong khi vẫn giữ lại các đặc trưng cũ. Hầu hết các mô hình này phải đánh đổi giữa sự ổn định về độ chính xác và tính linh hoạt khi thêm đặc trưng mới. Trong thời đại dữ liệu ngày càng tăng, nhu cầu về sự linh hoạt như các mô hình mạng mở rộng là rất cần thiết để xử lý việc thêm nhãn và đặc trưng mới.

Đa số các mô hình này được tối ưu để đảm bảo việc thêm nhãn mới không ảnh hưởng đến nhãn cũ thông qua các cơ chế như cố định đặc trưng cũ hoặc kết hợp đặc trưng vào các nhiệm vụ con. Ngoài ra, còn nhiều cơ chế khác cũng được áp dụng.

Để xây dựng các mô hình học sâu có khả năng nhận diện đa dạng bệnh trên cây thanh long và có thể học thêm lớp mới trong tương lai, chúng tôi đã tối ưu hóa một mạng mở rộng dựa trên học kết nối ngữ nghĩa, nhằm tránh mất mát đặc trưng giữa các nhiệm vụ. Vấn đề cốt lõi ở đây là xây dựng không gian kết nối ngữ nghĩa để biểu diễn các nhiệm vụ mà không phụ thuộc quá nhiều vào số lượng mẫu, từ đó giúp giảm chi phí tính toán.

## 1.1 Related Works 

### **Các mô hình học sâu cho học tăng cường lớp (Incremental Learning of Classes):**  
Một số mô hình dựa trên các đặc trưng của nhãn cũ, và khi thêm đặc trưng của nhãn mới thì không ảnh hưởng đến các lớp đã có. Một số phương pháp duy trì đặc trưng bằng cách lưu trữ chúng trong cơ sở dữ liệu, tuy nhiên cách tiếp cận này có nhược điểm là tốn nhiều bộ nhớ lưu trữ và cần huấn luyện lại từ đầu. Các nhóm nghiên cứu về nhận diện bệnh cây thường xuyên sử dụng phương pháp này trong huấn luyện mô hình.

Phương pháp ánh xạ dữ liệu thông qua các mô hình phản ánh đặc trưng cũ trong quá trình huấn luyện thường có chi phí tính toán cao. Một số nghiên cứu khác điều chỉnh tham số mô hình để tăng tính khách quan khi đánh giá các nhãn chứa nhiều đặc trưng hơn.

Gần đây, một số mô hình dựa trên khung “giáo viên – học sinh” giúp tích hợp nhãn mới và đảm bảo quá trình dự đoán được cân bằng, không thiên vị về nhãn cũ – đặc biệt hữu ích trong các bài toán nhận diện bệnh cây. Khi thêm nhãn mới, các mô hình học sâu thường tạo ra các khung bản đồ đặc trưng lớn để học bộ phân loại tương tự, từ đó điều chỉnh đặc trưng trên toàn bộ các lớp (bao gồm cả nhãn cũ và mới).

Hai yếu tố chính ảnh hưởng đến việc mở rộng mô hình là:

- Chi phí bộ nhớ khi huấn luyện nhiều đặc trưng
    
- Thời gian tính toán khi thêm nhãn mới
    

**Các mô hình học sâu đã được huấn luyện trước (Pre-trained Deep Learning Models):**  
Chúng ta có thể tận dụng các mô hình đã được huấn luyện trước từ các nhóm nghiên cứu trên toàn cầu để giải quyết vấn đề về đặc trưng nhãn mới và cũ. Trong những năm gần đây, nhiều mô hình gợi ý (recommendation models) đã được huấn luyện kỹ lưỡng và tỏ ra hiệu quả trong các bài toán nhận diện hình ảnh, do đó rất thích hợp để sử dụng lại.

Một số mô hình trích xuất đặc trưng để lưu trữ cũng đã cải thiện đáng kể khả năng học các đặc trưng quan trọng trong hình ảnh, đặc biệt là hình ảnh bệnh trên cây trồng. Nhiều nhóm nghiên cứu đã cải tiến hàm mất mát (loss function) để hỗ trợ tốt hơn cho việc nhận diện bệnh lá, và các hàm này có thể được tái sử dụng để đánh giá mô hình nhận diện bệnh trên cây thanh long.

Ngoài ra, việc kết hợp các phương pháp đa mô hình (multi-modal) để xử lý các loại sâu bệnh khác nhau từ các mô hình riêng biệt vào một mô hình tổng thể đang trở thành xu hướng. Sử dụng mô hình đã huấn luyện trước để truyền tham số cần thiết sang mô hình nhận diện bệnh cây thanh long là một phương pháp hiệu quả và chính xác.

## Motivation and contribution

Trong bài báo này, chúng tôi đề xuất một mô hình dựa trên phương pháp **Học dựa trên Kết nối Ngữ nghĩa (Semantic Connection Learning - SCL)** nhằm giải quyết các thách thức đã nêu. Để nâng cao hiệu quả giữa các nhiệm vụ cũ và mới, chúng tôi xây dựng một không gian kết nối đặc trưng – nhãn dựa trên ngữ nghĩa của từng đặc trưng, nhằm hỗ trợ mô hình trong việc nhận diện các nhiệm vụ riêng biệt cho từng nhãn.

Cách tiếp cận này cho phép mô hình học hiệu quả các lớp mới mà không ảnh hưởng đến các lớp đã học trước đó. Các không gian làm việc (working spaces) này được huấn luyện bằng cách điều chỉnh từ các mô hình đã được huấn luyện trước. Để giảm chi phí huấn luyện và giải quyết các vấn đề về dung lượng mô hình, chúng tôi sử dụng kỹ thuật sao chép tham số từ các mô hình đã được huấn luyện trước, nhằm cải thiện quá trình tiếp tục học với nhãn mới.

Bằng cách này, mô hình có thể tận dụng sức mạnh của các nhãn cũ cùng với nhãn mới, đồng thời tổng hợp một cách hiệu quả và tối ưu các đặc trưng thông tin từ nhiều nhãn khác nhau.

---

### **Các đóng góp và điểm mới chính của nghiên cứu này bao gồm:**

1. **Xây dựng không gian đặc trưng huấn luyện** để huấn luyện mô hình Học dựa trên Kết nối Ngữ nghĩa (SCL) được đề xuất. Chúng tôi tinh chỉnh các mô hình đã được huấn luyện trước nhằm tiết kiệm chi phí và tài nguyên huấn luyện, đồng thời vẫn đạt được độ chính xác mong muốn khi thêm các nhãn mới.
    
2. **Xây dựng tập dữ liệu bệnh trên cây thanh long**, được đặt tên là **D-Dragon**. D-Dragon là một tập dữ liệu do chúng tôi thu thập từ nhiều vườn thanh long khác nhau ở Việt Nam. Bộ ảnh bao gồm quả thanh long, thân cây thanh long, và các loại sâu bệnh khác nhau trên cây.
    
3. **Phân tích hiệu quả của mô hình huấn luyện** khi thêm nhãn mới, và đánh giá khách quan độ chính xác trên tập dữ liệu D-Dragon so với các mô hình tiên tiến hiện tại.
    

