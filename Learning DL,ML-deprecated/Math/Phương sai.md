---
aliases:
  - Variance
  - phương sai
tags:
  - variance
  - bias
---
# Tổng quan
**Phương sai (Variance)** là một đại lượng đo lường mức độ phân tán của một tập dữ liệu so với giá trị trung bình của nó. Nói một cách đơn giản, phương sai cho biết các giá trị trong tập dữ liệu phân tán rộng hay tập trung gần giá trị trung bình.

**Công thức tính phương sai:**

Có hai công thức tính phương sai tùy thuộc vào việc bạn đang tính phương sai của một **tổng thể** hay một **mẫu**.

- Phương sai của tổng thể (ký hiệu: σ2):
    
    $$\sigma^2=\frac{\sum\limits_{i=1}^{N}(x_{i}-\mu)^2}{N} ​$$
    
    Trong đó:
    - $x_i$​ là giá trị thứ i trong tổng thể
    - μ là giá trị trung bình của tổng thể
    - N là số lượng phần tử trong tổng thể
- Phương sai của mẫu (ký hiệu: s2):
    
    $$s^2=\frac{\sum_{i=1}^{n}(x_{i}-\hat x)^2}{n-1}$$
    
    Trong đó:
    
    - x$x_i$ là giá trị thứ i trong mẫu
    - $\hat x$ là giá trị trung bình của mẫu
    - n là số lượng phần tử trong mẫu
    - n−1 được gọi là bậc tự do, được sử dụng để hiệu chỉnh cho việc ước tính phương sai của tổng thể từ mẫu.

**Ý nghĩa của phương sai:**

- **Đo lường độ phân tán:** Phương sai càng lớn cho thấy dữ liệu càng phân tán rộng xung quanh giá trị trung bình, và ngược lại, phương sai càng nhỏ cho thấy dữ liệu tập trung gần giá trị trung bình.
- **Đánh giá rủi ro:** Trong lĩnh vực tài chính, phương sai của lợi nhuận một tài sản thường được sử dụng để đo lường mức độ rủi ro của tài sản đó. Phương sai cao đồng nghĩa với rủi ro cao hơn.
- **So sánh các tập dữ liệu:** Khi có hai hoặc nhiều tập dữ liệu có cùng đơn vị đo và giá trị trung bình tương đương, phương sai có thể được sử dụng để so sánh mức độ biến động giữa chúng. Tập dữ liệu có phương sai nhỏ hơn sẽ ổn định hơn.

**Lưu ý:**

- Đơn vị của phương sai là bình phương của đơn vị của dữ liệu gốc. Để có một thước đo độ phân tán có cùng đơn vị với dữ liệu gốc, người ta thường sử dụng **độ lệch chuẩn**, là căn bậc hai của phương sai (σ cho tổng thể và s cho mẫu).
# Vai trò trong ML, DL

Trong học máy, phương sai đóng một vai trò quan trọng, đặc biệt trong việc hiểu và giải quyết vấn đề **overfitting (quá khớp)** và **underfitting (thiếu khớp)** của mô hình. Dưới đây là các vai trò chính của phương sai:

**1. Đo lường độ nhạy của mô hình đối với sự thay đổi trong dữ liệu huấn luyện:**

- **Phương sai cao:** Một mô hình có phương sai cao có xu hướng rất nhạy cảm với những biến động nhỏ trong dữ liệu huấn luyện. Điều này có nghĩa là nếu bạn huấn luyện mô hình trên một tập dữ liệu khác một chút, kết quả có thể thay đổi đáng kể. Mô hình "học thuộc" cả những nhiễu và chi tiết không quan trọng trong dữ liệu huấn luyện, dẫn đến việc hoạt động kém trên dữ liệu mới (dữ liệu kiểm tra). Đây chính là hiện tượng **overfitting**.
- **Phương sai thấp:** Một mô hình có phương sai thấp thì ít nhạy cảm hơn với sự thay đổi trong dữ liệu huấn luyện. Dự đoán của mô hình sẽ ổn định hơn trên các tập dữ liệu khác nhau. Tuy nhiên, nếu phương sai quá thấp, mô hình có thể quá đơn giản và không nắm bắt được các mối quan hệ phức tạp trong dữ liệu, dẫn đến **underfitting**.

**2. Góp phần vào phân rã sai số (Bias-Variance Decomposition):**

Tổng sai số của một mô hình học máy có thể được phân thành ba thành phần:

- **Bias (Độ chệch):** Sai số do giả định đơn giản hóa của mô hình. Một mô hình có độ chệch cao có thể bỏ qua các mối quan hệ quan trọng trong dữ liệu.
- **Variance (Phương sai):** Sai số do độ nhạy của mô hình đối với sự thay đổi trong dữ liệu huấn luyện (như đã giải thích ở trên).
- **Irreducible Error (Sai số không thể giảm):** Sai số vốn có trong chính dữ liệu, không thể giảm thiểu bằng cách cải thiện mô hình.

Hiểu rõ về phương sai giúp chúng ta điều chỉnh mô hình để đạt được sự cân bằng tốt giữa độ chệch và phương sai, từ đó giảm thiểu tổng sai số và cải thiện khả năng khái quát hóa của mô hình trên dữ liệu mới.

**3. Lựa chọn mô hình và điều chỉnh siêu tham số:**

- Khi lựa chọn giữa các mô hình khác nhau, chúng ta thường xem xét hiệu suất của chúng trên tập kiểm tra. Một mô hình có phương sai quá cao có thể cho hiệu suất tốt trên tập huấn luyện nhưng kém trên tập kiểm tra.
- Trong quá trình điều chỉnh siêu tham số (hyperparameter tuning), chúng ta thường cố gắng tìm ra các giá trị siêu tham số giúp giảm phương sai (ví dụ: sử dụng regularization để giảm độ phức tạp của mô hình) mà không làm tăng đáng kể độ chệch.

**4. Đánh giá độ ổn định của mô hình:**

Phương sai thấp thường chỉ ra rằng mô hình ổn định hơn, đưa ra các dự đoán nhất quán hơn trên các tập dữ liệu khác nhau. Điều này đặc biệt quan trọng trong các ứng dụng thực tế, nơi dữ liệu có thể thay đổi theo thời gian.

Tóm lại, phương sai là một khái niệm then chốt trong học máy, giúp chúng ta hiểu rõ hơn về hành vi của mô hình, đặc biệt là khả năng khái quát hóa của nó. Việc kiểm soát và điều chỉnh phương sai là một phần quan trọng của quá trình xây dựng và đánh giá các mô hình học máy hiệu quả.