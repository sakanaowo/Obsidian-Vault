---
title: "Buổi 8 - Tuần 2: Introduction to Deep Learning (D2L)"
tags: [d2l, deep-learning, introduction, study-note]
created: 2026-03-16
session: "D2L Tuần 2, Buổi 8 — Introduction"
source:
  - "30_Resources/Books/Dive Into Deep Learning/d2l-en/chapter_introduction/index.md"
related:
  - "[[Tensor Operations]]"
  - "[[Linear Algebra for Deep Learning]]"
  - "[[Calculus for Deep Learning]]"
  - "[[Automatic Differentiation]]"
  - "[[Probability and Statistics for Deep Learning]]"
---

# Buổi 8 - Tuần 2: Introduction to Deep Learning

> [!NOTE] ELI5
> Lập trình truyền thống giống như viết sẵn luật chơi: nếu gặp tình huống A thì làm B. Deep learning thì khác: ta đưa rất nhiều ví dụ để máy tự tìm ra luật. Nghĩa là thay vì "viết tay toàn bộ quy tắc", ta "để dữ liệu dạy mô hình". Khi bài toán quá phức tạp để viết rule rõ ràng, cách này thường hiệu quả hơn.

## 1. Mục tiêu buổi học

Buổi này bám sát `chapter_introduction/index.md` của d2l.ai, với 4 mục tiêu chính:

1. Hiểu khi nào nên dùng **rule-based programming** và khi nào cần **machine learning**.
2. Nắm tư tưởng cốt lõi: **programming with data**.
3. Hiểu các thành phần nền: data, model, parameters, learning algorithm.
4. Chốt một training loop chuẩn để làm nền cho các chương sau.

## 2. Rule-Based vs Machine Learning

> [!NOTE] ELI5
> Nếu đề bài có luật rõ ràng, ổn định, cứ viết luật trực tiếp là nhanh và chắc nhất. Nếu đề bài mơ hồ, nhiều ngoại lệ, dữ liệu thay đổi liên tục, thì cho máy học từ dữ liệu sẽ hợp lý hơn.

Trong chương mở đầu, D2L nhấn mạnh một điểm quan trọng: **ML không thay thế mọi phần mềm**.

Claim: Không phải bài toán nào cũng nên dùng học máy.
Reasoning: nếu quan hệ input-output có thể mô tả chính xác bằng quy tắc, hệ rule-based sẽ dễ kiểm thử, dễ debug, và ổn định hơn.
Evidence: ví dụ logic thương mại điện tử (thêm sản phẩm vào giỏ, kiểm tra tồn kho, xác nhận đơn) thường có business rule rõ ràng.

Ngược lại, với các tác vụ như nhận dạng giọng nói, nhận dạng ảnh, trả lời câu hỏi tự nhiên, rất khó liệt kê đầy đủ quy tắc bằng tay. Đây là vùng đất của machine learning.

## 3. Ví dụ động lực: Wake Word

> [!NOTE] ELI5
> Bạn nghe "Hey Siri" và hiểu ngay. Nhưng bảo lập trình viên viết một danh sách luật cố định để nhận ra mọi giọng, mọi tiếng ồn, mọi ngữ điệu thì gần như bất khả thi. Vì vậy ta cho mô hình học từ hàng triệu ví dụ âm thanh có nhãn.

D2L dùng ví dụ wake-word để chỉ ra ranh giới giữa "biết làm" và "biết mô tả cách làm".

- Con người nhận ra wake word tốt, nhưng khó diễn đạt thành rule tường minh từ raw waveform.
- ML tiếp cận bằng cách xây dựng một **họ mô hình có tham số**, rồi tối ưu tham số trên dữ liệu gán nhãn.

Nội dung này kết nối trực tiếp với định nghĩa:

- **Model**: chương trình cụ thể sau khi chốt tham số.
- **Model family**: tập tất cả mô hình có thể tạo ra bằng thay đổi tham số.
- **Learning algorithm**: cơ chế tìm bộ tham số tốt.

## 4. Deep Concept: "Programming with Data"

> [!NOTE] ELI5
> Trước đây: "Tôi dạy máy bằng luật". Bây giờ: "Tôi dạy máy bằng ví dụ". Máy tự tinh chỉnh bên trong để khớp ví dụ đó.

### 4.1 Bản chất

Ý tưởng không phải bỏ lập trình, mà là **dịch trọng tâm lập trình**:

- Từ viết rule chi tiết
- Sang thiết kế cấu trúc mô hình + mục tiêu tối ưu + pipeline dữ liệu

### 4.2 Cơ chế hoạt động

Với dữ liệu huấn luyện $(x_i, y_i)$, ta chọn mô hình $f_\theta$ và tối ưu:

$$
	heta^* = \arg\min_\theta \frac{1}{n}\sum_{i=1}^{n} \mathcal{L}(f_\theta(x_i), y_i)
$$

Trong đó:

- $\theta$: tham số mô hình
- $\mathcal{L}$: hàm mất mát
- $f_\theta(x_i)$: dự đoán

### 4.3 Ví dụ cụ thể

Trong wake-word:

- $x_i$: đoạn âm thanh ngắn
- $y_i \in \{\text{yes}, \text{no}\}$
- $f_\theta$: mạng nhận diện

Mô hình học cách ánh xạ từ âm thanh sang nhãn thay vì ta viết đặc trưng thủ công từng quy tắc âm vị.

### 4.4 Ứng dụng thực tế trong DL

Tư tưởng này là nền cho mọi chương sau: linear regression, softmax, MLP, CNN, Transformer chỉ khác ở kiến trúc $f_\theta$ và dạng dữ liệu.

## 5. Các thành phần lõi (theo D2L)

> [!NOTE] ELI5
> Hãy xem mô hình như một bảng điều khiển có nhiều nút. Ban đầu nút đặt ngẫu nhiên nên dự đoán kém. Mỗi vòng học, máy nhìn mình sai bao nhiêu rồi vặn nút lại một chút. Lặp đủ nhiều, dự đoán sẽ tốt dần.

1. **Data**: nguồn kinh nghiệm để mô hình học.
2. **Model**: giả thuyết ánh xạ input-output.
3. **Objective/Loss**: tiêu chí đo sai.
4. **Optimization algorithm**: cách cập nhật tham số.
5. **Generalization target**: năng lực đúng trên dữ liệu chưa thấy.

## 6. Training Loop Chuẩn

> [!NOTE] ELI5
> Máy học theo vòng lặp: đoán, so với đáp án, sửa sai, rồi đoán lại. Cứ lặp vậy đến khi lỗi nhỏ xuống mức chấp nhận.

1. Khởi tạo model ngẫu nhiên.
2. Lấy minibatch từ dữ liệu.
3. Tính dự đoán, tính loss.
4. Backprop để lấy gradient.
5. Cập nhật tham số.
6. Lặp lại đến khi hội tụ hoặc đạt stopping criterion.

Khung này chính là skeleton cho toàn bộ D2L về sau.

## 7. Liên hệ với nền tảng Tuần 1-2

Buổi 8 không nặng công thức mới, nhưng là điểm nối logic quan trọng:

- `ndarray` + `linear algebra`: biểu diễn và biến đổi dữ liệu.
- `calculus` + `autograd`: tính gradient để tối ưu.
- `probability`: hiểu loss theo góc nhìn xác suất.
- `pandas`: làm sạch dữ liệu trước khi đưa vào model.

## 8. Kết luận buổi 8

Điểm quan trọng nhất của buổi này là đổi tư duy:

1. **ML = học từ dữ liệu**, không phải liệt kê mọi quy tắc.
2. **DL = mô hình tham số lớn + tối ưu gradient-based**.
3. Từ buổi 10 trở đi, mỗi mô hình chỉ là một hiện thực cụ thể của cùng một framework tối ưu.
