---
title: "Buổi 9 - Tuần 2: Review & Mini Test Preliminaries"
tags: [d2l, review, mini-test, preliminaries]
created: 2026-03-16
session: "D2L Tuần 2, Buổi 9 — Review Preliminaries"
source:
  - "30_Resources/Books/Dive Into Deep Learning/d2l-en/chapter_preliminaries"
  - "30_Resources/Books/Dive Into Deep Learning/d2l-en/chapter_introduction/index.md"
related:
  - "[[Tensor Operations]]"
  - "[[Linear Algebra for Deep Learning]]"
  - "[[Calculus for Deep Learning]]"
  - "[[Automatic Differentiation]]"
  - "[[Probability and Statistics for Deep Learning]]"
  - "[[Data Preprocessing with Pandas]]"
---

# Buổi 9 - Tuần 2: Review & Mini Test Preliminaries

> [!NOTE] ELI5
> Buổi này là checkpoint trước khi vào Linear Regression. Nếu nền chưa chắc, các chương sau sẽ dễ bị "học vẹt" công thức. Mục tiêu là kiểm tra xem bạn thật sự hiểu hay mới chỉ nhớ định nghĩa. Sau buổi này, bạn cần trả lời được không chỉ "cái gì" mà còn "vì sao".

## 1. Mục tiêu đánh giá

1. Kiểm tra mức hiểu bản chất của các khối nền tảng.
2. Phát hiện lỗ hổng tư duy trước khi sang mô hình đầu tiên.
3. Chuyển từ "biết thao tác" sang "biết giải thích".

## 2. Bản đồ kiến thức Tuần 1-2

> [!NOTE] ELI5
> Các buổi đầu không rời rạc. Chúng giống dây chuyền: dữ liệu đi vào, tensor biểu diễn dữ liệu, giải tích cho gradient, autograd tính gradient tự động, xác suất giúp hiểu loss, rồi mới huấn luyện mô hình.

- **Tensor operations**: ngôn ngữ dữ liệu số nhiều chiều.
- **Linear algebra**: phép biến đổi tuyến tính để mô hình tính dự đoán.
- **Calculus**: đạo hàm/gradient xác định hướng giảm loss.
- **Autograd**: tự động hóa chain rule trên computational graph.
- **Probability**: diễn giải uncertainty và loss theo likelihood.
- **Pandas preprocessing**: dữ liệu đầu vào sạch để tối ưu ổn định.
- **Introduction**: tư duy chung “programming with data”.

## 3. Mini Test Chuẩn Đoán (Diagnostic)

### Câu 1 (Tensor)

Cho `A.shape = (3, 1)` và `B.shape = (1, 4)`. Sau phép `A + B` thì shape kết quả là gì? Giải thích theo từng bước broadcasting.

### Câu 2 (Linear Algebra)

So sánh khác biệt toán học giữa Hadamard product và matrix multiplication. Mỗi phép dùng trong tình huống nào của DL?

### Câu 3 (Calculus)

Với $f(x)=3x^2-4x$, tính $f'(x)$ và diễn giải ý nghĩa của $f'(1)$ trong bối cảnh tối ưu.

### Câu 4 (Chain Rule)

Cho $y=g(u),\ u=h(x)$. Viết chain rule và giải thích tại sao đây là lõi của backpropagation.

### Câu 5 (Autograd)

Vì sao trong PyTorch cần reset gradient (`zero_()` hoặc `zero_grad()`) trước vòng cập nhật tiếp theo?

### Câu 6 (Probability)

Dùng Bayes để giải thích hiện tượng base-rate neglect: vì sao test dương tính không đồng nghĩa xác suất mắc bệnh là rất cao?

### Câu 7 (Data Preprocessing)

So sánh nhanh `dropna`, `fillna`, `get_dummies(dummy_na=True)`: khi nào nên dùng từng cách?

### Câu 8 (Conceptual)

Nêu 3 khác biệt giữa lập trình theo luật cố định và học máy theo dữ liệu.

## 4. Đáp án mẫu + lập luận

> [!NOTE] ELI5
> Đáp án đúng chưa đủ. Mục tiêu là bạn tự giải thích được vì sao đáp án đúng. Nếu không giải thích được, kiến thức chưa đủ chắc để đi tiếp.

1. Kết quả `(3, 4)`.
A đúng vì broadcasting so từng chiều từ phải sang trái: `(3,1)` và `(1,4)` tương thích ở cả hai chiều (1 có thể nở), được thể hiện qua việc một cột được copy theo chiều ngang và một hàng được copy theo chiều dọc.

2. Hadamard vs matrix multiplication.
A đúng vì Hadamard là phép nhân cục bộ từng phần tử còn matmul là phép tổng hợp tuyến tính giữa hàng-cột, được thể hiện qua công thức:
Hadamard: $(A \odot B)_{ij}=A_{ij}B_{ij}$.
Matmul: $(AB)_{ij}=\sum_k A_{ik}B_{kj}$.

3. Đạo hàm của $f(x)=3x^2-4x$.
$f'(x)=6x-4$, nên $f'(1)=2$.
A đúng vì đạo hàm là tốc độ thay đổi tức thời, được thể hiện qua xấp xỉ cục bộ: với $\Delta x$ nhỏ thì $\Delta f \approx 2\Delta x$ tại $x=1$.

4. Chain rule.
$\frac{dy}{dx}=\frac{dy}{du}\frac{du}{dx}$.
A đúng vì khi hàm lồng nhiều tầng, ảnh hưởng của biến đầu vào lên đầu ra đi qua từng tầng trung gian, được thể hiện qua backprop lan gradient từ output về input.

5. Vì sao cần `zero_grad()`.
A đúng vì PyTorch mặc định cộng dồn gradient qua nhiều lần `backward()`, được thể hiện qua việc không reset sẽ khiến bước cập nhật dùng gradient sai (to dần theo số vòng).

6. Bayes và base-rate neglect.
A đúng vì posterior không chỉ phụ thuộc test positive mà còn phụ thuộc prevalence (xác suất nền), được thể hiện qua:

$$
P(\text{disease}\mid +)=\frac{P(+\mid \text{disease})P(\text{disease})}{P(+)}
$$

Nếu bệnh hiếm, false positive vẫn có thể làm posterior thấp.

7. `dropna`, `fillna`, `dummy_na`.
A đúng vì mỗi chiến lược mã hóa một giả định khác nhau về missingness:
- `dropna`: chấp nhận mất dữ liệu để giữ sạch mẫu.
- `fillna`: giữ mẫu, chấp nhận bias từ imputing.
- `dummy_na=True`: coi missing là tín hiệu riêng.

8. Rule-based vs ML.
A đúng vì hai cách khác nhau ở nơi đặt tri thức: rule-based đặt tri thức trong code tường minh, ML đặt tri thức vào tham số học từ dữ liệu.

## 5. Tiêu chí qua buổi

1. Trả lời đúng tối thiểu 6/8 câu.
2. Có thể tự giải thích từng đáp án bằng ngôn ngữ của mình.
3. Không nhầm giữa khái niệm toán học (ví dụ Hadamard và matmul, derivative và gradient).

## 6. Quyết định sau buổi 9

Nếu làm đúng phần lớn câu hỏi, chuyển sang Tuần 3 với `chapter_linear-regression/linear-regression.md` (Buổi 10).
