---
tags:
  - resources/book-note
  - nlp/foundations
  - machine-learning/math
created: 2026-01-06
source: "[[NLP Book]]"
author: Tong Xiao, Jingbo Zhu
---

# Chapter 01: Foundations of Machine Learning (Part 1 & 2)

> [!ABSTRACT] Phạm vi Ghi chú
> Ghi chú này tổng hợp nội dung từ **Trang 1 đến Trang 20** của Chương 1.
> - **Phần 1 (Trang 1-10):** Các nguyên thủy toán học (Đại số tuyến tính, Xác suất).
> - **Phần 2 (Trang 11-20):** Lý thuyết thông tin (Entropy), Bài toán phân loại văn bản, và so sánh Mô hình Sinh (Generative) vs Mô hình Phân biệt (Discriminative).

---

## 1. Đại số Tuyến tính (Linear Algebra) - Nền tảng Biểu diễn
Ngôn ngữ mang tính biểu tượng (symbolic), nhưng mạng nơ-ron là các cỗ máy toán học hoạt động trên các con số liên tục. Đại số tuyến tính cung cấp khung lý thuyết để biểu diễn các ký hiệu này thành các đối tượng toán học—vectơ và ma trận—và thao tác chúng bằng các phép toán hình học.

### 1.1 Vectơ và Ma trận
Đơn vị cơ bản nhất là **số vô hướng (scalar)**, một đại lượng đơn lẻ (ví dụ: $a, b \in \mathbb{R}$) biểu thị độ lớn mà không có hướng. Trong NLP, một số vô hướng có thể đại diện cho giá trị của một đặc trưng cụ thể, chẳng hạn như số lần xuất hiện của một từ trong câu.

Một **vectơ (vector)** là một mảng có thứ tự các số vô hướng, được ký hiệu bằng chữ cái thường in đậm (ví dụ: $\mathbf{a}$). Một vectơ $n$-chiều $\mathbf{a}$ tập hợp $n$ số vô hướng thành một đối tượng duy nhất, được định nghĩa là:
$$\mathbf{a} = [a_1, a_2, ..., a_n] $$
trong đó $a_i$ (hoặc $a(i)$) là phần tử thứ $i$ của vectơ. Ta nói $\mathbf{a}$ là một vectơ thực nếu $\mathbf{a} \in \mathbb{R}^n$. Về mặt hình học, một vectơ đại diện cho một điểm hoặc một độ lớn có hướng trong không gian $n$-chiều. Trong cuốn sách này, vectơ được mặc định coi là **vectơ hàng** (row vector).

Một **ma trận (matrix)** là một mảng hình chữ nhật các số vô hướng, mở rộng khái niệm vectơ sang hai chiều. Chúng ta ký hiệu ma trận bằng chữ cái in hoa in đậm (ví dụ: $\mathbf{A}$). Một ma trận kích thước $m \times n$ có $m$ hàng và $n$ cột.

### 1.2 Các Phép toán trên Tensor
**Chuyển vị Ma trận (Transpose):**
Phép chuyển vị của một ma trận $\mathbf{A}$ lật nó qua đường chéo chính. Nếu $\mathbf{A}$ là ma trận $m \times n$, thì chuyển vị của nó $\mathbf{A}^\top$ là ma trận $n \times m$ với phần tử tại $(j, i)$ là $A_{ij}$.

**Tích Vô hướng (Dot Product):**
Tích vô hướng là một phép toán quan trọng để đo lường mối quan hệ giữa hai vectơ. Đối với hai vectơ $\mathbf{a}$ và $\mathbf{b}$ cùng kích thước $n$:
$$ \mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^n a_i b_i = \|\mathbf{a}\| \|\mathbf{b}\| \cos(\theta) $$
Mối quan hệ này ngụ ý rằng tích vô hướng là một thước đo độ tương đồng (similarity): nó đạt cực đại khi các vectơ cùng hướng và bằng 0 khi chúng trực giao (vuông góc).

**Tích Ma trận (Matrix Product):**
Phép nhân ma trận là sự kết hợp của các phép biến đổi tuyến tính. Đối với ma trận $\mathbf{A}$ ($m \times p$) và ma trận $\mathbf{B}$ ($p \times n$), tích $\mathbf{C} = \mathbf{AB}$ là một ma trận $m \times n$. Phần tử $C_{ij}$ là tích vô hướng của hàng thứ $i$ của $\mathbf{A}$ và cột thứ $j$ của $\mathbf{B}$.

### 1.3 Chuẩn (Norms): Đo lường Độ lớn
Để định lượng "kích thước" hoặc "độ dài" của một vectơ một cách chặt chẽ về mặt toán học, chúng ta sử dụng **chuẩn (norm)**. Chuẩn $L_p$ tổng quát cho một vectơ $\mathbf{a} \in \mathbb{R}^n$ được định nghĩa là:
$$ \|\mathbf{a}\|_p = \left( \sum_{i=1}^n |a_i|^p \right)^{1/p} $$
*   **Chuẩn $L_1$ (Manhattan):** Tổng các giá trị tuyệt đối. Thường dùng để khuyến khích tính thưa (sparsity).
*   **Chuẩn $L_2$ (Euclidean):** Khoảng cách đường thẳng tiêu chuẩn.
*   **Chuẩn $L_\infty$ (Maximum):** Giá trị tuyệt đối lớn nhất trong vectơ.

---

## 2. Lý thuyết Xác suất (Probability Theory) - Nền tảng của Sự Không Chắc Chắn
Trong NLP, sự không chắc chắn phát sinh từ tính mơ hồ của ngôn ngữ và nhiễu trong dữ liệu.

### 2.1 Các Định nghĩa Cơ bản
*   **Xác suất Đồng thời (Joint Probability)** $\text{Pr}(x, y)$: Xác suất để hai biến cố $x$ và $y$ xảy ra cùng lúc.
*   **Xác suất Có điều kiện (Conditional Probability)** $\text{Pr}(x|y)$: Xác suất $x$ xảy ra *với điều kiện* $y$ đã xảy ra. $\text{Pr}(x|y) = \frac{\text{Pr}(x, y)}{\text{Pr}(y)}$.
*   **Quy tắc Chuỗi (Chain Rule):** $\text{Pr}(x, y) = \text{Pr}(x|y)\text{Pr}(y)$.

### 2.2 Phân phối, Kỳ vọng và Phương sai
**Kỳ vọng (Expectation):**
Để tóm tắt một phân phối bằng một giá trị duy nhất, ta tính Giá trị Kỳ vọng. Đối với biến rời rạc $x$, kỳ vọng $\mathbb{E}[x]$ là trung bình có trọng số xác suất của tất cả các giá trị có thể:
$$ \mathbb{E}_{x \sim \text{Pr}(x)}[x] = \sum_{i=1}^n x_i \cdot \text{Pr}(x_i) $$
Giá trị này đại diện cho "trọng tâm" của phân phối. Trong ngữ cảnh học máy, hàm mục tiêu (loss function) thường được định nghĩa là kỳ vọng của sai số trên tập dữ liệu. Việc tối ưu hóa mô hình chính là việc thay đổi tham số để cực tiểu hóa giá trị kỳ vọng này. Đối với biến liên tục, tổng được thay thế bằng tích phân: $\mathbb{E}[x] = \int x \cdot \text{Pr}(x) dx$.

**Phương sai (Variance):**
Phương sai đo lường độ phân tán của dữ liệu xung quanh giá trị kỳ vọng.
$$ \text{Var}(x) = \mathbb{E}[(x - \mathbb{E}[x])^2] $$
Phương sai cao đồng nghĩa với việc mô hình hoặc dữ liệu có độ bất định lớn.

---

## 3. Lý thuyết Thông tin (Information Theory)
Lý thuyết thông tin cung cấp các công cụ định lượng để đo lường "lượng tin" hoặc "độ ngạc nhiên" trong các phân phối xác suất. Đây là nền tảng để xây dựng các hàm mất mát (loss functions) trong NLP.

### 3.1 Entropy (Độ bất định)
Entropy $H(x)$ là thước đo sự không chắc chắn của một biến ngẫu nhiên. Nếu một sự kiện là chắc chắn xảy ra ($\text{Pr}=1$), entropy bằng 0. Nếu mọi sự kiện đều có khả năng như nhau, entropy đạt cực đại.
$$ H(x) = - \sum_{i=1}^n \text{Pr}(x_i) \cdot \log_b \text{Pr}(x_i) $$
Trong NLP, cơ số $b$ thường là 2 (đơn vị bits) hoặc $e$ (đơn vị nats). Entropy càng cao, ta càng khó dự đoán kết quả của biến ngẫu nhiên đó.

### 3.2 Relative Entropy (KL Divergence)
Để đo lường sự khác biệt giữa hai phân phối xác suất $p$ (phân phối thực tế) và $q$ (phân phối dự đoán bởi mô hình), ta sử dụng **Kullback-Leibler (KL) Divergence**:
$$ D_{KL}(p||q) = \sum_{x} p(x) \cdot \log \frac{p(x)}{q(x)} $$
KL Divergence đo lường lượng thông tin bị mất khi dùng $q$ để xấp xỉ $p$. Lưu ý quan trọng là KL Divergence **không đối xứng** ($D_{KL}(p||q) \neq D_{KL}(q||p)$) và luôn không âm. $D_{KL}(p||q) = 0$ khi và chỉ khi $p = q$.

### 3.3 Cross-Entropy
Trong thực tế huấn luyện mô hình, chúng ta thường sử dụng **Cross-Entropy**. Nó liên quan chặt chẽ đến KL Divergence:
$$ H_{\text{cross}}(p, q) = - \sum_{x} p(x) \cdot \log q(x) = H(p) + D_{KL}(p||q) $$
Vì $H(p)$ (entropy của dữ liệu thực tế) là hằng số đối với mô hình, việc tối thiểu hóa Cross-Entropy tương đương với việc tối thiểu hóa KL Divergence, tức là làm cho phân phối dự đoán $q$ càng giống phân phối thực $p$ càng tốt. Đây là hàm mất mát tiêu chuẩn cho các bài toán phân loại.

---

## 4. Thiết kế Bộ Phân loại Văn bản (Designing a Text Classifier)
Bài toán cơ bản nhất trong NLP là gán nhãn cho văn bản (ví dụ: phân loại chủ đề, cảm xúc).

### 4.1 Phát biểu Bài toán
Giả sử có văn bản $x$ và tập nhãn $\mathcal{C}$. Chúng ta cần xây dựng mô hình để ước lượng xác suất có điều kiện $\text{Pr}(c|x)$. Nhãn dự đoán $\hat{c}$ sẽ là nhãn có xác suất cao nhất:
$$ \hat{c} = \underset{c \in \mathcal{C}}{\text{argmax}} \ \text{Pr}(c|x) $$

### 4.2 Biểu diễn: Bag-of-Words (BoW)
Để máy tính xử lý, văn bản phải được chuyển thành vectơ. Mô hình **Bag-of-Words** biểu diễn văn bản bằng vectơ tần suất từ, bỏ qua trật tự từ.
*   Gọi $V$ là từ điển.
*   Vectơ đặc trưng $\mathbf{x} \in \mathbb{R}^{|V|}$, với $x_i = \text{count}(w_i)$.
*   **Hạn chế:** Mất thông tin ngữ pháp và ngữ nghĩa ngữ cảnh (context); vectơ rất thưa (sparse).

### 4.3 Bộ Phân loại Tuyến tính (Linear Classifiers)
Mô hình đơn giản nhất sử dụng hàm tuyến tính để tính điểm số (score) cho mỗi lớp:
$$ s(\mathbf{x}, \mathbf{w}, b) = \mathbf{w} \cdot \mathbf{x} + b $$
Trong đó $\mathbf{w}$ là vectơ trọng số và $b$ là bias. Về mặt hình học, phương trình $s(\mathbf{x}) = 0$ định nghĩa một **siêu phẳng (hyperplane)** chia không gian thành các vùng tương ứng với các lớp.

---

## 5. Mô hình Sinh (Generative) vs Mô hình Phân biệt (Discriminative)
Có hai cách tiếp cận chính để giải quyết bài toán phân loại $\text{Pr}(c|\mathbf{x})$.

### 5.1 Mô hình Sinh (Generative Models) - Ví dụ: Naive Bayes
Mô hình sinh học cách dữ liệu được tạo ra bằng cách ước lượng phân phối đồng thời $\text{Pr}(\mathbf{x}, c)$. Sau đó dùng định lý Bayes để suy ra nhãn:
$$ \text{Pr}(c|\mathbf{x}) \propto \text{Pr}(c) \cdot \text{Pr}(\mathbf{x}|c) $$
*   **Naive Bayes:** Giả định rằng các từ trong văn bản độc lập với nhau khi đã biết nhãn $c$.
    $$ \text{Pr}(\mathbf{x}|c) \approx \prod_{i=1}^n \text{Pr}(x_i|c) $$
*   Mô hình này chuyển bài toán phức tạp thành việc đếm tần suất từ đơn giản.
*   **Log-Linear:** Để tránh tràn số, ta thường tính toán trên miền logarit, biến phép nhân thành phép cộng.

### 5.2 Mô hình Phân biệt (Discriminative Models) - Ví dụ: Logistic Regression
Mô hình phân biệt học trực tiếp ranh giới giữa các lớp, tức là ước lượng thẳng $\text{Pr}(c|\mathbf{x})$ mà không quan tâm đến phân phối của $\mathbf{x}$.
*   **Logistic Regression:** Sử dụng hàm **Sigmoid** để nén đầu ra tuyến tính thành xác suất:
    $$ \text{Pr}(c=1|\mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w} \cdot \mathbf{x} + b)}} $$
*   Mô hình này không giả định các từ độc lập, do đó thường chính xác hơn Naive Bayes khi có đủ dữ liệu.

### 5.3 So sánh
| | Generative (Naive Bayes) | Discriminative (Logistic Regression) |
|---|---|---|
| **Học** | Phân phối đồng thời $\text{Pr}(\mathbf{x}, c)$ | Phân phối điều kiện $\text{Pr}(c\|\mathbf{x})$ |
| **Giả định** | Mạnh (Độc lập có điều kiện) | Yếu (Không giả định về $\mathbf{x}$) |
| **Dữ liệu** | Tốt với ít dữ liệu | Cần nhiều dữ liệu hơn |
| **Tính chất** | Dễ dàng thêm dữ liệu mới | Độ chính xác thường cao hơn |

---

## 6. Tài liệu Tham khảo
- **[[Entropy (Information Theory)]]**
- **[[KL Divergence]]**
- **[[Generative vs Discriminative Models]]**