---
title: "Buổi 14 - Tuần 4: Softmax Regression — Từ Regression sang Classification (D2L)"
tags: [d2l, softmax, cross-entropy, classification, one-hot, study-note]
created: 2026-03-19
session: "D2L Tuần 4, Buổi 14 — Softmax Regression"
source:
  - "30_Resources/Books/Dive Into Deep Learning/d2l-en/chapter_linear-classification/softmax-regression.md"
related:
  - "[[Buổi 13 - Tuần 4]]"
  - "[[Linear Regression for Deep Learning]]"
  - "[[Softmax Function]]"
  - "[[Cross-Entropy Loss]]"
  - "[[One-Hot Encoding]]"
  - "[[Generalization]]"
---

# Buổi 14 — Softmax Regression: Từ "Bao nhiêu?" sang "Cái nào?"

> [!NOTE] ELI5
> Ở 3 buổi trước (buổi 10-12), bạn học dự đoán **một con số** (giá nhà, nhiệt độ) — đó là **regression**. Buổi 14 này, bạn học dự đoán **một nhóm** (ảnh này là mèo, gà hay chó?) — đó là **classification**.
>
> Ý tưởng: thay vì mô hình ra 1 số, nó ra **nhiều số** (mỗi số = điểm cho 1 nhóm). Hàm **softmax** biến các điểm thô thành **xác suất**. Hàm **cross-entropy** đo mô hình đoán đúng hay sai.

---

## 🎯 Mục tiêu buổi học

Sau buổi này, bạn cần biết:

1. Phân biệt **regression** vs **classification**
2. Hiểu **one-hot encoding** là gì và tại sao cần
3. Hiểu hàm **softmax** biến đổi điểm thô → xác suất
4. Hiểu **cross-entropy loss** đo lỗi phân loại như thế nào
5. Nắm ý nghĩa **information theory** của cross-entropy (ở mức cơ bản)

---

## Phần 1: Regression → Classification — Bước chuyển mình

> [!NOTE] ELI5
> **Regression** = hỏi "bao nhiêu?" (giá nhà bao nhiêu tiền?)
> **Classification** = hỏi "cái nào?" (ảnh này là con gì?)
>
> Cả hai đều dùng mô hình tuyến tính, nhưng khác nhau ở **đầu ra** và **cách đo lỗi**.

### 1.1 So sánh Regression vs Classification

| | Regression | Classification |
| --- | --- | --- |
| **Câu hỏi** | "Bao nhiêu?" | "Cái nào?" |
| **Output** | 1 con số liên tục | Xác suất cho mỗi class |
| **Loss** | MSE (lỗi bình phương) | Cross-entropy |
| **Ví dụ** | Giá nhà, nhiệt độ | Nhận diện ảnh, spam detection |
| **Buổi D2L** | Buổi 10-12 | Buổi 14-17 |

### 1.2 Ví dụ bài toán Classification

Cho ảnh xám $2 \times 2$ (4 pixel = 4 features), phân loại thành 3 nhóm: **Mèo**, **Gà**, **Chó**.

- Input: $\mathbf{x} = (x_1, x_2, x_3, x_4)$ — 4 giá trị pixel
- Output: xác suất thuộc mỗi class — ví dụ: P(mèo) = 0.7, P(gà) = 0.2, P(chó) = 0.1

---

## Phần 2: One-Hot Encoding — Biểu diễn nhãn đúng cách

> [!NOTE] ELI5
> Nếu đánh số Mèo=1, Gà=2, Chó=3 thì máy tính sẽ **hiểu nhầm** rằng "Gà lớn hơn Mèo". Thay vì vậy, ta dùng **one-hot**: mỗi class là 1 vector chỉ có đúng 1 vị trí bật.

![[assets/attachments/D2L/Buổi 14/onehot_encoding.png]]

| Class | Đánh số (❌ Sai) | One-hot (✅ Đúng) |
| --- | --- | --- |
| Mèo | 1 | $(1, 0, 0)$ |
| Gà | 2 | $(0, 1, 0)$ |
| Chó | 3 | $(0, 0, 1)$ |

**Tại sao one-hot?** Vì classification thường **không có thứ tự** giữa các class. Mèo không "nhỏ hơn" gà. One-hot đảm bảo mỗi class **độc lập và bình đẳng**.

> Xem thêm: [[One-Hot Encoding]]

---

## Phần 3: Mô hình tuyến tính cho Classification

> [!NOTE] ELI5
> Trong regression, bạn có **1 đầu ra** ($\hat{y} = \mathbf{w}^T\mathbf{x} + b$). Trong classification với 3 class, bạn cần **3 đầu ra** — mỗi đầu ra là 1 "điểm" cho 1 class. Mỗi đầu ra có bộ trọng số riêng.

### 3.1 Nhiều outputs

![[assets/attachments/D2L/Buổi 14/softmax_network.png]]

Với 4 features và 3 classes, mô hình tính 3 **logits** (điểm thô):

$$\begin{aligned}
o_1 &= x_1 w_{11} + x_2 w_{12} + x_3 w_{13} + x_4 w_{14} + b_1 \\
o_2 &= x_1 w_{21} + x_2 w_{22} + x_3 w_{23} + x_4 w_{24} + b_2 \\
o_3 &= x_1 w_{31} + x_2 w_{32} + x_3 w_{33} + x_4 w_{34} + b_3
\end{aligned}$$

### 3.2 Dạng ma trận

$$\mathbf{o} = \mathbf{W}\mathbf{x} + \mathbf{b}$$

| Ký hiệu | Kích thước | Ý nghĩa |
| --- | --- | --- |
| $\mathbf{x}$ | $(d, 1)$ = $(4, 1)$ | Vector features (4 pixels) |
| $\mathbf{W}$ | $(q, d)$ = $(3, 4)$ | Ma trận trọng số |
| $\mathbf{b}$ | $(q, 1)$ = $(3, 1)$ | Vector bias |
| $\mathbf{o}$ | $(q, 1)$ = $(3, 1)$ | Vector logits (điểm thô) |

**Lưu ý**: Đây vẫn là **single-layer neural network** — giống hệt linear regression nhưng có nhiều outputs. Mỗi output kết nối với **tất cả** inputs → **fully connected layer**.

### 3.3 Vectorization cho minibatch

Khi train với $n$ mẫu cùng lúc:

$$\mathbf{O} = \mathbf{X}\mathbf{W} + \mathbf{b}, \qquad \hat{\mathbf{Y}} = \text{softmax}(\mathbf{O})$$

Trong đó $\mathbf{X} \in \mathbb{R}^{n \times d}$, $\mathbf{W} \in \mathbb{R}^{d \times q}$, $\mathbf{O} \in \mathbb{R}^{n \times q}$.

---

## Phần 4: Softmax — Biến điểm thô thành xác suất

> [!NOTE] ELI5
> Logits $o_1, o_2, o_3$ là "điểm thô" — có thể âm, có thể lớn hơn 1, không cộng lại bằng 1. Softmax biến chúng thành **xác suất hợp lệ**: tất cả dương, tổng bằng 1.
>
> Cách làm: (1) lũy thừa $e^{o_i}$ để mọi thứ dương, (2) chia cho tổng để chuẩn hóa.

### 4.1 Công thức

$$\hat{y}_i = \text{softmax}(\mathbf{o})_i = \frac{\exp(o_i)}{\sum_{j=1}^q \exp(o_j)}$$

### 4.2 Ví dụ bằng số

![[assets/attachments/D2L/Buổi 14/softmax_visualization.png]]

Cho $\mathbf{o} = (2.0, 1.0, 0.1)$:

| Bước | Mèo ($o_1 = 2.0$) | Gà ($o_2 = 1.0$) | Chó ($o_3 = 0.1$) |
| --- | --- | --- | --- |
| **1. Lũy thừa** $e^{o_i}$ | $e^{2.0} = 7.39$ | $e^{1.0} = 2.72$ | $e^{0.1} = 1.11$ |
| Tổng | | $7.39 + 2.72 + 1.11 = 11.22$ | |
| **2. Chia cho tổng** | $7.39/11.22$ | $2.72/11.22$ | $1.11/11.22$ |
| **Kết quả** | **0.659 (65.9%)** | 0.242 (24.2%) | 0.099 (9.9%) |

→ Mô hình dự đoán: ảnh này **có lẽ là Mèo** (65.9%).

### 4.3 Tính chất quan trọng

- ✅ Tất cả $\hat{y}_i > 0$ (nhờ $e^x > 0$)
- ✅ $\sum_i \hat{y}_i = 1$ (nhờ chia cho tổng)
- ✅ **Giữ thứ tự**: class có logit cao nhất → xác suất cao nhất
- ✅ Không cần softmax để phân loại: $\arg\max_j \hat{y}_j = \arg\max_j o_j$

### 4.4 Softmax vs Sigmoid

| | Sigmoid | Softmax |
| --- | --- | --- |
| Dùng cho | Binary (2 classes) | Multi-class ($q$ classes) |
| Output | 1 số ∈ (0,1) | Vector ∈ (0,1)$^q$, tổng = 1 |
| Công thức | $\frac{1}{1+e^{-x}}$ | $\frac{e^{o_i}}{\sum_j e^{o_j}}$ |
| Quan hệ | = Softmax với 2 classes | = Sigmoid tổng quát |

> Xem thêm: [[Softmax Function]], [[Sigmoid Function]]

---

## Phần 5: Cross-Entropy Loss — Đo lỗi phân loại

> [!NOTE] ELI5
> Giả sử đáp án đúng là Mèo. Nếu mô hình đoán "90% là Mèo" → ít bất ngờ → **loss thấp**. Nếu mô hình đoán "10% là Mèo" → rất bất ngờ → **loss cao**. Cross-entropy đo chính xác mức "bất ngờ" đó.

### 5.1 Dẫn dắt từ MLE

Buổi 10 ta đã biết: MSE loss có nền tảng xác suất từ [[Maximum Likelihood Estimation|MLE]] (giả định nhiễu Gaussian). Tương tự, **cross-entropy loss có nền tảng từ MLE cho classification**:

1. **Maximize likelihood**: $\max \prod_{i=1}^n P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)})$
2. **Lấy $-\log$**: $\min \sum_{i=1}^n -\log P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)})$
3. **Thay softmax vào**: ra **cross-entropy loss**

### 5.2 Công thức

$$l(\mathbf{y}, \hat{\mathbf{y}}) = -\sum_{j=1}^q y_j \log \hat{y}_j$$

Vì $\mathbf{y}$ là one-hot (chỉ $y_c = 1$, còn lại = 0):

$$l(\mathbf{y}, \hat{\mathbf{y}}) = -\log \hat{y}_c$$

→ **Chỉ nhìn vào xác suất mà mô hình gán cho class đúng.**

### 5.3 Ví dụ bằng số

![[assets/attachments/D2L/Buổi 14/cross_entropy_concept.png]]

| Tình huống | Nhãn thật | Dự đoán | Loss |
| --- | --- | --- | --- |
| **Đoán đúng ✅** | Mèo $(1,0,0)$ | $(0.9, 0.05, 0.05)$ | $-\log(0.9) = 0.105$ |
| **Đoán sai ❌** | Mèo $(1,0,0)$ | $(0.1, 0.6, 0.3)$ | $-\log(0.1) = 2.303$ |
| **Sai nghiêm trọng** | Mèo $(1,0,0)$ | $(0.01, 0.5, 0.49)$ | $-\log(0.01) = 4.605$ |

Loss tăng **rất nhanh** khi dự đoán sai → ép mô hình phải sửa nhanh.

### 5.4 Gradient đẹp

$$\frac{\partial l}{\partial o_j} = \hat{y}_j - y_j = \text{softmax}(\mathbf{o})_j - y_j$$

Gradient = **dự đoán − thực tế** — giống MSE trong regression! Đơn giản, hiệu quả, dễ tính.

> Xem thêm: [[Cross-Entropy Loss]]

---

## Phần 6: Information Theory — Ý nghĩa sâu hơn (Tùy chọn)

> [!NOTE] ELI5
> **Entropy** = "mức độ bất ngờ trung bình" của dữ liệu. Tung đồng xu → khá bất ngờ (entropy cao). Mặt trời mọc hàng ngày → không bất ngờ (entropy thấp).
>
> **Cross-entropy** = "mức độ bất ngờ khi bạn dùng mô hình sai để đoán". Mô hình càng sai → càng bất ngờ → cross-entropy càng cao.

### 6.1 Entropy

$$H[P] = -\sum_j P(j) \log P(j)$$

Đo lượng **thông tin trung bình** cần để mã hóa dữ liệu từ phân phối $P$.

### 6.2 Surprisal (Mức bất ngờ)

$$\text{Surprisal}(j) = -\log P(j)$$

- Sự kiện xác suất cao → ít bất ngờ → surprisal thấp
- Sự kiện xác suất thấp → rất bất ngờ → surprisal cao

### 6.3 Cross-Entropy

$$H(P, Q) = -\sum_j P(j) \log Q(j) \geq H(P)$$

- $P$ = phân phối thật, $Q$ = phân phối mô hình dự đoán
- Cross-entropy **luôn ≥ Entropy** (đẳng thức khi $P = Q$)
- Training = **giảm cross-entropy** = đưa mô hình gần phân phối thật nhất

> [!TIP] Hai cách hiểu Cross-Entropy Loss
> 1. **MLE**: Maximize xác suất gán cho class đúng
> 2. **Information Theory**: Minimize mức bất ngờ khi dùng mô hình dự đoán

---

## 📖 Từ điển thuật ngữ Buổi 14

| Thuật ngữ | Dịch nghĩa | Nghĩa trong buổi này | Ví dụ |
| --- | --- | --- | --- |
| **Classification** | Phân loại | Dự đoán class (nhóm) thay vì số | Ảnh → mèo/chó/gà |
| **One-Hot Encoding** | Mã hóa one-hot | Vector chỉ có 0 và 1, đại diện class | Mèo = (1,0,0) |
| **Logits** | Điểm thô | Output chưa qua softmax | $o = (2.0, 1.0, 0.1)$ |
| **Softmax** | Hàm softmax | Biến logits → xác suất (dương, tổng=1) | $(0.66, 0.24, 0.10)$ |
| **Cross-Entropy Loss** | Hàm mất mát cross-entropy | Đo lỗi phân loại dựa trên MLE | $-\log(0.9) = 0.105$ |
| **Fully Connected** | Kết nối đầy đủ | Mỗi output nối tất cả inputs | Layer trong softmax regression |
| **Entropy** | Entropy (lượng tin) | Mức bất ngờ trung bình của phân phối | Tung xu → entropy cao |
| **Surprisal** | Mức bất ngờ | $-\log P(j)$ cho sự kiện $j$ | P(mưa)=0.01 → surprisal cao |
| **NLL** | Negative Log-Likelihood | $-\log P(\mathbf{Y} \mid \mathbf{X})$ | = tổng cross-entropy trên dataset |
| **Temperature** | Nhiệt độ | Hệ số điều chỉnh độ "sắc" của softmax | $T$ nhỏ → sắc, $T$ lớn → phẳng |

---

## ✅ Bài tự kiểm tra

1. Tại sao **không nên** dùng MSE loss cho bài toán classification? (Gợi ý: output không phải xác suất)
2. Cho logits $\mathbf{o} = (3, 1, -1)$. Tính softmax của vector này.
3. Nhãn thật là class 1 (one-hot $(1,0,0)$), mô hình dự đoán $(0.7, 0.2, 0.1)$. Tính cross-entropy loss.
4. Giải thích tại sao gradient của cross-entropy + softmax là $\hat{y}_j - y_j$ — đơn giản đến vậy có **lợi ích gì** cho training?
5. `nn.CrossEntropyLoss()` trong PyTorch nhận **logits** hay **probabilities**? Nếu bạn tự apply softmax trước rồi đưa vào thì sao?

---

## 🔗 Liên kết

- **Buổi trước**: [[Buổi 13 - Tuần 4]] — Generalization
- **Buổi sau**: [[Buổi 15 - Tuần 4]] — Image Classification Dataset (Fashion-MNIST)
- **Concept notes**: [[Softmax Function]], [[Cross-Entropy Loss]], [[One-Hot Encoding]], [[Maximum Likelihood Estimation]]

## 📝 Kết luận

Buổi 14 đánh dấu bước chuyển từ **regression** sang **classification** — từ "bao nhiêu?" sang "cái nào?". Ba thành phần mới cốt lõi:

1. **One-hot encoding** — cách biểu diễn nhãn phân loại không có thứ tự
2. **Softmax** — chuyển điểm thô thành phân phối xác suất hợp lệ
3. **Cross-entropy loss** — đo lỗi dự đoán, có nền tảng từ MLE và information theory

Gradient cực kỳ đẹp: $\hat{y} - y$ — giống hệt linear regression. Buổi 15 sẽ bắt tay vào dữ liệu thật: **Fashion-MNIST**.
