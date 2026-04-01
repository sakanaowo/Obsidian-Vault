---
title: "Buổi 20 - Tuần 5: Generalization in Deep Learning — Overfitting, Underfitting & Regularization (D2L)"
tags: [d2l, generalization, overfitting, underfitting, regularization, early-stopping, weight-decay, study-note]
created: 2026-03-25
session: "D2L Tuần 5, Buổi 20 — Generalization in Deep Learning"
source:
  - "30_Resources/Books/Dive Into Deep Learning/d2l-en/chapter_multilayer-perceptrons/generalization-deep.md"
related:
  - "[[Buổi 19 - Tuần 5]]"
  - "[[Buổi 18 - Tuần 5]]"
  - "[[Multilayer Perceptron]]"
---

# Buổi 20 — Generalization: Tại sao mạng "giỏi" trên data train lại "dốt" trên data mới?

> [!NOTE] ELI5
> Buổi 19, MLP đạt accuracy ~88% trên Fashion-MNIST. Nhưng có 1 câu hỏi quan trọng chưa trả lời: **mô hình chỉ "giỏi" trên data đã biết, hay thật sự "hiểu" quy luật?**
>
> Tưởng tượng bạn ôn thi bằng cách **học thuộc 100 câu hỏi đề cũ**. Nếu đề thi mới ra trúng 100 câu đó → điểm cao. Nhưng nếu đề ra câu mới → bạn không biết làm. Đó là **overfitting** (học tủ).
>
> Ngược lại, nếu bạn **chỉ đọc lướt** mục lục mà không đọc sâu → cả đề cũ lẫn đề mới đều làm dở. Đó là **underfitting** (học chưa đủ).
>
> Buổi 20 này sẽ giải thích: tại sao overfitting xảy ra, cách phát hiện, và **3 kỹ thuật** để ngăn chặn.

---

## 🎯 Mục tiêu buổi học

1. Phân biệt **underfitting** vs **overfitting** — nhận dạng qua training/test curves
2. Hiểu **generalization gap** — khoảng cách giữa train error và test error
3. Biết 3 kỹ thuật chống overfitting: **Early Stopping**, **Weight Decay**, **Dropout** (giới thiệu)
4. Hiểu tại sao deep learning **thách thức** lý thuyết cổ điển về generalization

---

## Phần 1: Underfitting vs Overfitting

> [!NOTE] ELI5
> Bạn đang chơi trò "nối các điểm" trên giấy:
> - **Underfitting**: bạn vẽ 1 đường thẳng qua đám điểm cong → đường thẳng **không khớp** gì cả
> - **Vừa đủ**: bạn vẽ đường cong mượt → khớp tốt với cả điểm đã biết lẫn điểm mới
> - **Overfitting**: bạn vẽ đường loằng ngoằng đi qua **từng điểm một** → khớp 100% điểm cũ, nhưng sai hoàn toàn ở những chỗ không có điểm

### 1.1 Biểu đồ kinh điển

![[assets/attachments/D2L/Buoi20/underfit_overfit.png]]

Giải thích biểu đồ:
- **Trục ngang**: độ phức tạp mô hình (từ linear đơn giản → MLP nhiều tầng)
- **Đường xanh (Training Error)**: error trên data đã biết — **luôn giảm** khi mô hình phức tạp hơn
- **Đường cam (Test Error)**: error trên data mới — lúc đầu giảm, sau đó **tăng lại**!
- **Sweet spot**: điểm mà test error thấp nhất — mô hình vừa đủ phức tạp

> [!question]- ❓ "Generalization Gap" là gì?
> **Generalization Gap** = khoảng cách giữa training error và test error.
>
> - Gap **nhỏ** → model khái quát tốt (học được quy luật chung)
> - Gap **lớn** → model overfit (chỉ giỏi trên data cũ, tệ trên data mới)
>
> Ví dụ cụ thể:
> - Training accuracy = 99%, Test accuracy = 88% → Gap = 11% → **overfit**
> - Training accuracy = 87%, Test accuracy = 85% → Gap = 2% → **bình thường**

### 1.2 Dấu hiệu nhận biết

| | Underfitting | Overfitting |
| --- | --- | --- |
| **Training error** | Cao | **Thấp** (gần 0) |
| **Test error** | Cao | **Cao** (dù train thấp) |
| **Generalization gap** | Nhỏ (cả hai đều tệ) | **Lớn** |
| **Nguyên nhân** | Model quá đơn giản, ít tham số, train ít epoch | Model quá phức tạp, nhiều tham số, train nhiều epoch |
| **Giống như** | Đọc lướt mục lục → không hiểu gì | Học thuộc đề cũ → không làm được đề mới |

> [!question]- ❓ 203K tham số của MLP (Buổi 19) có phải "quá nhiều" cho Fashion-MNIST?
> Fashion-MNIST có **60,000 ảnh train**, mỗi ảnh 784 pixels. MLP có 203,530 tham số — tức là **nhiều tham số hơn số lượng input pixels × 0.26**!
>
> Trong deep learning, đây là hiện tượng phổ biến gọi là **over-parametrization** (thừa tham số): số tham số **nhiều hơn** số dữ liệu. Theo lý thuyết cổ điển, model sẽ overfitting nặng.
>
> Nhưng thực tế? Deep networks **vẫn khái quát tốt** dù over-parametrized — đây là one of the biggest mysteries of deep learning! (Xem thêm Phần 4). Các kỹ thuật regularization ở Phần 3 giúp kiểm soát vấn đề này.

---

## Phần 2: Deep Learning thách thức lý thuyết cổ điển

> [!NOTE] ELI5
> Lý thuyết thống kê cổ điển nói: "model phức tạp → overfit". Nhưng deep learning phản bác:
> - MLP có **hàng triệu tham số** vẫn khái quát tốt
> - Tăng kích thước mạng đôi khi **cải thiện** generalization, không phải làm xấu đi
> - Mạng có thể fit **bất kỳ data nào**, kể cả data random — nhưng vẫn học được quy luật thật khi data có quy luật
>
> Đây là **bài toán mở** lớn nhất của deep learning lý thuyết.

### 2.1 Hiện tượng kỳ lạ: Fit 100% training data, vẫn generalize tốt

Zhang et al. (2017) chứng minh: mạng neural có thể **nhớ hoàn toàn** (memorize) bất kỳ dataset nào, kể cả labels hoàn toàn ngẫu nhiên. Tức là mạng đủ "sức chứa" để học thuộc mọi thứ.

Nhưng khi labels thật (có quy luật), cùng mạng đó **khái quát tốt** ra data mới. Tại sao?

> [!question]- ❓ Tại sao mạng có thể memorize nhưng không luôn memorize?
> Có 2 quan sát quan trọng:
>
> 1. **Mạng học quy luật trước, nhiễu sau**: Khi train, mạng **ưu tiên học các pattern đơn giản** (quy luật thật) trong vài epoch đầu. Chỉ khi train quá lâu, nó mới bắt đầu "nhớ" các điểm nhiễu/ngoại lệ.
>
> 2. **SGD có bias ngầm**: Stochastic Gradient Descent (SGD) không tìm **mọi** bộ trọng số fit training data — nó có xu hướng tìm bộ trọng số **"mượt"** nhất (simplest function that fits). Đây gọi là **implicit regularization**.
>
> → Đó là lý do Early Stopping hiệu quả: dừng trước khi mạng bắt đầu memorize nhiễu!

### 2.2 Mô hình phi tham số (Nonparametric Viewpoint)

> [!question]- ❓ Deep network là "parametric" hay "nonparametric"?
> Nghe mâu thuẫn nhưng: dù có **hàng triệu parameters**, deep network hành xử giống mô hình **nonparametric** (phi tham số) hơn:
>
> - **Parametric**: số tham số cố định, không phụ thuộc vào lượng data (ví dụ: Linear Regression luôn có d+1 tham số)
> - **Nonparametric**: "sức chứa" tăng theo lượng data (ví dụ: K-Nearest Neighbors lưu toàn bộ training data)
>
> Deep networks có quá nhiều tham số → fit được mọi data → giống nonparametric. Nghiên cứu gần đây (Neural Tangent Kernel) cho thấy mạng rộng vô hạn **tương đương** với kernel methods (1 dạng nonparametric).
>
> **Lesson**: đừng hoảng khi thấy model có triệu tham số. Quan trọng là **dùng kỹ thuật regularization đúng** chứ không phải giảm tham số.

---

## Phần 3: Ba kỹ thuật chống Overfitting

### 3.1 Early Stopping — "Dừng đúng lúc"

> [!NOTE] ELI5
> Giống nấu ăn: để lâu vừa đủ → chín ngon. Để quá lâu → cháy. **Dừng trước khi cháy** = early stopping.

**Cách hoạt động:**
1. Chia data thành **3 phần**: train, validation (kiểm tra), test
2. Mỗi epoch: train trên train set, **đo loss trên validation set**
3. Khi validation loss **bắt đầu tăng** (dù training loss vẫn giảm) → **DỪNG!**
4. Lấy model tại epoch có validation loss **thấp nhất**

![[assets/attachments/D2L/Buoi20/early_stopping.png]]
*Training loss (xanh) luôn giảm. Nhưng validation loss (cam) đến 1 lúc bắt đầu TĂNG → đó là lúc nên dừng!*

> [!question]- ❓ "Patience" là gì trong Early Stopping?
> Không dừng ngay khi validation loss tăng 1 epoch (có thể chỉ là noise). Thay vào đó:
>
> - Đặt **patience = N** (ví dụ N = 5)
> - Nếu validation loss **không cải thiện** trong N epoch liên tiếp → mới dừng
> - Trả về model tại epoch tốt nhất (không phải epoch cuối)
>
> ```python
> # Pseudo-code Early Stopping
> best_val_loss = float('inf')
> patience_counter = 0
> patience = 5
>
> for epoch in range(100):
>     train_loss = train_one_epoch()
>     val_loss = validate()
>     
>     if val_loss < best_val_loss:
>         best_val_loss = val_loss
>         save_model()           # Lưu model tốt nhất
>         patience_counter = 0   # Reset counter
>     else:
>         patience_counter += 1
>         if patience_counter >= patience:
>             print("Early stopping!")
>             break  # DỪNG
> ```

> [!question]- ❓ Khi nào Early Stopping hiệu quả nhất?
> - **Rất hiệu quả** khi data có **nhiễu** (noisy labels, dữ liệu không hoàn hảo)
> - **Ít hiệu quả** khi data sạch, classes rõ ràng (mèo vs chó rõ ràng)
>
> Lý do: mạng học quy luật trước, nhiễu sau. Early stopping cắt đúng giai đoạn mạng bắt đầu học nhiễu.

### 3.2 Weight Decay ($L_2$ Regularization) — "Phạt trọng số lớn"

> [!NOTE] ELI5
> Tưởng tượng bạn viết bài luận. Nếu bạn dùng **từ ngữ cực kỳ phức tạp** (trọng số lớn), bài có thể "fit" mọi ý tưởng nhưng rất rối. Nếu bạn bị **phạt mỗi khi dùng từ phức tạp**, bạn sẽ viết đơn giản hơn → bài rõ ràng hơn, dễ hiểu hơn cho người đọc mới. Đó là Weight Decay.

**Cách hoạt động:**

Thêm "hình phạt" vào loss function — **phạt trọng số lớn**:

$$\text{Loss}_{\text{mới}} = \text{Loss}_{\text{gốc}} + \frac{\lambda}{2} \|\mathbf{W}\|^2$$

Trong đó:
- $\text{Loss}_{\text{gốc}}$: cross-entropy loss (như cũ)
- $\|\mathbf{W}\|^2 = \sum w_i^2$: tổng bình phương tất cả trọng số
- $\lambda$: **hệ số phạt** — càng lớn → phạt càng nặng → trọng số bị "ép" nhỏ lại

![[assets/attachments/D2L/Buoi20/weight_decay.png]]
*Trái: không weight decay → đường cong loằng ngoằng (overfit). Phải: có weight decay → đường cong mượt, bám quy luật thật.*

> [!question]- ❓ Tại sao trọng số nhỏ → model đơn giản hơn?
> Trọng số nhỏ → mỗi feature chỉ ảnh hưởng **ít** đến output → model không "phản ứng thái quá" với từng điểm data → đường cong mượt hơn, ít nhạy với nhiễu.
>
> Ngược lại, trọng số lớn → model "nhảy nhót" mạnh → dễ khớp từng điểm nhưng sai ở giữa.

> [!question]- ❓ $\lambda$ chọn bao nhiêu?
> - $\lambda$ quá nhỏ (ví dụ $10^{-5}$) → phạt quá nhẹ → gần như không có tác dụng
> - $\lambda$ quá lớn (ví dụ $1$) → phạt quá nặng → trọng số gần 0 → **underfitting**
> - Thường dùng: $\lambda \in [10^{-4}, 10^{-2}]$
>
> **Trong PyTorch**: tham số `weight_decay` trong optimizer:
> ```python
> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-3)
> # weight_decay = lambda = 0.001
> ```

### 3.3 Dropout — "Random tắt neuron" (giới thiệu)

> [!NOTE] ELI5
> Khi thi nhóm, nếu **1 người giỏi** làm hết → nhóm phụ thuộc vào 1 người. Nếu **random cho vài người nghỉ** mỗi buổi → buộc MỌI NGƯỜI phải học → cả nhóm đều mạnh.
>
> Dropout = "random tắt" 1 số neurons **mỗi lần train**. Buộc mạng **không phụ thuộc** vào neurons cụ thể → robust hơn.

**Cách hoạt động:**
- Mỗi forward pass khi train: **random tắt** mỗi neuron với xác suất $p$ (thường $p = 0.5$)
- Neuron bị tắt → output = 0 (như thể nó không tồn tại)
- Khi test: **bật tất cả** neurons nhưng **nhân output với $(1-p)$** (để bù)

```python
model = nn.Sequential(
    nn.Flatten(),
    nn.LazyLinear(256), nn.ReLU(),
    nn.Dropout(0.5),         # ← Random tắt 50% neurons!
    nn.LazyLinear(10)
)
```

> [!IMPORTANT] Dropout sẽ được học chi tiết ở buổi sau (D2L section riêng). Ở đây chỉ giới thiệu.

---

## Phần 4: Tổng hợp — Khi nào dùng gì?

![[assets/attachments/D2L/Buoi20/generalization_gap.png]]
*Mô hình lớn hơn (nhiều tham số) → train accuracy tăng, nhưng test accuracy KHÔNG tăng: generalization gap mở rộng.*

| Kỹ thuật | Dùng khi nào | Code PyTorch |
| --- | --- | --- |
| **Early Stopping** | Hầu như luôn luôn — đặc biệt khi data có nhiễu | Tự viết patience logic |
| **Weight Decay** | Mặc định bật ($\lambda = 10^{-4}$) | `weight_decay=1e-4` trong optimizer |
| **Dropout** | MLP/Dense layers có nhiều tham số | `nn.Dropout(0.5)` sau ReLU |
| **Giảm model** | Data rất ít (< 1000 mẫu) | Giảm `num_hiddens`, bớt tầng |

> [!question]- ❓ Có thể dùng cả 3 cùng lúc không?
> **Có!** Và thực tế thường **phải** dùng nhiều kỹ thuật cùng lúc:
>
> ```python
> model = nn.Sequential(
>     nn.Flatten(),
>     nn.LazyLinear(256), nn.ReLU(), nn.Dropout(0.5),  # Dropout
>     nn.LazyLinear(10)
> )
> optimizer = torch.optim.SGD(
>     model.parameters(), lr=0.1,
>     weight_decay=1e-3  # Weight Decay
> )
> # + Early Stopping logic trong training loop
> ```
>
> Mỗi kỹ thuật chống overfitting theo **cách khác nhau**:
> - Early stopping: giới hạn **thời gian** train
> - Weight decay: giới hạn **kích thước** trọng số
> - Dropout: giới hạn **sự phụ thuộc** vào neurons cụ thể

---

## 📖 Từ điển thuật ngữ Buổi 20

| Thuật ngữ | Dịch sang tiếng Việt | Nghĩa dễ hiểu |
| --- | --- | --- |
| **Generalization** | Khái quát hóa | Khả năng model hoạt động tốt trên data **chưa thấy** |
| **Overfitting** | Quá khớp / Học tủ | Train error thấp nhưng test error cao |
| **Underfitting** | Dưới khớp / Học chưa đủ | Cả train lẫn test error đều cao |
| **Generalization gap** | Khoảng cách khái quát | Chênh lệch giữa train error và test error |
| **Early stopping** | Dừng sớm | Dừng train khi validation loss bắt đầu tăng |
| **Patience** | Sự kiên nhẫn | Số epoch chờ trước khi dừng |
| **Weight decay** | Suy giảm trọng số | Phạt trọng số lớn bằng cách cộng $\lambda\|\mathbf{W}\|^2$ vào loss |
| **$L_2$ regularization** | Chính quy hóa L2 | Tên khác của weight decay |
| **Dropout** | Bỏ ngẫu nhiên | Random tắt neurons khi train |
| **Over-parametrized** | Thừa tham số | Số tham số > số dữ liệu |
| **Interpolation** | Nội suy | Model fit **chính xác** mọi điểm train data |
| **Implicit regularization** | Chính quy hóa ngầm | SGD tự động ưu tiên "hàm đơn giản" |
| **Validation set** | Tập kiểm định | Phần data dùng để kiểm tra overfitting (không phải test) |

---

## ✅ Bài tự kiểm tra

1. Training accuracy = 99%, Test accuracy = 75%. Đây là **overfitting** hay **underfitting**? Generalization gap = ?
2. Tại sao **early stopping** được coi là 1 dạng regularization?
3. Weight decay thêm $\frac{\lambda}{2}\|\mathbf{W}\|^2$ vào loss. Nếu $\lambda = 0$, điều gì xảy ra?
4. Dropout tắt 50% neurons khi train. Khi test thì sao?
5. "Model có nhiều tham số → chắc chắn overfitting." Đúng hay sai?

> [!NOTE]- 📝 Đáp án
> 1. **Overfitting.** Gap = 99% - 75% = **24%** — rất lớn. Model "học thuộc" train data.
> 2. Vì nó **hạn chế** khả năng học của model — bằng cách giới hạn **số epoch** (thời gian train). Giống weight decay hạn chế **kích thước** trọng số. Cả hai đều ngăn model quá phức tạp.
> 3. $\lambda = 0$ → penalty = 0 → **không có weight decay** → loss function giữ nguyên. Model có thể có trọng số lớn tùy ý.
> 4. **Bật tất cả** neurons nhưng **nhân output × (1-p)** = 0.5. Vì khi train chỉ có 50% neurons hoạt động, output bị giảm một nửa. Khi test bật hết → phải scale xuống để output tổng thể không đổi.
> 5. **Sai.** Deep networks thường over-parametrized nhưng vẫn generalize tốt nhờ: implicit regularization của SGD, early stopping, weight decay, dropout. Số tham số ≠ overfitting — quan trọng là **cách train**, không chỉ kích thước model.

---

## 🔗 Liên kết

- **Buổi trước**: [[Buổi 19 - Tuần 5]] — MLP Implementation
- **Buổi sau**: [[Buổi 21 - Tuần 6]] — Weight Decay & Dropout (chi tiết + code)
- **Concept notes**: [[Multilayer Perceptron]], [[Activation Function]]

## 📝 Kết luận

Buổi 20 hoàn thành bức tranh **generalization** — khái niệm trung tâm của machine learning:
- **Underfitting**: model quá đơn giản → cần tăng phức tạp
- **Overfitting**: model quá phức tạp hoặc train quá lâu → cần regularization
- **3 kỹ thuật chống overfitting**: Early Stopping (dừng đúng lúc), Weight Decay (phạt trọng số lớn), Dropout (random tắt neuron)
- Deep learning **thách thức** trực giác: model over-parametrized vẫn generalize tốt — implicit regularization của SGD đóng vai trò quan trọng

Buổi 21 sẽ đi sâu vào **Weight Decay và Dropout** — với code PyTorch chi tiết.
