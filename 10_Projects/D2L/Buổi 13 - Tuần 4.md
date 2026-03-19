---
title: "Buổi 13 - Tuần 4: Generalization — Overfitting, Underfitting & Model Selection (D2L)"
tags: [d2l, generalization, overfitting, underfitting, model-selection, cross-validation, study-note]
created: 2026-03-18
session: "D2L Tuần 4, Buổi 13 — Generalization"
source:
  - "30_Resources/Books/Dive Into Deep Learning/d2l-en/chapter_linear-regression/generalization.md"
related:
  - "[[Buổi 12 - Tuần 3]]"
  - "[[Linear Regression for Deep Learning]]"
  - "[[Generalization]]"
  - "[[Overfitting and Underfitting]]"
  - "[[Bias-Variance Tradeoff]]"
  - "[[Cross-Validation]]"
  - "[[Training Error vs Generalization Error]]"
---

# Buổi 13 — Generalization: Vì sao train tốt chưa chắc test tốt?

> [!NOTE] ELI5
> Hãy tưởng tượng hai học sinh chuẩn bị thi cuối kỳ:
>
> - **Ellie** có trí nhớ siêu phàm — cô học thuộc lòng **toàn bộ đáp án** của các đề thi cũ. Nếu đề thi năm nay lặp lại câu cũ, Ellie đạt 100% ngay.
> - **Irene** trí nhớ bình thường nhưng giỏi **nhận ra quy luật**. Cô chỉ đạt 90% trên đề cũ, nhưng gặp đề mới vẫn giữ được 90%.
>
> Trong ML, chúng ta muốn xây **mô hình như Irene** — không chỉ "nhớ" dữ liệu cũ mà thật sự **hiểu quy luật** để dự đoán đúng trên dữ liệu mới. Khả năng đó gọi là **generalization** (tổng quát hóa).

---

## 🎯 Mục tiêu buổi học

Sau buổi này, bạn cần biết:

1. Phân biệt **training error** và **generalization error**
2. Nhận diện khi nào mô hình bị **overfitting** hoặc **underfitting**
3. Hiểu **model complexity** ảnh hưởng đến error thế nào
4. Biết cách chia dữ liệu thành **Training / Validation / Test**
5. Hiểu **K-fold cross-validation** dùng khi nào

---

## Phần 1: Training Error vs Generalization Error

> [!NOTE] ELI5
> **Training error** = điểm bài tập về nhà (tự chấm, dễ cao điểm). **Generalization error** = điểm bài thi thật (đề mới, mới phản ánh thực lực). Cái ta quan tâm là điểm thi thật.

### 1.1 Hai loại lỗi

| | Training Error $R_{\text{emp}}$ | Generalization Error $R$ |
| --- | --- | --- |
| **Đo trên** | Dữ liệu đã dùng để train | Dữ liệu mới chưa từng thấy |
| **Tính được không?** | ✅ Có, tính chính xác | ❌ Không, chỉ **ước lượng** |
| **Ẩn dụ** | Điểm bài tập về nhà | Điểm thi thật |
| **Công thức** | $\frac{1}{n} \sum_{i=1}^n l(\mathbf{x}^{(i)}, y^{(i)}, f(\mathbf{x}^{(i)}))$ | $E_{(\mathbf{x}, y) \sim P} [l(\mathbf{x}, y, f(\mathbf{x}))]$ |

> [!TIP] Tại sao không tính được generalization error chính xác?
> Vì ta không biết **phân phối thật** $p(\mathbf{x}, y)$ của dữ liệu. Ta chỉ có một **mẫu hữu hạn** rút từ phân phối đó. Giống như bạn không biết được **tất cả đề thi có thể ra** — bạn chỉ biết vài đề cũ.

### 1.2 Giả định IID — nền tảng của mọi đánh giá

Để việc tách train/test có ý nghĩa, ta cần giả định **IID (Independent and Identically Distributed)**:

- **Independent**: mỗi điểm dữ liệu được lấy **độc lập** với nhau
- **Identically Distributed**: train data và test data đến từ **cùng một phân phối**

> [!WARNING] Khi nào IID bị vi phạm?
> - 📈 Dữ liệu cổ phiếu: train trên 2023, test trên 2025 → phân phối thay đổi
> - 🏥 Y tế: train từ bệnh viện A, test từ bệnh viện B → dân số khác nhau
> - 📷 Ảnh: train trong phòng lab, test ngoài đời → ánh sáng/góc chụp khác

### 1.3 Generalization Gap

$$\text{Generalization Gap} = R - R_{\text{emp}}$$

Đây là khoảng cách giữa "điểm bài tập" và "điểm thi thật":

- **Gap nhỏ** → mô hình đang generalize tốt
- **Gap lớn** → mô hình đang **overfitting**
- **Cả hai lỗi đều cao** → mô hình đang **underfitting**

> Xem thêm: [[Training Error vs Generalization Error]]

---

## Phần 2: Overfitting vs Underfitting

> [!NOTE] ELI5
> - **Underfitting** = bạn chỉ đọc lướt 5 phút rồi đi thi → cái gì cũng sai vì chưa học đủ
> - **Overfitting** = bạn thuộc lòng từng dấu chấm phẩy trong sách → gặp đề mới là bó tay vì nhớ chi tiết thay vì hiểu bản chất

### 2.1 Nhận biết qua bảng dấu hiệu

| | Underfitting | Good Fit ✅ | Overfitting |
| --- | --- | --- | --- |
| Training Error | 🔴 Cao | 🟢 Thấp | 🟢 Rất thấp |
| Validation Error | 🔴 Cao | 🟢 Thấp | 🔴 Cao |
| Gap | Nhỏ | Nhỏ | 🔴 Lớn |
| Mô hình | Quá đơn giản | Vừa đủ | Quá phức tạp |

### 2.2 Ví dụ trực quan: Polynomial Curve Fitting

Giả sử dữ liệu thật tuân theo $y = x^2 + \text{noise}$. Ta thử fit bằng polynomial bậc khác nhau:

![[assets/attachments/D2L/Buổi 13/overfitting_underfitting.png]]

| Bậc polynomial | Chuyện gì xảy ra? | Loại lỗi |
| --- | --- | --- |
| **Bậc 1** (đường thẳng) | Không thể uốn cong → sai cả train lẫn test | **Underfitting** |
| **Bậc 4** (vừa đủ) | Bắt được xu hướng chung, bỏ qua nhiễu | **Good fit** ✅ |
| **Bậc 15** (quá cao) | Chui qua mọi điểm, kể cả nhiễu → đường ngoằn ngoèo | **Overfitting** |

> [!TIP] Ý nghĩa từ ví dụ này
> Bậc polynomial = **model complexity** (độ phức tạp mô hình). Polynomial bậc cao hơn luôn fit training data tốt hơn hoặc bằng bậc thấp. Nhưng fit tốt training data **không đảm bảo** dự đoán tốt trên data mới.

> Xem thêm: [[Overfitting and Underfitting]]

---

## Phần 3: Model Complexity & Dataset Size

> [!NOTE] ELI5
> Mô hình phức tạp giống **chiếc tủ nhiều ngăn** — có thể chứa được nhiều thứ hơn, nhưng nếu bạn chẳng có gì để bỏ vào (ít data), thì mấy ngăn trống sẽ được "nhét" bằng rác (nhiễu). Muốn dùng tủ lớn, bạn cần **đủ đồ** (đủ data).

### 3.1 Model Complexity ảnh hưởng thế nào?

![[assets/attachments/D2L/Buổi 13/complexity_vs_error.png]]

Biểu đồ trên là **đồ thị kinh điển** trong ML:

- **Trục X**: độ phức tạp mô hình (từ đơn giản → phức tạp)
- **Trục Y**: lỗi (error)
- **Đường xanh (Training Error)**: **luôn giảm** khi mô hình phức tạp hơn — vì mô hình mạnh hơn sẽ fit training data tốt hơn
- **Đường đỏ (Generalization Error)**: hình chữ **U** — giảm trước rồi **tăng lại** khi mô hình quá phức tạp

Vùng bên **trái** U-curve = **underfitting zone** (high bias)
Vùng bên **phải** U-curve = **overfitting zone** (high variance)
Đáy U-curve = **sweet spot** (optimal complexity)

> [!IMPORTANT] Insight từ Karl Popper
> Triết gia Karl Popper: *"Một lý thuyết có thể giải thích MỌI THỨ thì thực ra không giải thích được gì cả."* Mô hình ML cũng vậy — nếu mô hình đủ mạnh để fit **bất kỳ labels nào** (kể cả labels ngẫu nhiên), thì việc fit tốt training data không chứng minh được gì về khả năng generalize.

### 3.2 Dataset Size ảnh hưởng thế nào?

| Lượng data | Hệ quả |
| --- | --- |
| **Ít data** | Dễ overfit, generalization error cao |
| **Nhiều data** | Generalization error giảm |
| **Rất nhiều data** | Có thể dùng mô hình phức tạp hơn |

**Quy tắc chung**: model complexity không nên tăng nhanh hơn lượng data có sẵn.

Ví dụ thực tế:
- Với 100 mẫu → linear regression có thể đáng tin cậy hơn deep learning
- Với 1 triệu mẫu → deep learning bắt đầu phát huy lợi thế

> Xem thêm: [[Bias-Variance Tradeoff]]

---

## Phần 4: Model Selection — Cách chia dữ liệu đúng

> [!NOTE] ELI5
> Bạn có 100 bài tập. Bạn dùng 60 bài để **luyện**, 20 bài để **thử sức** (chọn cách học tốt nhất), và 20 bài **cất kín** chỉ mở ra 1 lần cuối cùng để biết thực lực.
>
> - **Training set** = bài để luyện
> - **Validation set** = bài thử sức (dùng nhiều lần để so sánh)
> - **Test set** = bài thi thật (chỉ dùng 1 lần)

### 4.1 Tại sao cần 3 tập?

![[assets/attachments/D2L/Buổi 13/data_split_diagram.png]]

**Vấn đề**: Nếu dùng training data để chọn model → ta sẽ chọn model overfit nhất. Nếu dùng test data để chọn model → ta đang **overfit test data** mà không biết.

**Giải pháp**: Thêm **validation set** ở giữa:

1. **Training set (60%)** → train nhiều mô hình khác nhau
2. **Validation set (20%)** → so sánh các mô hình, chọn mô hình tốt nhất
3. **Test set (20%)** → đánh giá mô hình cuối cùng **1 lần duy nhất**

> [!CAUTION] Quy tắc vàng
> **KHÔNG BAO GIỜ** dùng test set để quyết định bất kỳ điều gì trong quá trình train (chọn hyperparameters, chọn kiến trúc, quyết định khi nào dừng). Test set là "phong bì niêm phong" chỉ mở ra ở cuối.

### 4.2 Validation Error (Lỗi xác nhận)

Khi ta nói "lỗi trên holdout data" = **validation error**. Trong cuốn D2L, hầu hết "accuracy" được báo cáo thực ra là **validation accuracy**, không phải test accuracy.

---

## Phần 5: K-Fold Cross-Validation

> [!NOTE] ELI5
> Khi bạn chỉ có ít bài tập (ít data), tách ra 20% validation sẽ lãng phí. Thay vào đó, bạn **chia 5 phần**, mỗi lần dùng 1 phần khác nhau làm bài kiểm tra, rồi lấy **trung bình 5 lần**. Như vậy mọi bài tập đều được dùng **cả để học lẫn để kiểm tra**.

### 5.1 Quy trình K-fold (K = 5)

![[assets/attachments/D2L/Buổi 13/kfold_cross_validation.png]]

1. Chia dữ liệu thành **5 phần bằng nhau** (folds)
2. Lặp 5 lần:
   - Lần $i$: dùng fold $i$ làm validation, 4 folds còn lại làm training
   - Train model, đo lỗi → $\text{Score}_i$
3. Kết quả:

$$\text{CV Score} = \frac{1}{5} \sum_{i=1}^{5} \text{Score}_i$$

### 5.2 Khi nào dùng K-fold?

| Tình huống | Dùng K-fold? | Lý do |
| --- | --- | --- |
| Data < vài nghìn mẫu | ✅ Nên dùng | Tận dụng tối đa data |
| Data hàng triệu mẫu | ❌ Không cần | Split cố định đã đủ |
| Training nhanh (vài giây) | ✅ Nên dùng | Không tốn thêm nhiều thời gian |
| Training chậm (hàng giờ/ngày) | ❌ Cân nhắc | Phải train K lần = K lần thời gian |

### 5.3 Nhược điểm

- Tốn tính toán (train K lần)
- Mỗi lần chỉ train trên $\frac{K-1}{K}$ data → bias nhẹ
- Không phù hợp data theo thời gian (cần Time Series CV)

> Xem thêm: [[Cross-Validation]]

---

## Phần 6: Lưu ý quan trọng cho Deep Learning

> [!IMPORTANT] Overfitting trong DL không đơn giản
> Trong deep learning modern, mô hình tốt nhất thường vẫn có **gap** giữa training error và validation error. Mạng neural sâu đủ mạnh để fit **bất kỳ labels ngẫu nhiên nào**, nhưng trong thực tế vẫn generalize tốt. Đây là hiện tượng mà lý thuyết truyền thống chưa giải thích trọn vẹn.
>
> **Training error thấp** không **đảm bảo** generalization tốt, nhưng cũng **không NHẤT THIẾT** có nghĩa generalization xấu.

---

## 📖 Từ điển thuật ngữ Buổi 13

| Thuật ngữ | Dịch nghĩa | Nghĩa trong buổi này | Ví dụ |
| --- | --- | --- | --- |
| **Generalization** | Tổng quát hóa | Khả năng đúng trên data mới | Model đạt val accuracy 90% |
| **Training Error** ($R_{\text{emp}}$) | Lỗi huấn luyện | Trung bình lỗi trên training set | MSE = 0.001 trên train |
| **Generalization Error** ($R$) | Lỗi tổng quát | Kỳ vọng lỗi trên phân phối thật | Ước lượng qua val/test error |
| **Generalization Gap** | Khoảng cách tổng quát | Chênh lệch $R - R_{\text{emp}}$ | Gap = 0.3 → đáng lo |
| **Overfitting** | Quá khớp / Học vẹt | Train tốt, test kém | Polynomial bậc 15 |
| **Underfitting** | Kém khớp | Train kém, test kém | Đường thẳng fit data cong |
| **IID** | Độc lập & cùng phân phối | Train và test từ cùng nguồn | Giả định cơ bản khi split data |
| **Validation Set** | Tập xác nhận | Data để chọn model tốt nhất | 20% data dùng so sánh models |
| **Test Set** | Tập kiểm tra | Data chỉ dùng 1 lần cuối | 20% data "niêm phong" |
| **Cross-Validation** | Kiểm chứng chéo | Xoay vòng validation khi data ít | K-fold với K=5 |
| **Model Complexity** | Độ phức tạp mô hình | Số tham số + phạm vi giá trị | Polynomial bậc 1 vs bậc 15 |
| **Regularization** | Chính quy hóa | Ép buộc mô hình đơn giản hơn | L2, Dropout, Early Stopping |

---

## ✅ Bài tự kiểm tra

1. **Training error thấp** có đảm bảo **generalization error thấp** không? Vì sao?
2. Bạn train model, thấy training loss = 0.01 nhưng validation loss = 0.85. Đây là **overfitting** hay **underfitting**? Bạn nên làm gì?
3. Giải thích bằng lời tại sao **không nên dùng test set** để chọn hyperparameters.
4. Nếu bạn chỉ có 200 mẫu data, bạn nên dùng **split cố định** hay **K-fold CV**? Vì sao?
5. Theo biểu đồ complexity vs error: khi tăng model complexity, training error **luôn giảm**. Tại sao generalization error lại có lúc **tăng trở lại**?

---

## 🔗 Liên kết

- **Buổi trước**: [[Buổi 12 - Tuần 3]] — Linear Regression Concise
- **Buổi sau**: [[Buổi 14 - Tuần 4]] — Softmax Regression
- **Concept notes**: [[Generalization]], [[Overfitting and Underfitting]], [[Training Error vs Generalization Error]], [[Cross-Validation]], [[Bias-Variance Tradeoff]]

## 📝 Kết luận

Buổi 13 trả lời câu hỏi lớn nhất sau khi đã biết train model: **"Train tốt rồi, nhưng có thật sự tốt không?"** Training error thấp mới chỉ là bước đầu — cái thật sự quan trọng là mô hình có generalize được hay không. Để đánh giá đúng, ta cần chia data thành train/validation/test, hiểu khi nào mô hình đang underfit hoặc overfit, và biết dùng cross-validation khi data ít.

Buổi 14 sẽ chuyển sang **classification** — từ dự đoán số sang dự đoán nhóm (`softmax regression`).
