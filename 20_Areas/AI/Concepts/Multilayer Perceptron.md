---
title: "Multilayer Perceptron"
aliases: [MLP, multi-layer perceptron, mạng đa tầng, feedforward neural network, mạng truyền thẳng]
tags: [concept, deep-learning, neural-network, architecture]
created: 2026-03-24
modified: 2026-03-25
---

# Multilayer Perceptron (MLP) — Mạng đa tầng

> [!NOTE] ELI5
> Hãy tưởng tượng bạn muốn phân biệt **ảnh mèo** với **ảnh chó**. Cách đơn giản nhất: nhìn từng pixel, gán trọng số, cộng lại → ra 1 số → phân loại. Nhưng cách này chỉ "kẻ đường thẳng" — quá thô.
>
> **MLP** thêm "tầng trung gian" (tầng ẩn) ở giữa. Tầng này tự học cách nhìn ra những **nét đặc trưng** (viền tai, hình mũi) mà từng pixel riêng lẻ không thể hiện được. Sau đó, tầng cuối nhìn những nét đặc trưng này rồi mới phân loại.
>
> MLP = kiến trúc neural network **cơ bản nhất** của Deep Learning. Mọi mạng phức tạp hơn (CNN, Transformer) đều xây dựng **trên nền tảng** MLP.

---

## 1. MLP hoạt động thế nào?

### Luồng dữ liệu

```
Dữ liệu thô (pixels)  →  Tầng ẩn (tự học đặc trưng)  →  Output (phân loại)
     Input (d)                Hidden Layer (h)             Output (q)
       X             →     H = σ(XW¹ + b¹)         →   O = HW² + b²
```

Dữ liệu đi **1 chiều** từ trái sang phải (nên còn gọi là **feedforward** — truyền thẳng).

### Công thức

$$\begin{aligned}
\mathbf{H} &= \sigma(\mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)}) && \text{Bước 1: Tính tầng ẩn} \\
\mathbf{O} &= \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)} && \text{Bước 2: Tính output}
\end{aligned}$$

### Giải thích từng ký hiệu

| Ký hiệu | Kích thước | Ý nghĩa bằng lời |
| --- | --- | --- |
| $\mathbf{X}$ | $(n, d)$ | **Dữ liệu đầu vào**: $n$ mẫu, mỗi mẫu $d$ đặc trưng. Ví dụ: 64 ảnh, mỗi ảnh 784 pixels → $(64, 784)$ |
| $\mathbf{W}^{(1)}$ | $(d, h)$ | **Bảng trọng số thứ 1**: kết nối input → tầng ẩn. Mỗi ô = "pixel thứ $j$ quan trọng bao nhiêu với unit ẩn thứ $k$" |
| $\mathbf{b}^{(1)}$ | $(1, h)$ | **Bias thứ 1**: mỗi unit ẩn có 1 "mức mặc định" |
| $\sigma$ | — | **Hàm bẻ cong** ([[Activation Function]]) — ví dụ ReLU. Nếu không có, tầng ẩn vô nghĩa! |
| $\mathbf{H}$ | $(n, h)$ | **Kết quả tầng ẩn**: mỗi mẫu giờ được "mô tả" bằng $h$ đặc trưng mới thay vì $d$ pixels thô |
| $\mathbf{W}^{(2)}$ | $(h, q)$ | **Bảng trọng số thứ 2**: kết nối tầng ẩn → output |
| $\mathbf{b}^{(2)}$ | $(1, q)$ | **Bias thứ 2** |
| $\mathbf{O}$ | $(n, q)$ | **Output (logits)**: $q$ con số thô → đưa qua softmax ra xác suất |

> [!TIP] Hình dung bằng ví dụ cụ thể (Fashion-MNIST)
> - Input: 784 pixels (ảnh quần áo 28×28 duỗi phẳng)
> - Tầng ẩn: 256 units — mỗi unit "nhìn" toàn bộ 784 pixels, tổng hợp thành 1 đặc trưng
> - Output: 10 lớp (t-shirt, trouser, pullover, …, boot)
> - Tổng tham số: $784 \times 256 + 256 + 256 \times 10 + 10 = 203{,}530$

---

## 2. Tại sao cần tầng ẩn?

### Giới hạn của mô hình 1 tầng (linear)

Mô hình 1 tầng (như Softmax Regression) giả định **đơn điệu** (monotonic): tăng input → output **luôn tăng** hoặc **luôn giảm**. Nhưng thực tế có rất nhiều ngoại lệ:

| Ví dụ | Tại sao 1 tầng thất bại? |
| --- | --- |
| **Nhiệt cơ thể** | 37°C bình thường. Cao hơn **hoặc** thấp hơn đều nguy hiểm → quan hệ hình chữ U, không phải đường thẳng |
| **Nhận dạng ảnh** | Tăng pixel (13,17) không "luôn" tăng xác suất đó là chó. Context mới quan trọng |
| **Thu nhập vs hạnh phúc** | Từ 0→50tr: mỗi đồng thêm rất quan trọng. Từ 1 tỷ→1.05 tỷ: gần như không khác biệt |

![[assets/attachments/D2L/Buoi18/linear_limitation.png]]
*Trái: Nhiệt cơ thể vs Sức khỏe — đường thẳng hoàn toàn sai. Phải: Dữ liệu vòng tròn lồng nhau — đường thẳng không tách được.*

### Tầng ẩn giải quyết bằng cách nào?

Tầng ẩn + activation = **tự động học biểu diễn phi tuyến**:
- Thay vì nhìn 784 pixels thô (không có ý nghĩa), tầng ẩn tự tạo ra 256 đặc trưng mới
- Đặc trưng mới có thể là: "có đường viền ngang ở giữa", "có vùng tối ở góc trái", …
- Những đặc trưng này **có ý nghĩa** hơn pixel thô → tầng output dễ phân loại hơn

> [!IMPORTANT] Không có activation → tầng ẩn vô nghĩa!
> Nếu bỏ hàm $\sigma$, thì tầng ẩn + tầng output gộp thành **1 tầng tuyến tính duy nhất** (vì phép nhân ma trận chồng nhau = 1 phép nhân ma trận). Xem chứng minh ở [[Activation Function#1. Tại sao cần Activation Function?|Activation Function]].

---

## 3. Universal Approximation Theorem (Định lý xấp xỉ vạn năng)

> **Nội dung (Cybenko, 1989)**: MLP chỉ cần **1 tầng ẩn**, với đủ nhiều units và activation phi tuyến, có thể **xấp xỉ bất kỳ hàm liên tục nào**, với sai số nhỏ tùy ý.

### Nghe hay, nhưng có 3 giới hạn lớn:

| Giới hạn | Giải thích dễ hiểu |
| --- | --- |
| "Đủ nhiều" = bao nhiêu? | Có thể cần **hàng tỷ** units → bộ nhớ không đủ |
| Tìm được trọng số đúng | Định lý chỉ nói "tồn tại" bộ trọng số phù hợp — **không nói cách tìm**. Quá trình training có thể mắc kẹt |
| **Sâu > Rộng** | Thực tế: nhiều tầng ít units **hiệu quả hơn** 1 tầng nhiều units |

### Tại sao sâu hơn tốt hơn?

Tưởng tượng bạn viết chương trình phức tạp:
- **1 tầng rộng** = viết TẤT CẢ trong hàm `main()` → khó, dài, không tái sử dụng
- **Nhiều tầng** = chia thành hàm nhỏ, mỗi hàm xử lý 1 phần → gọn, dễ hiểu

Trong deep learning, mỗi tầng xây trên tầng trước (**compositional**):
- Tầng 1: học **đường viền** (edges)
- Tầng 2: ghép đường viền → **hình dạng** (shapes)
- Tầng 3: ghép hình dạng → **bộ phận** (mắt, mũi)
- Tầng 4: ghép bộ phận → **khuôn mặt**

> [!TIP] Ẩn dụ hay
> MLP giống ngôn ngữ C: *có thể viết bất kỳ chương trình nào*, nhưng *viết đúng chương trình bạn cần* mới là thách thức. **Universal approximation ≠ universal learnability.**

---

## 4. Cách đếm số tầng — Quy ước quan trọng

> [!WARNING] Input layer **không** tính!
> **Chỉ đếm tầng có trọng số cần học** (hidden + output).
>
> Ví dụ: input layer + 1 hidden layer + output layer → gọi là **2-layer MLP**.
>
> Tại sao? Vì input layer chỉ "nhận dữ liệu vào", không có trọng số nào cần học.

---

## 5. Mở rộng: MLP nhiều tầng ẩn

MLP không giới hạn ở 1 tầng ẩn. Có thể thêm nhiều tầng:

$$\begin{aligned}
\mathbf{H}^{(1)} &= \sigma(\mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)}) \\
\mathbf{H}^{(2)} &= \sigma(\mathbf{H}^{(1)}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}) \\
\mathbf{O} &= \mathbf{H}^{(2)}\mathbf{W}^{(3)} + \mathbf{b}^{(3)}
\end{aligned}$$

→ Đây là **3-layer MLP** (2 hidden + 1 output). Thêm tầng = thêm khả năng biểu diễn, nhưng cũng dễ overfit hơn và khó train hơn (cần kỹ thuật như [[Dropout]], [[Batch Normalization]]).

---

## TODO

- [ ] Thêm ví dụ decision boundary: so sánh linear vs MLP 1 tầng vs MLP 2 tầng
- [ ] Liên kết với CNN, Transformer (cũng dùng MLP bên trong)
- [ ] Thêm code ví dụ MLP đơn giản bằng PyTorch
