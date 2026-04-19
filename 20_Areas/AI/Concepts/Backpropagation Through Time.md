---
tags:
  - ai
  - deep-learning
  - rnn
  - optimization
  - gradient
aliases:
  - BPTT
  - Backprop Through Time
date: 2026-04-19
---

# Backpropagation Through Time (BPTT)

> [!NOTE] ELI5
> Hãy tưởng tượng một dây chuyền lắp ráp dài 100 người. Nếu sản phẩm cuối bị lỗi, quản lý phải truy ngược lại từng người để tìm ai gây ra lỗi. Nhưng qua mỗi người, "manh mối" (gradient) bị méo đi — qua 100 người thì hoặc biến mất hoàn toàn (vanishing) hoặc bị phóng đại phi lý (exploding). BPTT là quy trình "truy lỗi ngược" này trong RNN.

**Backpropagation Through Time (BPTT)** là thuật toán tính gradient cho Recurrent Neural Networks (RNN). Về bản chất, nó là backpropagation thông thường áp dụng lên computational graph của RNN **đã unrolled theo thời gian** — biến mạng hồi quy thành một mạng feedforward rất sâu ($T$ layers tương ứng $T$ time steps).

**Tại sao BPTT đặc biệt?** Trong feedforward networks, mỗi layer có trọng số riêng. Trong RNN, **cùng một ma trận** $W_{hh}$ được nhân lại $T$ lần. Gradient do đó chứa **lũy thừa** $(W_{hh})^k$, gây ra vấn đề vanishing/exploding gradient.

## Đây là gì, nhận gì, trả gì?

BPTT là **thuật toán tính đạo hàm** cho RNN khi dữ liệu là chuỗi.

- **Input của BPTT:** toàn bộ chuỗi $x_1, x_2, \ldots, x_T$, các hidden states $h_1, \ldots, h_T$, và loss ở từng bước.
- **Output của BPTT:** gradient đối với các tham số như $W_{xh}$, $W_{hh}$, $W_{qh}$.
- **Mục tiêu:** biết tham số nào đã làm loss tăng, và tăng bao nhiêu, để optimizer cập nhật theo hướng ngược lại.

## Tại sao lại có chữ "Through Time"?

Vì RNN vốn là một vòng lặp, nhưng khi học ta phải **mở nó ra theo trục thời gian**:

$$x_1 \to h_1 \to o_1,\; x_2 \to h_2 \to o_2,\; \ldots,\; x_T \to h_T \to o_T$$

Gradient không chỉ đi ngược qua các lớp, mà còn đi ngược qua **các thời điểm trước đó**. Do đó, cái khó của BPTT không phải là backprop mới, mà là **đồ thị quá sâu theo thời gian**.

## Từ điển nhanh các khái niệm đi kèm

| Khái niệm                 | Ý nghĩa                                                               |
| ------------------------- | --------------------------------------------------------------------- |
| **Hidden state $h_t$**    | Bộ nhớ tạm của RNN ở thời điểm $t$                                    |
| **Shared weights**        | Cùng một trọng số dùng lặp lại ở mọi bước thời gian                   |
| **Chain rule**            | Quy tắc nối các đạo hàm qua chuỗi phụ thuộc                           |
| **Jacobian**              | Ma trận cho biết đầu ra vector thay đổi ra sao khi đầu vào vector đổi |
| **Long-range dependency** | Thông tin rất xa trong quá khứ vẫn cần cho dự đoán hiện tại           |
| **Truncation**            | Cắt bớt độ dài mà gradient được phép đi ngược                         |
| **Bias của gradient**     | Gradient xấp xỉ bị lệch khỏi gradient thật                            |
| **Variance cao**          | Gradient thay đổi rất mạnh giữa các lần ước lượng                     |

## Công thức cốt lõi

Với mô hình RNN đơn giản $h_t = W_{hx} x_t + W_{hh} h_{t-1}$, $o_t = W_{qh} h_t$:

$$\frac{\partial L}{\partial h_t} = \sum_{i=t}^{T} \left(W_{hh}^T\right)^{T-i} \cdot W_{qh}^T \cdot \frac{\partial L}{\partial o_i}$$

Công thức này nói rằng: loss ở tương lai quay ngược về $h_t$ qua rất nhiều lần nhân với $W_{hh}^T$. Chính chuỗi nhân đó là gốc rễ của vấn đề.

## Các ma trận $W$ trong RNN thực chất làm gì?

Với công thức cơ bản:

$$h_t = W_{hx} x_t + W_{hh} h_{t-1}, \quad o_t = W_{qh} h_t$$

ta có thể hiểu từng ma trận như sau:

- **$W_{hx}$**: chuyển dữ liệu mới $x_t$ vào không gian hidden. Nó quyết định input hiện tại được "dịch" thành tín hiệu nội bộ như thế nào.
- **$W_{hh}$**: chuyển hidden state cũ $h_{t-1}$ thành một phần của hidden state mới. Đây là ma trận mô tả **động lực học của trí nhớ**.
- **$W_{qh}$**: chuyển từ biểu diễn ẩn ra output. Nó không trực tiếp quyết định khả năng nhớ dài hạn; vai trò đó nằm chủ yếu ở $W_{hh}$.

> [!NOTE] Ký hiệu có thể khác giữa các sách
> Có tài liệu dùng $W_{hq}$, có tài liệu dùng $W_{qh}$. Ý nghĩa không đổi: đây là ma trận ánh xạ từ hidden state sang output.

> [!NOTE] Điểm cần nhớ
> Khi ta nói RNN nhớ hay quên, ta đang nói chủ yếu về cách $W_{hh}$ tác động lặp đi lặp lại lên hidden state qua nhiều bước thời gian.

## Cơ chế chi tiết: tại sao gradient biến mất hoặc phát nổ?

Nếu xem $W_{hh}$ qua các eigenvalue của nó:

- $|\lambda_{\max}| < 1$ → mỗi lần nhân làm tín hiệu nhỏ thêm → **vanishing gradient**
- $|\lambda_{\max}| > 1$ → mỗi lần nhân làm tín hiệu lớn thêm → **exploding gradient**

Nói cách khác, RNN phải "nhớ" bằng cách lặp đi lặp lại cùng một phép biến đổi. Nếu phép biến đổi đó co lại quá mạnh, ký ức xa sẽ mờ dần. Nếu nó khuếch đại quá mạnh, training sẽ mất ổn định.

## Eigenvalue là gì theo trực giác?

Với một ma trận vuông $W$, nếu tồn tại vector $v \neq 0$ sao cho:

$$Wv = \lambda v$$

thì $v$ là eigenvector và $\lambda$ là eigenvalue.

Điều đó có nghĩa: theo hướng đặc biệt $v$, ma trận $W$ gần như chỉ làm một việc là **co lại, giữ nguyên, hoặc phóng to** theo hệ số $\lambda$.

- $|\lambda| < 1$ : co lại
- $|\lambda| = 1$ : giữ độ lớn tương đối ổn định
- $|\lambda| > 1$ : phóng to

Trong RNN, vì hidden state bị nhân lặp lại bởi $W_{hh}$, nên sau $k$ bước, ảnh hưởng dọc theo một eigenvector sẽ tỷ lệ với $\lambda^k$. Đây là lý do eigenvalue đóng vai trò trung tâm trong việc phân tích khả năng nhớ dài hạn và độ ổn định của gradient.

## Các chiến lược thực tế

| Chiến lược                | Mô tả                                 | Ưu/Nhược                                |
| ------------------------- | ------------------------------------- | --------------------------------------- |
| **Full BPTT**             | Truyền ngược toàn bộ $T$ bước         | Chính xác nhưng đắt bộ nhớ và dễ bất ổn |
| **Truncated BPTT**        | Chỉ truyền ngược $\tau$ bước gần nhất | Dùng phổ biến nhất vì ổn định và rẻ hơn |
| **Randomized truncation** | Cắt ngẫu nhiên theo xác suất          | Kỳ vọng đúng nhưng nhiễu cao            |

## Kết nối với PyTorch

Trong code PyTorch, truncated BPTT thường được thực hiện qua `state.detach_()`.

Điểm rất quan trọng: `detach_()` **không xóa hidden state**, mà chỉ cắt đứt đồ thị đạo hàm nối về quá khứ. Tức là mô hình vẫn tiếp tục mang "trí nhớ số học", nhưng không còn truy trách nhiệm gradient quá xa nữa.

## Liên kết

- [[Recurrent Neural Network]] — kiến trúc mà BPTT áp dụng
- [[Gradient Clipping]] — xử lý exploding gradient (không xử lý vanishing)
- [[Gradient Descent]] — thuật toán tối ưu sử dụng gradient từ BPTT

---

> [!TODO] Mở rộng
>
> - Thêm so sánh BPTT vs RTRL (Real-Time Recurrent Learning)
> - Phân tích chi tiết hơn về Jacobian spectrum trong thực tế
> - Kết nối với gate mechanism của LSTM/GRU
