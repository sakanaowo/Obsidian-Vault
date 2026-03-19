---
title: "Tensor Operations"
aliases: [tensor, tensors, ndarray, "data manipulation"]
tags: [concept, deep-learning, d2l, fundamentals, pytorch]
created: 2026-03-04
session: "D2L Tuần 1, Buổi 2 — Tensor Operations (Data Manipulation)"
source: "D2L Chapter Preliminaries - sec_ndarray"
related:
  - "[[Broadcasting]]"
  - "[[Automatic Differentiation]]"
  - "[[Linear Algebra]]"
---

# Tensor Operations

> [!NOTE] ELI5
> Hãy tưởng tượng bạn có một **hộp số**. Một con số đơn lẻ (scalar) giống như 1 viên bi. Xếp nhiều viên bi thành 1 hàng → đó là **vector**. Xếp nhiều hàng thành 1 bảng → đó là **matrix**. Chồng nhiều bảng lên nhau → đó là **tensor 3D**. **Tensor** chỉ là cách gọi tổng quát cho "hộp chứa số" ở bất kỳ số chiều nào. Deep learning gần như làm MỌI THỨ bằng tensor.

## 1. Tensor là gì?

**Tensor** là cấu trúc dữ liệu cốt lõi trong mọi framework deep learning. Về bản chất, tensor là một **mảng đa chiều (multidimensional array)** chứa các giá trị số.

| Số chiều (rank/order) | Tên gọi       | Ví dụ                         |
| --------------------- | ------------- | ----------------------------- |
| 0                     | **Scalar**    | Nhiệt độ: `42.0`              |
| 1                     | **Vector**    | Pixel 1 hàng: `[255, 128, 0]` |
| 2                     | **Matrix**    | Ảnh grayscale: `28×28`        |
| 3                     | **3D Tensor** | Ảnh RGB: `3×28×28`            |
| 4                     | **4D Tensor** | Batch ảnh: `32×3×28×28`       |

### Tại sao tensor quan trọng?

1. **GPU acceleration:** Tensor hỗ trợ tính toán trên GPU, nhanh hơn NumPy (chỉ CPU) hàng trăm lần
2. **Automatic differentiation:** Tensor tự động tính gradient — nền tảng của backpropagation
3. **Unified interface:** Mọi dữ liệu (ảnh, text, audio) đều được biểu diễn dưới dạng tensor

### Framework mapping:

| Framework      | Tên class | Import                         |
| -------------- | --------- | ------------------------------ |
| **PyTorch**    | `Tensor`  | `import torch`                 |
| **TensorFlow** | `Tensor`  | `import tensorflow as tf`      |
| **JAX**        | `Array`   | `from jax import numpy as jnp` |
| **NumPy**      | `ndarray` | `import numpy as np`           |

---

## 2. Khởi tạo Tensor

### 2.1 Tạo tensor cơ bản

```python
import torch

# Tạo vector tuần tự
x = torch.arange(12, dtype=torch.float32)
# tensor([ 0.,  1.,  2., ..., 11.])

# Số phần tử
x.numel()   # 12

# Shape
x.shape     # torch.Size([12])
```

### 2.2 Reshape — Thay đổi hình dạng

> [!NOTE] ELI5
> Reshape giống như lấy 12 viên bi xếp thành hàng dài, rồi xếp lại thành bảng 3 hàng × 4 cột. Số viên bi không đổi, chỉ cách sắp xếp thay đổi.

```python
X = x.reshape(3, 4)
# tensor([[ 0.,  1.,  2.,  3.],
#          [ 4.,  5.,  6.,  7.],
#          [ 8.,  9., 10., 11.]])

# Dùng -1 để tự suy luận
X = x.reshape(-1, 4)  # Tương đương reshape(3, 4)
X = x.reshape(3, -1)  # Tương đương reshape(3, 4)
```

**Quy tắc:** Tích các chiều phải bằng tổng số phần tử. Nếu $n$ phần tử và shape $(h, w)$, thì $h \times w = n$.

### 2.3 Tensor đặc biệt

```python
# Toàn số 0
torch.zeros((2, 3, 4))

# Toàn số 1
torch.ones((2, 3, 4))

# Phân phối chuẩn N(0,1)
torch.randn(3, 4)

# Từ Python list
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

---

## 3. Indexing & Slicing

> [!NOTE] ELI5
> Indexing giống như xem hàng thứ mấy trong bảng, slicing là cắt ra một phần. `X[1:3]` nghĩa là "lấy hàng số 1 và 2" (không lấy 3).

```python
X = torch.arange(12).reshape(3, 4)

# Hàng cuối cùng
X[-1]           # tensor([ 8,  9, 10, 11])

# Hàng 1 và 2 (index 1:3)
X[1:3]          # 2 hàng

# Gán giá trị cụ thể
X[1, 2] = 17    # Phần tử hàng 1, cột 2

# Gán nhiều phần tử
X[:2, :] = 12   # 2 hàng đầu toàn bộ = 12
```

**Lưu ý quan trọng với JAX:** JAX arrays là **immutable** — không thể thay đổi trực tiếp. Phải dùng `X.at[1, 2].set(17)` — tạo array mới.

---

## 4. Elementwise Operations

> [!NOTE] ELI5
> "Elementwise" nghĩa là **từng phần tử một**. Nếu bạn có 2 hàng bi giống nhau, cộng elementwise = lấy viên bi vị trí 1 của hàng A + viên bi vị trí 1 của hàng B, rồi vị trí 2, vị trí 3, v.v.

Ký hiệu toán học: Nếu $f: \mathbb{R} \rightarrow \mathbb{R}$ là phép toán unary (1 đầu vào), ta áp dụng nó lên **từng phần tử** của tensor.

### 4.1 Unary operations

```python
torch.exp(x)    # e^x cho từng phần tử
torch.sqrt(x)   # căn bậc 2
torch.abs(x)    # trị tuyệt đối
```

### 4.2 Binary operations

Cho $\mathbf{u}, \mathbf{v}$ cùng shape, và $f: \mathbb{R}, \mathbb{R} \rightarrow \mathbb{R}$:

$$\mathbf{c} = F(\mathbf{u}, \mathbf{v}) \quad \text{where} \quad c_i = f(u_i, v_i) \quad \forall i$$

```python
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])

x + y    # tensor([ 3.,  4.,  6., 10.])
x - y    # tensor([-1.,  0.,  2.,  6.])
x * y    # tensor([ 2.,  4.,  8., 16.])  — elementwise, KHÔNG phải dot product!
x / y    # tensor([0.5, 1.0, 2.0, 4.0])
x ** y   # tensor([ 1.,  4., 16., 64.])
```

### 4.3 Concatenation & Comparison

```python
# Nối tensor theo axis
torch.cat((X, Y), dim=0)   # Nối dọc (thêm hàng)
torch.cat((X, Y), dim=1)   # Nối ngang (thêm cột)

# So sánh elementwise → Boolean tensor
X == Y   # True/False cho từng vị trí

# Tổng toàn bộ phần tử
X.sum()
```

---

## 5. Broadcasting

> [!NOTE] ELI5
> Broadcasting giống như photocopy. Nếu bạn có 1 cột số (3×1) và 1 hàng số (1×2), máy tính sẽ "photocopy" cột đó thành 2 cột và hàng đó thành 3 hàng, để cả hai có cùng kích thước 3×2, rồi mới cộng.

### Cơ chế hoạt động:

**Bước 1:** Mở rộng (copy) tensor dọc theo trục có length = 1
**Bước 2:** Thực hiện phép toán elementwise trên kết quả

```python
a = torch.arange(3).reshape((3, 1))   # Shape: (3, 1)
b = torch.arange(2).reshape((1, 2))   # Shape: (1, 2)

# a:        b:
# [[0],     [[0, 1]]
#  [1],
#  [2]]

# Sau broadcasting:
# a → [[0, 0],    b → [[0, 1],
#      [1, 1],         [0, 1],
#      [2, 2]]         [0, 1]]

a + b
# tensor([[0, 1],
#          [1, 2],
#          [2, 3]])
```

### Quy tắc Broadcasting:

1. So sánh shape **từ phải qua trái** (trailing dimensions)
2. Hai chiều tương thích nếu: bằng nhau **HOẶC** một trong hai = 1
3. Nếu số chiều khác nhau, thêm chiều = 1 vào bên trái tensor nhỏ hơn

| Shape A     | Shape B  | Kết quả     | Hợp lệ? |
| ----------- | -------- | ----------- | ------- |
| `(3, 1)`    | `(1, 2)` | `(3, 2)`    | ✅      |
| `(4, 3)`    | `(3,)`   | `(4, 3)`    | ✅      |
| `(2, 3, 4)` | `(3, 4)` | `(2, 3, 4)` | ✅      |
| `(3, 4)`    | `(2, 3)` | ❌          | ❌      |

> [!WARNING] Cạm bẫy Broadcasting
> Broadcasting có thể gây bug ngầm nếu bạn vô tình cộng 2 tensor có shape khác nhau mà vẫn broadcast được. Luôn kiểm tra `.shape` trước khi tính toán!

---

## 6. Memory Efficiency (Saving Memory)

> [!NOTE] ELI5
> Mỗi lần bạn viết `Y = Y + X`, Python tạo một hộp số MỚI rồi dán nhãn `Y` lên đó. Hộp cũ bị bỏ đi. Nếu làm điều này hàng triệu lần/giây (training), sẽ rất lãng phí bộ nhớ.

### Vấn đề:

```python
before = id(Y)
Y = Y + X
id(Y) == before   # False! → Y giờ trỏ tới vùng nhớ MỚI
```

### Giải pháp: In-place operations

```python
# Cách 1: Slice assignment
Z = torch.zeros_like(Y)
Z[:] = X + Y     # Gán vào Z đã tồn tại

# Cách 2: Toán tử +=
before = id(X)
X += Y
id(X) == before   # True! → Cùng vùng nhớ
```

### Tại sao quan trọng?

- **Training loop:** Cập nhật parameters hàng triệu lần → in-place tiết kiệm memory
- **Multiple references:** Nếu nhiều biến trỏ tới cùng tensor, in-place đảm bảo tất cả đều thấy giá trị mới
- **GPU memory:** VRAM có hạn (6-24GB), phải dùng hiệu quả

---

## 7. Conversion — Chuyển đổi kiểu dữ liệu

```python
# Tensor ↔ NumPy
A = X.numpy()           # Tensor → NumPy (CHIA SẺ bộ nhớ trong PyTorch!)
B = torch.from_numpy(A) # NumPy → Tensor

# Tensor size-1 → Python scalar
a = torch.tensor([3.5])
a.item()    # 3.5
float(a)    # 3.5
int(a)      # 3
```

> [!WARNING] PyTorch vs TensorFlow Memory Sharing
>
> - **PyTorch:** `tensor.numpy()` **chia sẻ** bộ nhớ → thay đổi NumPy array sẽ thay đổi tensor gốc!
> - **TensorFlow/JAX:** Conversion **không chia sẻ** bộ nhớ → an toàn hơn nhưng tốn memory hơn

---

## 8. Tổng kết & So sánh

| Khái niệm           | Mục đích                                   | Khi nào dùng                             |
| ------------------- | ------------------------------------------ | ---------------------------------------- |
| **Reshape**         | Thay đổi shape, giữ nguyên data            | Chuẩn bị input cho model (flatten, etc.) |
| **Elementwise ops** | Tính toán từng phần tử                     | Activation functions, loss computation   |
| **Broadcasting**    | Tự mở rộng tensor cho phép toán khác shape | Bias addition, normalization             |
| **In-place ops**    | Tiết kiệm memory                           | Training loop, parameter updates         |
| **Concatenation**   | Nối tensor lại với nhau                    | Batch processing, feature combination    |

---

## Exercises (từ D2L)

1. Thay `X == Y` bằng `X < Y` hoặc `X > Y` → kết quả ra sao?
2. Thử broadcasting với 3D tensors → kết quả có đúng kỳ vọng không?

---

> [!TODO] Mở rộng
>
> - Liên kết với [[Linear Algebra]] (matrix multiply khác elementwise multiply)
> - Liên kết với [[Automatic Differentiation]] (tensor + autograd = backbone của DL)
> - Tìm hiểu thêm: `torch.einsum`, `torch.view` vs `torch.reshape`
