---
title: "Linear Algebra for Deep Learning"
aliases: [linear-algebra, linalg, "đại số tuyến tính"]
tags: [concept, deep-learning, d2l, math, fundamentals]
created: 2026-03-05
session: "D2L Tuần 1, Buổi 3 — Linear Algebra"
source: "D2L Chapter Preliminaries - sec_linear-algebra"
related:
  - "[[Tensor Operations]]"
  - "[[Calculus for Deep Learning]]"
  - "[[Matrix Multiplication in Neural Networks]]"
---

# Linear Algebra for Deep Learning

> [!NOTE] ELI5
> Linear algebra là "ngôn ngữ toán học" mà Deep Learning dùng để nói chuyện. Một ảnh = ma trận số. Weights của mạng nơ-ron = ma trận. Forward pass = nhân ma trận. Gradient = vector đạo hàm. Nếu Tensor Operations là "vật chứa dữ liệu", thì **Linear Algebra là các phép tính trên vật chứa đó** — đặc biệt là các phép nhân ma trận làm biến đổi dữ liệu qua từng lớp của mạng.

## 1. Hệ thống phân cấp đối tượng toán học

| Object     | Ký hiệu                                  | Code (PyTorch)                    | Rank | Ví dụ DL                  |
| ---------- | ---------------------------------------- | --------------------------------- | ---- | ------------------------- |
| **Scalar** | $x \in \mathbb{R}$                       | `torch.tensor(3.0)`               | 0D   | Learning rate, loss value |
| **Vector** | $\mathbf{x} \in \mathbb{R}^n$            | `torch.arange(3)`                 | 1D   | Bias, embedding vector    |
| **Matrix** | $\mathbf{A} \in \mathbb{R}^{m \times n}$ | `A.reshape(m, n)`                 | 2D   | Weight matrix của 1 layer |
| **Tensor** | $\mathsf{X}$                             | `torch.arange(24).reshape(2,3,4)` | nD   | Batch ảnh, activations    |

---

## 2. Phép toán Reduction

> [!NOTE] ELI5
> "Reduction" nghĩa là **nén lại**. Bạn có ma trận 3×4 = 12 số. Cộng tất cả → 1 số (reduce hết). Cộng theo hàng → vector 4 số (reduce axis 0). Cộng theo cột → vector 3 số (reduce axis 1).

### 2.1 Sum & Mean

```python
A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
# [[0, 1, 2],
#  [3, 4, 5]]

A.sum()            # 15.0 — tổng tất cả
A.sum(axis=0)      # tensor([3., 5., 7.]) — cộng theo hàng, kết quả shape (3,)
A.sum(axis=1)      # tensor([ 3., 12.]) — cộng theo cột, kết quả shape (2,)

A.mean()           # 2.5
A.mean(axis=0)     # tensor([1.5, 2.5, 3.5])
```

### 2.2 keepdims — Giữ số chiều

```python
# Không keepdims: shape (2,3) → sum(axis=1) → shape (2,)
A.sum(axis=1)            # tensor([ 3., 12.])

# Với keepdims: shape (2,3) → sum(axis=1) → shape (2,1)  ← GIỮ NGUYÊN SỐ CHIỀU
sum_A = A.sum(axis=1, keepdims=True)   # tensor([[ 3.], [12.]])

# Dùng cho broadcasting normalization:
A / sum_A   # Mỗi hàng chia cho tổng của nó → row-wise softmax thô
```

> [!TIP] Khi nào cần `keepdims=True`?
> Khi bạn muốn **chia/trừ/cộng** tensor với kết quả sum/mean của nó. Nếu không keepdims, broadcasting sẽ fail vì shape không tương thích. Đây là pattern cực phổ biến trong normalization.

### 2.3 Cumulative Sum

```python
A.cumsum(axis=0)
# [[0, 1,  2],
#  [3, 5,  7]]  ← hàng 1 = hàng 0 + hàng 1
```

---

## 3. Dot Product (Tích vô hướng)

> [!NOTE] ELI5
> Dot product $\mathbf{x}^\top \mathbf{y}$ giống như bạn "so sánh mức độ giống nhau" giữa 2 vector. Nhân từng cặp phần tử tương ứng rồi cộng lại. Nếu 2 vector cùng hướng → dot product lớn. Vuông góc → dot product = 0. Ngược chiều → dot product âm.

$$\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{d} x_i y_i$$

```python
x = torch.tensor([1.0, 2, 3])
y = torch.ones(3)

torch.dot(x, y)       # = 1*1 + 2*1 + 3*1 = 6.0
torch.sum(x * y)      # Tương đương: elementwise multiply rồi sum
```

### Ứng dụng trong DL:

- **Weighted sum:** $\mathbf{x}^\top \mathbf{w}$ khi $w_i \geq 0$ và $\sum w_i = 1$ → weighted average
- **Cosine similarity:** Sau khi normalize về unit length, dot product = $\cos(\theta)$
- **Attention score:** Query-Key dot product trong Transformer!

---

## 4. Matrix–Vector Product

> [!NOTE] ELI5
> Nhân ma trận $\mathbf{A}$ với vector $\mathbf{x}$ giống như **áp dụng $m$ phép đo khác nhau** lên cùng 1 vector $\mathbf{x}$. Kết quả là $m$ số, mỗi số là "mức độ khớp" của $\mathbf{x}$ với 1 trong $m$ hướng của $\mathbf{A}$.

$$\mathbf{A}\mathbf{x} = \begin{bmatrix} \mathbf{a}^\top_1 \mathbf{x} \\ \mathbf{a}^\top_2 \mathbf{x} \\ \vdots \\ \mathbf{a}^\top_m \mathbf{x} \end{bmatrix}$$

Yêu cầu: số cột $\mathbf{A}$ = số phần tử $\mathbf{x}$: $\mathbf{A} \in \mathbb{R}^{m \times n},\ \mathbf{x} \in \mathbb{R}^n \Rightarrow \mathbf{A}\mathbf{x} \in \mathbb{R}^m$

```python
A = torch.arange(6, dtype=torch.float32).reshape(2, 3)  # (2, 3)
x = torch.arange(3, dtype=torch.float32)                 # (3,)

torch.mv(A, x)   # (2,) — matrix-vector multiply
A @ x            # Cú pháp Python tiện hơn (cả mv và mm đều dùng được)
```

### Tại sao quan trọng trong DL?

**Mỗi layer trong neural network là một matrix-vector product:**

$$\text{output} = \mathbf{W} \cdot \text{input} + \mathbf{b}$$

Ở đây $\mathbf{W}$ là weight matrix, $\text{input}$ là activation vector từ layer trước.

---

## 5. Matrix–Matrix Multiplication

> [!NOTE] ELI5
> Nếu matrix-vector product là "1 phép biến đổi áp dụng lên 1 vector", thì matrix-matrix multiplication là **"1 phép biến đổi áp dụng lên nhiều vectors cùng lúc"** (mỗi cột của $\mathbf{B}$ là 1 vector). Hoặc: kết hợp 2 phép biến đổi liên tiếp thành 1.

$$\mathbf{C} = \mathbf{AB}, \quad c_{ij} = \mathbf{a}^\top_i \mathbf{b}_j$$

Yêu cầu: $\mathbf{A} \in \mathbb{R}^{n \times k}$, $\mathbf{B} \in \mathbb{R}^{k \times m}$ → $\mathbf{C} \in \mathbb{R}^{n \times m}$

```python
A = torch.arange(6, dtype=torch.float32).reshape(2, 3)  # (2, 3)
B = torch.ones(3, 4)                                     # (3, 4)

torch.mm(A, B)    # (2, 4) — matrix-matrix multiply
A @ B             # Cú pháp @  (tổng quát hơn, dùng cho cả batched matmul)
```

### Độ phức tạp tính toán

Tính $\mathbf{AB}$ với $\mathbf{A} \in \mathbb{R}^{n \times k}$, $\mathbf{B} \in \mathbb{R}^{k \times m}$:

$$\text{FLOPs} = O(n \cdot k \cdot m) \quad \text{— cubic complexity!}$$

> [!WARNING] Thứ tự nhân ma trận quan trọng!
> Với $\mathbf{A} \in \mathbb{R}^{2^{10} \times 2^{16}}$, $\mathbf{B} \in \mathbb{R}^{2^{16} \times 2^5}$, $\mathbf{C} \in \mathbb{R}^{2^5 \times 2^{14}}$:
>
> - $(\mathbf{AB})\mathbf{C}$: FLOPs = $2^{10} \cdot 2^{16} \cdot 2^5 + 2^{10} \cdot 2^5 \cdot 2^{14}$ — nhỏ hơn
> - $\mathbf{A}(\mathbf{BC})$: FLOPs = $2^{16} \cdot 2^5 \cdot 2^{14} + 2^{10} \cdot 2^{16} \cdot 2^{14}$ — lớn hơn nhiều!
>
> **Luôn nghĩ về thứ tự tối ưu khi nhân nhiều ma trận!**

---

## 6. Norms — Đo "độ lớn"

> [!NOTE] ELI5
> **Norm** là cách đo "vector dài bao nhiêu" hay "ma trận lớn bao nhiêu". Có nhiều cách đo khác nhau, giống như đo khoảng cách theo đường chim bay (L2) hay đo theo đường Manhattan đi qua các ô vuông (L1).

### Định nghĩa Norm

Một norm $\|\cdot\|$ phải thỏa 3 tính chất:

1. **Scaling:** $\|\alpha \mathbf{x}\| = |\alpha| \|\mathbf{x}\|$
2. **Triangle inequality:** $\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|$
3. **Non-negativity:** $\|\mathbf{x}\| \geq 0$, và $\|\mathbf{x}\| = 0 \Leftrightarrow \mathbf{x} = \mathbf{0}$

### 6.1 Vector Norms

#### L2 Norm (Euclidean norm) — Đường chim bay

$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2}$$

```python
u = torch.tensor([3.0, -4.0])
torch.norm(u)   # sqrt(9 + 16) = sqrt(25) = 5.0
```

#### L1 Norm (Manhattan distance) — Đường phố New York

$$\|\mathbf{x}\|_1 = \sum_{i=1}^n |x_i|$$

```python
torch.abs(u).sum()   # |3| + |-4| = 7.0
```

#### Lp Norm tổng quát

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n |x_i|^p\right)^{1/p}$$

### 6.2 Frobenius Norm (cho Matrix)

> [!NOTE] ELI5
> Frobenius norm giống như trải phẳng ma trận thành vector rồi tính L2 norm của vector đó.

$$\|\mathbf{X}\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}$$

```python
torch.norm(torch.ones((4, 9)))   # sqrt(4*9) = sqrt(36) = 6.0
```

### 6.3 So sánh L1 vs L2 trong ML

| Property           | L1 Norm             | L2 Norm                        |
| ------------------ | ------------------- | ------------------------------ |
| **Tên**            | Manhattan           | Euclidean                      |
| **Outliers**       | Robust hơn          | Nhạy cảm hơn                   |
| **Sparsity**       | Khuyến khích sparse | Không sparse                   |
| **Dùng trong**     | Lasso regression    | Ridge regression, weight decay |
| **Gradient tại 0** | Không xác định      | = 0                            |

### 🚀 Ứng dụng Norm trong DL

1. **Loss function:** MSE loss = $\|\hat{\mathbf{y}} - \mathbf{y}\|_2^2$ (bình phương L2)
2. **Regularization (L2/Weight decay):** Thêm $\lambda\|\mathbf{W}\|_2^2$ vào loss để tránh overfitting
3. **Gradient clipping:** Cắt gradient khi $\|\nabla\|_2 > \text{threshold}$ để ổn định training
4. **Similarity measure:** Cosine similarity = $\frac{\mathbf{x}^\top \mathbf{y}}{\|\mathbf{x}\|_2 \|\mathbf{y}\|_2}$

---

## 7. Transpose — Hoán vị

$$\mathbf{B} = \mathbf{A}^\top \Rightarrow b_{ij} = a_{ji}$$

```python
A = torch.arange(6).reshape(3, 2)   # shape (3, 2)
A.T                                   # shape (2, 3)
```

**Symmetric matrix:** $\mathbf{A} = \mathbf{A}^\top$ — rất hay xuất hiện trong covariance matrices, Gram matrices.

---

## 8. Tổng kết — Vai trò trong Deep Learning

```
Input x (vector)
    ↓  @ W₁  (matrix-vector product)
Hidden₁ (vector)
    ↓  @ W₂
Hidden₂ (vector)
    ↓  @ Wₙ
Output (vector)
```

| Phép toán           | Ứng dụng trong DL                                              |
| ------------------- | -------------------------------------------------------------- |
| **Matrix multiply** | Forward pass: $\mathbf{h} = \mathbf{W}\mathbf{x} + \mathbf{b}$ |
| **Transpose**       | Backward pass, weight tying                                    |
| **Dot product**     | Attention scores, cosine similarity                            |
| **L2 Norm**         | Loss function, regularization, gradient clipping               |
| **Reduction (sum)** | Global average pooling, loss aggregation                       |
| **keepdims**        | Broadcasting-compatible normalization                          |

---

## Exercises (từ D2L)

1. Chứng minh $(\mathbf{A}^\top)^\top = \mathbf{A}$
2. Chứng minh $\mathbf{A}^\top + \mathbf{B}^\top = (\mathbf{A} + \mathbf{B})^\top$
3. Với square matrix $\mathbf{A}$, $\mathbf{A} + \mathbf{A}^\top$ có luôn symmetric không? Tại sao?
4. Cho tensor `X` shape `(2, 3, 4)`, `len(X)` = ? (không chạy code, đoán trước)
5. Run `A / A.sum(axis=1)` → kết quả ra sao? Có đúng không? (Gợi ý: broadcasting failure)
6. Cho 3 matrices lớn $\mathbf{A}, \mathbf{B}, \mathbf{C}$, tính $\mathbf{ABC}$ theo thứ tự nào hiệu quả hơn?

---

> [!TODO] Mở rộng
>
> - [[Eigenvalues and Eigenvectors]] — cần cho PCA, SVD
> - [[Matrix Decomposition]] — SVD, Cholesky, QR
> - [[Gradient as a Vector]] — liên kết với [[Calculus for Deep Learning]]
> - Tìm hiểu `torch.einsum` — cách tổng quát để viết mọi phép toán tensor
