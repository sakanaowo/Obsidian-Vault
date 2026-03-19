

> [!NOTE] 
> Mọi bài toán DL đều quy về bài toán tối ưu ~ tìm nghiệm tối ưu nhất ~ ở đây là minimize loss

# 1. Tại sao giải tích quan trọng

Deep learning = tìm parameters $\theta$ để minimize loss $L(\theta)$.

Cách làm: truy ngược theo gradient bằng backpropagation
$$\theta \leftarrow \theta - \eta \cdot \nabla_\theta L(\theta)$$
- Gradient: vector các đạo hàm của các tham số (đạo hàm từng phần)

> [!NOTE] Gradient cho biết
> - Hướng tăng nhanh nhất của đạo hàm
> - Độ lớn của độ dốc (tốc độ biến đổi hàm ) ~ độ lớn của đạo hàm 

# 2. Đạo hàm - Derivative 


> [!NOTE] Định nghĩa
> Đạo hàm  = **tốc độ thay đổi tức thời** của f tại x. Nếu bạn tăng x thêm một chút , hàm f thay đổi khoảng $f'(x)*h$. Trực quan hơn:  chính là **độ dốc** (slope) của đường tiếp tuyến với đồ thị  tại điểm x

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

## Skipping

# Gradient

$$\nabla_{\mathbf{x}} f(\mathbf{x}) = \begin{bmatrix} \partial_{x_1} f \\ \partial_{x_2} f \\ \vdots \\ \partial_{x_n} f \end{bmatrix} \in \mathbb{R}^n$$
**Gradient của các biểu thức ma trận phổ biến**

| Biểu thức                                      | Gradient theo $\mathbf{x}$                 |
| ---------------------------------------------- | ------------------------------------------ |
| $\mathbf{A}\mathbf{x}$                         | $\mathbf{A}^\top$                          |
| $\mathbf{x}^\top\mathbf{A}$                    | $\mathbf{A}$                               |
| $\mathbf{x}^\top\mathbf{A}\mathbf{x}$          | $(\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$ |
| $\|\mathbf{x}\|^2 = \mathbf{x}^\top\mathbf{x}$ | $2\mathbf{x}$                              |
| $\|\mathbf{X}\|_F^2$                           | $2\mathbf{X}$                              |


# 4. Quy tắc chuỗi - Chain Rule

> [!NOTE]
> Neural Network là hàm lồng nhau nhiều tầng. VD: `loss(softmax(linear(relu(linear(x)))))`
> Chain rule cho phép tính gradient của mạng bằng cách nhân các gradient lại với nhau. Phần chính của **backpropagation**

## Hàm 1 biến
Nếu $y=f(g(x))$, $u=g(x)$
$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$
**Ví dụ:** $y = (3x^2 - 4x)^5$

- $u = 3x^2 - 4x$, $y = u^5$
- $\frac{dy}{du} = 5u^4$, $\frac{du}{dx} = 6x - 4$
- $\frac{dy}{dx} = 5(3x^2 - 4x)^4 \cdot (6x - 4)$

## Hàm nhiều biến
Nếu $y = f(\mathbf{u})$ và $\mathbf{u} = g(\mathbf{x})$ (vector):
$$\frac{\partial y}{\partial x_i} = \sum_{j=1}^m \frac{\partial y}{\partial u_j} \cdot \frac{\partial u_j}{\partial x_i}$$
Dạng ma trận:
$$\nabla_{\mathbf{x}} y = \mathbf{A} \cdot \nabla_{\mathbf{u}} y$$
Với $\mathbf{A} = \frac{\partial u_j}{\partial x_i}$ ~ **Jacobian matrix**
```lua 
Forward pass:  x → u₁ → u₂ → ... → y = L (loss)
                ↑      ↑      ↑
Backward:   ∂L/∂x ← ∂L/∂u₁ ← ∂L/∂u₂ ← ... ← ∂L/∂y = 1
```

Chain rule cho phép truyền gradient từ output ngược về input, **nhân qua từng layer**. Đây chính là lý do linear algebra quan trọng: mỗi bước backward = một **vector-matrix product**
