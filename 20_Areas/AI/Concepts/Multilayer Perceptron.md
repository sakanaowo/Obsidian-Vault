---
title: "Multilayer Perceptron"
aliases: [MLP, multi-layer perceptron, mạng đa tầng, feedforward neural network]
tags: [concept, deep-learning, neural-network, architecture]
created: 2026-03-24
---

# Multilayer Perceptron (MLP)

> [!NOTE] ELI5
> Softmax regression = 1 tầng tuyến tính → chỉ vẽ được đường thẳng. **MLP** thêm "tầng ẩn" ở giữa, kèm hàm kích hoạt phi tuyến. Mỗi tầng ẩn "uốn" không gian dữ liệu → cuối cùng đường phân chia có thể cong, phức tạp tùy ý. Đây là kiến trúc neural network **cơ bản nhất** của Deep Learning.

## 1. Kiến trúc

```
Input (d) → Hidden Layer (h) → Output (q)
   X    →   H = σ(XW¹ + b¹)  →  O = HW² + b²
```

$$\begin{aligned}
\mathbf{H} &= \sigma(\mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)}) \\
\mathbf{O} &= \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}
\end{aligned}$$

| Ký hiệu | Shape | Ý nghĩa |
| --- | --- | --- |
| $\mathbf{X}$ | $(n, d)$ | Input: $n$ mẫu, $d$ features |
| $\mathbf{W}^{(1)}$ | $(d, h)$ | Trọng số tầng ẩn |
| $\mathbf{b}^{(1)}$ | $(1, h)$ | Bias tầng ẩn |
| $\sigma$ | — | [[Activation Function]] (ReLU, Sigmoid...) |
| $\mathbf{H}$ | $(n, h)$ | Hidden representations |
| $\mathbf{W}^{(2)}$ | $(h, q)$ | Trọng số output |
| $\mathbf{O}$ | $(n, q)$ | Output (logits) |

## 2. Tại sao cần Hidden Layer?

Mô hình tuyến tính giả định **monotonic**: tăng feature → luôn tăng (hoặc luôn giảm) output. Nhiều bài toán vi phạm:
- **Nhiệt cơ thể**: 37°C bình thường. Cao hơn hoặc thấp hơn đều nguy hiểm.
- **Nhận dạng ảnh**: tăng pixel (13,17) không "luôn" tăng P(chó).

Hidden layer + activation = **học được biểu diễn phi tuyến** tự động.

## 3. Universal Approximation Theorem

> *Với đủ nhiều hidden units, MLP 1 tầng ẩn có thể xấp xỉ BẤT KỲ hàm liên tục nào.*

Nhưng:
- "Đủ nhiều" có thể là **cực kỳ nhiều** units
- **Tìm được** trọng số đúng mới là khó (optimization)
- Thực tế: **sâu hơn** (nhiều tầng) hiệu quả hơn **rộng hơn** (nhiều units)

## 4. Đếm số tầng

> [!WARNING] Input layer không tính
> MLP có 1 input layer + 1 hidden layer + 1 output layer → gọi là **2-layer MLP** (chỉ đếm tầng có tham số).

## TODO

- [ ] Thêm ví dụ decision boundary với MLP vs linear
- [ ] Liên kết với CNN, Transformer
