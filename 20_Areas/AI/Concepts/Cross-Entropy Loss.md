---
title: "Cross-Entropy Loss"
aliases: [cross entropy, categorical cross-entropy, log loss, hàm mất mát cross-entropy]
tags: [concept, machine-learning, loss-function, classification, information-theory]
created: 2026-03-19
---

# Cross-Entropy Loss

> [!NOTE] ELI5
> Bạn đoán ngày mai trời mưa 10%, nhưng thực tế trời mưa thật. Cross-entropy đo **bạn bất ngờ bao nhiêu**. Đoán 10% rồi mưa thật → **rất bất ngờ** → loss cao. Đoán 90% rồi mưa thật → **ít bất ngờ** → loss thấp. Mục tiêu training = giảm sự bất ngờ xuống thấp nhất.

## 1. Bản chất — Từ MLE đến Cross-Entropy

### Bước 1: Maximum Likelihood Estimation (MLE)

Ta muốn tìm mô hình $f$ sao cho xác suất quan sát được dữ liệu $(\mathbf{X}, \mathbf{Y})$ là **lớn nhất**:

$$P(\mathbf{Y} \mid \mathbf{X}) = \prod_{i=1}^n P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)})$$

### Bước 2: Negative Log-Likelihood (NLL)

Nhân nhiều xác suất < 1 → số rất nhỏ → khó tính. Lấy $-\log$:

$$-\log P(\mathbf{Y} \mid \mathbf{X}) = \sum_{i=1}^n -\log P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)})$$

Minimize NLL = Maximize Likelihood.

### Bước 3: Cross-Entropy Loss

Với nhãn [[One-Hot Encoding|one-hot]] $\mathbf{y}$ và dự đoán $\hat{\mathbf{y}}$ (từ [[Softmax Function]]):

$$l(\mathbf{y}, \hat{\mathbf{y}}) = -\sum_{j=1}^q y_j \log \hat{y}_j$$

Vì $\mathbf{y}$ là one-hot (chỉ 1 phần tử bằng 1, còn lại = 0), công thức rút gọn thành:

$$l(\mathbf{y}, \hat{\mathbf{y}}) = -\log \hat{y}_c$$

trong đó $c$ là class đúng. → **Chỉ quan tâm xác suất mà mô hình gán cho class đúng.**

## 2. Ý nghĩa trực quan

| Dự đoán cho class đúng ($\hat{y}_c$) | Loss $= -\log(\hat{y}_c)$ | Ý nghĩa |
| --- | --- | --- |
| 0.99 | 0.01 | Rất tự tin, đúng → loss gần 0 |
| 0.9 | 0.105 | Khá tự tin → loss thấp |
| 0.5 | 0.693 | Phân vân → loss trung bình |
| 0.1 | 2.303 | Khá sai → loss cao |
| 0.01 | 4.605 | Rất sai → loss rất cao |
| → 0 | → $\infty$ | Hoàn toàn sai → loss vô cực |

> [!IMPORTANT] Cross-entropy phạt **rất nặng** khi mô hình tự tin nhưng sai
> Nếu $\hat{y}_c \to 0$ (mô hình gần như chắc chắn đây **không phải** class đúng), loss → $\infty$. Đây là cơ chế "phạt" giúp mô hình học nhanh khi mắc lỗi nghiêm trọng.

## 3. Gradient — Tại sao cross-entropy kết hợp tuyệt vời với softmax?

Đạo hàm của cross-entropy loss theo logit $o_j$:

$$\frac{\partial l}{\partial o_j} = \hat{y}_j - y_j = \text{softmax}(\mathbf{o})_j - y_j$$

Gradient = **dự đoán − thực tế** — cực kỳ đơn giản và đẹp. Tương tự hệt gradient của MSE trong linear regression ($\hat{y} - y$). Đây không phải trùng hợp — cả hai đều thuộc **exponential family**.

## 4. Ý nghĩa Information Theory

### Entropy — thước đo "bất ngờ trung bình"

$$H[P] = -\sum_j P(j) \log P(j)$$

- Data dễ đoán (luôn cùng kết quả) → entropy thấp
- Data khó đoán (ngẫu nhiên hoàn toàn) → entropy cao

### Cross-Entropy — "bất ngờ khi dùng mô hình sai"

$$H(P, Q) = -\sum_j P(j) \log Q(j)$$

- $P$ = phân phối thật (ground truth)
- $Q$ = phân phối dự đoán (model output)
- Cross-entropy **luôn ≥ Entropy**: $H(P, Q) \geq H(P)$
- Bằng nhau khi $P = Q$ → mô hình hoàn hảo

## 5. Ứng dụng trong DL

- **Mọi classification models**: softmax regression, CNNs, Transformers
- **Language modeling**: dự đoán token tiếp theo (GPT, BERT)
- **Knowledge distillation**: soft labels = phân phối xác suất từ teacher model
- PyTorch: `nn.CrossEntropyLoss()` (đã gộp softmax + NLL bên trong)

> [!WARNING] Lưu ý PyTorch
> `nn.CrossEntropyLoss()` nhận **logits** (chưa qua softmax) làm input, không phải probabilities. Nếu bạn tự apply softmax trước rồi đưa vào `CrossEntropyLoss` → sai.

## TODO

- [ ] Thêm binary cross-entropy (BCELoss)
- [ ] So sánh cross-entropy vs focal loss
- [ ] Liên kết với KL divergence
