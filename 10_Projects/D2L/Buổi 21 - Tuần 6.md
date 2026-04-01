---
title: "Buổi 21 - Tuần 6: Weight Decay & Dropout — Code chi tiết (D2L)"
tags: [d2l, weight-decay, dropout, regularization, implementation, study-note]
created: 2026-03-26
session: "D2L Tuần 6, Buổi 21 — Weight Decay & Dropout Implementation"
source:
  - "30_Resources/Books/Dive Into Deep Learning/d2l-en/chapter_multilayer-perceptrons/dropout.md"
related:
  - "[[Buổi 20 - Tuần 5]]"
  - "[[Buổi 19 - Tuần 5]]"
  - "[[Multilayer Perceptron]]"
  - "[[Activation Function]]"
---

# Buổi 21 — Weight Decay & Dropout: Hai vũ khí chống Overfitting

> [!NOTE] ELI5
> Buổi 20 bạn đã hiểu **tại sao** overfitting xảy ra và biết 3 kỹ thuật chống nó. Buổi 21 sẽ **code thật** 2 kỹ thuật quan trọng nhất:
>
> 1. **Weight Decay**: phạt trọng số lớn → mô hình không "nhạy" quá mức → mượt hơn
> 2. **Dropout**: random tắt neurons khi train → mạng không phụ thuộc vào neuron nào cụ thể → robust hơn
>
> Cả hai đều rất dễ dùng trong PyTorch — chỉ cần 1 tham số mỗi cái!

---

## 🎯 Mục tiêu buổi học

1. Hiểu **công thức** Weight Decay và ý nghĩa tham số $\lambda$
2. Hiểu **cơ chế** Dropout: vì sao random tắt neuron lại giúp chống overfit
3. **Code from scratch** cả hai kỹ thuật
4. **Code concise** với PyTorch API (`weight_decay`, `nn.Dropout`)
5. Biết khi nào dùng từng cái, và cách **kết hợp** cả hai

---

## Phần 1: Weight Decay — Chi tiết

> [!NOTE] ELI5
> Bạn viết bài luận. Nếu dùng từ ngữ "bay bổng, hoa mỹ" (trọng số lớn) → bài nghe hay nhưng mơ hồ, áp dụng vào đề mới sẽ sai. Nếu bị **trừ điểm mỗi khi dùng từ phức tạp** → bạn viết đơn giản, rõ ràng hơn. Đó là Weight Decay.

### 1.1 Công thức

Loss function **trước**:

$$\mathcal{L}(\mathbf{W}) = \frac{1}{n}\sum_{i=1}^{n}\ell(f(\mathbf{x}_i, \mathbf{W}), y_i)$$

Loss function **sau** khi thêm Weight Decay:

$$\mathcal{L}_{\text{WD}}(\mathbf{W}) = \underbrace{\frac{1}{n}\sum_{i=1}^{n}\ell(f(\mathbf{x}_i, \mathbf{W}), y_i)}_{\text{Loss gốc (data fit)}} + \underbrace{\frac{\lambda}{2}\|\mathbf{W}\|^2}_{\text{Penalty (phạt W lớn)}}$$

Giải thích từng ký hiệu:

| Ký hiệu            | Ý nghĩa                                            | Ví dụ                                                       |
| ------------------ | -------------------------------------------------- | ----------------------------------------------------------- |
| $\mathcal{L}$      | Loss function                                      | Cross-entropy                                               |
| $\|\mathbf{W}\|^2$ | Tổng bình phương **tất cả** trọng số: $\sum w_i^2$ | Nếu W = [3, -2, 1] thì $\|W\|^2 = 9 + 4 + 1 = 14$           |
| $\lambda$ (lambda) | **Hệ số phạt** — càng lớn → kìm W càng mạnh        | Thường $10^{-4}$ đến $10^{-2}$                              |
| $\frac{1}{2}$      | Hệ số cho công thức đạo hàm gọn hơn                | Đạo hàm $\frac{\lambda}{2}w^2$ = $\lambda w$ (không cần ×2) |

> [!question]- ❓ Tại sao gọi là "Weight **Decay**" (suy giảm trọng số)?
> Khi cập nhật SGD **có** weight decay:
>
> $$w \leftarrow w - \eta \cdot \frac{\partial \mathcal{L}}{\partial w} - \eta \lambda w$$
>
> Viết lại:
>
> $$w \leftarrow (1 - \eta\lambda) \cdot w - \eta \cdot \frac{\partial \mathcal{L}}{\partial w}$$
>
> Mỗi bước, trọng số bị **nhân với $(1 - \eta\lambda)$** — một số **nhỏ hơn 1**. Tức là W **tự giảm dần** (decay) mỗi bước! Ví dụ: với $\eta = 0.1, \lambda = 0.01$:
> - $(1 - 0.1 \times 0.01) = 0.999$
> - Mỗi bước, W giảm 0.1% → trọng số lớn bị "kéo về 0" mạnh hơn trọng số nhỏ

### 1.2 Ảnh hưởng của $\lambda$

![[assets/attachments/D2L/Buoi21/weight_decay_lambda.png]]
*Lambda nhỏ → overfit (đường loằng ngoằng). Lambda vừa → mượt. Lambda quá lớn → đường phẳng (underfitting).*

| $\lambda$ | Hành vi | Kết quả |
| --- | --- | --- |
| 0 | Không penalty → W tự do | Overfit nếu model quá lớn |
| $10^{-4}$ | Penalty nhẹ | Thường là default tốt |
| $10^{-2}$ | Penalty vừa | Hiệu quả cho model lớn |
| 1+ | Penalty nặng → W ≈ 0 | **Underfitting** — model quá yếu |

### 1.3 Code: From Scratch

```python
# SGD update VỚI Weight Decay — tự viết
lambda_wd = 0.001
lr = 0.1

for epoch in range(10):
    for X, y in train_loader:
        loss = F.cross_entropy(model(X), y)
        loss.backward()
        
        with torch.no_grad():
            for param in model.parameters():
                # SGD + Weight Decay:
                param -= lr * (param.grad + lambda_wd * param)
                #                 ↑ gradient     ↑ penalty gradient
                param.grad.zero_()
```

> [!question]- ❓ Tại sao penalty gradient = $\lambda \cdot w$?
> Đạo hàm của penalty term $\frac{\lambda}{2}\|\mathbf{W}\|^2$ theo $w_i$:
>
> $$\frac{\partial}{\partial w_i}\left(\frac{\lambda}{2}\sum w_j^2\right) = \lambda w_i$$
>
> Nên gradient tổng = **gradient gốc + $\lambda w$**. Khi cập nhật SGD, ta trừ đi gradient tổng → W lớn bị phạt nhiều hơn.

### 1.4 Code: Concise (1 tham số!)

```python
# PyTorch API — chỉ cần thêm weight_decay vào optimizer!
optimizer = torch.optim.SGD(
    model.parameters(), 
    lr=0.1, 
    weight_decay=1e-3   # ← lambda = 0.001
)

# Training loop: KHÔNG ĐỔI GÌ CẢ!
for epoch in range(10):
    for X, y in train_loader:
        loss = F.cross_entropy(model(X), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  # SGD + WD tự động
```

> [!TIP] Chỉ 1 tham số `weight_decay` — còn lại giữ nguyên. Đây là lý do nên dùng concise API!

---

## Phần 2: Dropout — Chi tiết

### 2.1 Ý tưởng

> [!NOTE] ELI5
> Bạn có nhóm 10 người làm bài nhóm. Nếu **1 người giỏi nhất** luôn làm hết → cả nhóm phụ thuộc vào 1 người. Ngày đó người đó nghỉ → cả nhóm tịt.
>
> Dropout = mỗi buổi học, **random cho vài người nghỉ**. Vì không biết ai sẽ nghỉ → **mọi người buộc phải học** → cả nhóm đều mạnh, không phụ thuộc vào 1 cá nhân.
>
> Trong neural network: "người" = neuron. Dropout buộc mạng **không phụ thuộc** vào neurons cụ thể → tránh **co-adaptation** (các neuron phụ thuộc lẫn nhau quá chặt).

### 2.2 Cơ chế hoạt động

![[assets/attachments/D2L/Buoi21/dropout_visualization.png]]
*Trái: tất cả neurons hoạt động. Phải: một số neurons bị tắt (X đỏ) → kết nối bị cắt.*

**Công thức toán học:**

Với **dropout probability** $p$, mỗi activation $h$ được thay bằng:

$$h' = \begin{cases} 0 & \text{với xác suất } p \quad \text{(tắt)} \\ \frac{h}{1-p} & \text{với xác suất } 1-p \quad \text{(giữ + scale)} \end{cases}$$

> [!question]- ❓ Tại sao chia cho $(1-p)$ khi giữ neuron? Sao không để nguyên?
> Để **kỳ vọng** (expected value) **không đổi**:
>
> $$E[h'] = p \times 0 + (1-p) \times \frac{h}{1-p} = h$$
>
> Nếu không chia: khi train (có dropout), tổng output trung bình nhỏ hơn khi test (không dropout) → mạng **mất cân bằng**!
>
> **Ví dụ cụ thể**: neuron output = 4.0, dropout p = 0.5
> - Không scale: train → trung bình = $0.5 \times 0 + 0.5 \times 4 = 2.0$. Test → 4.0. Sai!
> - Có scale: train → trung bình = $0.5 \times 0 + 0.5 \times \frac{4}{0.5} = 4.0$. Test → 4.0. Đúng!
>
> Kỹ thuật này gọi là **inverted dropout** — scale khi train để test **không cần chỉnh gì**.

![[assets/attachments/D2L/Buoi21/dropout_scaling.png]]
*Trái: giá trị gốc. Giữa: dropout=0.5 — 3 neurons bị tắt, phần còn lại ×2. Phải: test — tất cả bật, giá trị gốc.*

### 2.3 Code: From Scratch

```python
def dropout_layer(X, dropout):
    """Tự viết dropout layer"""
    assert 0 <= dropout <= 1
    
    # Trường hợp đặc biệt
    if dropout == 0: return X           # Không dropout → giữ nguyên
    if dropout == 1: return torch.zeros_like(X)  # Tắt hết
    
    # Tạo mask: mỗi phần tử = 1 (giữ) hoặc 0 (tắt)
    mask = (torch.rand(X.shape) > dropout).float()
    
    # Nhân với mask + chia cho (1-p) để giữ kỳ vọng
    return mask * X / (1.0 - dropout)
```

**Test thử:**

```python
X = torch.arange(16, dtype=torch.float32).reshape(2, 8)

print('dropout=0.0:', dropout_layer(X, 0.0))
# → Giữ nguyên tất cả

print('dropout=0.5:', dropout_layer(X, 0.5))
# → ~50% giá trị = 0, phần còn lại ×2

print('dropout=1.0:', dropout_layer(X, 1.0))
# → Tất cả = 0
```

> [!question]- ❓ `torch.rand(X.shape) > dropout` — dòng này hoạt động thế nào?
> Chia nhỏ:
> 1. `torch.rand(X.shape)` → tensor cùng shape, mỗi phần tử là số random trong [0, 1)
> 2. `> dropout` → so sánh từng phần tử với `dropout` → tensor True/False
> 3. `.float()` → True → 1.0, False → 0.0
>
> Ví dụ: dropout = 0.5, rand = [0.7, 0.2, 0.8, 0.3]
> - `> 0.5` → [True, False, True, False]
> - `.float()` → [1.0, 0.0, 1.0, 0.0] ← mask!
>
> Xác suất giữ = $P(\text{rand} > p) = 1-p$. Đúng!

### 2.4 MLP với Dropout — From Scratch

```python
class DropoutMLPScratch(nn.Module):
    def __init__(self, num_inputs=784, num_hiddens_1=256, 
                 num_hiddens_2=256, num_outputs=10,
                 dropout_1=0.5, dropout_2=0.5):
        super().__init__()
        self.dropout_1 = dropout_1
        self.dropout_2 = dropout_2
        self.lin1 = nn.LazyLinear(num_hiddens_1)
        self.lin2 = nn.LazyLinear(num_hiddens_2)
        self.lin3 = nn.LazyLinear(num_outputs)
        self.relu = nn.ReLU()
    
    def forward(self, X):
        X = X.reshape(X.shape[0], -1)
        
        # Hidden 1 + ReLU + Dropout
        H1 = self.relu(self.lin1(X))
        if self.training:                          # ← CHỈ dropout khi train!
            H1 = dropout_layer(H1, self.dropout_1)
        
        # Hidden 2 + ReLU + Dropout
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, self.dropout_2)
        
        return self.lin3(H2)
```

> [!CAUTION] Nhắc lại quan trọng
> `if self.training:` — CHỈ apply dropout **khi train**. Khi test (`model.eval()`), dropout **tắt** hoàn toàn → giữ tất cả neurons.
>
> Đây là lý do `model.train()` / `model.eval()` quan trọng (đã nhắc ở Buổi 19)!

### 2.5 Code: Concise (PyTorch API)

```python
model = nn.Sequential(
    nn.Flatten(),
    nn.LazyLinear(256), nn.ReLU(), nn.Dropout(0.5),   # ← Dropout sau ReLU!
    nn.LazyLinear(256), nn.ReLU(), nn.Dropout(0.5),   # ← Layer 2
    nn.LazyLinear(10)
)
```

> [!question]- ❓ Dropout đặt **trước** hay **sau** ReLU?
> **Sau ReLU** (convention chuẩn): `Linear → ReLU → Dropout`
>
> Lý do:
> - ReLU đã tắt các neurons có output **âm** (đặt = 0)
> - Dropout tiếp tục random tắt thêm → tổng cộng **nhiều neurons bị tắt hơn**
> - Nếu dropout trước ReLU: có thể dropout tắt neuron vốn đã bị ReLU tắt → lãng phí
>
> Tuy nhiên, sự khác biệt thực tế rất nhỏ. Cái quan trọng hơn là **dropout đặt SAU activation, TRƯỚC layer tiếp theo**.

> [!question]- ❓ `nn.Dropout(0.5)` vs `nn.Dropout(0.2)` — chọn bao nhiêu?
> | p | Ý nghĩa | Khi nào dùng |
> | --- | --- | --- |
> | 0.1-0.2 | Nhẹ, tắt ít neurons | Model nhỏ, data nhiều |
> | **0.5** | **Standard**, tắt 50% | Default cho hầu hết MLP |
> | 0.7-0.8 | Nặng, tắt nhiều | Model rất lớn, data ít |
>
> **Rule of thumb:**
> - Tầng ẩn đầu (gần input): dropout **nhẹ** hơn (0.2-0.3) — giữ thông tin input
> - Tầng ẩn sau (gần output): dropout **mạnh** hơn (0.5) — chống co-adaptation
> - **KHÔNG** dùng dropout ở output layer

---

## Phần 3: Full Training — Kết hợp Weight Decay + Dropout

```python
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

# ====== DATA ======
trans = transforms.Compose([transforms.ToTensor()])
train_data = torchvision.datasets.FashionMNIST(
    './data', train=True, transform=trans, download=True)
test_data = torchvision.datasets.FashionMNIST(
    './data', train=False, transform=trans, download=True)
train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=256, shuffle=False)

# ====== MODEL: MLP + Dropout ======
model = nn.Sequential(
    nn.Flatten(),
    nn.LazyLinear(256), nn.ReLU(), nn.Dropout(0.5),
    nn.LazyLinear(256), nn.ReLU(), nn.Dropout(0.5),
    nn.LazyLinear(10)
)

# ====== OPTIMIZER: SGD + Weight Decay ======
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.1, weight_decay=1e-3
)

# ====== TRAIN ======
for epoch in range(10):
    model.train()   # ← BẬT dropout
    for X, y in train_loader:
        loss = F.cross_entropy(model(X), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    model.eval()    # ← TẮT dropout
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in test_loader:
            correct += (model(X).argmax(1) == y).sum().item()
            total += y.shape[0]
    print(f"Epoch {epoch+1:2d} | Test Acc: {correct/total:.4f}")
```

> [!question]- ❓ So sánh code này với Buổi 19 (MLP không regularization) — khác ở đâu?
> Chỉ **2 thay đổi**:
>
> ```diff
>  # Model
>  model = nn.Sequential(
>      nn.Flatten(),
> -    nn.LazyLinear(256), nn.ReLU(),
> +    nn.LazyLinear(256), nn.ReLU(), nn.Dropout(0.5),    # +Dropout
> +    nn.LazyLinear(256), nn.ReLU(), nn.Dropout(0.5),    # +Layer 2
>      nn.LazyLinear(10)
>  )
>  
>  # Optimizer
> -optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
> +optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-3)
>  #                                                       ↑ +Weight Decay
> ```
>
> Training loop: **không đổi gì cả**! Thiết kế modular tuyệt vời.

### 3.1 Kết quả kỳ vọng

![[assets/attachments/D2L/Buoi21/regularization_comparison.png]]
*WD + Dropout cùng nhau → validation loss thấp nhất, ổn định nhất.*

| Cấu hình | Test Accuracy (ước lượng) | Overfitting? |
| --- | --- | --- |
| MLP không regularization | ~87-88% | Có (gap ~5-10%) |
| MLP + Weight Decay | ~87-88% | Ít hơn (gap ~3-5%) |
| MLP + Dropout | ~87-88% | Ít hơn (gap ~3-5%) |
| **MLP + WD + Dropout** | ~**88-89%** | **Ít nhất** (gap ~2-3%) |

> [!question]- ❓ Tại sao accuracy tương đương nhưng regularization vẫn quan trọng?
> Fashion-MNIST quá đơn giản → accuracy không khác nhiều. Nhưng:
>
> 1. **Generalization gap** giảm rõ rệt → model **tin cậy hơn** trên data mới
> 2. Với bài toán khó hơn (CIFAR-10, ImageNet), regularization tạo **sự khác biệt lớn** (5-10%+)
> 3. Training ổn định hơn: validation loss không "nhảy" nhiều
> 4. Model **robust** hơn: không sụp đổ khi data input bị nhiễu nhẹ

---

## Phần 4: Tổng hợp — So sánh 2 kỹ thuật

| | Weight Decay | Dropout |
| --- | --- | --- |
| **Tác động lên** | **Trọng số** (W) | **Activations** (output neurons) |
| **Cách hoạt động** | Phạt $\|W\|^2$ → kéo W về 0 | Random tắt neurons → phá co-adaptation |
| **Khi train** | Luôn bật (trong loss) | Bật (tắt random neurons) |
| **Khi test** | Luôn bật (W đã nhỏ sẵn) | **TẮT** (giữ tất cả neurons) |
| **Tham số** | $\lambda$ (weight\_decay) | $p$ (dropout probability) |
| **Giá trị thường dùng** | $10^{-4}$ → $10^{-2}$ | 0.2 → 0.5 |
| **Code PyTorch** | `weight_decay=1e-3` trong optimizer | `nn.Dropout(0.5)` trong model |
| **Phù hợp với** | Mọi loại layer | Dense/FC layers (ít dùng cho Conv layers) |

> [!question]- ❓ Có nên dùng Dropout cho CNN (Convolutional Neural Networks)?
> **Ít phổ biến** cho CNN vì:
> - CNN dùng **weight sharing** (mỗi filter dùng chung cho toàn ảnh) → ít bị overfitting hơn MLP
> - Thay vào đó, CNN dùng **Batch Normalization** (sẽ học sau)
> - Nếu dùng, chỉ dùng dropout ở **FC layers cuối** (classifier head), KHÔNG phải conv layers
>
> Tuy nhiên, một biến thể gọi **Spatial Dropout** (tắt cả channel thay vì từng pixel) vẫn được dùng.

---

## 📖 Từ điển thuật ngữ Buổi 21

| Thuật ngữ | Dịch sang tiếng Việt | Nghĩa dễ hiểu |
| --- | --- | --- |
| **Weight Decay** | Suy giảm trọng số | Phạt W lớn → W tự nhỏ dần mỗi bước |
| **$L_2$ penalty** | Hình phạt L2 | $\frac{\lambda}{2}\sum w_i^2$ — tổng bình phương W |
| **$\lambda$ (lambda)** | Hệ số phạt | Càng lớn → phạt càng nặng |
| **Dropout** | Bỏ ngẫu nhiên | Random tắt neurons khi train |
| **Dropout probability $p$** | Xác suất tắt | p=0.5 → tắt 50% neurons |
| **Co-adaptation** | Đồng thích ứng | Neurons phụ thuộc lẫn nhau quá chặt |
| **Inverted dropout** | Dropout đảo | Scale ÷(1-p) khi train → test không cần chỉnh |
| **Mask** | Mặt nạ | Tensor 0/1 quyết định neuron nào tắt |
| **Bernoulli** | Phân phối Bernoulli | Random 0 hoặc 1 với xác suất cho trước |
| **self.training** | Đang train? | True khi `model.train()`, False khi `model.eval()` |

---

## ✅ Bài tự kiểm tra

1. Weight Decay thêm $\frac{\lambda}{2}\|\mathbf{W}\|^2$ vào loss. Gradient của penalty term theo $w_i$ = ?
2. Viết hàm `dropout_layer(X, dropout)` chỉ cần 4 dòng code.
3. Tại sao Dropout phải **chia cho $(1-p)$** khi giữ neuron?
4. `nn.Dropout(0.5)` đặt ở đâu trong Sequential? Trước hay sau ReLU?
5. Nếu quên `model.eval()` trước khi test → điều gì xảy ra?

> [!NOTE]- 📝 Đáp án
> 1. $\frac{\partial}{\partial w_i}\frac{\lambda}{2}\sum w_j^2 = \lambda w_i$. Gradient penalty **tỉ lệ thuận** với W → W lớn bị phạt nhiều hơn.
> 2. ```python
>    def dropout_layer(X, dropout):
>        if dropout == 0: return X
>        mask = (torch.rand(X.shape) > dropout).float()
>        return mask * X / (1.0 - dropout)
>    ```
> 3. Để **kỳ vọng không đổi**: $E[h'] = (1-p) \times \frac{h}{1-p} + p \times 0 = h$. Nếu không scale → output trung bình khi train **sẽ nhỏ hơn** khi test → kết quả sai.
> 4. **Sau ReLU, trước Linear tiếp theo**: `Linear → ReLU → Dropout → Linear → ...`
> 5. Dropout **vẫn bật** khi test → random tắt neurons → kết quả **không ổn định** (mỗi lần predict ra khác nhau) và accuracy **giảm** (vì mạng thiếu neurons). Đây là bug phổ biến!

---

## 🔗 Liên kết

- **Buổi trước**: [[Buổi 20 - Tuần 5]] — Generalization, Overfitting, Underfitting (lý thuyết)
- **Buổi sau**: [[Buổi 22 - Tuần 6]] — Numerical Stability & Weight Initialization
- **Concept notes**: [[Multilayer Perceptron]], [[Activation Function]]

## 📝 Kết luận

Buổi 21 hoàn thành **2 vũ khí** chống overfitting:
- **Weight Decay**: 1 tham số `weight_decay` trong optimizer → trọng số tự nhỏ dần
- **Dropout**: `nn.Dropout(p)` sau mỗi ReLU → buộc mạng không phụ thuộc neuron cụ thể
- Kết hợp cả hai = **hiệu quả nhất** — chỉ cần 2 dòng code thay đổi so với MLP cơ bản
- Nhớ: `model.train()` / `model.eval()` — **bắt buộc** khi dùng Dropout!

Buổi 22 sẽ giải quyết câu hỏi: "Tại sao khởi tạo trọng số quan trọng?" → **Numerical Stability & Weight Initialization**.
