---
title: "Buổi 19 - Tuần 5: MLP Implementation — Tự code và dùng PyTorch API (D2L)"
tags: [d2l, mlp, implementation, from-scratch, pytorch-api, study-note]
created: 2026-03-25
session: "D2L Tuần 5, Buổi 19 — MLP Implementation"
source:
  - "30_Resources/Books/Dive Into Deep Learning/d2l-en/chapter_multilayer-perceptrons/mlp-implementation.md"
related:
  - "[[Buổi 18 - Tuần 5]]"
  - "[[Multilayer Perceptron]]"
  - "[[Activation Function]]"
  - "[[Softmax Function]]"
---

# Buổi 19 — MLP Implementation: Tự lắp và dùng đồ lắp sẵn

> [!NOTE] ELI5
> Buổi 18 bạn đã hiểu **lý thuyết** MLP: tầng ẩn + activation function = mạng "sâu" có thể học bất kỳ quy luật nào.
>
> Buổi 19 bạn sẽ **code thật**. Chúng ta làm **2 lần**:
> 1. **From scratch** — tự viết từng dòng: khởi tạo trọng số, viết ReLU, tính forward, training loop → hiểu rõ bên trong.
> 2. **Concise** — dùng PyTorch API: 4 dòng code thay 40 dòng → nhanh, gọn, an toàn hơn.
>
> Kết quả kỳ vọng: accuracy **~87-88%** trên Fashion-MNIST — vượt ~85% của softmax regression (1 tầng)!

---

## 🎯 Mục tiêu buổi học

1. Tự **khởi tạo trọng số** cho MLP 1 hidden layer (784 → 256 → 10)
2. Tự **viết ReLU** (chỉ 1 dòng code!)
3. Hiểu **luồng dữ liệu** (forward pass) qua từng tầng
4. So sánh **from scratch vs concise**: code, performance, khi nào dùng cái nào
5. Thí nghiệm: thay đổi **số hidden units** → ảnh hưởng thế nào?

---

## Phần 1: MLP From Scratch — Tự lắp từng bộ phận

> [!NOTE] ELI5
> Giống buổi 16 (Softmax from scratch), bạn tự lắp xe đạp từ từng phụ tùng. Lần này xe phức tạp hơn — có thêm "hộp số" (hidden layer) ở giữa.

### 1.1 Khởi tạo trọng số (Parameters)

MLP của chúng ta có 3 tầng: **input (784)** → **hidden (256)** → **output (10)**.

Mỗi kết nối giữa 2 tầng cần 1 **bảng trọng số** (weight matrix) + 1 **bias vector**:

```python
import torch
from torch import nn

# ====== KHỞI TẠO TRỌNG SỐ ======
num_inputs = 784    # 28×28 pixels
num_hiddens = 256   # 256 neurons ẩn
num_outputs = 10    # 10 loại quần áo

# Tầng 1: input → hidden
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens))

# Tầng 2: hidden → output
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs))

params = [W1, b1, W2, b2]
```

> [!question]- ❓ Tại sao nhân `* 0.01` khi khởi tạo? Sao không để `torch.randn` mặc định?
> `torch.randn` tạo số ngẫu nhiên với trung bình = 0 và độ lệch chuẩn = **1**. Nếu trọng số ban đầu **quá lớn**:
> - Output của mỗi tầng sẽ rất lớn hoặc rất nhỏ
> - Sigmoid/Tanh sẽ bão hòa (vùng phẳng) → **gradient ≈ 0** → không học được
> - ReLU: nhiều neuron có input âm → **chết** ngay từ đầu
>
> Nhân `* 0.01` (còn gọi là `sigma`) để trọng số **nhỏ, gần 0** → output ban đầu ở vùng "nhạy" của activation → gradient tốt. Đây là kỹ thuật **weight initialization** cơ bản.
>
> Có kỹ thuật tốt hơn (Xavier, He init) sẽ học ở buổi sau.

> [!question]- ❓ `nn.Parameter` là gì? Sao không dùng `torch.tensor` thường?
> `nn.Parameter` = tensor đặc biệt, tự động:
> - Được **đăng ký** vào model → khi gọi `model.parameters()` sẽ liệt kê đầy đủ
> - Mặc định `requires_grad=True` → PyTorch tự tính gradient khi gọi `.backward()`
>
> Nếu dùng `torch.tensor` thường, bạn phải tự quản lý gradient — dễ quên, dễ sai.

> [!question]- ❓ Tại sao bias khởi tạo bằng 0 nhưng weight thì không?
> - **Weight = 0** sẽ gây **symmetry problem**: tất cả neurons học giống hệt nhau (vì cùng output, cùng gradient) → mạng không mạnh hơn 1 neuron duy nhất. Cần random để "phá đối xứng".
> - **Bias = 0** không gây vấn đề vì bias chỉ dịch output, không liên quan đến symmetry giữa các neurons.

### 1.2 Tự viết ReLU — đúng 1 dòng

```python
def relu(X):
    """ReLU: giữ nguyên giá trị dương, đổi âm thành 0"""
    return torch.max(X, torch.zeros_like(X))
```

> [!question]- ❓ `torch.zeros_like(X)` là gì? Sao không viết `torch.max(X, 0)`?
> - `torch.zeros_like(X)` tạo tensor **cùng shape, cùng device** (CPU/GPU) với `X`, toàn số 0.
> - Cần dùng vì `torch.max` so sánh **element-wise** (từng phần tử) giữa 2 tensors cùng shape.
> - `torch.max(X, 0)` trong PyTorch sẽ trả về **max theo dimension 0** (khác ý nghĩa!), nên phải dùng `torch.zeros_like`.
>
> Ngoài ra, bạn cũng có thể viết: `X * (X > 0)` hoặc `torch.clamp(X, min=0)` — kết quả tương đương.

### 1.3 Forward Pass — Dữ liệu đi qua mạng

```python
def forward(X):
    """Dữ liệu đi: Input → Hidden (ReLU) → Output"""
    # Bước 0: Duỗi ảnh 28×28 thành vector 784
    X = X.reshape(-1, num_inputs)  # (batch, 784)
    
    # Bước 1: Input → Hidden layer + ReLU
    H = relu(X @ W1 + b1)         # (batch, 256)
    
    # Bước 2: Hidden → Output (logits, KHÔNG apply activation!)
    O = H @ W2 + b2               # (batch, 10)
    
    return O  # Logits → đưa vào F.cross_entropy
```

Hãy đi qua từng bước với **1 ảnh** (batch_size = 1):

![[assets/attachments/D2L/Buoi19/forward_pass.png]]

```
Ảnh 28×28 → reshape → [1, 784]
                         ↓
     X @ W1 + b1 → [1, 256]  (nhân ma trận + cộng bias)
                         ↓
       ReLU → [1, 256]        (âm → 0, dương → giữ)
                         ↓
     H @ W2 + b2 → [1, 10]   (nhân ma trận + cộng bias)
                         ↓
              Output: 10 logits (1 số cho mỗi lớp quần áo)
```

> [!CAUTION] Nhắc lại: Output layer KHÔNG apply activation!
> Output trả về **logits** (giá trị thô). `F.cross_entropy()` sẽ tự tính softmax bên trong. Nếu bạn apply ReLU/sigmoid lên output → kết quả sai hoàn toàn.
>
> Lý do chi tiết: xem [[Buổi 17 - Tuần 4#Phần 2: Numerical Stability|Buổi 17 — LogSumExp trick]]

> [!question]- ❓ `@` nghĩa là gì trong Python?
> `@` là toán tử **nhân ma trận** (matrix multiplication) trong Python/NumPy/PyTorch.
> - `X @ W1` tương đương `torch.matmul(X, W1)` tương đương `X.mm(W1)`
> - Đây là phép nhân ma trận chuẩn, **KHÔNG phải** nhân từng phần tử (element-wise, dùng `*`).
>
> Ví dụ: `[1, 784] @ [784, 256] = [1, 256]` — mỗi output là tổng có trọng số của toàn bộ 784 pixels.

### 1.4 Full Training Code (From Scratch)

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

# ====== PARAMETERS ======
num_inputs, num_hiddens, num_outputs = 784, 256, 10
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens))
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs))
params = [W1, b1, W2, b2]

# ====== ReLU ======
def relu(X):
    return torch.max(X, torch.zeros_like(X))

# ====== FORWARD ======
def forward(X):
    X = X.reshape(-1, num_inputs)
    H = relu(X @ W1 + b1)
    return H @ W2 + b2

# ====== TRAIN ======
lr = 0.1
for epoch in range(10):
    for X, y in train_loader:
        loss = F.cross_entropy(forward(X), y)
        loss.backward()
        with torch.no_grad():
            for param in params:
                param -= lr * param.grad    # SGD update
                param.grad.zero_()          # Zero gradients
    
    # Evaluate
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in test_loader:
            correct += (forward(X).argmax(1) == y).sum().item()
            total += y.shape[0]
    print(f"Epoch {epoch+1:2d} | Test Acc: {correct/total:.4f}")
```

> [!question]- ❓ Training loop giống Softmax Regression (Buổi 16) hay khác?
> **Gần như giống hệt!** Sự khác biệt duy nhất:
>
> | | Softmax Regression | MLP |
> | --- | --- | --- |
> | Forward | `softmax(X @ W + b)` | `relu(X @ W1 + b1) @ W2 + b2` |
> | Số params | W, b (2 tensors) | W1, b1, W2, b2 (4 tensors) |
> | Loss | Giống | Giống (`F.cross_entropy`) |
> | Backward | Giống | Giống (`.backward()`) |
> | Update | Giống | Giống (SGD) |
>
> Điều hay: **training loop không đổi** khi thêm tầng. Bạn chỉ thay hàm `forward()` — phần còn lại (loss, backward, update) y nguyên.

---

## Phần 2: MLP Concise — 4 dòng code thay 40 dòng

> [!NOTE] ELI5
> Giống buổi 17, bạn chuyển từ "tự lắp xe" sang "mua xe có sẵn". PyTorch đã gói gọn MLP thành vài dòng — bạn chỉ cần nói mạng gồm những gì.

### 2.1 Định nghĩa Model

```python
import torch
from torch import nn

model = nn.Sequential(
    nn.Flatten(),           # (batch, 1, 28, 28) → (batch, 784)
    nn.LazyLinear(256),     # Linear: 784 → 256
    nn.ReLU(),              # Activation: ReLU
    nn.LazyLinear(10)       # Linear: 256 → 10 (output logits)
)
```

**4 dòng** — so với ~20 dòng khởi tạo + forward ở phiên bản scratch!

> [!question]- ❓ `nn.Sequential` hoạt động thế nào?
> `nn.Sequential` là **container** — nó xếp các layer theo thứ tự, rồi khi bạn gọi `model(X)`:
> 
> ```
> X → Flatten → LazyLinear(256) → ReLU → LazyLinear(10) → Output
> ```
> 
> Mỗi layer nhận output của layer trước làm input. Bạn **không cần viết hàm `forward()`** — Sequential tự động xếp pipeline.
>
> Tương tự một dây chuyền sản xuất: đưa nguyên liệu vào đầu → qua từng máy → ra sản phẩm cuối.

> [!question]- ❓ So sánh scratch vs concise — khác ở đâu?
> 
> | | From Scratch | Concise (PyTorch API) |
> | --- | --- | --- |
> | **Khởi tạo W, b** | Tự viết (`torch.randn`, `torch.zeros`) | `nn.LazyLinear` tự lo |
> | **Activation** | Tự viết hàm `relu()` | `nn.ReLU()` |
> | **Forward** | Tự viết `X @ W1 + b1`, `relu(...)` | `model(X)` — Sequential tự chạy |
> | **Optimizer** | Tự viết `param -= lr * param.grad` | `torch.optim.SGD(model.parameters(), lr)` |
> | **Numerical stability** | ❌ Không (tự viết thì tự lo) | ✅ PyTorch xử lý sẵn |
> | **Số dòng** | ~40 dòng | ~10 dòng |
> | **Khi nào dùng** | Học, hiểu, debug, research | Prototype, production |

### 2.2 Full Training Code (Concise)

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

# ====== MODEL (4 dòng!) ======
model = nn.Sequential(
    nn.Flatten(),
    nn.LazyLinear(256),
    nn.ReLU(),
    nn.LazyLinear(10)
)

# ====== OPTIMIZER ======
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# ====== TRAIN ======
for epoch in range(10):
    model.train()
    for X, y in train_loader:
        loss = F.cross_entropy(model(X), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Evaluate
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in test_loader:
            correct += (model(X).argmax(1) == y).sum().item()
            total += y.shape[0]
    print(f"Epoch {epoch+1:2d} | Test Acc: {correct/total:.4f}")
```

> [!question]- ❓ `model.train()` vs `model.eval()` — lại xuất hiện, tại sao quan trọng?
> Cả hai buổi 17 và 19 đều dùng `model.train()` / `model.eval()`. Hiện tại MLP đơn giản **chưa** cần — nhưng khi thêm **Dropout** hoặc **Batch Normalization** (học ở buổi sau):
> - `model.train()`: Dropout **bật** (random tắt neuron), Batch Norm dùng **batch statistics**
> - `model.eval()`: Dropout **tắt** (giữ tất cả neuron), Batch Norm dùng **running statistics**
>
> **Tập thói quen** dùng từ bây giờ sẽ tránh bug khó tìm sau này.

---

## Phần 3: Kết quả và phân tích

### 3.1 Kết quả kỳ vọng

| Metric | Softmax Regression (Buổi 17) | MLP (Buổi 19) | Cải thiện |
| --- | --- | --- | --- |
| **Test Accuracy** | ~82-85% | ~**87-88%** | +3-5% |
| **Số tham số** | 7,850 | 203,530 | ×26 |
| **Thời gian train** | Nhanh | Chậm hơn ~2-3× | Nhiều phép tính hơn |

![[assets/attachments/D2L/Buoi19/accuracy_comparison.png]]
*MLP (xanh) vượt Softmax Regression (cam) ~3-5% accuracy trên Fashion-MNIST*

> [!question]- ❓ Chỉ tăng 3-5% mà gấp 26× tham số — có đáng không?
> **Có**, vì:
> 1. Fashion-MNIST là bài toán **đơn giản** — 85% đã khá tốt, khó cải thiện thêm nhiều
> 2. MLP mới chỉ có **1 hidden layer** — thêm tầng, thêm kỹ thuật (dropout, learning rate schedule) sẽ tốt hơn
> 3. Với bài toán khó hơn (ImageNet, NLP), sự khác biệt giữa linear và deep model là **trời vs đất** (70% vs 99%+)
> 4. Tham số nhiều ≠ luôn tệ — miễn có đủ dữ liệu và kỹ thuật regularization

### 3.2 Training loop không đổi — thiết kế modular

Điều hay nhất: khi thêm tầng ẩn, **training loop gần như giữ nguyên**:

```diff
 # Softmax Regression (Buổi 17)
 model = nn.Sequential(nn.Flatten(), nn.LazyLinear(10))

 # MLP 1 hidden layer (Buổi 19)
-model = nn.Sequential(nn.Flatten(), nn.LazyLinear(10))
+model = nn.Sequential(nn.Flatten(), nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))

 # MLP 2 hidden layers (tương lai)
+model = nn.Sequential(
+    nn.Flatten(),
+    nn.LazyLinear(256), nn.ReLU(),
+    nn.LazyLinear(128), nn.ReLU(),
+    nn.LazyLinear(10)
+)

 # Training loop: KHÔNG ĐỔI GÌ CẢ!
 for epoch in range(10):
     ...
```

> [!TIP] Tại sao thiết kế này hay?
> Đây là **separation of concerns** (tách biệt mối quan tâm):
> - Model: "mạng trông như thế nào" — thay đổi tự do
> - Training: "cách huấn luyện" — giữ nguyên
> 
> Bạn chỉ cần thay **1 dòng** `model = ...` để thử kiến trúc khác mà không sửa bất kỳ thứ gì ở training loop.

---

## Phần 4: Thí nghiệm — Thay đổi Hidden Units

Thay đổi số hidden units ảnh hưởng ra sao?

| `num_hiddens` | Test Accuracy (ước lượng) | Số tham số | Nhận xét |
| --- | --- | --- | --- |
| 32 | ~84-85% | ~25K | Quá ít → ít hơn ~1% so với linear |
| 128 | ~86-87% | ~103K | Tốt hơn linear nhưng chưa peak |
| **256** | ~**87-88%** | ~**203K** | **Sweet spot** cho Fashion-MNIST |
| 512 | ~87-88% | ~407K | Không tốt hơn 256 nhiều → diminishing returns |
| 1024 | ~87-88% | ~813K | Tương tự 512, lãng phí. Dễ overfit hơn |

![[assets/attachments/D2L/Buoi19/hidden_units_experiment.png]]
*Tăng hidden units: accuracy tăng nhanh đến 256, sau đó gần như bão hòa (diminishing returns)*

> [!question]- ❓ "Diminishing returns" nghĩa là gì? Tại sao tăng units không tăng accuracy mãi?
> **Diminishing returns** = "lợi nhuận giảm dần". Giống như: ăn 1 bát cơm → no 50%. Ăn 2 bát → no 80%. Ăn 3 bát → no 90%. Ăn 10 bát? Vẫn chỉ no 100% thôi, mà còn đau bụng.
>
> Trong ML:
> - 256 units đã đủ "sức chứa" cho Fashion-MNIST (bài toán không phức tạp)
> - Thêm units → mạng **có thể** biểu diễn thêm, nhưng **data không đủ phức tạp** để tận dụng
> - Quá nhiều units → **overfit** (học thuộc train data, sai trên data mới)
>
> **Rule of thumb**: bắt đầu nhỏ (128-256), tăng dần, dừng khi validation accuracy không cải thiện nữa.

> [!question]- ❓ Thêm 1 hidden layer nữa (2 hidden layers) thì sao?
> Có thể thử:
> ```python
> model = nn.Sequential(
>     nn.Flatten(),
>     nn.LazyLinear(256), nn.ReLU(),  # Hidden 1
>     nn.LazyLinear(128), nn.ReLU(),  # Hidden 2
>     nn.LazyLinear(10)               # Output
> )
> ```
> 
> Trên Fashion-MNIST: accuracy **tăng rất ít** (~0.5-1%) hoặc **không tăng**. Vì bài toán đơn giản — 1 hidden layer đã đủ.
>
> Với bài toán khó hơn (CIFAR-10, ImageNet): nhiều tầng **quan trọng rất nhiều** — đó là lý do mạng có 50, 100, 1000+ tầng tồn tại (ResNet, GPT).

---

## 📖 Từ điển thuật ngữ Buổi 19

| Thuật ngữ | Dịch sang tiếng Việt | Nghĩa dễ hiểu |
| --- | --- | --- |
| **from scratch** | Làm từ đầu | Tự viết mọi thứ, không dùng thư viện |
| **concise** | Súc tích | Dùng API sẵn, code ngắn gọn |
| **nn.Sequential** | Container tuần tự | Xếp layers thành pipeline tự động |
| **nn.LazyLinear** | Linear layer "lười" | Tự detect input size lần đầu |
| **nn.ReLU()** | ReLU layer | Activation function dạng layer (dùng trong Sequential) |
| **nn.Parameter** | Tham số đăng ký | Tensor tự động tracked gradient |
| **forward pass** | Lượt truyền xuôi | Input đi qua mạng → ra output |
| **weight initialization** | Khởi tạo trọng số | Cách đặt giá trị ban đầu cho W, b |
| **sigma** | Độ lệch chuẩn | Mức "phân tán" khi random khởi tạo |
| **symmetry breaking** | Phá đối xứng | Random W để mỗi neuron học khác nhau |
| **diminishing returns** | Lợi nhuận giảm dần | Tăng resource nhưng cải thiện ít dần |
| **model.train()** | Chế độ huấn luyện | Bật Dropout/BatchNorm (nếu có) |
| **model.eval()** | Chế độ đánh giá | Tắt Dropout, dùng running stats |

---

## ✅ Bài tự kiểm tra

1. Viết hàm `relu(X)` chỉ bằng **1 dòng code** PyTorch (2 cách khác nhau).
2. MLP scratch: tại sao `W1` khởi tạo random nhưng `b1` khởi tạo = 0?
3. Viết `nn.Sequential` cho MLP với **2 hidden layers**: 784 → 512 → 256 → 10.
4. Training loop MLP có khác gì training loop Softmax Regression? Nếu không, tại sao?
5. `num_hiddens = 2` (chỉ 2 neurons ẩn) → vấn đề gì xảy ra?

> [!NOTE]- 📝 Đáp án
> 1. Cách 1: `torch.max(X, torch.zeros_like(X))`. Cách 2: `torch.clamp(X, min=0)`. Bonus: `X * (X > 0)`.
> 2. Random W để **phá đối xứng** — nếu W = 0, mọi neuron output giống nhau → mạng không mạnh hơn 1 neuron. Bias = 0 OK vì bias chỉ dịch output, không ảnh hưởng đối xứng.
> 3. `nn.Sequential(nn.Flatten(), nn.LazyLinear(512), nn.ReLU(), nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))`
> 4. **Không khác.** Chỉ hàm `forward()` / model definition thay đổi. Loss, backward, optimizer giữ nguyên. Đây là ưu điểm của thiết kế modular.
> 5. Chỉ 2 neurons ẩn = "bottleneck" — mạng buộc phải nén 784 features thành 2 số → mất quá nhiều thông tin → accuracy rất thấp (có thể ≤ 60-70%). Cũng gần như tương đương **2D projection** của dữ liệu → chỉ tốt nếu dữ liệu thật sự nằm trong không gian 2D.

---

## 🔗 Liên kết

- **Buổi trước**: [[Buổi 18 - Tuần 5]] — MLP Lý thuyết (hidden layers, activation functions)
- **Buổi sau**: [[Buổi 20 - Tuần 5]] — Underfitting, Overfitting, Regularization
- **Concept notes**: [[Multilayer Perceptron]], [[Activation Function]]

## 📝 Kết luận

Buổi 19 đưa MLP từ **lý thuyết sang thực hành**:
- Code từ scratch → hiểu rõ **3 bộ phận**: init params, forward (ReLU), training loop
- Code concise → **4 dòng** với `nn.Sequential`: Flatten → Linear → ReLU → Linear
- Accuracy **~87-88%** vượt softmax regression (~85%) trên Fashion-MNIST
- Training loop **không đổi** khi thêm/bớt tầng → thiết kế modular tuyệt vời

Buổi 20 sẽ trả lời câu hỏi: "203K tham số có phải **quá nhiều** cho Fashion-MNIST?" → **Overfitting, Underfitting, và Regularization**.
