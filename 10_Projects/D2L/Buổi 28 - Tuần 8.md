---
title: "Buổi 28 - Tuần 8: LeNet — Xây CNN đầu tiên từ A đến Z"
tags: [d2l, cnn, lenet, fashion-mnist, training, study-note]
created: 2026-04-02
modified: 2026-04-02
session: "D2L Tuần 8, Buổi 28 — LeNet: CNN đầu tiên"
source:
  - "30_Resources/Books/Dive Into Deep Learning/d2l-en/chapter_convolutional-neural-networks/lenet.md"
related:
  - "[[Buổi 27 - Tuần 8]]"
  - "[[Buổi 29 - Tuần 8]]"
---

# Buổi 28 — LeNet: Xây CNN đầu tiên từ A đến Z

> [!NOTE] ELI5
> Ba buổi trước bạn đã học từng linh kiện riêng lẻ:
> - **Buổi 26**: Convolution (kernel trượt qua ảnh)
> - **Buổi 27**: Pooling (rút gọn kích thước) + Multi-channel (ảnh RGB)
>
> Bây giờ đến bước **quan trọng nhất**: ghép tất cả lại thành **1 model hoạt động được** — train nó và xem nó phân loại ảnh quần áo!
>
> Model chúng ta xây là **LeNet-5** (1998) — CNN đầu tiên được dùng trong **máy ATM** để đọc chữ viết tay trên séc ngân hàng. Kiến trúc rất đơn giản nhưng đặt nền móng cho mọi CNN đến nay.

---

## 🎯 Mục tiêu buổi học

Sau buổi này, bạn sẽ **tự tay**:
1. Xây 1 CNN hoàn chỉnh từ **dòng code đầu tiên**
2. Hiểu data đi qua mỗi layer **thay đổi shape** như thế nào
3. **Train** CNN trên Fashion-MNIST và đọc kết quả
4. **Nâng cấp** LeNet cổ điển → bản hiện đại (ReLU + MaxPool + BN)
5. Hiểu tại sao CNN **ít tham số hơn nhưng chính xác hơn** MLP

---

## Phần 1: LeNet gồm những gì?

> [!NOTE] ELI5
> Hãy nghĩ LeNet như một **nhà máy** có 2 phân xưởng:
>
> 🔍 **Phân xưởng 1 — "Mắt" (Encoder)**: Nhìn ảnh → tìm edges, hình dạng → tạo ra **bản tóm tắt** ảnh
> 🧠 **Phân xưởng 2 — "Não" (Classifier)**: Đọc bản tóm tắt → quyết định "đây là áo phông" hay "đây là giày"
>
> Mỗi phân xưởng gồm vài bước xử lý đơn giản xếp nối tiếp nhau.

![[assets/attachments/D2L/Buoi28/lenet_architecture.png]]
*Kiến trúc LeNet-5: Ảnh đi qua 2 tầng Conv+Pool (Mắt), rồi 3 tầng FC (Não) → ra kết quả.*

### 1.1 Liệt kê từng bước — Từ ảnh → Class

Hãy theo dõi **1 bức ảnh áo phông 28×28** đi qua LeNet:

```
Bước 0: Ảnh gốc                    → (1, 28, 28)    "1 tấm ảnh trắng-đen"
  ↓ Conv 5×5, padding=2, 6 filters
Bước 1: 6 bản feature maps         → (6, 28, 28)    "6 cách nhìn khác nhau"
  ↓ Sigmoid (activation)
Bước 2: Giữ nguyên size            → (6, 28, 28)
  ↓ AvgPool 2×2
Bước 3: Thu nhỏ 1/4 diện tích      → (6, 14, 14)    "rút gọn"
  ↓ Conv 5×5, 16 filters
Bước 4: 16 bản feature maps        → (16, 10, 10)   "16 cách nhìn tinh hơn"
  ↓ Sigmoid → AvgPool 2×2
Bước 5: Thu nhỏ tiếp               → (16, 5, 5)
  ↓ Flatten (duỗi phẳng)
Bước 6: Vector 1 chiều             → (400,)          "bản mô tả ảnh"
  ↓ FC 400→120 → Sigmoid
Bước 7: Rút gọn                    → (120,)
  ↓ FC 120→84 → Sigmoid
Bước 8: Rút gọn                    → (84,)
  ↓ FC 84→10
Bước 9: Điểm cho 10 class          → (10,)          "T-shirt=7.2, Dress=1.1..."
```

> [!question]- ❓ Mối liên hệ giữa shape và ý nghĩa?
> | Shape | Ý nghĩa dễ hiểu |
> | --- | --- |
> | **(1, 28, 28)** | 1 ảnh, mỗi pixel = 1 số (grayscale) |
> | **(6, 28, 28)** | 6 "bản đồ đặc trưng" — mỗi bản nhìn ảnh theo 1 cách khác |
> | **(6, 14, 14)** | Giống 6 bản đồ trên nhưng thu nhỏ — giữ thông tin chính, bỏ chi tiết thừa |
> | **(16, 5, 5)** | 16 "bản tóm tắt" nhỏ — mỗi bản mô tả 1 đặc điểm (có viền? có túi? có dây?) |
> | **(400,)** | Duỗi thẳng 16×5×5 thành 1 hàng — chuẩn bị cho FC |
> | **(10,)** | 10 điểm số — số nào cao nhất = class được chọn |

### 1.2 Đếm tham số — Mỗi layer tốn bao nhiêu?

| Layer             | Tính                                 | Kết quả    | %     |
| ----------------- | ------------------------------------ | ---------- | ----- |
| Conv1 (1→6, 5×5)  | $6 \times 1 \times 5 \times 5 + 6$   | **156**    | 0.3%  |
| Conv2 (6→16, 5×5) | $16 \times 6 \times 5 \times 5 + 16$ | **2,416**  | 3.9%  |
| FC1 (400→120)     | $400 \times 120 + 120$               | **48,120** | 78.0% |
| FC2 (120→84)      | $120 \times 84 + 84$                 | **10,164** | 16.5% |
| FC3 (84→10)       | $84 \times 10 + 10$                  | **850**    | 1.4%  |
| **Tổng**          |                                      | **61,706** | 100%  |

> [!IMPORTANT] Nhận xét quan trọng
> **96%** tham số nằm ở FC layers (Não), nhưng phần **làm việc nặng nhất** (phát hiện features) là Conv layers (Mắt) — chỉ chiếm **4%** tham số.
>
> Giống một công ty: bộ phận **R&D** (Conv) ít người nhưng tạo ra sản phẩm. Bộ phận **hành chính** (FC) nhiều người nhưng chỉ xử lý giấy tờ.
>
> → Sau này (ResNet, EfficientNet) sẽ thay FC bằng **Global Average Pooling** → giảm 96% tham số classifier!

---

## Phần 2: Code LeNet từng dòng

> [!NOTE] ELI5
> Đọc code bên dưới **từ trên xuống**, mỗi dòng tương ứng 1 bước trong sơ đồ ở trên. Comment giải thích **input → output shape** cho từng dòng.

### 2.1 LeNet Classic (đúng bản gốc 1998)

```python
import torch
from torch import nn
from torch.nn import functional as F

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()

        # ═══ PHẦN MẮT (Encoder) ═══
        # Mỗi block: Conv → Activation → Pool
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        #     Input: (1, 28, 28) → Output: (6, 28, 28)
        #     padding=2 giữ nguyên 28×28

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        #     Input: (6, 14, 14) → Output: (16, 10, 10)
        #     không padding → nhỏ lại

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        #     Giảm nửa H và W mỗi lần gọi

        # ═══ PHẦN NÃO (Classifier) ═══
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 400 → 120
        self.fc2 = nn.Linear(120, 84)            # 120 → 84
        self.fc3 = nn.Linear(84, 10)             # 84 → 10

    def forward(self, x):
        # x shape: (batch, 1, 28, 28)

        # Block 1: Conv → Sigmoid → Pool
        x = self.pool(torch.sigmoid(self.conv1(x)))
        # (batch, 1, 28, 28) → Conv → (batch, 6, 28, 28)
        #                    → Sigmoid → (batch, 6, 28, 28)
        #                    → Pool → (batch, 6, 14, 14)

        # Block 2: Conv → Sigmoid → Pool
        x = self.pool(torch.sigmoid(self.conv2(x)))
        # (batch, 6, 14, 14) → Conv → (batch, 16, 10, 10)
        #                    → Sigmoid → (batch, 16, 10, 10)
        #                    → Pool → (batch, 16, 5, 5)

        # Duỗi phẳng
        x = x.flatten(1)  # (batch, 16, 5, 5) → (batch, 400)

        # 3 tầng FC
        x = torch.sigmoid(self.fc1(x))  # (batch, 400) → (batch, 120)
        x = torch.sigmoid(self.fc2(x))  # (batch, 120) → (batch, 84)
        x = self.fc3(x)                  # (batch, 84)  → (batch, 10)

        return x  # 10 scores, chưa qua softmax
```

### 2.2 Kiểm tra: Chạy thử 1 ảnh

```python
model = LeNet()

# Tạo 1 ảnh giả: batch=1, channel=1, H=28, W=28
fake_image = torch.randn(1, 1, 28, 28)

# Forward qua model
output = model(fake_image)
print(output.shape)  # torch.Size([1, 10]) ✓ — 10 scores cho 10 classes!
print(output)         # tensor([[-0.0432, 0.0891, ...]])  — số ngẫu nhiên (chưa train)
```

### 2.3 Theo dõi shape qua từng layer

```python
# Cách đơn giản nhất để debug CNN:
X = torch.randn(1, 1, 28, 28)
print(f"Input:       {X.shape}")

X = model.conv1(X);          print(f"After Conv1: {X.shape}")
X = torch.sigmoid(X);        print(f"After Sigm:  {X.shape}")
X = model.pool(X);           print(f"After Pool1: {X.shape}")
X = model.conv2(X);          print(f"After Conv2: {X.shape}")
X = torch.sigmoid(X);        print(f"After Sigm:  {X.shape}")
X = model.pool(X);           print(f"After Pool2: {X.shape}")
X = X.flatten(1);            print(f"After Flat:  {X.shape}")
X = torch.sigmoid(model.fc1(X)); print(f"After FC1:   {X.shape}")
X = torch.sigmoid(model.fc2(X)); print(f"After FC2:   {X.shape}")
X = model.fc3(X);            print(f"After FC3:   {X.shape}")
```

Output:
```
Input:       torch.Size([1, 1, 28, 28])
After Conv1: torch.Size([1, 6, 28, 28])     ← channels 1→6, size giữ nguyên
After Sigm:  torch.Size([1, 6, 28, 28])
After Pool1: torch.Size([1, 6, 14, 14])     ← size giảm nửa: 28→14
After Conv2: torch.Size([1, 16, 10, 10])    ← channels 6→16, size 14→10
After Sigm:  torch.Size([1, 16, 10, 10])
After Pool2: torch.Size([1, 16, 5, 5])      ← size giảm nửa: 10→5
After Flat:  torch.Size([1, 400])            ← 16×5×5 = 400
After FC1:   torch.Size([1, 120])
After FC2:   torch.Size([1, 84])
After FC3:   torch.Size([1, 10])             ← 10 class scores ✓
```

> [!TIP] Mẹo debug CNN
> Khi xây CNN mới, **luôn in shape** sau mỗi layer! Lỗi phổ biến nhất = shape mismatch giữa Flatten và FC đầu tiên.
>
> ```python
> # Nếu bạn không chắc flatten cho bao nhiêu:
> x = torch.randn(1, 1, 28, 28)
> x = encoder(x)
> print(x.shape)  # (1, 16, 5, 5)
> flat_size = x.shape[1] * x.shape[2] * x.shape[3]  # = 400
> ```

---

## Phần 3: Train LeNet — Từng bước chi tiết

> [!NOTE] ELI5
> Training = cho model **xem hàng nghìn ảnh**, mỗi lần sai thì **sửa trọng số** — giống học sinh làm bài tập và sửa lỗi. Lặp lại nhiều lần (epochs) cho đến khi model "thuộc bài".

### 3.1 Chuẩn bị data

```python
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Tải Fashion-MNIST
# transforms.ToTensor() chuyển ảnh [0,255] → [0.0, 1.0] và shape (28,28) → (1,28,28)
train_data = datasets.FashionMNIST(
    root='./data', train=True, download=True,
    transform=transforms.ToTensor()
)
test_data = datasets.FashionMNIST(
    root='./data', train=False,
    transform=transforms.ToTensor()
)

# DataLoader: chia data thành batches
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=128)

print(f"Training: {len(train_data):,} ảnh")   # 60,000
print(f"Testing:  {len(test_data):,} ảnh")     # 10,000
```

### 3.2 Chuẩn bị model + optimizer

```python
# Device: GPU nếu có, không thì CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

# Tạo model và chuyển sang device
model = LeNet().to(device)

# Xavier initialization — giúp training ổn định hơn
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
model.apply(init_weights)

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.9)
# lr=0.9 cao vì Sigmoid converge chậm, cần bước nhảy lớn
```

### 3.3 Training loop

```python
EPOCHS = 10

for epoch in range(EPOCHS):
    # ══════ TRAIN ══════
    model.train()  # bật Dropout/BN (nếu có)
    train_loss, train_correct, train_total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        # images shape: (128, 1, 28, 28), labels shape: (128,)

        # Forward
        outputs = model(images)           # (128, 10)
        loss = F.cross_entropy(outputs, labels)

        # Backward + Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Ghi nhận
        train_loss += loss.item() * images.size(0)
        train_correct += (outputs.argmax(1) == labels).sum().item()
        train_total += images.size(0)

    # ══════ EVALUATE ══════
    model.eval()  # tắt Dropout/BN
    test_correct = 0
    with torch.no_grad():  # không cần gradient khi evaluate
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_correct += (outputs.argmax(1) == labels).sum().item()

    # ══════ IN KẾT QUẢ ══════
    train_acc = train_correct / train_total * 100
    test_acc  = test_correct / len(test_data) * 100
    avg_loss  = train_loss / train_total
    print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
          f"Loss: {avg_loss:.4f} | "
          f"Train: {train_acc:.1f}% | "
          f"Test: {test_acc:.1f}%")
```

### 3.4 Giải thích từng dòng quan trọng

| Dòng code | Làm gì | Tại sao cần |
| --- | --- | --- |
| `images.to(device)` | Chuyển data sang GPU/CPU | Model và data phải cùng device |
| `model(images)` | Forward pass — tính output | Gọi `forward()` qua `__call__` |
| `F.cross_entropy(...)` | Tính loss | So sánh output với đáp án đúng |
| `optimizer.zero_grad()` | Xóa gradient cũ | Gradient tích lũy nếu không xóa |
| `loss.backward()` | Tính gradient mới | Backpropagation |
| `optimizer.step()` | Cập nhật trọng số | W = W - lr × gradient |
| `model.eval()` | Chuyển sang eval mode | Tắt Dropout, BN dùng running stats |
| `torch.no_grad()` | Không tính gradient | Tiết kiệm bộ nhớ khi test |
| `outputs.argmax(1)` | Chọn class có score cao nhất | Prediction |

### 3.5 Kết quả mong đợi

```
Epoch  1/10 | Loss: 2.2981 | Train: 11.2% | Test: 10.0%    ← random!
Epoch  2/10 | Loss: 1.8234 | Train: 32.1% | Test: 45.3%
Epoch  3/10 | Loss: 0.8456 | Train: 65.7% | Test: 68.2%
Epoch  5/10 | Loss: 0.5123 | Train: 80.4% | Test: 80.8%
Epoch 10/10 | Loss: 0.4012 | Train: 84.5% | Test: 83.6%    ← ~84%
```

> [!WARNING] Vấn đề: ~84% cho LeNet Classic — thấp hơn MLP (~87%)!
> **Lý do**: Sigmoid + AvgPool là công nghệ 1998 — rất chậm hội tụ.
> → **Giải pháp**: nâng cấp lên bản Modern!

---

## Phần 4: Nâng cấp → Modern LeNet

> [!NOTE] ELI5
> Bạn có chiếc xe cổ 1998. Xe chạy được nhưng chậm và tốn xăng. Bạn thay:
> - **Động cơ xăng cũ** (Sigmoid) → **Động cơ điện** (ReLU) — nhanh hơn, mạnh hơn
> - **Phanh thường** (AvgPool) → **Phanh ABS** (MaxPool) — giữ features tốt hơn
> - Thêm **dây an toàn** (Dropout) → chống overfitting
> - Thêm **cân bằng điện tử** (BatchNorm) → training ổn định
>
> Cùng khung xe, nhưng performance **tăng vọt**!

![[assets/attachments/D2L/Buoi28/lenet_classic_vs_modern.png]]
*4 nâng cấp: Sigmoid→ReLU, AvgPool→MaxPool, thêm BatchNorm, thêm Dropout.*

### 4.1 Code Modern LeNet

```python
class ModernLeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # Block 1: Conv → BatchNorm → ReLU → MaxPool
            nn.Conv2d(1, 6, kernel_size=5, padding=2),  # (1,28,28)→(6,28,28)
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),                   # (6,28,28)→(6,14,14)

            # Block 2: Conv → BatchNorm → ReLU → MaxPool
            nn.Conv2d(6, 16, kernel_size=5),             # (6,14,14)→(16,10,10)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),                   # (16,10,10)→(16,5,5)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                 # (16,5,5)→(400)
            nn.Linear(400, 120), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(120, 84),  nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        return self.classifier(self.encoder(x))
```

### 4.2 Thay đổi gì? Tại sao?

| # | Classic (1998) | Modern (2020s) | Lợi ích |
| --- | --- | --- | --- |
| 1 | **Sigmoid** $\frac{1}{1+e^{-x}}$ | **ReLU** $\max(0,x)$ | Gradient không vanish → train nhanh 5-10× |
| 2 | **AvgPool** (lấy trung bình) | **MaxPool** (lấy max) | Giữ features nổi bật, bỏ noise |
| 3 | Không có | **BatchNorm** | Chuẩn hóa → lr cao hơn, training ổn định |
| 4 | Không có | **Dropout 0.5** | Tắt 50% neuron ngẫu nhiên → chống overfitting |

### 4.3 Training Modern LeNet

```python
model = ModernLeNet().to(device)
model.apply(init_weights)

# Adam tốt hơn SGD cho modern architectures
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ... (cùng training loop)
```

Kết quả:
```
Epoch  1/10 | Loss: 0.5834 | Train: 78.4% | Test: 84.5%    ← ngay epoch 1 đã 84%!
Epoch  5/10 | Loss: 0.2891 | Train: 89.2% | Test: 89.8%
Epoch 10/10 | Loss: 0.2134 | Train: 92.1% | Test: 91.2%    ← 91%!
```

### 4.4 So sánh kết quả

| Model | Tham số | Fashion-MNIST Accuracy | Epochs |
| --- | --- | --- | --- |
| MLP 2 layers (Buổi 19) | ~200K | ~87% | 10 |
| LeNet Classic (1998) | ~62K | ~84% | 10 |
| **Modern LeNet** | ~62K | **~91%** | 10 |

> [!TIP] Kết luận
> Modern LeNet đạt **91%** — cao hơn MLP 4% — mà chỉ dùng **1/3 tham số**!
>
> CNN thắng vì có **inductive bias đúng**: locality (nhìn vùng nhỏ), translation invariance (dùng chung kernel mọi nơi), hierarchical features (edges → textures → shapes).

---

## Phần 5: Tại sao CNN thắng MLP? (Giải thích đơn giản)

### 5.1 Vấn đề với MLP

```
Ảnh 28×28 → Flatten → vector 784 phần tử → FC layer

Vấn đề 1: MLP KHÔNG BIẾT pixel nào cạnh pixel nào!
  pixel (0,0) và pixel (27,27) được đối xử GIỐNG NHAU
  như pixel (0,0) và pixel (0,1).

Vấn đề 2: Tham số KHỔNG LỒ!
  FC 784→256: 200,704 tham số
  Mỗi pixel kết nối với MỌI neuron → lãng phí
```

### 5.2 CNN giải quyết thế nào

```
Ảnh 28×28 → GIỮ NGUYÊN 2D → Conv 5×5 kernel

Giải pháp 1: Locality — mỗi neuron CHỈ nhìn vùng 5×5
  → Biết pixel nào gần pixel nào ✓

Giải pháp 2: Weight sharing — CÙNG kernel cho MỌI vị trí
  → Conv 5×5: chỉ 25+1=26 tham số (thay vì 200K) ✓

Giải pháp 3: Hierarchical — nhiều tầng chồng lên nhau
  → Tầng 1 tìm edges, tầng 2 tìm shapes → không cần 1 FC lớn ✓
```

---

## Phần 6: CNN Pipeline chuẩn — Template cho mọi bài toán ảnh

```
Input Image
  → [Conv → BN → ReLU → Pool] × N        ← Thu thập features
  → Global Average Pool (hoặc Flatten)     ← Chuyển sang classifier
  → [FC → ReLU → Dropout] × M             ← Phân loại
  → FC → Output                            ← Kết quả
```

```python
class SimpleCNN(nn.Module):
    """Template CNN tổng quát — dùng cho mọi bài toán ảnh."""
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            self._block(in_channels, 32),  # Block 1
            self._block(32, 64),           # Block 2
            self._block(64, 128),          # Block 3
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Bất kỳ size → (n, 128, 1, 1)
            nn.Flatten(),             # (n, 128, 1, 1) → (n, 128)
            nn.Linear(128, num_classes),
        )

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

> [!TIP] AdaptiveAvgPool2d(1) là gì?
> Bất kể feature map đang là 7×7 hay 5×5 hay 14×14, nó **rút gọn xuống 1×1** bằng cách lấy trung bình toàn bộ.
>
> Lợi ích: FC layer sau đó chỉ cần $128 \to 10 = 1,290$ params thay vì $128 \times 7 \times 7 \times 10 = 62,730$ params!

---

## 📖 Từ điển thuật ngữ

| Thuật ngữ                | Nghĩa dễ hiểu                                                      |
| ------------------------ | ------------------------------------------------------------------ |
| **LeNet-5**              | CNN đầu tiên thành công (1998, Yann LeCun). Đọc chữ tay trên ATM   |
| **Encoder**              | Phần Conv+Pool — trích xuất features từ ảnh. "Con mắt" của model   |
| **Classifier**           | Phần FC — phân loại. "Bộ não" quyết định                           |
| **Flatten**              | Duỗi tensor 3D (C,H,W) thành vector 1D. Nối Encoder → Classifier   |
| **Xavier init**          | Khởi tạo W sao cho variance input ≈ variance output                |
| **AdaptiveAvgPool2d(1)** | Pool toàn bộ H×W → 1×1. Thay thế Flatten+FC lớn                    |
| **Inductive bias**       | "Niềm tin" built-in (locality, invariance) giúp model học hiệu quả |
| **Feature hierarchy**    | Tầng thấp → edges, tầng giữa → textures, tầng cao → objects        |

---

## ✅ Bài tự kiểm tra

1. LeNet gồm mấy tầng Conv, mấy tầng FC? Tổng bao nhiêu tham số?
2. Input (1, 28, 28) đi qua `Conv2d(1, 6, 5, padding=2)` rồi `AvgPool2d(2)` → shape là gì?
3. Tại sao 96% tham số nằm ở FC mà Conv chỉ 4%? Cách giảm FC?
4. Modern LeNet thay đổi 4 thứ gì? Mỗi thứ có lợi ích gì?
5. CNN 62K params đạt 91%, MLP 200K params đạt 87%. Giải thích.
6. **Thực hành**: Copy code `ModernLeNet` ở trên, train trên Fashion-MNIST, in kết quả.

> [!NOTE]- 📝 Đáp án
> 1. **2 Conv + 3 FC**. Tổng **≈62K** tham số.
> 2. Conv: $(1,28,28) \to (6,28,28)$ (padding=2 giữ size). Pool: $(6,28,28) \to (6,14,14)$. **Đáp án: (6, 14, 14)**.
> 3. FC1 có $400 \times 120 = 48K$ params vì Flatten tạo vector dài 400. **Giảm bằng**: `nn.AdaptiveAvgPool2d(1)` rút feature map xuống 1×1 → FC chỉ cần 16→10 = 170 params.
> 4. | Thay đổi | Lợi ích |
>    | --- | --- |
>    | Sigmoid → **ReLU** | Gradient không vanish, train nhanh 5-10× |
>    | AvgPool → **MaxPool** | Giữ features nổi bật |
>    | + **BatchNorm** | Ổn định training, lr cao hơn |
>    | + **Dropout** | Chống overfitting |
> 5. CNN có **inductive bias đúng** cho ảnh: **(a)** locality — chỉ xét vùng nhỏ → ít params, **(b)** weight sharing — cùng kernel mọi vị trí, **(c)** hierarchical features — edges → textures → objects.
> 6. (Tự thực hành)

---

## 🔗 Liên kết

- **Buổi trước**: [[Buổi 27 - Tuần 8]] — Pooling & Multiple Channels
- **Buổi sau**: [[Buổi 29 - Tuần 8]] — AlexNet: Deep CNN & GPU Revolution
