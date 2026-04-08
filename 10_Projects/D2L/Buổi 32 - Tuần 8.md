---
title: "Buổi 32 - Tuần 8: Multi-Branch Networks (GoogLeNet)"
tags: [d2l, cnn, googlenet, inception, modern-cnn, multi-branch, study-note]
created: 2026-04-05
session: "D2L Tuần 8, Buổi 32 — 8.4 Multi-Branch Networks (GoogLeNet)"
d2l_section: "8.4"
source:
  - "https://d2l.ai/chapter_convolutional-modern/googlenet.html"
  - "30_Resources/Books/Dive Into Deep Learning/d2l-en/chapter_convolutional-modern/googlenet.md"
related:
  - "[[Buổi 31 - Tuần 8]]"
  - "[[Buổi 33 - Tuần 9]]"
aliases: ["GoogLeNet", "Inception", "8.4 GoogLeNet"]
---

# Buổi 32 — 8.4 Multi-Branch Networks (GoogLeNet)

> [!NOTE] ELI5
> Các mạng trước (AlexNet, VGG, NiN) đều phải **chọn** kernel size cố định: dùng 3x3? hay 5x5? hay 11x11? Mỗi kích thước nhìn thấy "chi tiết" ở mức khác nhau — 3x3 nhìn cận cảnh (texture, cạnh), 5x5 nhìn rộng hơn (hình dạng nhỏ), 11x11 nhìn toàn cảnh.
>
> GoogLeNet hỏi: **"Tại sao phải chọn? Dùng hết cùng lúc!"** Nó tạo ra **Inception block** — 4 nhánh chạy song song (1x1, 3x3, 5x5, MaxPool), mỗi nhánh nhìn ảnh ở scale khác nhau, rồi **ghép kết quả lại**. Giống 4 thợ thẩm định đồng thời: người nhìn gần, người nhìn xa, người nhìn trung bình — rồi tổng hợp ý kiến.

---

## 1. Bối cảnh — Tại sao cần Multi-Branch?

> [!NOTE] ELI5
> AlexNet dùng 11x11, VGG dùng toàn 3x3, NiN dùng các kernel khác nhau. Ai đúng? Câu trả lời: **tất cả đều đúng cho các mục đích khác nhau**. Chi tiết nhỏ (texture) cần filter nhỏ, chi tiết lớn (hình dạng tổng thể) cần filter lớn. GoogLeNet là mạng đầu tiên nói: "không cần chọn — dùng hết!".

Năm 2014, **GoogLeNet** (Szegedy et al., 2015) thắng ImageNet Challenge bằng kiến trúc **kết hợp sức mạnh** từ các mạng trước:
- Từ **NiN**: Conv 1x1 để giảm channels (bottleneck) + Global Average Pooling
- Từ **VGG**: Repeated blocks (sử dụng cùng block nhiều lần)
- Ý tưởng mới: **Song song nhiều kernel sizes** rồi concatenate

**GoogLeNet** (được viết với chữ "L" hoa để hướng về LeNet) cũng là mạng **đầu tiên** phân biệt rõ 3 phần trong CNN:

| Phần | Tên gọi | Vai trò | Ví dụ trong GoogLeNet |
| --- | --- | --- | --- |
| **Stem** | Data ingest | Tiền xử lý, trích low-level features | Conv 7x7, Conv 1x1+3x3 |
| **Body** | Data processing | Trích features phức tạp (main compute) | 9 Inception blocks |
| **Head** | Prediction | Phân loại cuối | GAP + FC |

> [!TIP] Design pattern Stem-Body-Head
> Pattern này trở thành **chuẩn mực** bất biến kể từ GoogLeNet. Mọi CNN hiện đại (ResNet, EfficientNet, ConvNeXt...) đều tổ chức theo 3 phần: Stem (vài conv đơn giản), Body (nhiều blocks lặp lại), Head (GAP + classifier). Khi đọc code bất kỳ mạng nào, hãy xác định 3 phần này trước.

**Đóng góp chính** (key contribution) của GoogLeNet: thiết kế **network body** bằng Inception blocks — giải quyết bài toán "chọn kernel size nào" theo cách khéo léo: **không chọn, concatenate hết**.

---

## 2. Inception Block — Khối xây dựng cốt lõi

> [!NOTE] ELI5
> Hãy tưởng tượng bạn thuê 4 thợ chụp ảnh cùng lúc:
> - Thợ 1: chụp **cận cảnh** (Conv 1x1 — nhìn từng pixel)
> - Thợ 2: chụp **trung cảnh** (Conv 3x3 — nhìn vùng 3x3)
> - Thợ 3: chụp **toàn cảnh** (Conv 5x5 — nhìn vùng rộng)
> - Thợ 4: chụp **nét nhất** (MaxPool — giữ chi tiết nổi bật nhất)
>
> Rồi bạn **ghép 4 bức ảnh lại** thành 1 bức ảnh tổng hợp. Mỗi thợ bắt được thông tin mà thợ khác bỏ sót!

**Inception block** là block conv đa nhánh (multi-branch): 4 nhánh xử lý **song song** cùng input, mỗi nhánh dùng kernel size khác nhau, rồi **concatenate** outputs theo chiều channels.

- **Đây là gì?** Một convolutional block gồm **4 nhánh song song**: (1) Conv 1x1, (2) Conv 1x1 + Conv 3x3, (3) Conv 1x1 + Conv 5x5, (4) MaxPool 3x3 + Conv 1x1. Outputs từ 4 nhánh được concatenate theo channel dimension.
- **Input/Output:** Input $(C, H, W)$ → Output $(c_1 + c_2 + c_3 + c_4, H, W)$. Spatial dimensions **giữ nguyên** (nhờ padding), chỉ channels thay đổi.
- **Tại sao cần?** Thay vì phải chọn 1 kernel size duy nhất (3x3 hay 5x5?), Inception block **dùng tất cả cùng lúc** → trích features ở **nhiều scale** đồng thời. Conv 1x1 trước Conv 3x3 và 5x5 đóng vai trò **bottleneck** — giảm channels để tiết kiệm compute.

![[assets/attachments/d2l-buoi-32/inception_block.png]]

### 2.1 Chi tiết 4 nhánh

| Nhánh | Cấu trúc | Vai trò | Tại sao cần Conv 1x1 trước? |
| --- | --- | --- | --- |
| **1** | Conv 1x1 | Trích features **xuyên channels** tại mỗi pixel | Không cần — đã là 1x1 |
| **2** | Conv 1x1 → Conv 3x3 (pad=1) | Trích features **spatial cục bộ** (3x3) | Giảm channels trước conv 3x3 → tiết kiệm compute |
| **3** | Conv 1x1 → Conv 5x5 (pad=2) | Trích features **spatial rộng** (5x5) | Giảm channels trước conv 5x5 → **rất quan trọng** vì 5x5 tốn 25x compute per channel |
| **4** | MaxPool 3x3 (pad=1, s=1) → Conv 1x1 | Giữ **features nổi bật nhất**, rồi điều chỉnh channels | Conv 1x1 sau MaxPool để kiểm soát output channels |

> [!IMPORTANT] Bottleneck — Vai trò cốt yếu của Conv 1x1
> Conv 1x1 trước Conv 3x3 và 5x5 là **chìa khóa** giúp GoogLeNet "rẻ" hơn VGG dù phức tạp hơn.
> 
> **Ví dụ cụ thể** — Nhánh 3 của Inception block đầu tiên (b3):
> - Không có bottleneck: Conv 5x5, 192→32 channels = $5^2 \times 192 \times 32 = 153{,}600$ params
> - Có bottleneck: Conv 1x1 192→16 ($1^2 \times 192 \times 16 = 3{,}072$) + Conv 5x5 16→32 ($5^2 \times 16 \times 32 = 12{,}800$) = **15,872 params** → giảm **~10x**!
>
> Đây chính là di sản Conv 1x1 từ NiN (Buổi 31) — ở đây nó được dùng chủ yếu để **giảm channels** (dimensionality reduction) thay vì thêm nonlinearity.

> [!question]- Tại sao padding khác nhau ở mỗi nhánh?
> Mục tiêu: **tất cả 4 nhánh** phải ra **cùng spatial dimensions** $(H, W)$ để có thể concatenate.
> - Conv 1x1: không cần padding (kernel 1x1, stride 1 → output = input size)
> - Conv 3x3, pad=1: $(H + 2 \times 1 - 3)/1 + 1 = H$ → giữ nguyên
> - Conv 5x5, pad=2: $(H + 2 \times 2 - 5)/1 + 1 = H$ → giữ nguyên
> - MaxPool 3x3, pad=1, stride=1: $(H + 2 \times 1 - 3)/1 + 1 = H$ → giữ nguyên
>
> Nếu bất kỳ nhánh nào ra size khác → `torch.cat` sẽ lỗi!

### 2.2 Implementation

```python
import torch
from torch import nn
from torch.nn import functional as F

class Inception(nn.Module):
    """Inception block — 4 nhánh song song, concatenate theo channels.
    
    Args:
        c1: output channels cho nhánh 1 (Conv 1x1)
        c2: tuple (reduce, out) cho nhánh 2 (Conv 1x1 → Conv 3x3)
        c3: tuple (reduce, out) cho nhánh 3 (Conv 1x1 → Conv 5x5)
        c4: output channels cho nhánh 4 (MaxPool → Conv 1x1)
    
    Total output channels = c1 + c2[1] + c3[1] + c4
    """
    def __init__(self, c1, c2, c3, c4):
        super().__init__()
        # Nhánh 1: Conv 1x1
        self.b1_1 = nn.LazyConv2d(c1, kernel_size=1)
        # Nhánh 2: Conv 1x1 (bottleneck) → Conv 3x3
        self.b2_1 = nn.LazyConv2d(c2[0], kernel_size=1)
        self.b2_2 = nn.LazyConv2d(c2[1], kernel_size=3, padding=1)
        # Nhánh 3: Conv 1x1 (bottleneck) → Conv 5x5
        self.b3_1 = nn.LazyConv2d(c3[0], kernel_size=1)
        self.b3_2 = nn.LazyConv2d(c3[1], kernel_size=5, padding=2)
        # Nhánh 4: MaxPool 3x3 → Conv 1x1
        self.b4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b4_2 = nn.LazyConv2d(c4, kernel_size=1)

    def forward(self, x):
        b1 = F.relu(self.b1_1(x))
        b2 = F.relu(self.b2_2(F.relu(self.b2_1(x))))
        b3 = F.relu(self.b3_2(F.relu(self.b3_1(x))))
        b4 = F.relu(self.b4_2(self.b4_1(x)))
        return torch.cat((b1, b2, b3, b4), dim=1)
```

> [!question]- Tại sao concatenate thay vì cộng (add)?
> Concatenate theo channel dimension giữ **tách biệt** thông tin từ mỗi nhánh — mạng có thể học cách sử dụng thông tin từ scale khác nhau một cách độc lập. Cộng (element-wise addition) sẽ trộn lẫn thông tin → mất khả năng phân biệt thông tin từ scale nào.
>
> Tuy nhiên, add cũng có ưu điểm: không tăng channels → ít params ở layer sau. **ResNet** (Buổi 34) sẽ dùng add thay vì concatenate — nhưng với mục đích khác (skip connections).

### 2.3 Hyperparameters — Phân bổ channels

Hyperparameter chính của Inception block: **phân bổ bao nhiêu channels cho mỗi nhánh?** Ví dụ Inception block đầu tiên trong b3 của GoogLeNet:

```python
Inception(c1=64, c2=(96, 128), c3=(16, 32), c4=32)
```

Phân tích:
- Nhánh 1: 64 channels (tỉ lệ: 2)
- Nhánh 2: 96→128 channels (tỉ lệ: 4) ← **chiếm nhiều nhất**
- Nhánh 3: 16→32 channels (tỉ lệ: 1)
- Nhánh 4: 32 channels (tỉ lệ: 1)
- **Tổng output**: $64 + 128 + 32 + 32 = 256$ channels
- **Tỉ lệ**: $2:4:1:1$

> [!TIP] Pattern phân bổ channels
> Nhánh 2 (Conv 3x3) luôn chiếm **nhiều channels nhất** — vì Conv 3x3 là "sweet spot" giữa receptive field và compute cost. Nhánh 1 (Conv 1x1) đứng thứ hai. Nhánh 3 (Conv 5x5) và 4 (MaxPool) ít nhất — Conv 5x5 tốn compute, MaxPool không trích features mới.
>
> Bottleneck reduction: Nhánh 2 giảm $192 \to 96$ ($\div 2$), nhánh 3 giảm $192 \to 16$ ($\div 12$). Nhánh 3 giảm **mạnh hơn nhiều** vì Conv 5x5 rất tốn compute.

---

## 3. GoogLeNet Model — Kiến trúc đầy đủ

> [!NOTE] ELI5
> GoogLeNet = 1 phần "nhận ảnh" (Stem) + 9 Inception blocks xếp thành 3 nhóm (Body) + GAP và 1 FC nhỏ (Head). Tổng cộng 22 layers sâu — gấp đôi VGG-11 — nhưng **ít parameters hơn** nhờ bottleneck Conv 1x1 và GAP thay FC layers lớn.

**GoogLeNet** gồm 3 phần rõ ràng: **Stem**, **Body**, **Head**.

- **Đây là gì?** Một CNN 22 layers sâu với 9 Inception blocks, sử dụng multi-scale feature extraction song song + GAP.
- **Input/Output:** Input $1 \times 96 \times 96$ (grayscale, resize từ 28x28 cho Fashion-MNIST) → Output vector 10 chiều (logits cho 10 classes).
- **Tại sao quan trọng?** GoogLeNet **rẻ hơn** VGG mà **chính xác hơn** — nhờ bottleneck Conv 1x1 giảm compute. Đây là bắt đầu xu hướng "thiết kế mạng hiệu quả" (efficient network design).

![[assets/attachments/d2l-buoi-32/googlenet_architecture.png]]

![[Pasted image 20260406143102.png]]
### 3.1 Stem (b1 + b2) — Tiền xử lý

```python
class GoogleNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            self._b1(), self._b2(), self._b3(), 
            self._b4(), self._b5(),
            nn.LazyLinear(num_classes)
        )
    
    def _b1(self):
        """Stem part 1: giống AlexNet — Conv 7x7 giảm resolution nhanh."""
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    def _b2(self):
        """Stem part 2: Conv 1x1 bottleneck + Conv 3x3 tăng channels.
        Giống nhánh 2 trong Inception block — 64→192 channels."""
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=1), nn.ReLU(),
            nn.LazyConv2d(192, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    def forward(self, x):
        return self.net(x)
```

> [!question]- Tại sao Stem không dùng Inception blocks?
> Ở giai đoạn đầu, ảnh còn resolution cao (96x96) → compute per pixel lớn. Inception block với 4 nhánh sẽ **quá tốn** ở resolution này. Stem dùng conv lớn (7x7) + pool để **giảm resolution nhanh** trước khi vào Inception blocks. Đây là lý do chuẩn: stem luôn đơn giản, body mới phức tạp.

### 3.2 Body (b3 + b4 + b5) — 9 Inception blocks

```python
    def _b3(self):
        """Body group 1: 2 Inception blocks + MaxPool."""
        return nn.Sequential(
            Inception(64, (96, 128), (16, 32), 32),   # 256ch
            Inception(128, (128, 192), (32, 96), 64),  # 480ch
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    def _b4(self):
        """Body group 2: 5 Inception blocks + MaxPool.
        Phần nặng nhất — channels tăng từ 512 → 832."""
        return nn.Sequential(
            Inception(192, (96, 208), (16, 48), 64),   # 512ch
            Inception(160, (112, 224), (24, 64), 64),   # 512ch
            Inception(128, (128, 256), (24, 64), 64),   # 512ch
            Inception(112, (144, 288), (32, 64), 64),   # 528ch
            Inception(256, (160, 320), (32, 128), 128),  # 832ch
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    def _b5(self):
        """Body group 3: 2 Inception blocks + GAP (head)."""
        return nn.Sequential(
            Inception(256, (160, 320), (32, 128), 128),  # 832ch
            Inception(384, (192, 384), (48, 128), 128),  # 1024ch
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten())
```

**Channels progression qua 9 Inception blocks:**

| Block | Group | Output channels | Tính toán |
| --- | --- | --- | --- |
| Inception 1 | b3 | 256 | 64+128+32+32 |
| Inception 2 | b3 | 480 | 128+192+96+64 |
| Inception 3 | b4 | 512 | 192+208+48+64 |
| Inception 4 | b4 | 512 | 160+224+64+64 |
| Inception 5 | b4 | 512 | 128+256+64+64 |
| Inception 6 | b4 | 528 | 112+288+64+64 |
| Inception 7 | b4 | 832 | 256+320+128+128 |
| Inception 8 | b5 | 832 | 256+320+128+128 |
| Inception 9 | b5 | 1024 | 384+384+128+128 |

### 3.3 Head — GAP + 1 FC

GoogLeNet kết thúc bằng **Global Average Pooling** (giống NiN) → Flatten → **1 FC layer** (khác NiN: NiN dùng conv 1x1 + GAP, không có FC). FC duy nhất này chỉ có $1024 \times 10 = 10{,}240$ params — so với VGG FC1 alone: ~103M params!

### 3.4 Data flow analysis

```python
model = GoogleNet()
X = torch.randn(1, 1, 96, 96)

for i, block in enumerate(model.net):
    X = block(X)
    print(f"Block {i} ({block.__class__.__name__:12s}) -> {str(X.shape):30s}")
```

```
Block 0 (Sequential ) -> torch.Size([1, 64, 24, 24])      <- b1 (Stem)
Block 1 (Sequential ) -> torch.Size([1, 192, 12, 12])     <- b2 (Stem)
Block 2 (Sequential ) -> torch.Size([1, 480, 6, 6])       <- b3 (Body: 2 Inception)
Block 3 (Sequential ) -> torch.Size([1, 832, 3, 3])       <- b4 (Body: 5 Inception)
Block 4 (Sequential ) -> torch.Size([1, 1024])             <- b5 (Body: 2 Inception + GAP)
Block 5 (LazyLinear ) -> torch.Size([1, 10])               <- FC (Head)
```

> [!TIP] Patterns
> 1. **Spatial**: 96 → 24 → 12 → 6 → 3 → **1** (giảm qua MaxPool giữa groups + GAP cuối)
> 2. **Channels**: 64 → 192 → 480 → 832 → **1024** (tăng dần, mỗi group tăng đáng kể)
> 3. **Stem** giảm resolution nhanh (96→24→12) trước khi vào Body → tiết kiệm compute
> 4. **b4 nặng nhất**: 5 Inception blocks tại resolution 6x6 — đây là nơi tính toán chính

---

## 4. So sánh GoogLeNet vs các mạng trước

### 4.1 Bảng tổng hợp

| Tiêu chí | AlexNet | VGG-16 | NiN | **GoogLeNet** |
| --- | --- | --- | --- | --- |
| **Năm** | 2012 | 2014 | 2013 | **2014** |
| **Depth** | 8 layers | 16 layers | ~12 conv layers | **22 layers** |
| **Block type** | Ad-hoc | VGG block (conv 3x3) | NiN block (mlpconv) | **Inception block** (4 nhánh) |
| **Kernel sizes** | 11, 5, 3 | Toàn 3x3 | 11, 5, 3 + Conv 1x1 | **1x1, 3x3, 5x5 song song** |
| **FC layers** | 3 (nặng) | 3 (rất nặng) | 0 (GAP) | **1** (nhẹ, sau GAP) |
| **Total params** | ~62M | ~138M | ~2M | **~5M** |
| **Conv 1x1** | Khong | Khong | Nonlinearity | **Bottleneck** (giảm channels) |
| **Architecture** | Sequential | Sequential | Sequential | **Multi-branch parallel** |
| **Design insight** | Deep CNN works | Deep + narrow + blocks | No FC, GAP | **Multi-scale + efficient** |

> [!IMPORTANT] GoogLeNet "rẻ" hơn VGG nhưng tốt hơn
> GoogLeNet chỉ có **~5M params** (so với VGG-16: ~138M, giảm **~28x**!) mà đạt accuracy cao hơn trên ImageNet 2014. Bí quyết:
> 1. **Bottleneck Conv 1x1**: giảm channels trước conv lớn → giảm compute
> 2. **GAP thay FC layers**: bỏ ~120M params
> 3. **Multi-branch**: trích features đa scale hiệu quả hơn single-scale

---

## 5. Training trên Fashion-MNIST

d2l.ai giảm resolution xuống **96x96** (thay vì 224x224) để tiết kiệm compute cho Fashion-MNIST.

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((96, 96)),   # 96x96 thay vì 224x224
    transforms.ToTensor(),
])

train_data = datasets.FashionMNIST('./data', train=True,
                                    download=True, transform=transform)
test_data  = datasets.FashionMNIST('./data', train=False,
                                    transform=transform)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=128)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GoogleNet().to(device)

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
model.apply(init_weights)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        y_hat = model(X)
        loss = F.cross_entropy(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += (y_hat.argmax(1) == y).sum().item()
        total += y.size(0)
    
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            test_correct += (model(X).argmax(1) == y).sum().item()
    
    print(f"Epoch {epoch+1:2d} | "
          f"Loss: {total_loss/total:.4f} | "
          f"Train: {correct/total*100:.1f}% | "
          f"Test: {test_correct/len(test_data)*100:.1f}%")
```

> [!TIP] Phân tích training
> 1. **Resolution 96x96** thay vì 224x224 — giảm compute ~5x, đủ cho Fashion-MNIST.
> 2. **lr = 0.01** — giống VGG. GoogLeNet có ít params nên có thể thử lr cao hơn (0.05).
> 3. **Accuracy kỳ vọng**: ~91-92% test accuracy — tương đương VGG nhưng **nhanh hơn** vì: (a) input nhỏ hơn (96 vs 224), (b) ít params hơn (5M vs 138M).

---

## 6. Discussion

### 6.1 Tại sao GoogLeNet đánh dấu bước ngoặt?

d2l.ai nhận định: GoogLeNet đánh dấu bắt đầu xu hướng thiết kế mạng **có chủ đích** (deliberate design) — **trade-off** giữa chi phí tính toán (compute cost) và độ chính xác (accuracy). Trước GoogLeNet, mạng mới thường lớn hơn = tốt hơn (AlexNet → VGG). GoogLeNet chứng minh rằng **thiết kế thông minh** có thể vừa nhỏ hơn vừa tốt hơn.

### 6.2 Hạn chế

1. **Quá nhiều hyperparameters "tùy ý"**: Channels cho mỗi nhánh, số blocks mỗi group, tỉ lệ bottleneck... — tất cả đều được chọn thủ công qua thực nghiệm (brute-force, genetic algorithms). Chưa có công cụ **tự động** thiết kế.

2. **Chưa có batch normalization**: Training 22 layers sâu mà không có BN rất khó ổn định. Paper gốc dùng "auxiliary classifiers" (loss functions phụ ở giữa mạng) để giúp gradient flow — trick này **không còn cần thiết** khi có BN (Ioffe & Szegedy, 2015 — cùng nhóm tác giả!).

3. **Chưa có residual connections**: Giống VGG và NiN, vẫn gặp degradation problem khi tăng depth quá mức.

### 6.3 Các phiên bản sau (Inception v2, v3, v4)

| Version | Cải tiến | Paper |
| --- | --- | --- |
| **Inception v2** | Thêm Batch Normalization | Ioffe & Szegedy, 2015 |
| **Inception v3** | Factorize 5x5 → 2x 3x3, asymmetric convolutions | Szegedy et al., 2016 |
| **Inception v4** | Thêm residual connections (Inception-ResNet) | Szegedy et al., 2017 |

### 6.4 Nhìn về phía trước

Các chương tiếp theo sẽ giới thiệu:
- **Batch Normalization** (Buổi 33): ổn định training cho mạng sâu
- **ResNet** (Buổi 34): residual connections giải quyết degradation problem
- Cả hai đều cải thiện trực tiếp GoogLeNet.

---

## 7. Exercises (từ d2l.ai)

1. **Cải tiến GoogLeNet:** Thử implement các phiên bản sau:
   - (a) Thêm Batch Normalization
   - (b) Thay đổi Inception block (width, thứ tự convolutions)
   - (c) Dùng label smoothing cho regularization
   - (d) Thêm residual connections

2. **Minimum image size?** Tính resolution tối thiểu để GoogLeNet hoạt động (không bị spatial dimension = 0).

3. **Thiết kế GoogLeNet cho 28x28** (Fashion-MNIST gốc): Cần thay đổi Stem, Body, Head thế nào?

4. **So sánh params**: AlexNet vs VGG vs NiN vs GoogLeNet. NiN và GoogLeNet giảm params bằng cách nào?

5. **So sánh compute**: GoogLeNet vs AlexNet — ảnh hưởng thế nào đến thiết kế chip (memory size, bandwidth, cache, specialized operations)?

---

## Từ điển thuật ngữ

| Thuật ngữ | Nghĩa tiếng Việt | Chi tiết |
| --- | --- | --- |
| **GoogLeNet** | Google + LeNet | Tên mạng, viết "L" hoa hướng về LeNet |
| **Inception block** | Khối Inception | Block 4 nhánh song song, tên từ phim "Inception" ("we need to go deeper") |
| **Multi-branch** | Đa nhánh | Kiến trúc xử lý song song nhiều nhánh rồi ghép kết quả |
| **Bottleneck** | Cổ chai | Conv 1x1 giảm channels trước conv lớn (3x3, 5x5) để tiết kiệm compute |
| **Channel concatenation** | Ghép kênh | Nối outputs theo channel dimension: `torch.cat(..., dim=1)` |
| **Stem** | Thân gốc | Phần đầu CNN: tiền xử lý, trích low-level features, giảm resolution |
| **Body** | Thân chính | Phần giữa CNN: trích features phức tạp (Inception blocks) |
| **Head** | Đầu (phân loại) | Phần cuối CNN: GAP + classifier |
| **Auxiliary classifier** | Bộ phân loại phụ | Loss functions phụ ở giữa mạng — giúp gradient flow (không còn cần nhờ có BN) |
| **Deliberate design** | Thiết kế có chủ đích | Xu hướng trade-off compute vs accuracy thay vì chỉ tăng size |
| **Label smoothing** | Làm mượt nhãn | Kỹ thuật regularization: thay one-hot bằng soft targets |

---

## Bài tự kiểm tra

1. Inception block có bao nhiêu nhánh? Mỗi nhánh dùng gì?
2. Tại sao Conv 1x1 trước Conv 3x3 và 5x5 quan trọng? Cho ví dụ giảm params cụ thể.
3. Tại sao 4 nhánh cần cùng spatial output? Cách đạt được?
4. GoogLeNet chia mạng thành 3 phần gì? Vai trò từng phần?
5. So sánh params: VGG-16 (~138M) vs GoogLeNet (~5M). GoogLeNet giảm bằng cách nào?
6. GoogLeNet dùng concatenate hay add để ghép branches? Tại sao?
7. Tại sao Stem không dùng Inception blocks?
8. Kể 2 hạn chế của GoogLeNet gốc mà các phiên bản sau cải thiện.

> [!NOTE]- Đáp án gợi ý
> 1. 4 nhánh: (1) Conv 1x1, (2) Conv 1x1 + Conv 3x3, (3) Conv 1x1 + Conv 5x5, (4) MaxPool 3x3 + Conv 1x1.
> 2. Conv 1x1 **giảm channels** (bottleneck) trước conv lớn. Ví dụ: Conv 5x5 trực tiếp 192→32 = 153,600 params. Có bottleneck 192→16→32 = 15,872 params → giảm ~10x.
> 3. Phải cùng $(H, W)$ để `torch.cat` theo dim=1. Đạt được bằng padding: Conv 3x3 pad=1, Conv 5x5 pad=2, MaxPool 3x3 pad=1 stride=1.
> 4. **Stem** (data ingest): Conv 7x7 + Conv 1x1+3x3, giảm resolution nhanh. **Body** (data processing): 9 Inception blocks, trích features. **Head** (prediction): GAP + 1 FC.
> 5. VGG ~138M, GoogLeNet ~5M (giảm ~28x). Cách: (a) Bottleneck Conv 1x1 giảm compute, (b) GAP thay 3 FC layers (~120M), (c) Chỉ 1 FC nhỏ (1024x10 = 10K params).
> 6. **Concatenate** theo channels (`torch.cat`). Lý do: giữ tách biệt thông tin từ mỗi scale → mạng có thể học sử dụng từng scale riêng.
> 7. Stem ở resolution cao (96x96), Inception 4 nhánh sẽ rất tốn compute. Stem dùng conv lớn + pool để giảm resolution nhanh trước.
> 8. (a) Chưa có Batch Normalization → training khó ổn định (dùng auxiliary classifiers thay). (b) Chưa có residual connections → degradation problem ở depth rất sâu.

---

## Liên kết

- **Buổi trước**: [[Buổi 31 - Tuần 8]] — 8.3 NiN: Conv 1x1 + GAP
- **Buổi sau**: [[Buổi 33 - Tuần 9]] — 8.5 Batch Normalization
- **Concepts**: [[Activation Function]], [[Dropout]]
- **Source**: [d2l.ai — 8.4 GoogLeNet](https://d2l.ai/chapter_convolutional-modern/googlenet.html)
