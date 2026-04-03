---
title: "Buổi 30 - Tuần 8: Networks Using Blocks (VGG)"
tags: [d2l, cnn, vgg, modern-cnn, blocks, architecture-design, study-note]
created: 2026-04-02
modified: 2026-04-03
session: "D2L Tuần 8, Buổi 30 — 8.2 Networks Using Blocks (VGG)"
d2l_section: "8.2"
source:
  - "https://d2l.ai/chapter_convolutional-modern/vgg.html"
  - "30_Resources/Books/Dive Into Deep Learning/d2l-en/chapter_convolutional-modern/vgg.md"
related:
  - "[[Buổi 29 - Tuần 8]]"
  - "[[Buổi 31 - Tuần 8]]"
aliases: ["VGG", "VGGNet", "8.2 VGG"]
---

# Buổi 30 — 8.2 Networks Using Blocks (VGG)

> [!NOTE] ELI5
> Hãy tưởng tượng bạn xây nhà bằng LEGO. 
> - **AlexNet** giống xây nhà bằng cách đặt **từng viên gạch** — mỗi viên có hình dạng khác nhau (11×11, 5×5, 3×3…), rất khó lặp lại và không rõ vì sao chọn viên nào.
> - **VGG** giống xây nhà bằng **các khối module giống nhau** — cứ lặp đi lặp lại: 2 viên gạch nhỏ + 1 tấm sàn, 2 viên gạch nhỏ + 1 tấm sàn… Đơn giản, dễ hiểu, dễ mở rộng.
> 
> Ý tưởng cốt lõi của VGG: **thay vì thiết kế từng layer riêng lẻ, hãy thiết kế "blocks" và lặp lại chúng.** Ý tưởng này tưởng đơn giản nhưng thay đổi cách thiết kế mọi mạng CNN về sau — kể cả ResNet, GoogLeNet, DenseNet.

---

## 1. Bối cảnh — Từ AlexNet đến VGG

> [!NOTE] ELI5
> AlexNet (Buổi 29) chứng minh CNN sâu hoạt động tốt. Nhưng AlexNet được thiết kế theo kiểu "thủ công": mỗi layer có kernel size, số channels, stride **khác nhau** — không có pattern rõ ràng. Nếu ai muốn thay đổi hay mở rộng AlexNet, không biết nên thêm/bớt gì.
>
> VGG đặt câu hỏi: **Có thể thiết kế CNN theo một nguyên tắc đơn giản, có quy luật không?** Và câu trả lời là có.

AlexNet đã cho thấy **empirical evidence** rằng deep CNNs đạt kết quả tốt, nhưng nó **không cung cấp template chung** để hướng dẫn các nhà nghiên cứu sau thiết kế mạng mới. Nhìn lại kiến trúc AlexNet: Conv1 dùng kernel 11×11, Conv2 dùng 5×5, Conv3-5 dùng 3×3, stride khác nhau ở mỗi tầng — **tại sao?** Không có lý do rõ ràng ngoài trial-and-error. Nếu ai muốn làm mạng "sâu hơn AlexNet một chút" hoặc "rộng hơn AlexNet một chút", không có hướng dẫn nào để biết nên thêm layer ở đâu, kernel size bao nhiêu.

Đây chính là vấn đề mà VGG giải quyết: đưa ra một **nguyên tắc thiết kế có hệ thống** (principled design) cho CNN.

### 1.1 Phép loại suy: VLSI trong thiết kế chip

Sự phát triển của thiết kế kiến trúc neural network **phản ánh** tiến triển trong VLSI (Very Large Scale Integration) trong thiết kế chip (Mead, 1980). Đây không phải trùng hợp — cả hai đều bắt đầu từ thành phần nhỏ nhất, rồi dần dần trừu tượng hóa lên:

| Thế hệ | Chip Design | Neural Network Design | Tương đồng |
| --- | --- | --- | --- |
| 1 | Đặt từng **transistor** | Đặt từng **neuron** | Thao tác ở mức primitives |
| 2 | Nhóm thành **logic elements** (AND, OR) | Nhóm thành **layers** (Conv, FC) | Abstraction đầu tiên |
| 3 | Nhóm thành **logic blocks** (ALU, Register) | Nhóm thành **blocks** (VGG!) | Pattern lặp lại |
| 4 | **IP cores** tái sử dụng (CPU core, GPU core) | **Foundation models** (GPT, ViT) | Tái sử dụng toàn bộ thành phần |

> [!question]- ❓ Tại sao phép loại suy này quan trọng?
> Trong chip design, việc chuyển từ transistors sang logic blocks giúp **tăng độ phức tạp** mà vẫn **quản lý được**. Tương tự, VGG cho phép xây mạng sâu hơn AlexNet rất nhiều (11-19 layers thay vì 8) mà vẫn **dễ hiểu, dễ implement, dễ modify**.
> 
> Đây là bước chuyển paradigm: từ "suy nghĩ ở mức layer" sang "suy nghĩ ở mức block". Mọi kiến trúc CNN hiện đại sau VGG (Inception blocks trong GoogLeNet, Residual blocks trong ResNet, Dense blocks trong DenseNet…) đều dựa trên ý tưởng blocks này.

> [!TIP] Insight quan trọng
> Gần đây (thập niên 2020), paradigm đã tiến thêm một bước: thay vì thiết kế blocks mới, người ta **tái sử dụng toàn bộ model đã train sẵn** (**foundation models**) — GPT, ViT, CLIP — rồi fine-tune cho task mới. Đây là tương đương với việc tái sử dụng IP cores trong chip design (Bommasani et al., 2021).

Ý tưởng block-based design xuất phát từ nhóm **Visual Geometry Group (VGG)** tại Đại học Oxford, trong mạng mang tên chính nhóm — **VGG network** (Simonyan & Zisserman, 2014).

---

## 2. VGG Blocks — Khối xây dựng cơ bản

> [!NOTE] ELI5
> Một "VGG block" cực đơn giản: **vài conv 3×3 + 1 MaxPool**. Chỉ vậy thôi! Nhưng sức mạnh nằm ở chỗ: khi xếp chồng nhiều blocks giống nhau, ta được một mạng rất sâu và rất mạnh — giống cách xếp nhiều tầng nhà giống nhau tạo nên tòa chung cư. Mà quan trọng nhất: bạn chỉ cần thiết kế **1 block**, rồi lặp lại.

### 2.1 Vấn đề với cách tiếp cận cũ

Khối xây dựng cơ bản của CNN truyền thống là chuỗi: (i) conv layer + (ii) activation (nonlinearity) + (iii) pooling layer. Mỗi pooling layer **giảm resolution** (thường giảm 2×).

Vấn đề: nếu mỗi conv layer đi kèm 1 pooling, spatial resolution giảm **quá nhanh**. Cụ thể, sau $k$ pooling layers (mỗi cái giảm 2×), resolution từ $d$ giảm xuống $d / 2^k$. Với ảnh ImageNet 224×224:

$$224 \div 2^k = \begin{cases} 112 & k=1 \\ 56 & k=2 \\ 28 & k=3 \\ 14 & k=4 \\ 7 & k=5 \\ 3.5 & k=6 \\ 1.75 & k=7 \end{cases}$$

Sau $k = \log_2(224) \approx 7{-}8$ lần pool → resolution hết! Điều này **giới hạn cứng** (hard limit) số conv layers tối đa trong mạng (~7-8 layers). AlexNet với 5 conv layers + 3 pools đã gần chạm giới hạn này.

> [!IMPORTANT] Insight cốt lõi — Giải pháp của VGG
> **Dùng nhiều convolutions liên tiếp giữa mỗi lần downsampling (pooling).**
> 
> Thay vì: `Conv → Pool → Conv → Pool → Conv → Pool`
> VGG dùng: `Conv → Conv → Pool → Conv → Conv → Pool`
> 
> Cách này cho phép mạng **sâu hơn nhiều** mà không giảm resolution quá nhanh. Với 5 lần pool (resolution từ 224 → 7), ta có thể nhét 2-4 conv layers vào mỗi khoảng giữa → tổng 8-20 conv layers!

### 2.2 Deep and Narrow > Shallow and Wide

Đây là câu hỏi trung tâm mà VGG đặt ra và trả lời một cách thuyết phục: **Mạng sâu (deep) với kernel nhỏ tốt hơn mạng nông (shallow) với kernel lớn.**

Ý tưởng then chốt (key insight): **hai conv 3×3 liên tiếp "nhìn" cùng vùng pixels như một conv 5×5** — nhưng với ít tham số hơn và nhiều tính phi tuyến hơn.

#### Receptive Field Analysis

![[assets/attachments/d2l-buoi-30/receptive_field_comparison.png]]

**Chứng minh receptive field tương đương:**

- Conv 3×3 đầu tiên: mỗi pixel output "nhìn" 3×3 = 9 pixels đầu vào
- Conv 3×3 thứ hai: mỗi pixel output "nhìn" 3×3 pixels của feature map trước → mỗi pixel đó lại "nhìn" 3×3 pixels gốc → tổng cộng "nhìn" $(3 + 3 - 1) \times (3 + 3 - 1) = 5 \times 5$ pixels gốc

Công thức tổng quát: Receptive field sau $n$ conv $k \times k$ liên tiếp (stride 1):

$$r = n(k - 1) + 1$$

Kiểm tra:
- $n=2, k=3$: $r = 2 \times 2 + 1 = 5$ → receptive field $5 \times 5$ ✓
- $n=3, k=3$: $r = 3 \times 2 + 1 = 7$ → receptive field $7 \times 7$ ✓

#### So sánh chi phí tham số

| Cấu hình | Receptive field | Parameters ($c$ channels) | ReLU activations | Tiết kiệm params |
| --- | --- | --- | --- | --- |
| 1× Conv $5 \times 5$ | $5 \times 5$ | $25 c^2$ | 1 | — |
| **2× Conv $3 \times 3$** | $5 \times 5$ | $2 \times 9 c^2 = 18 c^2$ | **2** | **28% ít hơn** |
| 1× Conv $7 \times 7$ | $7 \times 7$ | $49 c^2$ | 1 | — |
| **3× Conv $3 \times 3$** | $7 \times 7$ | $3 \times 9 c^2 = 27 c^2$ | **3** | **45% ít hơn** |

> [!question]- ❓ Tại sao công thức parameters là $k^2 \cdot c^2$?
> Mỗi conv layer với $c$ input channels và $c$ output channels có kernel size $k \times k$. Mỗi output channel cần 1 filter kích thước $k \times k \times c$, và có $c$ output channels → tổng parameters = $k^2 \times c \times c = k^2 c^2$ (bỏ qua bias).
> 
> Ví dụ: Conv 3×3 với 256 channels: $9 \times 256^2 = 589{,}824$ params.
> Conv 5×5 với 256 channels: $25 \times 256^2 = 1{,}638{,}400$ params — gấp gần 3×!

#### Tại sao nhiều conv 3×3 tốt hơn 1 conv lớn? — Ba lý do

**Lý do 1: Ít tham số hơn** — Như bảng trên, 2× Conv 3×3 dùng $18c^2$ params, ít hơn 1× Conv 5×5 ($25c^2$) đến **28%**. Ba Conv 3×3 tiết kiệm **45%** so với 1 Conv 7×7. Ít params → ít memory, ít compute, ít khả năng overfit.

**Lý do 2: Nhiều ReLU hơn → Model "biểu cảm" hơn** — Mỗi conv đi kèm 1 ReLU activation. Hai conv 3×3 → 2 lần ReLU, trong khi 1 conv 5×5 → chỉ 1 lần ReLU. Thêm ReLU nghĩa là thêm **tính phi tuyến (nonlinearity)** vào model → model có thể biểu diễn các hàm phức tạp hơn (expressive power cao hơn).

> [!question]- ❓ Tại sao thêm ReLU lại tăng expressive power?
> Nếu không có activation, 2 conv liên tiếp chỉ là 1 linear transformation (nhân ma trận 2 lần = nhân ma trận 1 lần). Thêm ReLU giữa chúng **phá vỡ** tính tuyến tính này — mỗi ReLU tạo ra một "ngưỡng quyết định" (decision boundary) mới. Nhiều ReLU = nhiều ngưỡng = khả năng phân biệt features phức tạp hơn.
> 
> Theo lý thuyết Universal Approximation: mạng sâu hơn (nhiều layers + activations hơn) cần **ít neurons hơn** để xấp xỉ cùng một hàm so với mạng nông rộng.

**Lý do 3: Regularize tốt hơn** — Nhiều layers nhỏ → cùng receptive field nhưng mỗi layer bị "ép" phải học representation nhỏ gọn hơn (do ít params mỗi layer). Điều này hoạt động như **implicit regularization** — giảm overfit mà không cần thêm Dropout hay Weight Decay.

> [!TIP] Kết luận quan trọng từ VGG paper
> Trong phân tích chi tiết, Simonyan & Zisserman chứng minh rằng deep and narrow networks **significantly outperform** shallow counterparts. Phát hiện này đưa deep learning vào cuộc đua "ai sâu hơn" — mạng 100+ layers (ResNet) trở thành chuẩn mực.
> 
> **Stacking conv 3×3** trở thành **gold standard** trong thiết kế CNN sâu — một quyết định thiết kế **chỉ mới bị xem xét lại gần đây** bởi ConvNeXt (Liu et al., 2022), khi họ quay lại thử kernel 7×7 kết hợp với các kỹ thuật hiện đại khác. Các framework GPU hiện đại cũng được tối ưu riêng cho phép nhân conv 3×3 nhỏ (Lavin & Gray, 2016).

### 2.3 Cấu trúc VGG Block

Một VGG block gồm 2 thành phần:
1. **Chuỗi $n$ conv $3 \times 3$**, padding 1 (giữ nguyên height × width) + ReLU sau mỗi conv
2. **Một MaxPool $2 \times 2$**, stride 2 (giảm height × width còn một nửa)

![[assets/attachments/d2l-buoi-30/vgg_block_detail.png]]

**Implementation trong PyTorch:**

```python
import torch
from torch import nn

def vgg_block(num_convs: int, out_channels: int) -> nn.Sequential:
    """Tạo một VGG block: num_convs lớp Conv 3×3 + MaxPool.
    
    Args:
        num_convs: Số conv layers trong block
        out_channels: Số output channels cho mỗi conv
    
    Returns:
        nn.Sequential chứa [Conv+ReLU] × num_convs + MaxPool
    """
    layers = []
    for _ in range(num_convs):
        layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)
```

> [!question]- ❓ Tại sao dùng `nn.LazyConv2d` thay vì `nn.Conv2d`?
> `LazyConv2d` không cần chỉ định `in_channels` — nó tự suy ra từ dữ liệu đầu tiên đi qua. Rất tiện khi xếp nhiều blocks lại vì không cần tính thủ công `in_channels` cho từng layer.
> 
> Trong production code, ta thường dùng `nn.Conv2d` với `in_channels` tường minh để code rõ ràng hơn.

> [!question]- ❓ Tại sao padding=1 với kernel 3×3?
> Công thức output size: $\text{out} = \frac{\text{in} + 2p - k}{s} + 1$
> 
> Với $k=3, p=1, s=1$: $\text{out} = \frac{\text{in} + 2 - 3}{1} + 1 = \text{in}$
> 
> → Conv 3×3 pad 1 **giữ nguyên** spatial dimensions. Chỉ MaxPool mới giảm size.
> 
> Đây là thiết kế có chủ đích: **tách biệt nhiệm vụ** — Conv lo "trích features", MaxPool lo "giảm resolution". Mỗi thành phần có 1 vai trò rõ ràng.

---

## 3. VGG Network — Kiến trúc đầy đủ

> [!NOTE] ELI5
> VGG Network = vài VGG blocks xếp chồng + 3 FC layers (giống hệt AlexNet). Điểm đặc biệt nhất: bạn chỉ cần **1 list nhỏ** (gọi là `arch`) để mô tả **toàn bộ** kiến trúc. Ví dụ: `[(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)]` — đọc là "block 1 có 1 conv, 64 channels; block 2 có 1 conv, 128 channels; block 3 có 2 convs, 256 channels; …". Chỉ 1 dòng Python mô tả hàng triệu tham số!

VGG Network, giống AlexNet và LeNet, gồm 2 phần chính:
1. **Convolutional part**: Nhiều VGG blocks nối tiếp — trích xuất features phân cấp
2. **Fully connected part**: 3 FC layers (giống **hệt** AlexNet) — phân loại

Khác biệt mấu chốt so với AlexNet: conv layers được **nhóm trong các blocks** — mỗi block giữ nguyên spatial dimensions (nhờ padding=1), chỉ có MaxPool cuối mỗi block mới giảm resolution.

### 3.1 VGG-11 — Phiên bản gốc

![[assets/attachments/d2l-buoi-30/vgg11_architecture.png]]

VGG-11 (còn gọi là VGG-A trong paper gốc) là phiên bản đơn giản nhất của họ VGG:

```python
# arch = [(num_convs, channels), ...]
VGG_11_arch = [
    (1, 64),    # Block 1: 1 conv, 64 channels  → 224→112
    (1, 128),   # Block 2: 1 conv, 128 channels → 112→56
    (2, 256),   # Block 3: 2 convs, 256 channels → 56→28
    (2, 512),   # Block 4: 2 convs, 512 channels → 28→14
    (2, 512),   # Block 5: 2 convs, 512 channels → 14→7
]
# Tổng conv layers: 1+1+2+2+2 = 8
# + 3 FC layers = 11 layers → VGG-"11"
```

**Design patterns quan trọng:**

1. **Block đầu**: ít conv, ít channels (vì ảnh còn lớn → mỗi conv tốn nhiều compute)
2. **Block sau**: nhiều conv hơn, nhiều channels hơn (ảnh nhỏ hơn → conv tốn ít compute hơn, nhưng cần nhiều channels để biểu diễn features phức tạp)
3. **Channels gấp đôi** sau mỗi block: 64 → 128 → 256 → 512 → 512

> [!question]- ❓ Tại sao channels gấp đôi mỗi lần spatial giảm 2×?
> Khi MaxPool giảm height và width đi 2× → lượng thông tin spatial giảm 4× (vì diện tích giảm $2 \times 2 = 4$ lần). Để **bù đắp** thông tin bị mất, ta tăng số channels lên 2× → tổng "capacity" ($\text{channels} \times H \times W$) giảm chậm hơn, giữ được nhiều thông tin hơn.
> 
> Tuy nhiên, sau 512 channels, channels **không tăng nữa** — vì lúc này tính toán đã rất nặng ($512 \times 7 \times 7 = 25{,}088$ phần tử mỗi sample).

**Full implementation:**

```python
class VGG(nn.Module):
    """VGG network — xây dựng từ arch specification.
    
    Arch là list of tuples: [(num_convs, channels), ...]
    Mỗi tuple định nghĩa 1 VGG block.
    """
    def __init__(self, arch, num_classes=10):
        super().__init__()
        # Phần conv: xếp chồng các VGG blocks
        conv_blks = []
        for (num_convs, out_channels) in arch:
            conv_blks.append(vgg_block(num_convs, out_channels))
        
        self.net = nn.Sequential(
            *conv_blks,          # Tất cả VGG blocks
            nn.Flatten(),        # Flatten: (512, 7, 7) → 25088
            # FC layers — giống AlexNet
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
            nn.LazyLinear(num_classes),
        )
    
    def forward(self, x):
        return self.net(x)
```

> [!IMPORTANT] Ý nghĩa của biến `arch` — Ý tưởng thiết kế cách mạng
> Đây là ý tưởng thiết kế cực kỳ quan trọng: VGG định nghĩa một **family of networks** (họ mạng) thay vì một network cụ thể. Bằng cách thay đổi `arch`, ta tạo ra VGG-11, VGG-13, VGG-16, VGG-19… **từ cùng một codebase**. 
>
> Đây là **lần đầu tiên** trong lịch sử deep learning: khi introduce kiến trúc mới, tác giả đề xuất một **họ mạng** với các trade-off speed-accuracy khác nhau, thay vì chỉ 1 mạng đơn lẻ. Pattern này trở thành **chuẩn mực** sau VGG — mọi paper kiến trúc hiện đại đều đề xuất nhiều biến thể (EfficientNet-B0 đến B7, ResNet-18/34/50/101/152…).
>
> Ở góc nhìn code: kiến trúc VGG có thể được **định nghĩa bằng 1 dòng Python**. Đây là minh chứng cho sức mạnh của modern deep learning frameworks — thay vì viết file XML config dài dòng, ta chỉ cần composable Python code.

### 3.2 VGG Family — Các phiên bản

Simonyan & Zisserman mô tả nhiều biến thể VGG trong paper gốc (Table 1). Đây là các phiên bản phổ biến nhất:

| Model | Architecture `arch` | Conv layers | Total layers | Parameters |
| --- | --- | --- | --- | --- |
| **VGG-11** (A) | `(1,64),(1,128),(2,256),(2,512),(2,512)` | 8 | 11 | ~133M |
| **VGG-13** (B) | `(2,64),(2,128),(2,256),(2,512),(2,512)` | 10 | 13 | ~133M |
| **VGG-16** (D) | `(2,64),(2,128),(3,256),(3,512),(3,512)` | 13 | 16 | ~138M |
| **VGG-19** (E) | `(2,64),(2,128),(4,256),(4,512),(4,512)` | 16 | 19 | ~144M |

> [!question]- ❓ Tại sao VGG-11 và VGG-13 có cùng ~133M params?
> Phần lớn params nằm ở **FC layers** (Flatten 25088 → 4096 → 4096 → 10), mà FC layers giống hệt nhau ở tất cả biến thể. Thêm 2 conv layers ở VGG-13 chỉ tăng rất ít params (vì conv layers ít params hơn FC layers rất nhiều).
>
> Cụ thể: FC1 alone = $25{,}088 \times 4{,}096 \approx 102M$ params. Tất cả 8 conv layers trong VGG-11 chỉ có ~9M params. Tỉ lệ: FC chiếm ~**92%** tổng params!

> [!question]- ❓ Tên VGG-11 nghĩa là gì? Tại sao `layer_summary` chỉ hiện 8 blocks?
> "11" = 8 conv layers + 3 FC layers = **11 trainable layers tổng**.
> 
> Khi xem output shape, mỗi VGG block được hiển thị như **1 đơn vị** (vì nó là `nn.Sequential`). Bên trong mỗi block có thể có 1-4 conv layers — nhưng chúng ẩn bên trong. Vì vậy `layer_summary` chỉ hiện 5 VGG blocks + Flatten + 3 FC (chỉ ~8 "top-level" items).

### 3.3 Data flow analysis — VGG-11

Kiểm tra output shape qua từng block — **rất quan trọng** để hiểu dòng chảy dữ liệu:

```python
model = VGG(arch=[(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)])
X = torch.randn(1, 1, 224, 224)

for i, block in enumerate(model.net):
    X = block(X)
    print(f"Block {i:2d} ({block.__class__.__name__:12s}) → {str(X.shape):30s}")
```

```
Block  0 (Sequential ) → torch.Size([1, 64, 112, 112])    ← VGG Block 1
Block  1 (Sequential ) → torch.Size([1, 128, 56, 56])     ← VGG Block 2
Block  2 (Sequential ) → torch.Size([1, 256, 28, 28])     ← VGG Block 3
Block  3 (Sequential ) → torch.Size([1, 512, 14, 14])     ← VGG Block 4
Block  4 (Sequential ) → torch.Size([1, 512, 7, 7])       ← VGG Block 5
Block  5 (Flatten    ) → torch.Size([1, 25088])            ← Flatten
Block  6 (LazyLinear ) → torch.Size([1, 4096])             ← FC1
Block  7 (ReLU       ) → torch.Size([1, 4096])
Block  8 (Dropout    ) → torch.Size([1, 4096])
Block  9 (LazyLinear ) → torch.Size([1, 4096])             ← FC2
Block 10 (ReLU       ) → torch.Size([1, 4096])
Block 11 (Dropout    ) → torch.Size([1, 4096])
Block 12 (LazyLinear ) → torch.Size([1, 10])               ← Output
```

> [!TIP] Đọc patterns — 3 quy luật rõ ràng
> 1. **Spatial halving**: 224 → 112 → 56 → 28 → 14 → **7** (giảm đúng 2× mỗi block nhờ MaxPool 2×2)
> 2. **Channel doubling**: 64 → 128 → 256 → 512 → 512 (gấp 2 cho đến 512 rồi giữ nguyên)
> 3. **Information flow**: Spatial giảm 32× (224→7), channels tăng 8× (64→512). Tổng tensor size:
>    - Block 1 output: $64 \times 112 \times 112 = 802{,}816$ elements
>    - Block 5 output: $512 \times 7 \times 7 = 25{,}088$ elements → giảm **32×**
>    - Flatten: $25{,}088$ → gấp **~4× so với AlexNet** (6,400). Đây là nguyên nhân VGG có **rất nhiều params** ở FC layers!

---

## 4. So sánh VGG vs AlexNet

### 4.1 Kiến trúc — Sự khác biệt tư duy thiết kế

```mermaid
flowchart TD
    subgraph AlexNet["AlexNet — Ad-hoc Design 🔧"]
        A1["Conv 11×11, 96, s=4"] --> A2["Conv 5×5, 256"]
        A2 --> A3["Conv 3×3, 384"]
        A3 --> A4["Conv 3×3, 384"]
        A4 --> A5["Conv 3×3, 256"]
        A5 --> A6["FC 4096"]
        A6 --> A7["FC 4096"]
        A7 --> A8["FC 10"]
    end
    
    subgraph VGG["VGG-11 — Block Design 🧱"]
        V1["🧱 Block 1: 1×Conv3, 64"] --> V2["🧱 Block 2: 1×Conv3, 128"]
        V2 --> V3["🧱 Block 3: 2×Conv3, 256"]
        V3 --> V4["🧱 Block 4: 2×Conv3, 512"]
        V4 --> V5["🧱 Block 5: 2×Conv3, 512"]
        V5 --> V6["FC 4096"]
        V6 --> V7["FC 4096"]
        V7 --> V8["FC 10"]
    end
    
    style A1 fill:#ef476f,color:#fff
    style A2 fill:#E8A838,color:#fff
    style A3 fill:#4A90D9,color:#fff
    style A4 fill:#4A90D9,color:#fff
    style A5 fill:#4A90D9,color:#fff
    style V1 fill:#06D6A0,color:#fff
    style V2 fill:#06D6A0,color:#fff
    style V3 fill:#06D6A0,color:#fff
    style V4 fill:#06D6A0,color:#fff
    style V5 fill:#06D6A0,color:#fff
```

**Điểm khác biệt quan trọng nhất**: AlexNet — mỗi layer có kernel size khác nhau (11, 5, 3), không có pattern rõ ràng. VGG — tất cả blocks đều dùng cùng cấu trúc (conv 3×3 + MaxPool), chỉ khác **số convs** và **số channels**. Nếu AlexNet là "ngôi nhà xây thủ công", VGG là "tòa nhà prefab".

### 4.2 Bảng so sánh chi tiết

| Tiêu chí | AlexNet | VGG-11 | VGG-16 |
| --- | --- | --- | --- |
| **Conv layers** | 5 (kích thước khác nhau: 11, 5, 3) | 8 (**toàn bộ 3×3**) | 13 (**toàn bộ 3×3**) |
| **FC layers** | 3 | 3 | 3 |
| **Total params** | ~62M | ~133M (2.1×) | ~138M (2.2×) |
| **Max kernel** | **11×11** | 3×3 | 3×3 |
| **Design principle** | Ad-hoc (mỗi layer khác nhau) | **Block-based** (pattern lặp lại) | **Block-based** |
| **Configurable?** | Không | Có (qua `arch`) | Có (qua `arch`) |
| **Flatten size** | 6,400 | **25,088** (3.9×) | **25,088** |
| **FC1 params** | ~26M | **~103M** | **~103M** |

> [!WARNING] VGG tốn tài nguyên hơn AlexNet rất nhiều!
> VGG-11 có **gấp 2× params** so với AlexNet. Phần lớn đến từ FC layers: flatten $512 \times 7 \times 7 = 25{,}088$ → FC 4096 = **~103M params** chỉ riêng FC1! So với AlexNet: $256 \times 5 \times 5 = 6{,}400$ → FC 4096 = **~26M params**.
> 
> Đây là vấn đề mà **NiN** (Buổi 31) sẽ giải quyết bằng **Global Average Pooling** — bỏ hoàn toàn 3 FC layers, giảm hàng chục triệu params.

---

## 5. Training trên Fashion-MNIST

> [!NOTE] ELI5
> VGG-11 gốc quá nặng để train trên một bài demo — nó cần GPU tốt và nhiều thời gian. Vì vậy ta dùng **phiên bản thu nhỏ**: giảm channels 4× (64→16, 128→32…). Mạng vẫn giữ **đúng cấu trúc**, chỉ "mỏng hơn" — giống thu nhỏ tòa nhà nhưng giữ nguyên bản thiết kế.

VGG-11 tốn nhiều compute hơn AlexNet, nên khi train trên Fashion-MNIST — một bài toán tương đối đơn giản — ta dùng **phiên bản channels nhỏ hơn** (d2l.ai gọi đây là sufficient cho Fashion-MNIST):

```python
# VGG-11 thu nhỏ: giảm channels 4× để train khả thi
small_arch = [(1, 16), (1, 32), (2, 64), (2, 128), (2, 128)]

model = VGG(arch=small_arch)
```

So sánh channels gốc vs thu nhỏ:

| Block | VGG-11 gốc | VGG-11 thu nhỏ | Tỉ lệ |
| --- | --- | --- | --- |
| 1 | 64 | 16 | ÷4 |
| 2 | 128 | 32 | ÷4 |
| 3 | 256 | 64 | ÷4 |
| 4 | 512 | 128 | ÷4 |
| 5 | 512 | 128 | ÷4 |

**Training code:**

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize 28×28 → 224×224 (wasteful nhưng giữ nguyên arch)
    transforms.ToTensor(),
])

train_data = datasets.FashionMNIST('./data', train=True,
                                    download=True, transform=transform)
test_data  = datasets.FashionMNIST('./data', train=False,
                                    transform=transform)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=128)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VGG(arch=small_arch).to(device)

# Xavier initialization — thiết kế chuẩn cho ReLU networks
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
model.apply(init_weights)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
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
    
    train_acc = correct / total
    test_acc = test_correct / len(test_data)
    print(f"Epoch {epoch+1:2d} | "
          f"Loss: {total_loss/total:.4f} | "
          f"Train: {train_acc*100:.1f}% | "
          f"Test: {test_acc*100:.1f}%")
```

> [!TIP] Phân tích training
> 1. **Training process**: Giống AlexNet — lr=0.01, SGD, 10 epochs. Nhưng **chậm hơn** vì VGG có nhiều conv layers hơn (8 vs 5), mỗi conv đều 3×3 pad 1.
> 2. **Overfitting**: Training loss và validation loss **rất sát nhau** → rất ít overfitting. Điều này nhờ Dropout 0.5 ở FC layers + tính regularization implicit từ kiến trúc deep narrow.
> 3. **Accuracy kỳ vọng**: ~91-92% test accuracy trên Fashion-MNIST, tương đương hoặc nhỉnh hơn AlexNet.
> 4. **Resize 28→224 là lãng phí**: Phóng to ảnh 28×28 lên 224×224 **không thêm thông tin** — chỉ thêm compute. Exercise 4 sẽ yêu cầu bạn thử giảm resolution để tiết kiệm.

---

## 6. Discussion — Tại sao VGG quan trọng?

### 6.1 VGG = CNN hiện đại đầu tiên "thực sự"

> *"One might argue that VGG is the **first truly modern** convolutional neural network. While AlexNet introduced many of the components of what make deep learning effective at scale, it is VGG that arguably introduced **key properties** such as **blocks of multiple convolutions** and a **preference for deep and narrow networks**."*
> 
> — d2l.ai, Section 8.2

Câu nhận định này từ d2l.ai rất chính xác: AlexNet là **breakthrough** (đột phá), nhưng VGG mới là **blueprint** (bản vẽ thiết kế) cho tất cả CNN sau này. AlexNet chứng minh "CNN sâu hoạt động", VGG trả lời "CNN sâu nên được **thiết kế như thế nào**".

### 6.2 Bốn di sản của VGG

1. **Block-based design**: Mọi kiến trúc CNN sau đều tổ chức theo blocks — Inception blocks (GoogLeNet), Residual blocks (ResNet), Dense blocks (DenseNet), Inverted Residual blocks (MobileNet). VGG block là "model block" đơn giản nhất.

2. **3×3 conv standard**: Conv 3×3 trở thành **kernel size mặc định** — và kiến trúc GPU cũng được tối ưu riêng cho phép nhân 3×3 nhỏ (Lavin & Gray, 2016). Khi bạn viết `nn.Conv2d(C, C_out, 3, padding=1)`, bạn đang theo di sản VGG.

3. **Network families**: Giới thiệu ý tưởng đề xuất một **họ mạng** (VGG-11/13/16/19) thay vì 1 mạng đơn lẻ khi publish. Pattern này trở thành chuẩn mực — ResNet-18/34/50/101/152, EfficientNet-B0→B7.

4. **Programmable architecture**: Kiến trúc VGG có thể được định nghĩa bằng 1 variable (`arch`), mở đường cho era "define-network-by-code" — không cần XML config, không cần hardcode. Đây cũng là lý do modern deep learning frameworks (PyTorch, TF) **tỏa sáng**: implement VGG trong 20 dòng Python.

### 6.3 Hạn chế

1. **Quá nhiều parameters**: 133M params, phần lớn (~92%) ở FC layers → tốn memory, chậm inference. → **NiN** (Buổi 31) sẽ thay FC bằng **Global Average Pooling**.
2. **Chưa có skip connections**: Mạng quá sâu vẫn gặp **gradient degradation** (degradation problem — accuracy giảm khi tăng depth, dù training error không giảm). → **ResNet** (Buổi 34) sẽ thêm residual connections.
3. **FC layers vẫn lớn**: VGG giữ 3 FC layers như AlexNet — đây là bottleneck về memory và parameter count.

### 6.4 ParNet — Hướng phát triển thay thế

Gần đây, **ParNet** (Goyal et al., 2021) cho thấy có thể đạt performance cạnh tranh bằng kiến trúc **nông hơn nhiều** nhờ **parallel computations** — nhiều nhánh tính toán song song thay vì xếp chồng tuần tự. Đây là hướng phát triển thú vị: **sâu** không phải cách duy nhất để "mạnh" — **rộng và song song** cũng có thể. Tuy nhiên, cho đến nay, cách tiếp cận sâu tuần tự (VGG → ResNet) vẫn chiếm ưu thế.

---

## 7. Exercises (từ d2l.ai)

> [!NOTE] Bài tập gốc từ sách — nên làm để hiểu sâu hơn.

1. **So sánh AlexNet vs VGG:**
   - (a) Tính số parameters AlexNet vs VGG-11. Layer nào chiếm nhiều nhất?
   - (b) Tính FLOPs cho conv layers vs FC layers ở mỗi mạng. Phần nào tốn compute hơn?
   - (c) Làm sao giảm chi phí tính toán từ FC layers? (Gợi ý: dùng technique gì thay FC?)

2. **VGG-11 có 11 layers nhưng `layer_summary` hiện 8 blocks.** 3 layers còn lại ở đâu? Tại sao chúng "ẩn"?

3. **Xây VGG-16 và VGG-19** bằng cách thay `arch`:
   ```python
   VGG_16 = [(2,64), (2,128), (3,256), (3,512), (3,512)]
   VGG_19 = [(2,64), (2,128), (4,256), (4,512), (4,512)]
   ```
   Verify bằng cách đếm: VGG-16 = 2+2+3+3+3 = **13** conv + **3** FC = **16**. ✓

4. **Giảm resolution input:** Resize 28×28 → 224×224 rất lãng phí. Thử resize thành 56×56 hoặc 84×84 rồi điều chỉnh network. Accuracy có giảm không? Tham khảo VGG paper về ý tưởng thêm nonlinearities trước downsampling.

---

## 📖 Từ điển thuật ngữ

| Thuật ngữ | Nghĩa tiếng Việt | Chi tiết |
| --- | --- | --- |
| **VGG** | Visual Geometry Group | Nhóm nghiên cứu tại Oxford, cũng là tên mạng |
| **Block** | Khối xây dựng | Nhóm layers lặp lại, có cấu trúc giống nhau — thành phần cơ bản để xây kiến trúc |
| **Deep and narrow** | Sâu và hẹp | Chiến lược dùng nhiều layers kernel nhỏ (3×3) thay vì ít layers kernel lớn (5×5, 7×7) |
| **Network family** | Họ mạng | Nhiều biến thể (VGG-11/13/16/19) từ cùng 1 template code |
| **arch** | Architecture specification | List tuples `(num_convs, channels)` định nghĩa toàn bộ kiến trúc VGG |
| **VLSI** | Very Large Scale Integration | Công nghệ thiết kế chip tích hợp quy mô lớn — phép loại suy cho sự phát triển kiến trúc CNN |
| **Foundation model** | Mô hình nền tảng | Model pretrained lớn (GPT, ViT) được tái sử dụng cho nhiều tasks — tương đương IP cores |
| **Receptive field** | Trường tiếp nhận | Vùng pixels đầu vào mà 1 neuron output "nhìn thấy" được |
| **Gold standard** | Chuẩn vàng | Phương pháp được coi là tốt nhất, được mọi người áp dụng rộng rãi |
| **Implicit regularization** | Chính quy hóa ngầm | Hiệu ứng regularization tự nhiên xuất phát từ kiến trúc (deep narrow), không cần thêm penalty |
| **Expressiveness** | Sức biểu diễn | Khả năng model biểu diễn các hàm phức tạp — tăng nhờ thêm nonlinearity |

---

## ✅ Bài tự kiểm tra

1. VGG giải quyết vấn đề gì mà AlexNet chưa có?
2. Tại sao CNN không thể chỉ xếp Conv → Pool → Conv → Pool liên tiếp? Giới hạn toán học là gì?
3. VGG block gồm những thành phần gì? Vẽ/mô tả sơ đồ 1 block.
4. Tại sao 2 conv $3 \times 3$ tốt hơn 1 conv $5 \times 5$? (3 lý do chi tiết)
5. VGG-11 có bao nhiêu conv layers và FC layers? Tính tổng. Tại sao gọi là "VGG-11"?
6. Channels thay đổi thế nào qua các blocks? Tại sao tăng gấp đôi?
7. Viết `arch` cho VGG-16 mà không nhìn đáp án. Verify bằng cách đếm layers.
8. FC layers chiếm bao nhiêu % tổng params của VGG-11? Tại sao đây là hạn chế? Giải pháp?

> [!NOTE]- 📝 Đáp án gợi ý
> 1. AlexNet thiết kế **ad-hoc** (mỗi layer khác nhau, không pattern). VGG giới thiệu **block-based design**: thiết kế theo blocks lặp lại, có quy luật → dễ mở rộng, dễ thay đổi. VGG cũng lần đầu đề xuất **network family** thay vì 1 mạng đơn lẻ.
> 2. Mỗi Pool giảm resolution 2×. Sau $k$ pools: $d / 2^k$. Với $d = 224$: $\log_2(224) \approx 7.8$ → tối đa ~7-8 conv layers. VGG giải quyết bằng cách dùng **nhiều conv giữa mỗi lần pool**.
> 3. VGG block = $n$ × (Conv $3 \times 3$, pad 1 + ReLU) + MaxPool $2 \times 2$, stride 2. Conv giữ nguyên size, MaxPool giảm 2×.
> 4. **(a)** Ít tham số hơn ($18c^2$ vs $25c^2$, tiết kiệm 28%). **(b)** Nhiều ReLU hơn → tăng tính phi tuyến → model expressive hơn. **(c)** Implicit regularization tốt hơn → ít overfit.
> 5. 8 conv layers + 3 FC layers = **11 layers** → VGG-11.
> 6. 64 → 128 → 256 → 512 → 512 (**gấp đôi** mỗi block, dừng ở 512). Lý do: khi spatial giảm 2× (diện tích giảm 4×), channels tăng 2× để **bù thông tin** bị mất.
> 7. `[(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)]` — Block 3-5 có **3 convs** mỗi block. Đếm: 2+2+3+3+3 = 13 conv + 3 FC = **16**. ✓
> 8. FC giữ nguyên ở mọi biến thể VGG. FC1 = $25{,}088 \times 4{,}096 \approx 103M$, FC2 = $4{,}096 \times 4{,}096 \approx 17M$, FC3 = $4{,}096 \times 10 \approx 0.04M$. Tổng FC ≈ **120M** / 133M ≈ **~90%** params! Hạn chế: tốn memory, chậm. Giải pháp: **NiN** (Buổi 31) thay FC bằng **Global Average Pooling** → giảm hàng chục triệu params.

---

## 🔗 Liên kết

- **Buổi trước**: [[Buổi 29 - Tuần 8]] — 8.1 AlexNet: Deep CNN đầu tiên
- **Buổi sau**: [[Buổi 31 - Tuần 8]] — 8.3 Network in Network (NiN): giải quyết FC problem
- **Concepts**: [[Activation Function]], [[Dropout]]
- **Source**: [d2l.ai — 8.2 VGG](https://d2l.ai/chapter_convolutional-modern/vgg.html)
