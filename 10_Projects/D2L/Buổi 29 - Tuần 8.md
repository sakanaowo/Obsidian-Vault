---
title: "Buổi 29 - Tuần 8: Deep Convolutional Neural Networks (AlexNet)"
tags: [d2l, cnn, alexnet, modern-cnn, gpu, representation-learning, imagenet, study-note]
created: 2026-04-02
session: "D2L Tuần 8, Buổi 29 — 8.1 Deep Convolutional Neural Networks (AlexNet)"
d2l_section: "8.1"
source:
  - "https://d2l.ai/chapter_convolutional-modern/alexnet.html"
  - "30_Resources/Books/Dive Into Deep Learning/d2l-en/chapter_convolutional-modern/alexnet.md"
related:
  - "[[Buổi 28 - Tuần 8]]"
  - "[[Buổi 30 - Tuần 8]]"
aliases: ["AlexNet", "8.1 AlexNet"]
---

# Buổi 29 — 8.1 Deep Convolutional Neural Networks (AlexNet)

> [!NOTE] ELI5
> Bạn đã biết cách nhận diện **con mèo** từ bé: nhìn tai nhọn, bộ lông, mắt to… Não bạn **tự học** những đặc điểm này sau khi nhìn hàng nghìn con mèo.
> 
> Trước năm 2012, máy tính không biết **tự học** mô tả ảnh. Thay vào đó, con người phải **tự viết công thức** để mô tả "tai nhọn", "lông mượt"… rồi dùng chúng để phân loại — gọi là **feature engineering**. Cách này rất tốn công và chỉ tốt cho một bài toán cụ thể.
> 
> **AlexNet** (2012) thay đổi tất cả: nó là CNN "to" đầu tiên, tự học features từ pixels thô, đánh bại mọi phương pháp truyền thống trong cuộc thi ImageNet. Sau AlexNet, cả thế giới AI chuyển sang Deep Learning.

---

## 1. Bối cảnh — Tại sao mất 14 năm từ LeNet đến AlexNet?

> [!NOTE] ELI5
> LeNet (1998) đã **chứng minh** CNN hoạt động được — nhưng rồi bị lãng quên suốt 14 năm. Tại sao? Vì thời đó thiếu 3 thứ cốt yếu: dữ liệu lớn, phần cứng mạnh, và kỹ thuật training hiện đại. Giống như bạn có bản vẽ máy bay nhưng chưa có động cơ phản lực — ý tưởng đúng nhưng chưa thể bay.

### 1.1 CNN và các phương pháp truyền thống

Mặc dù CNN đã được biết đến rộng rãi trong cộng đồng Computer Vision và Machine Learning kể từ khi LeNet ra đời (LeCun et al., 1995), chúng **không ngay lập tức thống trị** lĩnh vực này. Trên thực tế, trong phần lớn thời gian từ đầu thập niên 1990 đến kết quả đột phá năm 2012 (Krizhevsky et al., 2012), neural networks thường **bị vượt mặt** bởi các phương pháp ML khác như:

- **Kernel methods** (SVM — Support Vector Machine): Dựa trên lý thuyết tối ưu lồi (convex optimization), có nền tảng toán học chặt chẽ, hoạt động tốt trên dữ liệu nhỏ.
- **Ensemble methods** (Random Forest, Boosting): Kết hợp nhiều model yếu thành model mạnh.
- **Structured estimation** (CRF — Conditional Random Fields): Tận dụng cấu trúc dữ liệu.

> [!TIP] Tại sao lại thua?
> Điểm mấu chốt không nằm ở việc CNN "kém" — mà nằm ở việc so sánh này **không công bằng**. Các phương pháp truyền thống **không** nhận pixels thô làm đầu vào. Thay vào đó, chúng dựa vào một pipeline được thiết kế thủ công:

```mermaid
flowchart LR
    A["📷 Ảnh thô<br/>(pixels)"] --> B["🔧 Feature Extraction<br/>SIFT / SURF / HOG<br/><i>(thiết kế bằng tay)</i>"]
    B --> C["📊 Feature Vector<br/>(mô tả toán học)"]
    C --> D["🤖 Classifier<br/>SVM / Random Forest"]
    D --> E["📋 Nhãn"]

    style B fill:#ef476f,color:#fff
    style D fill:#4A90D9,color:#fff
```

Pipeline này gồm 4 bước:

1. **Thu thập dữ liệu** — Thời kỳ đầu, camera còn đắt đỏ. Apple QuickTake 100 (1994) chỉ chụp được ảnh 0.3 megapixel, lưu tối đa 8 ảnh, giá 1000 USD.
2. **Tiền xử lý** — Dùng kiến thức quang học, hình học, và nhiều kỹ thuật thủ công để trích xuất thông tin.
3. **Feature extraction** — Đưa ảnh qua các bộ trích features đã thiết kế sẵn: SIFT (Scale-Invariant Feature Transform), SURF (Speeded Up Robust Features), HOG (Histograms of Oriented Gradient), Bags of Visual Words.
4. **Classification** — Đưa vector đặc trưng vào SVM hoặc kernel method để phân loại.

> [!IMPORTANT] Vấn đề cốt lõi
> **Features được "chế tạo" (crafted), không phải "học" (learned).**
> 
> Phần lớn tiến bộ đến từ việc nghĩ ra cách trích features thông minh hơn, chứ **không phải** từ thuật toán học. Nói cách khác: thuật toán ML chỉ là "afterthought" — thứ được nghĩ đến cuối cùng. Nhà nghiên cứu CV tin rằng một dataset sạch hơn hoặc pipeline features tốt hơn sẽ quan trọng hơn bất kỳ thuật toán ML mới nào.

### 1.2 Missing Ingredient #1: Data (Dữ liệu)

> [!NOTE] ELI5
> Bạn muốn dạy con robot nhận dạng 1000 loại vật thể. Nhưng bạn chỉ có 60.000 ảnh nhỏ xíu 28×28 pixels (MNIST) — mỗi loại chỉ 60 ảnh. Quá ít! Robot cần **hàng triệu** ảnh lớn, rõ nét, đa dạng mới học được.

Deep models với nhiều layers cần **lượng dữ liệu lớn** mới có thể vượt trội hơn các phương pháp truyền thống dựa trên tối ưu lồi. Tuy nhiên, trong thập niên 1990, do hạn chế về bộ nhớ máy tính, chi phí cảm biến (sensor), và ngân sách nghiên cứu eo hẹp, hầu hết nghiên cứu dựa trên **những bộ dữ liệu rất nhỏ**: vài trăm đến vài ngàn ảnh, độ phân giải thấp, nền sạch nhân tạo.

Bước ngoặt đến năm **2009** khi bộ dữ liệu **ImageNet** được phát hành (Deng et al., 2009):

| Đặc tính | MNIST (1998) | ImageNet (2009) |
| --- | --- | --- |
| **Số ảnh training** | 60,000 | **1,281,167** (gấp 21×) |
| **Số lớp** | 10 | **1,000** |
| **Resolution** | 28×28 grayscale | **224×224 RGB** (gấp 64× pixels) |
| **Nguồn gốc** | Chữ viết tay | Ảnh thật từ internet |
| **Diversity** | Rất thấp | Cực cao (chó, mèo, xe, máy bay…) |

ImageNet được xây dựng bằng cách:
1. Lấy danh sách 1000 loại từ **WordNet** (ontology ngôn ngữ).
2. Dùng **Google Image Search** để tìm ảnh ứng viên.
3. Dùng **Amazon Mechanical Turk** (crowdsourcing) để xác nhận nhãn mỗi ảnh.

Cuộc thi **ILSVRC** (ImageNet Large Scale Visual Recognition Challenge) mỗi năm thúc đẩy cộng đồng phát triển model ngày một tốt hơn. Quy mô này chưa từng có — vượt các bộ dữ liệu trước đó **hơn 1 bậc** (CIFAR-100 chỉ có 60.000 ảnh). Ngày nay, các bộ dữ liệu lớn nhất như LAION-5B chứa **hàng tỷ** ảnh.

> [!question]- ❓ Tại sao resolution 224×224 quan trọng?
> Ảnh 224×224 pixel cho phép hình thành **higher-level features** — các đặc trưng bậc cao như khuôn mặt, bộ phận cơ thể, cấu trúc vật thể. Với ảnh 28×28 (MNIST), thông tin quá thô sơ, chỉ đủ phân biệt nét chữ đơn giản. Ảnh TinyImages (32×32) cũng quá nhỏ để CNN sâu phát huy sức mạnh.

### 1.3 Missing Ingredient #2: Hardware (Phần cứng)

> [!NOTE] ELI5
> Deep Learning cần tính toán **cực nhanh** — mỗi bước training phải nhân hàng triệu ma trận. CPU (bộ xử lý thông thường) giống con dao đa năng: làm được mọi thứ nhưng không nhanh thứ gì. GPU (card đồ họa) giống dây chuyền sản xuất: chỉ làm được một loại phép tính nhưng **nhanh gấp hàng nghìn lần**.

Deep learning models là "kẻ ngốn tính toán": training có thể mất hàng trăm epochs, mỗi iteration phải đưa dữ liệu qua nhiều lớp phép nhân ma trận. Đây là lý do chính vì sao trong thập niên 1990–2000, các thuật toán đơn giản hơn dựa trên tối ưu lồi (convex optimization) được ưa chuộng.

**GPU** (Graphical Processing Units) là game changer. Ban đầu chúng được phát triển để tăng tốc xử lý đồ họa cho game, đặc biệt tối ưu cho phép nhân ma trận $4 \times 4$ — và may mắn thay, toán học này **gần giống hệt** toán cần cho CNN.

#### CPU vs GPU — Khác biệt bản chất

| | CPU | GPU |
|---|---|---|
| **Số cores** | 4–64 (laptop 4-8) | **Hàng nghìn** (A100: 6912 CUDA cores) |
| **Clock speed** | 3–5 GHz (nhanh) | ~1 GHz (chậm hơn mỗi core) |
| **Kiến trúc** | Phức tạp: branch prediction, pipeline sâu, cache lớn (MB), speculative execution | Đơn giản: ALU thuần túy, cache nhỏ |
| **Thế mạnh** | Code đa dạng, control flow phức tạp | **Tính toán song song** cùng 1 phép tính |
| **FP32 FLOPS** | ~1 TFLOPS | ~20 TFLOPS (A100) |
| **BF16 FLOPS** | ~2 TFLOPS | **~300 TFLOPS** (A100) |

> [!TIP] Tại sao GPU nhanh hơn CPU cho DL?
> 1. **Công suất**: Công suất tiêu thụ tăng theo **bình phương** tần số clock. Thay vì 1 core chạy 4GHz, ta dùng 16 cores chạy 1GHz → $16 \times \frac{1}{4} = 4\times$ performance với cùng điện năng.
> 2. **Đơn giản**: GPU cores đơn giản hơn → tiết kiệm năng lượng hơn cho mỗi phép tính. Chúng không hỗ trợ speculative evaluation, không lập trình từng core riêng.
> 3. **Băng thông bộ nhớ**: GPU có bus rộng hơn CPU **10 lần** — quan trọng vì DL cần đọc/ghi dữ liệu liên tục.

#### Bước đột phá 2012

Alex Krizhevsky và Ilya Sutskever nhận ra rằng **bottleneck tính toán** của CNN — convolutions và nhân ma trận — đều có thể **song song hóa** trên GPU. Họ dùng **2 card NVIDIA GTX 580** (mỗi card 3GB RAM, 1.5 TFLOPS) để implement fast convolutions. Thư viện [cuda-convnet](https://code.google.com/archive/p/cuda-convnet/) trở thành chuẩn mực của ngành trong vài năm đầu Deep Learning bùng nổ.

> [!IMPORTANT] Perspective: Sự tiến bộ của phần cứng
> NVIDIA GeForce 256 (1999): max **480 MFLOPS** — không có framework lập trình ngoài game.
> NVIDIA GTX 580 (2010): **1.5 TFLOPS** — cuda-convnet cho DL.
> NVIDIA A100 (2020): **300+ TFLOPS** (BF16) — training GPT-3 trong vài tuần.
> 
> Tốc độ tăng **hơn 600.000×** trong 20 năm. Không có GPU, AlexNet **không thể tồn tại**.

### 1.4 Missing Ingredient #3: Techniques (Kỹ thuật)

Ngoài data và hardware, còn thiếu những kỹ thuật training quan trọng chỉ được phát minh sau năm 2010:

| Kỹ thuật | Năm | Tác dụng | Tham khảo |
| --- | --- | --- | --- |
| **Xavier initialization** | 2010 | Giữ variance ổn định qua các layers → training ổn | Glorot & Bengio |
| **ReLU activation** | 2010 | Không bị gradient vanishing (khác sigmoid) → train mạng sâu | Nair & Hinton |
| **Adam optimizer** | 2014 | Adaptive learning rate → converge nhanh hơn SGD | Kingma & Ba |
| **Dropout** | 2014 | Regularization bằng cách tắt ngẫu nhiên neurons → chống overfitting | Srivastava et al. |

> [!question]- ❓ Tại sao ReLU thay đổi tất cả?
> **Sigmoid**: $\sigma(x) = \frac{1}{1+e^{-x}}$ có gradient gần 0 khi $|x|$ lớn → **gradient vanishing**: backpropagation "chết" dần qua các layers sâu → không train được mạng nhiều layers.
> 
> **ReLU**: $f(x) = \max(0, x)$ có gradient = 1 khi $x > 0$ → gradient **luôn truyền ngược được** qua các layers sâu. Thêm nữa, ReLU tính đơn giản (không cần hàm mũ), nhanh hơn sigmoid ~6× trên GPU.
> 
> Đây là lý do AlexNet dùng ReLU thay sigmoid — quyết định tưởng đơn giản nhưng **thay đổi bản chất** khả năng training mạng sâu.

---

## 2. Representation Learning — Bước chuyển tư duy

> [!NOTE] ELI5
> **Feature Engineering** (trước 2012) = Con người phải tự nghĩ cách mô tả ảnh. Giống dạy robot bằng sách hướng dẫn: "mèo có tai nhọn, chó có mũi dài…" — mỗi loại vật cần viết sách khác nhau.
> 
> **Representation Learning** (sau 2012) = Cho robot xem hàng triệu ảnh, robot **tự rút ra** đặc điểm nào quan trọng. Không cần sách hướng dẫn. Cùng 1 robot (CNN) có thể nhận dạng mèo, chó, xe hơi… chỉ bằng cách thay đổi dữ liệu huấn luyện.

Nói cách khác: trước 2012, phần quan trọng nhất của pipeline CV là **representation** — và nó được tạo ra **bằng tay** (mechanically). Việc thiết kế bộ features mới, cải thiện kết quả, sau đó viết paper mô tả phương pháp là công việc chính của các nhà nghiên cứu CV.

Một nhóm nghiên cứu khác — gồm **Yann LeCun, Geoff Hinton, Yoshua Bengio, Andrew Ng, Shun-ichi Amari, và Juergen Schmidhuber** — tin rằng features **nên được học tự động**. Hơn nữa, chúng nên được tổ chức **phân cấp** (hierarchically composed) với nhiều lớp cùng học, mỗi lớp có tham số riêng:

```
Tầng thấp nhất  → Edges, colors, textures
                   (tương tự cách hệ thần kinh thị giác xử lý)
Tầng trung gian → Corners, contours, textures phức tạp
Tầng cao        → Parts (mắt, mũi, bánh xe)
Tầng cuối       → Objects (người, chó, máy bay)
```

AlexNet — CNN "hiện đại" đầu tiên (Krizhevsky et al., 2012) — đã **chứng minh** ý tưởng này đúng. Nó thắng cuộc thi ImageNet 2012 với cách biệt lớn, cho thấy rằng features **tự học** có thể vượt trội features **thiết kế tay**, phá vỡ paradigm cũ trong Computer Vision.

> [!TIP] Features AlexNet tự học ở tầng đầu
> Ở tầng thấp nhất, AlexNet học ra các bộ lọc (filters) **giống hệt** một số filters truyền thống được thiết kế tay (Gabor filters, edge detectors). Nhưng khác biệt là: CNN tự phát hiện chúng từ dữ liệu, không cần ai thiết kế. Các tầng cao hơn xây dựng lên từ đó: mắt, mũi, lá cỏ… và cuối cùng: toàn bộ vật thể.

---

## 3. Kiến trúc AlexNet

> [!NOTE] ELI5
> AlexNet = LeNet **phóng to** + **nâng cấp linh kiện**.
> - **Phóng to**: 2 conv → **8 layers** (5 conv + 3 FC), 28×28 → **224×224**, 62K → **~47 triệu** tham số
> - **Nâng cấp**: Sigmoid → **ReLU**, AvgPool → **MaxPool**, thêm **Dropout**
> - Như việc sửa xe đạp thành ô tô: cùng nguyên lý di chuyển, nhưng to hơn, nhanh hơn, an toàn hơn.

Kiến trúc AlexNet và LeNet **giống nhau đáng ngạc nhiên**. Tuy nhiên có những khác biệt quan trọng.

### 3.1 Architecture chi tiết

AlexNet gồm **8 layers**: 5 convolutional layers, 2 fully connected hidden layers, và 1 fully connected output layer.

> [!WARNING] Lưu ý
> Phiên bản AlexNet trong d2l.ai là **phiên bản đơn giản hóa**, bỏ bớt một số quirk thiết kế của bản gốc 2012 (vốn cần chia model qua 2 GPU nhỏ). Chúng ta sẽ học phiên bản đơn giản nhưng giữ nguyên bản chất.

**Layer-by-layer analysis:**

| # | Layer | Details | Output Shape | Ghi chú |
| --- | --- | --- | --- | --- |
| 1 | **Conv2d** | 96 filters, k=11, s=4, p=1 | (96, 54, 54) | Kernel rất lớn, stride 4 giảm kích thước mạnh |
| — | ReLU + MaxPool | pool k=3, s=2 | (96, 26, 26) | |
| 2 | **Conv2d** | 256 filters, k=5, p=2 | (256, 26, 26) | |
| — | ReLU + MaxPool | pool k=3, s=2 | (256, 12, 12) | |
| 3 | **Conv2d** | 384 filters, k=3, p=1 | (384, 12, 12) | Ba conv 3×3 liên tiếp |
| 4 | **Conv2d** | 384 filters, k=3, p=1 | (384, 12, 12) | không có pooling ở giữa |
| 5 | **Conv2d** | 256 filters, k=3, p=1 | (256, 12, 12) | |
| — | ReLU + MaxPool | pool k=3, s=2 | (256, 5, 5) | |
| — | Flatten | — | (6400,) | $256 \times 5 \times 5 = 6400$ |
| 6 | **Linear** + ReLU + Dropout(0.5) | 4096 outputs | (4096,) | **26.2M params** — chiếm phần lớn model! |
| 7 | **Linear** + ReLU + Dropout(0.5) | 4096 outputs | (4096,) | |
| 8 | **Linear** | num_classes outputs | (10,) | Output layer |

### 3.2 Tại sao thiết kế như vậy?

#### Tại sao Conv1 dùng kernel 11×11?

Ảnh ImageNet 224×224 **lớn hơn 8×** so với MNIST 28×28. Các vật thể trong ảnh ImageNet chiếm **nhiều pixels hơn** và có chi tiết thị giác phong phú hơn. Do đó cần **convolution window lớn hơn** để bắt được vật thể.

Kernel 11×11 stride 4 ở tầng đầu có 2 tác dụng:
1. **Receptive field rộng** — "nhìn" vùng 11×11 pixels ngay tầng đầu ≈ 5% ảnh
2. **Giảm kích thước nhanh** — 224 → 54 ngay tầng 1, tiết kiệm compute cho các tầng sau

Từ tầng 2 trở đi, kernel **giảm dần**: 5×5 → 3×3 → 3×3 → 3×3. Tầng đầu "nhìn rộng", tầng sau "nhìn chi tiết".

#### Tại sao ReLU thay Sigmoid?

Hai lý do chính mà d2l.ai nhấn mạnh:

1. **Tính đơn giản**: ReLU không cần phép tính mũ ($e^{-x}$) như sigmoid → nhanh hơn trên phần cứng.
2. **Training dễ hơn**: Sigmoid có gradient gần 0 khi output gần 0 hoặc 1 → backpropagation không thể cập nhật tham số (**gradient vanishing**). ReLU có gradient = 1 trên miền dương → gradient luôn truyền được.

> [!question]- ❓ Chi tiết: Gradient vanishing với Sigmoid
> Sigmoid: $\sigma(x) = \frac{1}{1+e^{-x}}$, đạo hàm: $\sigma'(x) = \sigma(x)(1 - \sigma(x))$
> 
> Giá trị lớn nhất của $\sigma'$ là $0.25$ (tại $x=0$). Qua $n$ layers, gradient bị nhân liên tục: $0.25^n$. Với 5 layers: $0.25^5 = 0.001$ — gradient gần như **biến mất**.
> 
> ReLU: $f(x) = \max(0,x)$, đạo hàm: $f'(x) = 1$ khi $x > 0$. Qua $n$ layers: $1^n = 1$ — gradient **giữ nguyên**.
> 
> Đây là lý do căn bản tại sao ReLU cho phép train mạng sâu hơn nhiều so với sigmoid.

#### Capacity Control: Dropout thay Weight Decay

AlexNet kiểm soát model complexity của FC layers bằng **[[Dropout]]** (Section 5.6 đã học ở Buổi 22), trong khi LeNet chỉ dùng weight decay.

Ngoài ra, training loop của AlexNet còn dùng **data augmentation** (tăng cường dữ liệu): lật ảnh (flipping), cắt ảnh (clipping), thay đổi màu sắc (color changes). Điều này giúp model robust hơn — lượng mẫu lớn hơn hiệu quả giảm overfitting.

### 3.3 So sánh AlexNet vs LeNet

| | LeNet (1998) | AlexNet (2012) |
| --- | --- | --- |
| **Input** | 28×28 grayscale | 224×224 (RGB) |
| **Total layers** | 5 (2 Conv + 3 FC) | **8** (5 Conv + 3 FC) |
| **Activation** | Sigmoid | **ReLU** |
| **Pooling** | Average Pooling | **Max Pooling** |
| **Regularization** | Weight Decay | **Dropout 0.5** + Data Augmentation |
| **Parameters** | ~62K | ~**47M** (gấp ~750×) |
| **Hardware** | CPU | **2× GPU** (GTX 580) |
| **Data** | MNIST (60K, 28×28) | ImageNet (1.2M, 224×224) |

---

## 4. Implementation

### 4.1 Định nghĩa AlexNet trong PyTorch

```python
import torch
from torch import nn

class AlexNet(nn.Module):
    """AlexNet đơn giản hóa (d2l version) cho Fashion-MNIST.
    
    8 layers: 5 Conv + 3 FC.
    Input: (batch, 1, 224, 224) — ảnh grayscale resize lên 224.
    Output: (batch, num_classes)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            # === CONVOLUTIONAL LAYERS ===
            # Conv1: kernel lớn 11×11, stride 4 → giảm 224 → 54
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),           # 54 → 26
            
            # Conv2: kernel 5×5, padding giữ size
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),           # 26 → 12
            
            # Conv3, 4, 5: ba conv 3×3 liên tiếp, KHÔNG pool ở giữa
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),           # 12 → 5
            
            # === FULLY CONNECTED LAYERS ===
            nn.Flatten(),                                     # 256×5×5 = 6400
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        return self.net(x)
```

### 4.2 Kiểm tra output shape qua từng layer

Đây là bước quan trọng để hiểu dòng chảy dữ liệu (data flow):

```python
model = AlexNet()
X = torch.randn(1, 1, 224, 224)  # 1 ảnh grayscale 224×224

for layer in model.net:
    X = layer(X)
    print(f"{layer.__class__.__name__:15s} output shape: {X.shape}")
```

```
Conv2d          output shape: torch.Size([1, 96, 54, 54])
ReLU            output shape: torch.Size([1, 96, 54, 54])
MaxPool2d       output shape: torch.Size([1, 96, 26, 26])
Conv2d          output shape: torch.Size([1, 256, 26, 26])
ReLU            output shape: torch.Size([1, 256, 26, 26])
MaxPool2d       output shape: torch.Size([1, 256, 12, 12])
Conv2d          output shape: torch.Size([1, 384, 12, 12])
ReLU            output shape: torch.Size([1, 384, 12, 12])
Conv2d          output shape: torch.Size([1, 384, 12, 12])
ReLU            output shape: torch.Size([1, 384, 12, 12])
Conv2d          output shape: torch.Size([1, 256, 12, 12])
ReLU            output shape: torch.Size([1, 256, 12, 12])
MaxPool2d       output shape: torch.Size([1, 256, 5, 5])
Flatten         output shape: torch.Size([1, 6400])
Linear          output shape: torch.Size([1, 4096])
ReLU            output shape: torch.Size([1, 4096])
Dropout         output shape: torch.Size([1, 4096])
Linear          output shape: torch.Size([1, 4096])
ReLU            output shape: torch.Size([1, 4096])
Dropout         output shape: torch.Size([1, 4096])
Linear          output shape: torch.Size([1, 10])
```

> [!TIP] Đọc shape — pattern quan trọng
> - **Spatial size giảm dần**: 224 → 54 → 26 → 12 → 5 (nhờ stride và pooling)
> - **Channels tăng dần**: 1 → 96 → 256 → 384 → 384 → 256 (bù thông tin bị mất khi giảm spatial)
> - **Flatten**: từ tensor 3D (256, 5, 5) → vector 1D (6400)
> - **FC layers**: 6400 → 4096 → 4096 → 10 (nén dần về số classes)

### 4.3 Training trên Fashion-MNIST

Mặc dù AlexNet ban đầu được train trên ImageNet, ta dùng **Fashion-MNIST** vì training ImageNet cần hàng giờ—hàng ngày kể cả trên GPU hiện đại.

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

# QUAN TRỌNG: resize 28×28 → 224×224
# Đây KHÔNG phải best practice — chỉ để ảnh vừa với AlexNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_data = datasets.FashionMNIST('./data', train=True,
                                    download=True, transform=transform)
test_data  = datasets.FashionMNIST('./data', train=False,
                                    transform=transform)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=128)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AlexNet().to(device)

# Xavier initialization — khởi tạo trọng số chuẩn
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
model.apply(init_weights)

# Optimizer: SGD với learning rate nhỏ hơn LeNet
# vì model sâu hơn + phức tạp hơn
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
    
    # Evaluation
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

> [!WARNING] Resize 28→224 là lãng phí!
> Phóng to ảnh 28×28 lên 224×224 **không thêm thông tin** — chỉ thêm compute. Đây là workaround để demo AlexNet trên Fashion-MNIST mà vẫn giữ nguyên kiến trúc gốc.
> 
> So với LeNet, thay đổi chính ở đây là dùng **learning rate nhỏ hơn** (0.01 vs 0.1) và training **chậm hơn nhiều** do mạng sâu hơn, ảnh lớn hơn, convolutions tốn kém hơn.

---

## 5. Discussion — Bài học và hạn chế

AlexNet có cấu trúc **giống LeNet đáng ngạc nhiên**, nhưng với cải tiến quan trọng về accuracy (Dropout) và tốc độ training (ReLU). Điều đáng chú ý không kém là sự tiến bộ của **công cụ** deep learning: những gì mất vài tháng làm trong năm 2012 nay có thể accomplish trong vài chục dòng code.

### 5.1 Achilles Heel: FC layers quá lớn

Xét riêng về hiệu quả, AlexNet có điểm yếu lớn ở **hai FC layers cuối**:

| Ma trận | Kích thước | Memory | Compute |
| --- | --- | --- | --- |
| FC1 (Flatten → 4096) | $6400 \times 4096$ | ~100 MB | ~52 MFLOPs |
| FC2 (4096 → 4096) | $4096 \times 4096$ | ~64 MB | ~34 MFLOPs |
| **Tổng FC** | | **~164 MB** | **~86 MFLOPs** |

164 MB memory và 86 MFLOPs chỉ cho 2 layers — đây là chi phí đáng kể, đặc biệt trên thiết bị nhỏ (phones). Đây là lý do chính khiến AlexNet nhanh chóng bị thay thế bởi các kiến trúc hiệu quả hơn (NiN → GoogLeNet dùng **Global Average Pooling** thay FC, sẽ học ở Buổi 31-32).

### 5.2 Paradox: Tại sao không overfit?

Một điều đáng ngạc nhiên: dù số tham số (>40 triệu) **vượt xa** lượng training data (60.000 ảnh Fashion-MNIST), model hầu như **không overfit** — training loss và validation loss gần như giống nhau suốt quá trình training.

Lý do: Dropout + Data augmentation + improved regularization inherent trong thiết kế deep network hiện đại.

### 5.3 Tốc độ adoption chậm

Mặc dù AlexNet chỉ thêm vài dòng code so với LeNet, cộng đồng học thuật mất **nhiều năm** để chấp nhận sự thay đổi tư duy này. Phần lớn do thiếu công cụ hiệu quả — khi đó chưa có TensorFlow (2016), chỉ mới có Theano (còn thiếu nhiều tính năng) và DistBelief (nội bộ Google). TensorFlow ra đời đã thay đổi tình hình **đáng kể**.

---

## 6. Exercises (từ d2l.ai)

> [!NOTE] Bài tập gốc từ sách — nên làm để hiểu sâu hơn.

1. **Phân tích tính toán của AlexNet:**
   - (a) Tính memory footprint cho conv layers vs FC layers. Layer nào chiếm nhiều hơn?
   - (b) Tính chi phí tính toán (FLOPs) cho conv vs FC.
   - (c) Memory (bandwidth, latency, size) ảnh hưởng compute thế nào? Khác nhau giữa training và inference?

2. **Trade-off thiết kế chip:** Nếu bạn là chip designer, bạn phải cân bằng giữa compute và memory bandwidth. Chip nhanh hơn → cần nhiều điện hơn, diện tích lớn hơn. Memory bandwidth cao hơn → cần nhiều pins và control logic. Bạn tối ưu thế nào?

3. **Tại sao không ai còn benchmark trên AlexNet?**

4. **Tăng epochs:** So với LeNet, kết quả khác gì? Tại sao?

5. **AlexNet quá phức tạp cho Fashion-MNIST** (do ảnh input 28×28 quá nhỏ):
   - (a) Đơn giản hóa model để training nhanh hơn, giữ accuracy.
   - (b) Thiết kế model tốt hơn **trực tiếp** trên 28×28 (không resize).

6. **Batch size:** Thay đổi batch size, quan sát throughput (images/s), accuracy, GPU memory.

7. **Áp dụng Dropout + ReLU vào LeNet.** Có cải thiện không? Có thể cải thiện thêm bằng preprocessing tận dụng invariance của ảnh?

8. **Làm AlexNet overfit:** Feature nào cần bỏ/thay đổi để phá training?

---

## 📖 Từ điển thuật ngữ

| Thuật ngữ | Nghĩa tiếng Việt | Chi tiết |
| --- | --- | --- |
| **Representation Learning** | Học biểu diễn | Model tự học cách mô tả data, thay cho feature engineering thủ công |
| **Feature Engineering** | Kỹ thuật trích đặc trưng | Con người tự thiết kế cách trích features (SIFT, HOG, SURF) |
| **SIFT** | Scale-Invariant Feature Transform | Trích keypoints + orientations bất biến với scale |
| **SURF** | Speeded Up Robust Features | Phiên bản nhanh hơn của SIFT |
| **HOG** | Histograms of Oriented Gradient | Đếm hướng gradient trong từng vùng ảnh |
| **ImageNet** | — | Bộ dữ liệu 1.2M ảnh, 1000 classes (Deng et al., 2009) |
| **ILSVRC** | ImageNet Large Scale Visual Recognition Challenge | Cuộc thi CV hàng năm trên ImageNet (2010-2017) |
| **Dropout** | — | Tắt ngẫu nhiên neurons khi training để chống overfitting |
| **ReLU** | Rectified Linear Unit | $\max(0, x)$ — activation không gây gradient vanishing |
| **CUDA** | Compute Unified Device Architecture | Platform lập trình GPU của NVIDIA |
| **TFLOPS** | Tera Floating-Point Operations Per Second | $10^{12}$ phép tính/giây |
| **End-to-end learning** | Học đầu-cuối | Pixel → Class trực tiếp, không qua bước trung gian |
| **Data Augmentation** | Tăng cường dữ liệu | Tạo thêm data bằng biến đổi (lật, cắt, đổi màu) |

---

## ✅ Bài tự kiểm tra

1. Kể 3 yếu tố ("missing ingredients") khiến CNN "ngủ đông" từ 1998 đến 2012.
2. **Representation Learning** khác **Feature Engineering** ở điểm căn bản nào? Tại sao RL thắng?
3. GPU nhanh hơn CPU cho Deep Learning vì 3 lý do gì?
4. AlexNet có bao nhiêu conv layers và FC layers? Tổng bao nhiêu params? FC layers chiếm bao nhiêu % params?
5. Tại sao Conv1 dùng kernel 11×11 stride 4?
6. Tại sao ReLU thay được Sigmoid cho training mạng sâu? Giải thích bằng gradient.
7. Achilles heel lớn nhất của AlexNet là gì? Các kiến trúc sau giải quyết nó bằng cách nào?

> [!NOTE]- 📝 Đáp án gợi ý
> 1. **(a)** Data thiếu — chỉ có MNIST 60K ảnh nhỏ. **(b)** Hardware yếu — CPU ~1 GFLOPS, chưa có GPU programming framework cho DL. **(c)** Techniques thiếu — chưa có ReLU (gradient vanishing với sigmoid), chưa có Dropout, Xavier init, Adam.
> 2. Feature Engineering: con người **tự nghĩ** cách mô tả data → tốn thời gian, chỉ tốt cho 1 bài toán cụ thể. Representation Learning: model **tự học** representations từ data → nhanh, tổng quát. RL thắng vì model có thể khám phá patterns mà con người không nghĩ ra, và tự tối ưu features cho bài toán cụ thể.
> 3. **(a)** Công suất: dùng nhiều core đơn giản thay 1 core phức tạp → hiệu quả năng lượng hơn. **(b)** Core đơn giản → tiết kiệm transistors cho compute thay vì control logic. **(c)** Bus rộng hơn 10× → bandwidth cao cho DL (cần đọc/ghi dữ liệu liên tục).
> 4. **5 Conv + 3 FC = 8 layers.** ~47M params. FC layers chiếm **>56%** tổng params (FC1 alone: 6400×4096 = 26.2M).
> 5. Ảnh 224×224 lớn gấp 8× so với 28×28. Cần receptive field lớn ở tầng đầu để "nhìn" features thô. Stride 4 giảm 224→54 ngay, tiết kiệm compute cho các tầng sau.
> 6. Sigmoid: đạo hàm max 0.25. Qua $n$ layers: $0.25^n \to 0$. ReLU: đạo hàm = 1 khi $x>0$. Qua $n$ layers: $1^n = 1$ → gradient giữ nguyên → train được mạng sâu.
> 7. FC layers quá lớn (~164MB, >56% params). NiN/GoogLeNet thay FC bằng **Global Average Pooling** → giảm hàng triệu params.

---

## 🔗 Liên kết

- **Buổi trước**: [[Buổi 28 - Tuần 8]] — 7.6 LeNet: CNN cơ bản
- **Buổi sau**: [[Buổi 30 - Tuần 8]] — 8.2 VGG: Networks Using Blocks
- **Concepts**: [[Activation Function]], [[Multilayer Perceptron]], [[Dropout]]
- **Source**: [d2l.ai — 8.1 AlexNet](https://d2l.ai/chapter_convolutional-modern/alexnet.html)
