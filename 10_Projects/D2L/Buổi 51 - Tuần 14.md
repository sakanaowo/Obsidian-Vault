---
session: "D2L Tuần 14, Buổi 51 — 11.2 Attention Pooling by Similarity"
aliases: ["Buổi 51"]
tags: [d2l, deep-learning, attention, nadaraya-watson, kernel, nlp]
status: growth
source: "D2L Chapter 11.2 — Attention Pooling by Similarity"
created: 2026-04-25
related:
  - "[[Buổi 50 - Tuần 14]]"
  - "[[Buổi 52 - Tuần 14]]"
---

# Buổi 51 — 11.2 Attention Pooling by Similarity

> [!NOTE] Mục tiêu buổi học
>
> - [ ] Hiểu Nadaraya-Watson estimator — tiền thân phi tham số của attention
> - [ ] Nắm bốn kernel functions (Gaussian, Boxcar, Epanechikov, Constant)
> - [ ] Hiểu kết nối: NW = attention với hand-crafted kernel, learned attention = NW với learned kernel
> - [ ] Phân biệt được giữa hand-crafted attention và learned attention
> - [ ] Thấy trực quan: kernel width ảnh hưởng thế nào đến smoothness vs local adaptation

---

## Active Recall — Ôn lại Buổi 50

### Câu hỏi truy hồi (không nhìn tài liệu)

1. Attention pooling formula (11.1.1) là gì? Cho biết ý nghĩa từng thành phần QKV.
2. Tại sao softmax đảm bảo gradient không biến mất? (So sánh với sigmoid)
3. Phân biệt convex cone (chỉ $\alpha_i \geq 0$) vs convex combination ($\sum \alpha_i = 1$ VÀ $\alpha_i \geq 0$)
4. Tại sao attention gọi là "khả vi"? Liên hệ Nadaraya-Watson estimator.
5. Nếu dùng raw scores (không softmax) làm trọng số chú ý (attention weights) — điều gì sai?

### Trả lời nhanh

- **Q1:** $\textbf{Attention}(\mathbf{q}, \mathcal{D}) = \sum_i \alpha(\mathbf{q}, \mathbf{k}_i)\mathbf{v}_i$. **Claim** → **Reasoning** → Query decoder hidden state tìm "câu hỏi", Keys encoder hidden states đóng vai trò "định danh", Values encoder hidden states chứa "nội dung". **Evidence** → Công thức tổng quát Eq. 11.1.1 trong D2L.

- **Q2:** $\nabla \exp(x) = \exp(x) > 0$. **Claim** → **Reasoning** → Sigmoid có $\sigma'(x) = \sigma(x)(1-\sigma(x)) \to 0$ khi $x \gg 0$ (saturation). Softmax gradient luôn dương và tỷ lệ với giá trị. **Evidence** → Chain rule với softmax: $\nabla_{s_i} \text{softmax}_i = \text{softmax}_i(1 - \text{softmax}_i)$.

- **Q3:** Convex cone chỉ nonnegative ($\alpha_i \geq 0$), output có thể "phóng đại". **Claim** → **Reasoning** → Không có ràng buộc tổng = 1, nên output có thể vượt ra ngoài range của values gốc. Convex combination thêm $\sum \alpha_i = 1$, output bị chặn trong bao lồi. **Evidence** → D2L Section 11.1.1.

- **Q4:** Vì tất cả phép toán đều khả vi: dot product, softmax, weighted sum. **Claim** → **Reasoning** → Gradient flow được qua trọng số chú ý (attention weights). NW cũng tương tự: dùng kernel similarity làm trọng số chú ý (attention weights), nhưng kernel là **cố định** (hand-crafted), không có tham số có thể học. **Evidence** → D2L Section 11.1.2.

- **Q5:** (a) Không sum-to-1 → output scale phụ thuộc score magnitude; (b) Có thể âm → "trừ" information; (c) Gradient không bị chặn → training unstable. **Claim** → **Reasoning** → Softmax sinh ra convex combination với 2 điều kiện: nonnegative và sum=1. Raw scores không đảm bảo điều này. **Evidence** → Phân tích gradient trong D2L.

### Liên kết cần ôn lại

- [[Buổi 50 - Tuần 14|QKV — Queries, Keys, and Values]]
- [[Softmax Function]]

---

# PHẦN I — NADARAYA-WATSON: TIỀN THÂN CỦA ATTENTION

## 1.1 Từ Attention về Nadaraya-Watson

> [!NOTE] ELI5
>
> Nadaraya-Watson (NW) giống như khi bạn muốn đoán giá tiền một căn nhà mới. Thay vì dùng công thức phức tạp, bạn nhìn vào những căn nhà **gần đó** đã bán, lấy trung bình có trọng số — nhà càng gần thì trọng số càng lớn. "Khoảng cách" ở đây được đo bằng **kernel** (một loại "thước đo sự tương tự"). Attention hiện đại chính là phiên bản **có thể học được** của ý tưởng này.

**Định nghĩa kỹ thuật:**

- **Đây là gì?** Nadaraya-Watson estimator là phương pháp **phi tham số** (non-parametric) để ước lượng hàm regression, sử dụng tổng có trọng số (weighted sum) của các điểm training gần với query.
- **Input/Output là gì?** Input: training data $(\mathbf{x}_i, y_i)$ và query $\mathbf{q}$. Output: giá trị dự đoán $f(\mathbf{q}) = \sum_i y_i \cdot \frac{K(\mathbf{q}, \mathbf{x}_i)}{\sum_j K(\mathbf{q}, \mathbf{x}_j)}$.
- **Giải quyết vấn đề gì?** Dự đoán giá trị tại điểm mới mà không cần giả định về dạng hàm cụ thể (non-parametric).
- **Thay thế/gợi ý giải pháp nào trước đây?** Các phương pháp parametric regression (linear, polynomial) đòi hỏi giả định về dạng hàm.

### Kết nối với Buổi 50

| Khía cạnh | Nadaraya-Watson (11.2) | Attention (11.1) |
|-----------|----------------------|-----------------|
| Query $\mathbf{q}$ | Vị trí cần predict | Decoder hidden state |
| Keys $\mathbf{k}_i$ | Training features $x_i$ | Encoder hidden states |
| Values $\mathbf{v}_i$ | Training labels $y_i$ | Encoder hidden states |
| Trọng số chú ý (attention weights) | $\alpha = K(\mathbf{q}, \mathbf{k}_i)$ (cố định kernel) | $\alpha = \text{softmax}(a(\mathbf{q}, \mathbf{k}_i))$ (có thể học) |
| Có học được không? | **Không** — kernel cố định | **Có** — $W_Q, W_K, W_V$ có thể học |
| Output | $f(\mathbf{q}) = \sum_i y_i \cdot \frac{K}{\sum K}$ | $\sum_i \mathbf{v}_i \cdot \text{softmax}(a)$ |

> [!KEY]- Key Insight (D2L)
>
> "Người đọc tinh ý có thể tự hỏi tại sao chúng ta đi sâu vào một phương pháp đã hơn nửa thế kỷ tuổi. Thứ nhất, đây là một trong những tiền thân sớm nhất của attention mechanisms hiện đại. Thứ hai, nó rất tốt cho việc trực quan hóa. Thứ ba, và quan trọng không kém, nó cho thấy **giới hạn của attention mechanisms được thiết kế thủ công**. Một chiến lược tốt hơn nhiều là **học cơ chế này**, bằng cách học các biểu diễn cho queries và keys."

> [!CHECKLIST]-
> Reader tự kiểm tra:
>
> - [ ] Tôi giải thích được: NW = weighted sum của labels gần query, weights từ kernel similarity
> - [ ] Tôi phân biệt được: kernel cố định (NW) vs learned projections ($W_Q, W_K, W_V$)
> - [ ] Tôi hiểu tại sao NW gọi là "non-parametric" — không có learnable parameters

---

# PHẦN II — BỐN KERNEL FUNCTIONS (D2L Eq. 11.2.1)

## 2.1 Giới thiệu Kernels

> [!NOTE] ELI5
>
> Kernel giống như "thước đo khoảng cách" trong một bữa tiệc. Khi bạn hỏi "Ai gần tôi nhất?", có nhiều cách đo:
>
> - **Gaussian**: Dùng thước "đường chim bay", điểm nào cũng có contribution nhưng giảm dần theo khoảng cách
> - **Boxcar**: Chỉ nhìn những người trong bán kính 1m, ngoài ra không quan tâm
> - **Epanechikov**: Như Boxcar nhưng có đường cong mềm mại hơn, giảm từ từ về 0
> - **Constant**: Không phân biệt ai, mọi người bằng nhau

**Định nghĩa kỹ thuật:**

- **Đây là gì?** Kernel function là hàm đo "độ tương tự" (similarity) giữa query và key, dựa trên khoảng cách.
- **Input/Output là gì?** Input: query $\mathbf{q}$ và key $\mathbf{k}$. Output: scalar $\alpha \in [0, 1]$ hoặc $[0, \infty)$ tùy kernel.
- **Giải quyết vấn đề gì?** Cung cấp cách tính trọng số chú ý (attention weights) mà không cần học từ data (hand-crafted).
- **Thay thế/gợi ý giải pháp nào trước đây?** Trước đây dùng hard boundaries hoặc uniform weights.

### Công thức bốn Kernels (D2L Eq. 11.2.1)

Cho query $\mathbf{q}$ và key $\mathbf{k}$. Các kernel functions định nghĩa similarity:

$$\alpha(\mathbf{q}, \mathbf{k}) \stackrel{\text{def}}{=} \begin{cases} \exp\!\left(-\frac{1}{2} \|\mathbf{q} - \mathbf{k}\|^2\right) & \text{Gaussian} \\[6pt] 1 \quad \text{nếu } \|\mathbf{q} - \mathbf{k}\| \leq 1 & \text{Boxcar} \\[6pt] \max\!\left(0, 1 - \|\mathbf{q} - \mathbf{k}\|\right) & \text{Epanechikov} \end{cases}$$

**Từ điển ký hiệu:**

| Ký hiệu | Định nghĩa |
|---------|-------------|
| $\mathbf{q}$ | Query vector — vị trí cần predict |
| $\mathbf{k}$ | Key vector — vị trí training point |
| $\|\mathbf{q} - \mathbf{k}\|$ | Euclidean distance giữa query và key |
| $\exp(x)$ | Hàm exponential |
| $\max(0, x)$ | Rectified linear — giá trị không âm |

### Bảng so sánh bốn kernels

| Kernel          | Công thức                                                   | Đặc điểm                     | Phạm vi                               | Dạng hình                                             |
| --------------- | ----------------------------------------------------------- | ---------------------------- | ------------------------------------- | ----------------------------------------------------- |
| **Gaussian**    | $\exp\!\left(-\frac{\|\mathbf{q}-\mathbf{k}\|^2}{2}\right)$ | Mượt, đường cong hình chuông | Toàn bộ không gian (infinite support) | ![](assets/attachments/d2l-buoi-51/kernel-shapes.png) |
| **Boxcar**      | $\mathbb{1}(\|\mathbf{q}-\mathbf{k}\| \leq 1)$              | Cứng, "bậc thang"            | Chỉ trong bán kính 1                  |                                                       |
| **Epanechikov** | $\max(0, 1 - \|\mathbf{q}-\mathbf{k}\|)$                    | Vòm, tuyến tính từng khúc    | Trong bán kính 1, 0 ngoài             |                                                       |
| **Constant**    | $1$                                                         | Không phân biệt              | Mọi nơi (uniform)                     |                                                       |

**Phân tích hình dạng từng kernel:**

- **Gaussian**: Mượt nhất — mọi điểm đều có contribution, giảm dần theo khoảng cách. $\sigma$ càng nhỏ → càng "nhọn" (ít smooth).
- **Boxcar**: Cứng — hoặc 1 (trong bán kính) hoặc 0 (ngoài). Không có transition mềm.
- **Epanechikov**: Vòm mềm — giảm tuyến tính về 0. "Mềm hơn" Boxcar.
- **Constant**: Không phân biệt — mọi điểm bằng nhau. Tương ứng với **uniform attention** (Buổi 50).

**Tính chất quan trọng:** Tất cả kernels trong D2L đều **translation and rotation invariant** — nếu dịch hoặc xoay cả query và key cùng cách, kernel value không đổi. Vì lý do đó, D2L đơn giản hóa thành scalar arguments $k, q \in \mathbb{R}$.

> [!WARNING]- Dấu hiệu nhồi nhét
>
> Nếu bạn nhớ "Gaussian là phổ biến nhất" mà không hiểu **tại sao**: vì nó mượt và khả vi ở mọi nơi → gradient flow tốt, dễ phân tích. Hãy suy nghĩ về trade-off giữa Boxcar (đơn giản nhưng không liên tục) và Gaussian (mượt nhưng infinite support).

> [!CHECKLIST]-
> Reader tự kiểm tra:
>
> - [ ] Tôi viết được công thức cả bốn kernels
> - [ ] Tôi biết tại sao Gaussian "mượt" hơn Boxcar — vì khả vi ở mọi điểm
> - [ ] Tôi giải thích được tại sao Constant kernel thất bại — không có weighting

---

# PHẦN III — NW FORMULA VÀ REGRESSION CONTEXT

## 3.1 Nadaraya-Watson Regression Formula (D2L Eq. 11.2.2)

> [!NOTE] ELI5
>
> Để đoán giá nhà tại $q$: nhìn tất cả nhà đã bán $(x_i, y_i)$, dùng "thước đo" (kernel) để xem nhà nào "gần" $q$ nhất, rồi lấy trung bình có trọng số — nhà càng gần thì đóng góp càng nhiều.

**Định nghĩa kỹ thuật (D2L 11.2.2):**

- **Đây là gì?** NW estimator là tổng có trọng số (weighted sum) của labels, với weights từ kernel similarity.
- **Input/Output là gì?** Input: training data $(\mathbf{x}_i, y_i)$ và query $\mathbf{q}$. Output: $f(\mathbf{q}) = \sum_i \mathbf{v}_i \cdot \frac{\alpha(\mathbf{q}, \mathbf{k}_i)}{\sum_j \alpha(\mathbf{q}, \mathbf{k}_j)}$.
- **Giải quyết vấn đề gì?** Regression không cần giả định về dạng hàm.

$$f(\mathbf{q}) = \sum_i \mathbf{v}_i \frac{\alpha(\mathbf{q}, \mathbf{k}_i)}{\sum_j \alpha(\mathbf{q}, \mathbf{k}_j)}$$

**Từ điển ký hiệu:**

| Ký hiệu | Định nghĩa |
|---------|-------------|
| $\mathbf{q}$ | Query vector — vị trí cần predict |
| $\mathbf{k}_i$ | Key vector thứ $i$ (trong NW = $\mathbf{x}_i$) |
| $\mathbf{v}_i$ | Value vector thứ $i$ (trong NW = $y_i$) |
| $\alpha(\mathbf{q}, \mathbf{k}_i)$ | Kernel similarity score |
| $m$ | Số lượng training points |

**Trong regression setting:**

- Training data: $(\mathbf{x}_i, y_i)$ — features và labels
- $\mathbf{k}_i = \mathbf{x}_i$ — keys = training features
- $\mathbf{v}_i = y_i$ — values = training labels (scalars)
- $\mathbf{q}$ — vị trí mới cần dự đoán

**Trong classification (multiclass):**
- $\mathbf{v}_i$ = one-hot encoding của $y_i$
- Kết quả là phân phối xác suất over classes

> [!IMPORTANT]- NW estimator **không cần training**
>
> Đây là **phi tham số** (non-parametric) method. Không có learnable parameters. Kết quả hoàn toàn phụ thuộc vào data và kernel được chọn.
>
> Nếu thu hẹp kernel width khi data tăng (consistency condition, Mack & Silverman 1982), NW **sẽ hội tụ** về giải pháp tối ưu về mặt thống kê.

**Dataset cho Demo (D2L Eq. 11.2.3):**

D2L tạo dataset: $y_i = 2\sin(x_i) + x_i + \epsilon$, với $\epsilon \sim \mathcal{N}(0, 1)$. 40 training examples, $x_i \in [0, 5]$.

```python
def f(x):
    return 2 * torch.sin(x) + x

n = 40
x_train, _ = torch.sort(torch.rand(n) * 5)
y_train = f(x_train) + torch.randn(n)
x_val = torch.arange(0, 5, 0.1)
y_val = f(x_val)
```

**Từ điển ký hiệu:**

| Ký hiệu | Định nghĩa |
|---------|-------------|
| $x_i$ | Training feature (scalar) |
| $y_i$ | Training label (scalar) |
| $\epsilon$ | Noise term |
| $\mathcal{N}(0, 1)$ | Standard normal distribution |

> [!CHECKLIST]-
> Reader tự kiểm tra:
>
> - [ ] Tôi viết được công thức NW (Eq. 11.2.2)
> - [ ] Tôi giải thích được: tử số = tổng có trọng số (weighted sum), mẫu số = normalization
> - [ ] Tôi hiểu tại sao NW không cần training — không có learnable parameters

---

# PHẦN IV — IMPLEMENTATION VÀ VISUALIZATION

## 4.1 Hàm nadaraya_watson

> [!NOTE] ELI5
>
> Code tính NW giống như một công thức nấu ăn: (1) Tính "khoảng cách" giữa query và mỗi training point, (2) Đưa vào máy xay (kernel) để ra similarity score, (3) Normalize để tổng = 1, (4) Nhân với labels rồi cộng lại.

**Định nghĩa kỹ thuật:**

- **Đây là gì?** Implementation của NW estimator với kernel tùy chọn.
- **Input/Output là gì?** Input: x_train $(n)$, y_train $(n)$, x_val $(m)$, kernel function. Output: y_hat $(m)$, attention_weights $(n \times m)$.
- **Giải quyết vấn đề gì?** Tính NW regression estimate một cách vectorized.

```python
def nadaraya_watson(x_train, y_train, x_val, kernel):
    """NW estimator với kernel tùy chọn.

    Args:
        x_train: (n_train,) — training features
        y_train: (n_train,) — training labels
        x_val: (n_val,) — queries cần predict
        kernel: callable — hàm kernel nhận khoảng cách, trả về similarity

    Returns:
        y_hat: (n_val,) — predicted values
        attention_w: (n_train, n_val) — trọng số chú ý (attention weights)
    """
    dists = x_train.reshape((-1, 1)) - x_val.reshape((1, -1))
    # dists[i, j] = x_train[i] - x_val[j]
    k = kernel(dists).type(torch.float32)
    # Normalize over keys (columns) cho mỗi query (row)
    attention_w = k / k.sum(0)
    # Matrix multiply: y_train @ attention_w
    y_hat = y_train @ attention_w
    return y_hat, attention_w
```

**Phân tích từng bước:**

1. `dists = x_train.reshape((-1,1)) - x_val.reshape((1,-1))` — tạo ma trận khoảng cách: hàng = query, cột = key. Shape: `(n_train, n_val)`.
2. `k = kernel(dists)` — tính kernel similarity cho mỗi cặp (query, key).
3. `attention_w = k / k.sum(0)` — **normalize over keys (axis=0)** — tức $\alpha_i / \sum_j \alpha_j$. Đây là **softmax normalization** nhưng không có exp — chỉ simple ratio normalization.
4. `y_hat = y_train @ attention_w` — tổng có trọng số (weighted sum). Shape: `(n_val,)`.

> [!CRITICAL]- Axis normalization trong attention
>
> Bình thường trọng số chú ý (attention weights) normalize over keys (column), nhưng tùy convention. Trong transformer (Buổi 55-56), ta normalize over keys bằng `softmax` theo last axis. Ở đây D2L dùng simple divide vì kernels đã đảm bảo nonnegative.

> [!CHECKLIST]-
> Reader tự kiểm tra:
>
> - [ ] Tôi giải thích được tại sao `k.sum(0)` normalize over keys (columns)
> - [ ] Tôi hiểu kích thước (shape) `(n_train, n_val)`: hàng = key, cột = query
> - [ ] Tôi biết đây là "simple ratio" normalization, không phải softmax thật sự

## 4.2 Regression Estimates — So sánh bốn Kernels

**Phân tích từng kernel:**

| Kernel | $y_{\text{predict}}$ | Nhận xét |
|--------|---------------------|----------|
| **Gaussian** | Khá sát ground truth (đường đứt) | Mượt, khớp tốt. Đây là default choice. |
| **Boxcar** | Tương đương Gaussian, hơi gồ ghề | Chỉ nhìn trong bán kính 1 → local, nhưng discontinuous edges |
| **Epanechikov** | Tương tự Gaussian/Boxcar | Rất giống hai cái trên về kết quả — trọng số chú ý (attention weights) tương tự dù hàm khác nhau |
| **Constant** | Đường ngang ~mean của y | **Thất bại** — trả về $\frac{1}{n}\sum_i y_i$ cho mọi $q$ (uniform attention). Không có chọn lọc. |

![[assets/attachments/d2l-buoi-51/nw-regression-comparison.png]]
_Hình 1. So sánh regression estimates với bốn kernels. Gaussian, Boxcar, Epanechikov đều fit tốt. Constant chỉ trả về mean — không có local adaptation._

> [!KEY]- D2L observation
>
> "Điều đầu tiên nổi bật là cả ba kernel nontrivial (Gaussian, Boxcar, và Epanechikov) đều tạo ra các ước lượng khá khả dụng, không quá xa hàm thực. Chỉ có constant kernel... tạo ra kết quả khá phi thực tế."
>
> Quan trọng: Gaussian, Boxcar, Epanechikov cho kết quả **rất giống nhau** dù hàm kernel khác nhau. Điều này gợi ý: **kernel shape không quan trọng bằng việc có kernel hay không**.

## 4.3 Attention Weights Heatmap — Trực quan hóa Attention

**Phân tích heatmap:**

| Kernel | Heatmap pattern | Giải thích |
|--------|----------------|------------|
| **Gaussian** | Diagonal band rộng | Query tại $x$ chú ý nhiều vào keys gần $x$ (Gaussian tails) |
| **Boxcar** | Diagonal band cứng | Chỉ nhìn trong bán kính 1 — hard boundary |
| **Epanechikov** | Diagonal band vừa | Tương tự Gaussian nhưng sharp edges |
| **Constant** | Màu đều (uniform) | Mọi query nhìn mọi key bằng nhau — không có chọn lọc |

![[assets/attachments/d2l-buoi-51/nw-attention-weights.png]]
_Hình 2. Trọng số chú ý (attention weights) heatmap cho bốn kernels. Trục x = queries, trục y = keys. Màu sáng = attention cao. Gaussian/Boxcar/Epanechikov có pattern diagonal (query attend vào nearby keys), Constant là uniform._

> [!CHECKLIST]-
> Reader tự kiểm tra:
>
> - [ ] Tôi đọc được heatmap: diagonal = "query attend vào nearby keys"
> - [ ] Tôi giải thích được tại sao Constant heatmap đều màu
> - [ ] Tôi biết "diagonal band" nghĩa là gì — attention locality

---

# PHẦN V — ADAPTING ATTENTION POOLING: KERNEL WIDTH

## 5.1 Tại sao cần điều chỉnh Width?

> [!NOTE] ELI5
>
> Gaussian kernel có "độ rộng" $\sigma$ giống như zoom của camera. Zoom gần ($\sigma$ nhỏ) → nhìn rõ chi tiết nhỏ nhưng bỏ lỡ bức tranh toàn cảnh. Zoom xa ($\sigma$ lớn) → thấy toàn cảnh nhưng bỏ lỡ chi tiết.

**Định nghĩa kỹ thuật:**

- **Đây là gì?** Kernel width (hoặc bandwidth) $\sigma$ là hyperparameter điều chỉnh "độ rộng" của kernel.
- **Input/Output là gì?** Input: khoảng cách $\|\mathbf{q} - \mathbf{k}\|$. Output: $\alpha = \exp(-\|\mathbf{q} - \mathbf{k}\|^2 / 2\sigma^2)$.
- **Giải quyết vấn đề gì?** Điều chỉnh trade-off giữa bias và variance.
- **Thay thế/gợi ý giải pháp nào trước đây?** Trước đây dùng fixed kernel width.

D2L Section 11.2.3: Gaussian kernel với width parameter:

$$\alpha(\mathbf{q}, \mathbf{k}) = \exp\!\left(-\frac{\|\mathbf{q} - \mathbf{k}\|^2}{2\sigma^2}\right)$$

**Từ điển ký hiệu:**

| Ký hiệu | Định nghĩa |
|---------|-------------|
| $\sigma$ | Kernel width (bandwidth parameter) |
| $\|\mathbf{q} - \mathbf{k}\|^2$ | Squared Euclidean distance |

$\sigma$ càng nhỏ → kernel càng "nhọn" → attention càng tập trung (narrow). $\sigma$ càng lớn → kernel càng "phẳng" → attention càng lan tỏa (wide).

## 5.2 Effect của Sigma trên Regression

**Bảng phân tích:**

| $\sigma$ | Bandwidth | Smoothness | Local adaptation | Overfitting risk |
|----------|-----------|-----------|----------------|-----------------|
| **0.1** | Rất hẹp | Thấp | Rất cao (nhọn) | Cao — đi theo noise |
| **0.2** | Hẹp | Thấp | Cao | Đáng kể |
| **0.5** | Trung bình | Tốt | Trung bình | Thấp |
| **1.0** | Rộng | Rất cao | Thấp | Thấp — có thể underfit |

![[assets/attachments/d2l-buoi-51/nw-gaussian-width.png]]
_Hình 3. Gaussian kernel với các giá trị $\sigma$ khác nhau. $\sigma$ càng nhỏ → đường càng nhọn, fit noise. $\sigma$ càng lớn → đường càng mượt, có thể underfit._

> [!KEY]- D2L observation
>
> "Rõ ràng, kernel càng hẹp thì ước lượng càng ít mượt. Đồng thời, nó thích nghi tốt hơn với các biến thiên cục bộ."
>
> Trade-off kinh điển: **bias-variance tradeoff**. Kernel hẹp → bias thấp (fit các patterns cục bộ) nhưng variance cao (nhạy với noise). Kernel rộng → bias cao (smooth away details) nhưng variance thấp (dự đoán ổn định).

## 5.3 Heuristics cho Kernel Width

D2L đề cập Silverman (1986) heuristics cho adaptive width theo local density. Ngoài ra:

- **Per-coordinate width**: $\sigma_d$ khác nhau cho mỗi dimension — dùng trong attention scaling (Buổi 52)
- **Learned width**: thay vì hand-crafted, cho model tự học width qua attention scaling factor $\sqrt{d_k}$ (Vaswani 2017)
- **Data-driven**: cross-validation để chọn $\sigma$

> [!CHECKLIST]-
> Reader tự kiểm tra:
>
> - [ ] Tôi vẽ được hình dạng Gaussian với $\sigma$ lớn và nhỏ
> - [ ] Tôi giải thích được bias-variance tradeoff: hẹp → low bias, high variance
> - [ ] Tôi biết $\sqrt{d_k}$ (Buổi 52) là cách attention học width thay vì hand-crafted

---

# PHẦN VI — QUAN SÁT TRỰC QUAN: QUERY TẠI MỘT ĐIỂM

## 6.1 Attention tại một Query cụ thể

![[assets/attachments/d2l-buoi-51/nw-query-diagram.png]]
_Hình 4. Query tại $x = 2.0$. Điểm xanh = 40 training points, đường đứt = ground truth. Độ dày đường kết nối = trọng số chú ý (attention weights) $\alpha_i$._

**Giải thích hình:**

- Điểm xanh: 40 training points $(x_i, y_i)$
- Đường đứt hồng: ground truth $f(x) = 2\sin(x) + x$
- Đường dọc cam: query $q = 2.0$
- Độ dày của đường kết nối: trọng số chú ý (attention weights) $\alpha_i$ — đường càng dày → weight càng lớn

**Observation:** Query tại $x=2.0$ chủ yếu "chú ý" đến các training points gần $x=2.0$. Điểm ở xa gần như không contribute (đường mỏng).

> [!CHECKLIST]-
> Reader tự kiểm tra:
>
> - [ ] Tôi giải thích được: đường dày = trọng số chú ý (attention weights)
> - [ ] Tôi nhận ra: query chú ý nhiều vào nearby points
> - [ ] Tôi hiểu: "nhìn xa" có nghĩa là trọng số chú ý (attention weights) nhỏ

---

# PHẦN VII — HAND-CRAFTED VS LEARNED ATTENTION

## 7.1 NW Estimator = Attention với Hand-Crafted Kernel

**Từ điển ký hiệu NW → Attention:**

| NW terminology | Attention terminology | Notes |
|---------------|---------------------|-------|
| Query $q$ | Query $\mathbf{q}$ | Vị trí cần predict / decoder state |
| Keys $k_i$ | Keys $\mathbf{k}_i$ | Training features / encoder states |
| Values $v_i$ | Values $\mathbf{v}_i$ | Training labels / encoder states |
| Kernel $\alpha(q, k_i) = K(q - k_i)$ | Compatibility $a(\mathbf{q}, \mathbf{k}_i)$ | Fixed distance function vs learned projection |
| $f(q) = \sum_i v_i \cdot \frac{\alpha}{\sum \alpha}$ | $\text{Attention}(\mathbf{q}, \mathcal{D}) = \sum_i \mathbf{v}_i \cdot \text{softmax}(a)$ | Cùng cấu trúc tổng có trọng số (weighted sum) |

**Sự khác biệt then chốt:**

| Khía cạnh | Nadaraya-Watson | Learned Attention (Buổi 52+) |
|-----------|----------------|------------------------------|
| Similarity function | **Cố định** kernel (ví dụ: Gaussian) | **Có thể học** dot product hoặc MLP |
| Parameters | Không có | $W_Q, W_K, W_V$ có thể học |
| Adaptability | Phụ thuộc vào hyperparameter $\sigma$ | Tự học từ data |
| Generalization | Chỉ nội suy mượt | Học các patterns phức tạp |
| Use case | Regression/classification cổ điển | Seq2Seq, Transformer, NLP |

> [!WARNING]- Dấu hiệu nhồi nhét
>
> Nếu bạn nghĩ "attention = NW với dot product thay kernel" → **đúng về mặt công thức** nhưng **thiếu bản chất**. Điểm khác biệt quan trọng nhất: attention có **learnable projections** ($W_Q, W_K, W_V$) — nghĩa là model tự học **cách biểu diễn** query, key, value, không chỉ học cách tính similarity.

> [!CHECKLIST]-
> Reader tự kiểm tra:
>
> - [ ] Tôi phân biệt được: hand-crafted (NW) vs learned attention ($W_Q, W_K, W_V$)
> - [ ] Tôi giải thích được: NW không cần training — không có learnable parameters
> - [ ] Tôi biết: key insight là **learnable projections** chứ không phải chỉ thay kernel

---

# PHẦN VIII — TÓM TẮT VÀ LIÊN KẾT

## 8.1 Tóm tắt buổi

| Khái niệm | Hiểu | Cần ôn |
|-----------|------|--------|
| NW estimator = attention precursor | ✅ | |
| Bốn kernels (Gaussian, Boxcar, Epanechikov, Constant) | ✅ | |
| Eq 11.2.1 và Eq 11.2.2 | ✅ | |
| NW không cần training (non-parametric) | ✅ | |
| Gaussian width $\sigma$ effect | ✅ | |
| Bias-variance trade-off | ✅ | |
| Hand-crafted vs learned attention | ✅ | |
| Kết nối với QKV (Buổi 50) | ✅ | |

## 8.2 Liên kết với các buổi tiếp theo

| Buổi | Chủ đề | Liên kết |
|------|--------|---------|
| **Buổi 52** (11.3) | Attention Scoring Functions | Từ hand-crafted kernel → learned scoring function ($q^T k_i$, MLP) |
| **Buổi 53** (11.4) | Bahdanau Attention | Cross-attention trong Seq2Seq: Query=decoder, Keys/Values=encoder |
| **Buổi 54** (11.5) | Multi-Head Attention | Học đa quan điểm: nhiều $W_Q, W_K, W_V$ song song |

## 8.3 Bảng thuật ngữ

| Thuật ngữ | Định nghĩa |
|-----------|-----------|
| Nadaraya-Watson estimator | Non-parametric regression: $f(q) = \sum_i y_i \cdot \frac{K(q, x_i)}{\sum_j K(q, x_j)}$ |
| Kernel function $K(\mathbf{q}, \mathbf{k})$ | Hàm similarity giữa query và key — đo "khoảng cách" |
| Gaussian kernel | $K(q,k) = \exp(-\|q-k\|^2/2)$ — mượt, infinite support |
| Boxcar kernel | $K(q,k) = 1$ nếu $\|q-k\| \leq 1$, else 0 — hard boundary |
| Epanechikov kernel | $K(q,k) = \max(0, 1-\|q-k\|)$ — piecewise linear |
| Kernel width $\sigma$ | Hyperparameter điều chỉnh "độ rộng" của kernel — trade-off mượt vs cục bộ |
| Translation invariant | $K(q, k) = K(q - k, 0)$ — không phụ thuộc absolute position |
| Non-parametric | Không có fixed functional form — complexity tăng với data size |
| Consistency (Mack & Silverman 1982) | NW hội tụ về optimal khi data $\to \infty$ và kernel width $\to 0$ appropriately |
| Hand-crafted attention | Attention với kernel/function được chọn cố định, không học |
| Learned attention | Attention với $W_Q, W_K, W_V$ có thể học — model tự học representations |
| Trọng số chú ý (attention weights) | Trọng số $\alpha_i$ trong tổng có trọng số (weighted sum), quyết định mức độ "chú ý" vào mỗi value |

---

## Bài tập D2L 11.2.5

1. **Parzen windows density estimation**: $\hat{p}(\mathbf{x}) = \frac{1}{n}\sum_i k(\mathbf{x}, \mathbf{x}_i)$. Chứng minh rằng trong binary classification, hàm $\hat{p}(\mathbf{x}, y=1) - \hat{p}(\mathbf{x}, y=-1)$ từ Parzen windows **tương đương** với NW classification.

2. **Tối ưu kernel width bằng SGD**: Implement gradient descent để tìm giá trị tốt cho kernel width $\sigma$ trong NW regression.

3. **Overfitting khi minimize MSE trực tiếp**: Điều gì xảy ra nếu minimize $(f(\mathbf{x}_i) - y_i)^2$ trực tiếp, với $y_i$ nằm trong các terms tính $f$? → Gợi ý: $y_i$ là part of the terms used to compute $f$ → circular dependency.

4. **Leave-one-out overfitting**: Remove $(\mathbf{x}_i, y_i)$ khỏi estimate cho $f(\mathbf{x}_i)$, rồi optimize kernel widths. Observation: overfitting vẫn xảy ra nếu kernel quá narrow.

5. **Đơn giản hóa khi $\|\mathbf{x}\|=1$ (unit sphere)**: Nếu mọi $\mathbf{x}$ nằm trên unit sphere, đơn giản hóa $\|\mathbf{x} - \mathbf{x}_i\|^2$ trong exponential. → $\|\mathbf{x} - \mathbf{x}_i\|^2 = 2 - 2\mathbf{x}^\top\mathbf{x}_i$ → Gaussian kernel becomes $\exp(-\sigma^2(1 - \mathbf{x}^\top\mathbf{x}_i))$ → **rất liên quan đến dot-product attention** (Buổi 52).

6. **Consistency và tốc độ giảm scale**: Khi data tăng, kernel width nên giảm với tốc độ nào? → $\sigma \to 0$ khi $n \to \infty$ theo tốc độ phù hợp. Trong high dimensions, cần giảm nhanh hơn (curse of dimensionality). Gợi ý: Mack & Silverman 1982.

---

## Active Recall — Câu hỏi về Buổi 51

1. **Cho dataset $(\mathbf{x}_i, y_i)$ và Gaussian kernel với $\sigma=0.5$. Tính $f(2.0)$ cho hai điểm: $x_1=1.9, y_1=3.0$ và $x_2=3.5, y_2=1.0$.**

→ **Claim**: $f(2.0) \approx 2.97$.
→ **Reasoning**: $K(q=2.0, k_1=1.9) = \exp(-(0.1)^2/0.5) \approx 0.980$, $K(q=2.0, k_2=3.5) = \exp(-(1.5)^2/0.5) \approx 0.011$. $w_1 = 0.980/(0.980+0.011) \approx 0.989$, $w_2 \approx 0.011$. $f(2.0) \approx 0.989 \cdot 3.0 + 0.011 \cdot 1.0 \approx 2.97$.
→ **Evidence**: Áp dụng công thức NW Eq. 11.2.2.

2. **Tại sao Gaussian, Boxcar, Epanechikov cho kết quả regression gần như giống nhau?**

→ **Claim**: Vì chúng đều "ưu tiên điểm gần hơn điểm xa".
→ **Reasoning**: Sự khác biệt về functional form ít quan trọng bằng việc có weighting scheme hay không. Constant kernel thất bại vì không có weighting.
→ **Evidence**: D2L Section 11.2.2: cả ba đều fit ground truth tốt, chỉ có Constant thất bại.

3. **Cho $\|\mathbf{x}\|=1$ (unit sphere). Đơn giản hóa Gaussian kernel $\|\mathbf{x}-\mathbf{x}_i\|^2$ và nêu implication cho attention.**

→ **Claim**: Gaussian kernel reduces to scaled dot-product.
→ **Reasoning**: $\|\mathbf{x}-\mathbf{x}_i\|^2 = 2 - 2\mathbf{x}^\top\mathbf{x}_i$. Gaussian kernel becomes $\exp(-\sigma^2(1-\mathbf{x}^\top\mathbf{x}_i)) = \exp(-\sigma^2) \cdot \exp(\sigma^2 \cdot \mathbf{x}^\top\mathbf{x}_i)$. Constant factor $\exp(-\sigma^2)$ cancels out in softmax → attention score essentially reduces to **scaled dot-product** $\mathbf{q}^\top\mathbf{k}$.
→ **Evidence**: Đây là foundation của dot-product attention trong Transformer (Buổi 52)!

4. **Tại sao NW estimator không cần training?**

→ **Claim**: Vì không có learnable parameters.
→ **Reasoning**: Kết quả hoàn toàn determined bởi: (a) data points $(x_i, y_i)$ và (b) kernel function + width. Không có gradient flow nào cần thiết.
→ **Evidence**: NW là non-parametric method — complexity tăng với data size, không có weights cố định.

5. **Bias-variance trade-off của $\sigma$:**

→ **Claim**: $\sigma$ nhỏ → low bias, high variance; $\sigma$ lớn → high bias, low variance.
→ **Reasoning**: Kernel hẹp $\sigma$ (0.1): bias thấp (fit cục bộ) nhưng variance cao (nhạy với noise). Kernel rộng $\sigma$ (1.0): bias cao (smooth away detail) nhưng variance thấp (dự đoán ổn định).
→ **Evidence**: D2L Section 11.2.3: "kernel càng hẹp thì ước lượng càng ít mượt. Đồng thời, nó thích nghi tốt hơn với các biến thiên cục bộ."

---

## TODO

- [ ] Implement NW estimator với bốn kernels từ scratch bằng NumPy
- [ ] Experiment với different kernel widths và visualize overfitting/underfitting
- [ ] Đọc thêm về Parzen windows (Parzen 1957)
- [ ] Đọc Mack & Silverman (1982) về consistency của NW estimator

---

## Liên kết

- [[Buổi 50 - Tuần 14|QKV — Queries, Keys, and Values]]
- [[Buổi 52 - Tuần 14|Attention Scoring Functions]]
- [[Buổi 53 - Tuần 14|Bahdanau Attention]]
- [[Softmax Function]]
- [[Attention Mechanism]] *(concept note — cần tạo)*
- [[Transformer Architecture]] *(concept note đã có)*
