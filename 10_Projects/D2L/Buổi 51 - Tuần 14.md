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
> - Hiểu Nadaraya-Watson estimator — tiền thân phi tham số của attention
> - Nắm bốn kernel functions (Gaussian, Boxcar, Epanechikov, Constant)
> - Hiểu kết nối: NW = attention với hand-crafted kernel, learned attention = NW với learned kernel
> - Phân biệt được giữa hand-crafted attention và learned attention
> - Thấy trực quan: kernel width ảnh hưởng thế nào đến smoothness vs local adaptation

---

## Active Recall — Ôn lại Buổi 50

### Câu hỏi truy hồi

1. Attention pooling formula (11.1.1) là gì? Cho biết ý nghĩa từng thành phần QKV.
2. Tại sao softmax đảm bảo gradient không biến mất? (So sánh với sigmoid)
3. Phân biệt convex cone (chỉ $\alpha_i \geq 0$) vs convex combination ($\sum \alpha_i = 1$ VÀ $\alpha_i \geq 0$)
4. Tại sao attention gọi là "khả vi"? Liên hệ Nadaraya-Watson estimator.
5. Nếu dùng raw scores (không softmax) làm attention weights — điều gì sai?

### Tự trả lời

1. $\textbf{Attention}(\mathbf{q}, \mathcal{D}) = \sum_i \alpha(\mathbf{q}, \mathbf{k}_i)\mathbf{v}_i$. Query = decoder hidden state, Keys = encoder hidden states, Values = encoder hidden states.
2. $\nabla \exp(x) = \exp(x) > 0$ — không có saturation như sigmoid ($\sigma'(x) = \sigma(x)(1-\sigma(x)) \to 0$ khi $x \gg 0$).
3. Convex cone chỉ nonnegative ($\alpha_i \geq 0$), output có thể "phóng đại". Convex combination thêm $\sum \alpha_i = 1$, output bị chặn trong bao lồi.
4. Attention dùng softmax (khả vi) → gradient flow được qua attention weights. NW cũng tương tự: dùng kernel similarity làm attention weights, nhưng kernel là **cố định** (hand-crafted), không có tham số có thể học.
5. Raw scores: (a) không sum-to-1 → output scale phụ thuộc score magnitude; (b) có thể âm → "trừ" information; (c) gradient không bị chặn.

### Liên kết cần ôn lại

- [[Buổi 50 - Tuần 14|QKV — Queries, Keys, and Values]]
- [[Softmax Function]]

---

# PHẦN I — NADARAYA-WATSON: TIỀN THÂN CỦA ATTENTION

## 1.1 Từ Attention về Nadaraya-Watson

> [!NOTE] ELI5
> Nadaraya-Watson (NW) estimator là cách cổ điển để dự đoán giá trị tại một điểm mới ($q$), bằng cách lấy **trung bình có trọng số** của các điểm training gần đó. "Gần" được đo bằng **kernel** (hàm similarity). Attention mechanism chính là phiên bản **có thể học được** của ý tưởng này.

**Định nghĩa kỹ thuật:**

- **Đây là gì?** Nadaraya-Watson estimator là phương pháp phi tham số (non-parametric) để ước lượng hàm regression, sử dụng weighted average của các điểm training gần với query.
- **Input/Output là gì?** Input: training data $(\mathbf{x}_i, y_i)$ và query $\mathbf{q}$. Output: giá trị dự đoán $f(\mathbf{q}) = \sum_i y_i \cdot \frac{K(\mathbf{q}, \mathbf{x}_i)}{\sum_j K(\mathbf{q}, \mathbf{x}_j)}$.
- **Giải quyết vấn đề gì?** Dự đoán giá trị tại điểm mới mà không cần giả định về dạng hàm cụ thể (non-parametric).

**Kết nối với Buổi 50:**

| Khía cạnh | Nadaraya-Watson (11.2) | Attention (11.1) |
|-----------|----------------------|-----------------|
| Query $\mathbf{q}$ | Vị trí cần predict | Decoder hidden state |
| Keys $\mathbf{k}_i$ | Training features $x_i$ | Encoder hidden states |
| Values $\mathbf{v}_i$ | Training labels $y_i$ | Encoder hidden states |
| Attention weights | $\alpha = K(\mathbf{q}, \mathbf{k}_i)$ (cố định kernel) | $\alpha = \text{softmax}(a(\mathbf{q}, \mathbf{k}_i))$ (có thể học) |
| Có học được không? | **Không** — kernel cố định | **Có** — $W_Q, W_K, W_V$ có thể học |
| Output | $f(\mathbf{q}) = \sum_i y_i \cdot \frac{K}{\sum K}$ | $\sum_i \mathbf{v}_i \cdot \text{softmax}(a)$ |

> [!KEY]- Key Insight (D2L)
> "Người đọc tinh ý có thể tự hỏi tại sao chúng ta đi sâu vào một phương pháp đã hơn nửa thế kỷ tuổi. Thứ nhất, đây là một trong những tiền thân sớm nhất của attention mechanisms hiện đại. Thứ hai, nó rất tốt cho việc trực quan hóa. Thứ ba, và quan trọng không kém, nó cho thấy **giới hạn của attention mechanisms được thiết kế thủ công**. Một chiến lược tốt hơn nhiều là **học cơ chế này**, bằng cách học các biểu diễn cho queries và keys."

---

# PHẦN II — BỐN KERNEL FUNCTIONS (D2L Eq. 11.2.1)

## 2.1 Giới thiệu Kernels

> [!NOTE] ELI5
> Kernel giống như "thước đo khoảng cách" — nó cho biết hai điểm "gần nhau" hay "xa nhau" theo những cách khác nhau. Gaussian dùng đường cong hình chuông, Boxcar dùng cửa sổ cứng, Epanechikov dùng mái vòm.

**Định nghĩa kỹ thuật:**

- **Đây là gì?** Kernel function là hàm đo "độ tương tự" (similarity) giữa query và key, dựa trên khoảng cách.
- **Input/Output là gì?** Input: query $\mathbf{q}$ và key $\mathbf{k}$. Output: scalar $\alpha \in [0, 1]$ hoặc $[0, \infty)$ tùy kernel.
- **Giải quyết vấn đề gì?** Cung cấp cách tính attention weights mà không cần học từ data.

Cho query $\mathbf{q}$ và key $\mathbf{k}$. Các kernel functions định nghĩa similarity:

$$\alpha(\mathbf{q}, \mathbf{k}) \stackrel{\text{def}}{=} \begin{cases} \exp\!\left(-\frac{1}{2} \|\mathbf{q} - \mathbf{k}\|^2\right) & \text{Gaussian} \\[6pt] 1 \quad \text{nếu } \|\mathbf{q} - \mathbf{k}\| \leq 1 & \text{Boxcar} \\[6pt] \max\!\left(0, 1 - \|\mathbf{q} - \mathbf{k}\|\right) & \text{Epanechikov} \end{cases}$$

**Bốn kernels trong bảng so sánh:**

| Kernel | Công thức | Đặc điểm | Phạm vi |
|--------|-----------|---------|---------|
| **Gaussian** | $\exp\!\left(-\frac{\|\mathbf{q}-\mathbf{k}\|^2}{2}\right)$ | Mượt, đường cong hình chuông | Toàn bộ không gian (infinite support) |
| **Boxcar** | $\mathbb{1}(\|\mathbf{q}-\mathbf{k}\| \leq 1)$ | Cứng, "bậc thang" | Chỉ trong bán kính 1 |
| **Epanechikov** | $\max(0, 1 - \|\mathbf{q}-\mathbf{k}\|)$ | Vòm, tuyến tính từng khúc | Trong bán kính 1, 0 ngoài |
| **Constant** | $1$ | Không phân biệt | Mọi nơi (uniform) |

**Tính chất quan trọng:** Tất cả kernels trong D2L đều **translation and rotation invariant** — nếu dịch hoặc xoay cả query và key cùng cách, kernel value không đổi. Vì lý do đó, D2L đơn giản hóa thành scalar arguments $k, q \in \mathbb{R}$.

**Phân tích hình dạng từng kernel:**

- **Gaussian**: Mượt nhất — mọi điểm đều có contribution, giảm dần theo khoảng cách. $\sigma$ càng nhỏ → càng "nhọn" (ít smooth).
- **Boxcar**: Cứng — hoặc 1 (trong bán kính) hoặc 0 (ngoài). Không có transition mềm.
- **Epanechikov**: Vòm mềm — giảm tuyến tính về 0. "Mềm hơn" Boxcar.
- **Constant**: Không phân biệt — mọi điểm bằng nhau. Tương ứng với **uniform attention** (Buổi 50).

> [!WARNING]- Dấu hiệu nhồi nhét
> Nếu bạn nhớ "Gaussian là phổ biến nhất" mà không hiểu **tại sao**: vì nó mượt và khả vi ở mọi nơi → gradient flow tốt, dễ phân tích. Hãy suy nghĩ về trade-off giữa Boxcar (đơn giản nhưng không liên tục) và Gaussian (mượt nhưng infinite support).

---

# PHẦN III — NW FORMULA VÀ REGRESSION CONTEXT

## 3.1 Nadaraya-Watson Regression Formula (D2L Eq. 11.2.2)

> [!NOTE] ELI5
> Để dự đoán giá trị tại $q$: xem tất cả điểm training $(x_i, y_i)$, tính similarity giữa $q$ và mỗi $x_i$ (dùng kernel), normalize để tổng weights = 1, rồi lấy weighted average của $y_i$.

**Định nghĩa kỹ thuật (D2L 11.2.2):**

- **Đây là gì?** NW estimator là weighted average của labels, với weights từ kernel similarity.
- **Input/Output là gì?** Input: training data $(\mathbf{x}_i, y_i)$ và query $\mathbf{q}$. Output: $f(\mathbf{q}) = \sum_i \mathbf{v}_i \cdot \frac{\alpha(\mathbf{q}, \mathbf{k}_i)}{\sum_j \alpha(\mathbf{q}, \mathbf{k}_j)}$.
- **Giải quyết vấn đề gì?** Regression không cần giả định về dạng hàm.

$$f(\mathbf{q}) = \sum_i \mathbf{v}_i \frac{\alpha(\mathbf{q}, \mathbf{k}_i)}{\sum_j \alpha(\mathbf{q}, \mathbf{k}_j)}$$

**Trong regression setting:**

- Training data: $(\mathbf{x}_i, y_i)$ — features và labels
- $\mathbf{k}_i = \mathbf{x}_i$ — keys = training features
- $\mathbf{v}_i = y_i$ — values = training labels (scalars)
- $\mathbf{q}$ — vị trí mới cần dự đoán

**Trong classification (multiclass):**
- $\mathbf{v}_i$ = one-hot encoding của $y_i$
- Kết quả là phân phối xác suất over classes

**Đặc điểm quan trọng:**

> [!IMPORTANT]- NW estimator **không cần training**
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

---

# PHẦN IV — IMPLEMENTATION VÀ VISUALIZATION

## 4.1 Hàm nadaraya_watson

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
        attention_w: (n_train, n_val) — attention weights
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
4. `y_hat = y_train @ attention_w` — weighted sum. Shape: `(n_val,)`.

> [!CRITICAL]- Axis normalization trong attention
> Bình thường attention weights normalize over keys (column), nhưng tùy convention. Trong transformer (Buổi 55-56), ta normalize over keys bằng `softmax` theo last axis. Ở đây D2L dùng simple divide vì kernels đã đảm bảo nonnegative.

## 4.2 Regression Estimates — So sánh bốn Kernels

**Phân tích từng kernel:**

| Kernel | $y_{\text{predict}}$ | Nhận xét |
|--------|---------------------|---------|
| **Gaussian** | Khá sát ground truth (đường đứt) | Mượt, khớp tốt. Đây là default choice. |
| **Boxcar** | Tương đương Gaussian, hơi gồ ghề | Chỉ nhìn trong bán kính 1 → local, nhưng discontinuous edges |
| **Epanechikov** | Tương tự Gaussian/Boxcar | Rất giống hai cái trên về kết quả — attention weights tương tự dù hàm khác nhau |
| **Constant** | Đường ngang ~mean của y | **Thất bại** — trả về $\frac{1}{n}\sum_i y_i$ cho mọi $q$ (uniform attention). Không có chọn lọc. |

> [!KEY]- D2L observation
> "Điều đầu tiên nổi bật là cả ba kernel nontrivial (Gaussian, Boxcar, và Epanechikov) đều tạo ra các ước lượng khá khả dụng, không quá xa hàm thực. Chỉ có constant kernel... tạo ra kết quả khá phi thực tế."
>
> Quan trọng: Gaussian, Boxcar, Epanechikov cho kết quả **rất giống nhau** dù hàm kernel khác nhau. Điều này gợi ý: **kernel shape không quan trọng bằng việc có kernel hay không**.

## 4.3 Attention Weights Heatmap — Trực quan hóa Attention

**Đọc heatmap:**

- **Trục x (ngang)** = Queries ($q$ = validation $x$ values)
- **Trục y (dọc)** = Keys ($k_i$ = training $x_i$ values)
- **Màu sáng (đỏ)** = Attention weight cao (model "chú ý" nhiều vào key đó)
- **Hàng = 1 query**: cho biết model nhìn vào những training points nào để dự đoán tại $q$

**Pattern quan sát được:**

| Kernel | Heatmap pattern | Giải thích |
|--------|---------------|-----------|
| **Gaussian** | Diagonal band rộng | Query tại $x$ chú ý nhiều vào keys gần $x$ (Gaussian tails) |
| **Boxcar** | Diagonal band cứng | Chỉ nhìn trong bán kính 1 — hard boundary |
| **Epanechikov** | Diagonal band vừa | Tương tự Gaussian nhưng sharp edges |
| **Constant** | Màu đều (uniform) | Mọi query nhìn mọi key bằng nhau — không có chọn lọc |

---

# PHẦN V — ADAPTING ATTENTION POOLING: KERNEL WIDTH

## 5.1 Tại sao cần điều chỉnh Width?

> [!NOTE] ELI5
> Gaussian kernel có "độ rộng" $\sigma$. $\sigma$ lớn → nhìn xa (smooth nhưng có thể blur). $\sigma$ nhỏ → chỉ nhìn gần (local adaptation tốt nhưng có thể overfit noise).

**Định nghĩa kỹ thuật:**

- **Đây là gì?** Kernel width (hoặc bandwidth) $\sigma$ là hyperparameter điều chỉnh "độ rộng" của kernel.
- **Input/Output là gì?** Input: khoảng cách $\|\mathbf{q} - \mathbf{k}\|$. Output: $\alpha = \exp(-\|\mathbf{q} - \mathbf{k}\|^2 / 2\sigma^2)$.
- **Giải quyết vấn đề gì?** Điều chỉnh trade-off giữa bias và variance.

D2L Section 11.2.3: Gaussian kernel với width parameter:

$$\alpha(\mathbf{q}, \mathbf{k}) = \exp\!\left(-\frac{\|\mathbf{q} - \mathbf{k}\|^2}{2\sigma^2}\right)$$

$\sigma$ càng nhỏ → kernel càng "nhọn" → attention càng tập trung (narrow). $\sigma$ càng lớn → kernel càng "phẳng" → attention càng lan tỏa (wide).

## 5.2 Effect của Sigma trên Regression

**Bảng phân tích:**

| $\sigma$ | Bandwidth | Smoothness | Local adaptation | Overfitting risk |
|---------|-----------|-----------|----------------|-----------------|
| **0.1** | Rất hẹp | Thấp | Rất cao (nhọn) | Cao — đi theo noise |
| **0.2** | Hẹp | Thấp | Cao | Đáng kể |
| **0.5** | Trung bình | Tốt | Trung bình | Thấp |
| **1.0** | Rộng | Rất cao | Thấp | Thấp — có thể underfit |

> [!KEY]- D2L observation
> "Rõ ràng, kernel càng hẹp thì ước lượng càng ít mượt. Đồng thời, nó thích nghi tốt hơn với các biến thiên cục bộ."
>
> Trade-off kinh điển: **bias-variance tradeoff**. Kernel hẹp → bias thấp (fit các patterns cục bộ) nhưng variance cao (nhạy với noise). Kernel rộng → bias cao (smooth away details) nhưng variance thấp (dự đoán ổn định).

## 5.3 Heuristics cho Kernel Width

D2L đề cập Silverman (1986) heuristics cho adaptive width theo local density. Ngoài ra:

- **Per-coordinate width**: $\sigma_d$ khác nhau cho mỗi dimension — dùng trong attention scaling (Buổi 52)
- **Learned width**: thay vì hand-crafted, cho model tự học width qua attention scaling factor $\sqrt{d_k}$ (Vaswani 2017)
- **Data-driven**: cross-validation để chọn $\sigma$

---

# PHẦN VI — QUAN SÁT TRỰC QUAN: QUERY TẠI MỘT ĐIỂM

## 6.1 Attention tại một Query cụ thể

**Giải thích hình:**

- Điểm xanh: 40 training points $(x_i, y_i)$
- Đường đứt hồng: ground truth $f(x) = 2\sin(x) + x$
- Đường dọc cam: query $q = 2.0$
- Độ dày của đường kết nối: attention weight $\alpha_i$ — đường càng dày → weight càng lớn

**Observation:** Query tại $x=2.0$ chủ yếu "chú ý" đến các training points gần $x=2.0$. Điểm ở xa gần như không contribute (đường mỏng).

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
| $f(q) = \sum_i v_i \cdot \frac{\alpha}{\sum \alpha}$ | $\text{Attention}(\mathbf{q}, \mathcal{D}) = \sum_i \mathbf{v}_i \cdot \text{softmax}(a)$ | Cùng cấu trúc weighted sum |

**Sự khác biệt then chốt:**

| Khía cạnh | Nadaraya-Watson | Learned Attention (Buổi 52+) |
|-----------|----------------|------------------------------|
| Similarity function | **Cố định** kernel (ví dụ: Gaussian) | **Có thể học** dot product hoặc MLP |
| Parameters | Không có | $W_Q, W_K, W_V$ có thể học |
| Adaptability | Phụ thuộc vào hyperparameter $\sigma$ | Tự học từ data |
| Generalization | Chỉ nội suy mượt | Học các patterns phức tạp |
| Use case | Regression/classification cổ điển | Seq2Seq, Transformer, NLP |

> [!WARNING]- Dấu hiệu nhồi nhét
> Nếu bạn nghĩ "attention = NW với dot product thay kernel" → **đúng về mặt công thức** nhưng **thiếu bản chất**. Điểm khác biệt quan trọng nhất: attention có **learnable projections** ($W_Q, W_K, W_V$) — nghĩa là model tự học **cách biểu diễn** query, key, value, không chỉ học cách tính similarity.

> [!CHECKLIST]-
> Reader tự kiểm tra:
>
> - [ ] Tôi giải thích được bằng lời: NW = weighted average của labels gần query, weights từ kernel similarity
> - [ ] Tôi biết bốn kernel functions và hình dạng của chúng
> - [ ] Tôi biết tại sao Gaussian, Boxcar, Epanechikov cho kết quả tương tự
> - [ ] Tôi hiểu trade-off bias-variance khi thay đổi $\sigma$
> - [ ] Tôi phân biệt được hand-crafted (NW) vs learned attention ($W_Q, W_K, W_V$)
> - [ ] Tôi biết NW không cần training — không có learnable parameters

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

1. **Cho dataset $(\mathbf{x}_i, y_i)$ và Gaussian kernel với $\sigma=0.5$. Tính $f(2.0)$ cho hai điểm: $x_1=1.9, y_1=3.0$ và $x_2=3.5, y_2=1.0$.** → $K(q=2.0, k_1=1.9) = \exp(-(0.1)^2/0.5) \approx 0.980$, $K(q=2.0, k_2=3.5) = \exp(-(1.5)^2/0.5) \approx 0.011$. $w_1 = 0.980/(0.980+0.011) \approx 0.989$, $w_2 \approx 0.011$. $f(2.0) \approx 0.989 \cdot 3.0 + 0.011 \cdot 1.0 \approx 2.97$.

2. **Tại sao Gaussian, Boxcar, Epanechikov cho kết quả regression gần như giống nhau?** → Vì chúng đều "ưu tiên điểm gần hơn điểm xa". Sự khác biệt về functional form ít quan trọng bằng việc có weighting scheme hay không. Constant kernel thất bại vì không có weighting.

3. **Cho $\|\mathbf{x}\|=1$ (unit sphere). Đơn giản hóa Gaussian kernel $\|\mathbf{x}-\mathbf{x}_i\|^2$ và nêu implication cho attention.** → $\|\mathbf{x}-\mathbf{x}_i\|^2 = 2 - 2\mathbf{x}^\top\mathbf{x}_i$. Gaussian kernel becomes $\exp(-\sigma^2(1-\mathbf{x}^\top\mathbf{x}_i)) = \exp(-\sigma^2) \cdot \exp(\sigma^2 \cdot \mathbf{x}^\top\mathbf{x}_i)$. Constant factor $\exp(-\sigma^2)$ cancels out in softmax → attention score essentially reduces to **scaled dot-product** $\mathbf{q}^\top\mathbf{k}$. Đây là foundation của dot-product attention trong Transformer!

4. **Tại sao NW estimator không cần training?** → Vì không có learnable parameters. Kết quả hoàn toàn determined bởi: (a) data points $(x_i, y_i)$ và (b) kernel function + width. Không có gradient flow nào cần thiết.

5. **Bias-variance trade-off của $\sigma$:** → Kernel hẹp $\sigma$ (0.1): bias thấp (fit cục bộ) nhưng variance cao (nhạy với noise). Kernel rộng $\sigma$ (1.0): bias cao (smooth away detail) nhưng variance thấp (dự đoán ổn định). Optimal $\sigma$ nằm đâu đó ở giữa.

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
