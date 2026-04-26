---
session: "D2L Tuần 14, Buổi 50 — 11.1 Queries, Keys, and Values"
aliases: ["Buổi 50"]
tags: [d2l, deep-learning, attention, qkv, nlp]
status: growth
source: "D2L Chapter 11.1 — Queries, Keys, and Values"
created: 2026-04-23
related:
  - "[[Buổi 49 - Tuần 13]]"
  - "[[Buổi 51 - Tuần 14]]"
  - "[[Tổng ôn RNN]]"
---

# Buổi 50 — 11.1 Queries, Keys, and Values

> [!NOTE] Mục tiêu buổi học
> - Hiểu bản chất QKV — tại sao cần 3 thành phần riêng biệt
> - Nắm vững công thức attention pooling (11.1.1) và softmax normalization (11.1.3)
> - Phân biệt 4 special cases của attention weights
> - Nắm connection từ RNN bottleneck (buổi 48-49) sang attention mechanism
> - Hiểu attention = differentiable version của Nadaraya-Watson estimator (Buổi 51)

---

## Active Recall — Ôn lại Buổi 49

### Câu hỏi truy hồi

1. Beam Search khác Greedy Search ở điểm cốt lõi nào? Khi nào dùng k=1 (tương đương greedy)?
2. Tại sao beam search dùng log-probability thay vì raw probability? Công thức cụ thể?
3. Length normalization trong beam search giải quyết vấn đề gì? Tại sao dùng $\alpha=0.75$?
4. So sánh độ phức tạp: Greedy ($O(T')$), Exhaustive ($O(V^{T'})$), Beam ($O(k \cdot V \cdot T')$)?
5. Trong ví dụ beam search với $k=2$, tại sao path B (0.036) thắng path A (0.030) dù bước 1 của B (0.6) nhỏ hơn A (0.8)?

### Tự trả lời

1. **Greedy chỉ giữ 1 path mỗi step** → miss global optimum. **Beam giữ $k$ paths** → explore song song, khả năng cao tìm global optimum.
2. **Log-probability tránh underflow**: $\log(a \cdot b) = \log a + \log b$. Với $T'=10$, raw prob có thể $10^{-40}$ → numerical underflow.
3. **Ngắn sequences có lợi thế tự nhiên** (ít bước → ít penalty). Length normalization $\frac{1}{L^\alpha} \sum \log P$ phạt sequences ngắn. $\alpha=0.75$ là heuristic từ thực nghiệm.
4. Mỗi beam step: $k$ branches × $V$ tokens → $O(kV)$. Giữ top-$k$ → giảm từ $V^{T'}$ xuống $O(kVT')$.
5. Path score là tích/tổng log **toàn bộ sequence**, không phải chỉ bước đầu. Beam chọn dựa trên **global optimum**, không greedy local.

---

# PHẦN I — MOTIVATION: FIXED INPUT SIZE VÀ RNN BOTTLENECK

## 1.1 Vấn đề mà Attention giải quyết

>[!NOTE] ELI5
> CNN/RNN đòi hỏi input cố định — ảnh $224 \times 224$, chuỗi xử lý từng token. Nhưng thế giới thực **không cố định**: câu dịch có thể 5 từ hoặc 50 từ, thông tin quan trọng nằm ở bất kỳ đâu. Attention giúp model **tự chọn** phần nào của input cần tập trung, bất kể độ dài.

**Định nghĩa kỹ thuật:**

D2L nêu rõ: CNN/RNN hoạt động với input cố định. RNN Seq2Seq (Buổi 48) xử lý từng token, nhưng **rất khó theo dõi** toàn bộ chuỗi đã sinh hoặc đã thấy — đặc biệt với chuỗi dài.

**Bottleneck cốt lõi của Seq2Seq không có attention:**

Trong Seq2Seq không attention, encoder nén **toàn bộ** chuỗi nguồn vào 1 context vector $c$ cố định. Decoder dùng cùng 1 $c$ cho mọi bước sinh token.

Ví dụ: câu nguồn 10 tokens, hidden size 256:
- Encoder có: $10 \times 256 = 2560$ dims thông tin
- Context vector: $1 \times 256 = 256$ dims
- **Mất ~90% thông tin!**

> [!QUESTION]- Câu hỏi then chốt
> Thay vì nén toàn bộ vào 1 vector, có cách nào decoder **chủ động nhìn lại** phần nào của chuỗi nguồn cần thiết khi sinh mỗi token?

**Attention mechanism chính là câu trả lời.**

---

# PHẦN II — DATABASE ANALOGY

## 2.1 Thế giới trước khi có Attention

> [!NOTE] ELI5
> Cơ sở dữ liệu truyền thống giống tra từ điển: hỏi "Li" → trả "Mu". Khớp đúng → có kết quả. Không khớp → không có gì. Máy tính không "suy nghĩ" hay "đoán" — chỉ so khớp chính xác.

**Định nghĩa kỹ thuật:**

Cơ sở dữ liệu truyền thống là tập hợp các cặp **(key, value)**:

$$\mathcal{D} = \{(\text{"Zhang"}, \text{"Aston"}), (\text{"Lipton"}, \text{"Zachary"}), (\text{"Li"}, \text{"Mu"}), (\text{"Smola"}, \text{"Alex"}), (\text{"Hu"}, \text{"Rachel"}), (\text{"Werness"}, \text{"Brent"})\}$$

Query $q$ = "Li" → trả về "Mu". Query $q$ = "Lipton" → trả về "Zachary". Query $q$ = "Lipt" → không có kết quả (exact match).

**4 tính chất quan trọng:**

| # | Tính chất | Ý nghĩa |
|---|-----------|---------|
| 1 | Query độc lập với database size | Cùng query hoạt động bất kể DB lớn hay nhỏ |
| 2 | Cùng query → kết quả khác nhau theo DB | Query không hard-coded với output |
| 3 | "Code" xử lý rất đơn giản | Exact/approximate match, không cần học phức tạp |
| 4 | Không cần nén database | Query trực tiếp operate trên data |

> [!WARNING]- Dấu hiệu nhồi nhét
> Nếu bạn nhớ "attention giống database lookup" mà không hiểu 4 tính chất trên — bạn đang nhồi nhét. Attention **không phải** chỉ là "lookup tốt hơn". Nó là **differentiable soft selection** — có thể học từ data.

![[Pasted image 20260426165415.png]]

**Database truyền thống vs Attention:**

| Khía cạnh | Traditional DB | Attention |
|-----------|---------------|-----------|
| Match type | Exact (hoặc approximate đơn giản) | Soft, weighted |
| Output | 1 value hoặc nothing | Weighted combination của tất cả values |
| Gradient | Không cần (không học) | Differentiable end-to-end |
| Flexibility | Cố định | Learnable từ data |

---

# PHẦN III — ATTENTION MECHANISM: CÔNG THỨC CỐT LÕI

## 3.1 Attention Pooling — Công thức (11.1.1)

> [!NOTE] ELI5
> Attention giống như bạn có đội ngũ 5 chuyên gia, mỗi người có kiến thức (value) khác nhau. Khi hỏi câu hỏi (query), mỗi chuyên gia cho điểm (attention weight) dựa trên câu hỏi đó. Câu trả lời là **tổ hợp có trọng số** — người giỏi nhất về chủ đề được hỏi có trọng số cao nhất.

**Định nghĩa kỹ thuật (D2L 11.1.1):**

Cho database $\mathcal{D} \stackrel{\text{def}}{=} \{(\mathbf{k}_1, \mathbf{v}_1), \ldots, (\mathbf{k}_m, \mathbf{v}_m)\}$ gồm $m$ cặp key-value, với $\mathbf{k}_i \in \mathbb{R}^d$ và $\mathbf{v}_i \in \mathbb{R}^d$. Cho query $\mathbf{q} \in \mathbb{R}^d$. Attention pooling:

$$\boxed{\textbf{Attention}(\mathbf{q}, \mathcal{D}) \stackrel{\text{def}}{=} \sum_{i=1}^{m} \alpha(\mathbf{q}, \mathbf{k}_i) \, \mathbf{v}_i}$$

**Từ điển ký hiệu (D2L):**

| Ký hiệu | Định nghĩa | Nguồn gốc |
|---------|-----------|-----------|
| $\mathbf{q}$ | Query vector — "câu hỏi" đặt ra | Decoder hidden state hoặc input |
| $\mathbf{k}_i$ | Key vector thứ $i$ — "định danh" của value $i$ | Encoder hidden states |
| $\mathbf{v}_i$ | Value vector thứ $i$ — "nội dung" cần lấy | Thường = $\mathbf{k}_i$ hoặc $\mathbf{h}_i$ |
| $\alpha(\mathbf{q}, \mathbf{k}_i) \in \mathbb{R}$ | Attention weight — mức q quan tâm đến key $i$ | Từ compatibility function |
| $m$ | Số lượng key-value pairs | Độ dài sequence |

**Output** là **linear combination** của các values, với weights $\alpha$ quyết định tỷ lệ đóng góp của mỗi value. Attention chính là **weighted sum**, không phải gì khác.

> [!IMPORTANT]- Tại sao gọi là "Attention"?
> Tên gọi đến từ việc operation này **"pay particular attention"** đến các terms có weight $\alpha$ lớn (significant). Weight càng lớn → attention càng nhiều vào value đó.

---

## 3.2 4 Special Cases của Attention Weights (D2L 11.1.1)

> [!WARNING]- Dấu hiệu nhồi nhét
> Nếu bạn chỉ nhớ công thức mà không biết **khi nào và tại sao** mỗi special case được dùng → dừng lại, suy nghĩ.

| Case | Điều kiện trên $\alpha$ | Ràng buộc | Ý nghĩa | Ví dụ dùng |
|------|------------------------|-----------|---------|-----------|
| **Nonnegative** | $\alpha_i \geq 0$ | Output nằm trong **convex cone** của values $\mathbf{v}_i$ | Không "trừ" thông tin, chỉ "cộng thêm" | Softmax base case |
| **Convex combination** | $\sum_i \alpha_i = 1$, $\alpha_i \geq 0$ | Output là **weighted average** | Interpolate giữa values | Phổ biến nhất |
| **Hard attention** | $\alpha_i \in \{0, 1\}$, đúng 1 = 1 | Chọn đúng 1 value | Traditional DB lookup | Image captioning (Xu 2015) |
| **Uniform** | $\alpha_i = 1/m$ | Equal weights | Average pooling | Baseline |

> [!CRITICAL]- Phân biệt "convex cone" vs "convex combination"
> - **Convex cone**: $\alpha_i \geq 0$ nhưng **không yêu cầu** $\sum \alpha_i = 1$. Output có thể "phóng đại" — nằm ngoài range của values gốc.
> - **Convex combination**: $\sum \alpha_i = 1$ **VÀ** $\alpha_i \geq 0$. Output **luôn nằm trong** convex hull (bao lồi) của values — bounded.
>
> Trong deep learning, ta thường dùng **convex combination** (softmax đảm bảo cả 2 điều kiện).

> [!CHECKLIST]-
> Reader tự kiểm tra:
>
> - [ ] Tôi giải thích được: attention = weighted sum của values, weights từ similarity giữa q và k
> - [ ] Tôi biết softmax đảm bảo 2 điều kiện: nonnegative + sum-to-1
> - [ ] Tôi biết khi nào dùng hard attention (cần hard selection, dùng RL) vs soft attention ( differentiable, dùng softmax)
> - [ ] Tôi phân biệt được convex cone (chỉ nonnegative) vs convex combination (nonnegative + sum=1)

---

## 3.3 Softmax Normalization — Công thức (11.1.3)

>[!NOTE] ELI5
> Như điểm thi: mỗi chuyên gia cho điểm (score), softmax biến điểm đó thành phần trăm — cộng lại bằng 100%, không ai có điểm âm. Ai được điểm cao → trọng số lớn, nhưng vẫn có contribution từ người khác (soft selection).

**Định nghĩa kỹ thuật:**

Bước 1 — Tính raw scores bằng **compatibility function** $a(\mathbf{q}, \mathbf{k}_i)$:

$$s_i = a(\mathbf{q}, \mathbf{k}_i)$$

Bước 2 — Chuẩn hóa bằng softmax (D2L 11.1.3):

$$\alpha(\mathbf{q}, \mathbf{k}_i) = \frac{\exp(a(\mathbf{q}, \mathbf{k}_i))}{\sum_{j=1}^{m} \exp(a(\mathbf{q}, \mathbf{k}_j))}$$

**Tại sao dùng exp? 4 lý do:**

| Lý do | Giải thích |
|-------|-----------|
| **Nonnegative** | $\exp(s_i) > 0$ với mọi $s_i$ → đảm bảo $\alpha \geq 0$ |
| **Sum-to-1** | Softmax tự động normalize → $\sum_i \alpha_i = 1$ |
| **Differentiable everywhere** | $\nabla \exp(x) = \exp(x) > 0$ → gradient không vanish (khác sigmoid) |
| **Competitive / "soft" argmax** | Scores cạnh tranh với nhau — lớn hơn nhiều → chiếm gần như toàn bộ weight |

**Edge case quan trọng:** Nếu $s_i$ quá lớn (e.g., $\exp(1000)$ overflow), gradient vẫn ổn nhờ **softmax stability trick** (trừ max trước khi exp). PyTorch/TensorFlow tự xử lý.

**Tại sao dùng softmax thay vì chỉ normalize đơn giản ($\alpha_i = s_i / \sum s_j$)?**

- $s_i$ có thể âm → $\alpha_i$ âm → "trừ" information (vô nghĩa)
- $s_i$ có thể rất lớn → numerical instability
- Gradient của $s_i / \sum s_j$ không bounded → training unstable

---

## 3.4 Attention ≠ Softmax Attention

> [!IMPORTANT]- Phân biệt hai khái niệm
> **Attention mechanism** = tổng có trọng số $\sum_i \alpha_i \mathbf{v}_i$ (Eq. 11.1.1). Đây là **định nghĩa tổng quát**.
>
> **Softmax attention** = cách tính $\alpha_i = \text{softmax}(a(\mathbf{q}, \mathbf{k}_i))$ — cách phổ biến nhất vì differentiable.
>
> Attention có thể dùng cách khác: hard attention với argmax, RL-based attention (Mnih 2014), hoặc bất kỳ method nào sinh ra $\alpha_i$.

---

# PHẦN IV — QKV: TẠI SAO TÁCH RIÊNG 3 THÀNH PHẦN?

## 4.1 Intuition — "Hỏi-Đáp" trong câu

> [!NOTE] ELI5
> Khi đọc "Con mèo ngồi trên bàn":
>
> - **Query**: "Từ nào liên quan đến 'mèo'?" — bạn đang hỏi
> - **Key**: mỗi từ có "nhãn" mô tả nó là gì ("con mèo", "ngồi", "trên bàn")
> - **Value**: bản thân từ đó mang thông tin gì
>
> Attention so sánh query với keys → quyết định "mèo" nên "nghe" từ nào nhiều nhất.

**Định nghĩa kỹ thuật — QKV trong RNN Seq2Seq:**

| Thành phần | Trong Seq2Seq (Buổi 48) | Vai trò |
|-----------|------------------------|---------|
| **Query $\mathbf{q}$** | Decoder hidden state $s_{t'}$ | "Tôi đang dịch từ gì, cần thông tin gì?" |
| **Keys $\mathbf{k}_i$** | Encoder hidden states $\mathbf{h}_t$ | "Tôi chứa thông tin gì, tôi nói về chủ đề gì?" |
| **Values $\mathbf{v}_i$** | Encoder hidden states $\mathbf{h}_t$ | "Tôi thực sự chứa thông tin gì để truyền đi?" |

Trong **self-attention** (Buổi 55-56), cả Q, K, V đến từ cùng một nguồn — mỗi token "hỏi" tất cả tokens khác.

**Tại sao tách Q, K, V thay vì dùng chung 1 vector?**

1. **Flexibility**: query cần tìm thứ khác với key chứa → cần 2 không gian biểu diễn khác nhau
2. **Learnable**: $W_Q, W_K, W_V$ là learnable parameters → model tự học cách query nên "hỏi" gì
3. **Non-trivial similarity**: nếu Q = K = V, dot product Q·K chỉ đo similarity của cùng vector, không đủ expressive

---

## 4.2 Data Flow — Fig 11.1.1 (D2L)

![[assets/attachments/d2l-buoi-50/d2l-fig-11-1-1.png]]

**Giải thích từng bước:**

1. **Query** → đi vào **Compatibility function** $a(\mathbf{q}, \mathbf{k}_i)$ (tính "độ phù hợp" giữa q và k)
2. **Keys** → đi vào **Compatibility function** (cùng với Query)
3. **Compatibility function** → output scores → đi vào **Softmax** (Eq. 11.1.3)
4. **Softmax** → sinh **Attention Weights** $\alpha_1, \ldots, \alpha_m$ (convex combination)
5. **Keys/Values** → đi vào **Weighted Sum**
6. **Attention Weights + Values** → kết hợp → sinh **Output** $\sum_i \alpha_i \mathbf{v}_i$

---

# PHẦN V — VISUALIZATION: ATTENTION WEIGHTS

## 5.1 4 Special Cases — Heatmap

![[assets/attachments/d2l-buoi-50/attention-special-cases.png]]

**Phân tích từng heatmap:**

| Hình | Case | Pattern | Giải thích |
|------|------|---------|-----------|
| (a) | Hard attention | Đường chéo = 1, các ô khác = 0 | Query $i$ chỉ attend đúng 1 key tại vị trí trùng. Giống hệt traditional DB lookup. |
| (b) | Softmax (sharp) | 1 peak rõ, xung quanh gần 0 | Attention tập trung mạnh vào 1-2 keys. Khi similarity score chênh lệch lớn. |
| (c) | Softmax (soft) | Phân bố rộng, peak không rõ | Attention "lan tỏa" nhiều keys. Scores gần nhau → softmax giữ contribution từ nhiều nơi. |
| (d) | Uniform | Mọi ô = 1/m | Query nhận equal contribution từ tất cả values. Không có "chọn lọc" gì cả. Baseline. |

> [!CHECKLIST]-
> Reader tự kiểm tra:
>
> - [ ] Tôi biết identity/hard = "chọn đúng 1 key" — giống exact match
> - [ ] Tôi biết uniform = "lấy trung bình tất cả" — không selective
> - [ ] Tôi biết softmax = "học được từ data" — linh hoạt nhất, là default trong deep learning
> - [ ] Tôi biết "soft" trong softmax không phải soft attention — nó là competitive normalization

---

# PHẦN VI — CODE: VISUALIZATION VÀ SANITY CHECK

## 6.1 show_heatmaps — Công cụ visualize attention

```python
import torch
from d2l import torch as d2l

@d2l.add_to_class(d2l.HyperParameters)
def save_hyperparameters(self, *args):
    pass

#@save
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    """Show heatmaps of matrices.
    
    matrices: 4D tensor (num_rows, num_cols, height, width)
              thường reshape từ attention weights (n_queries, n_keys)
    """
    d2l.use_svg_display()
    num_rows, num_cols, _, _ = matrices.shape
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)
```

**Input shape**: `(num_rows, num_cols, n_queries, n_keys)` — cho phép hiển thị array of heatmaps.

## 6.2 Sanity check: Identity Matrix

```python
# Identity matrix = perfect match (query i only attends to key i)
attention_weights = torch.eye(10).reshape((1, 1, 10, 10))
show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')
```

Output: đường chéo chính sáng (weight=1), các ô khác tối (weight=0). Nếu attention hoạt động đúng, ta sẽ thấy patterns có ý nghĩa thay vì chỉ đường chéo.

---

# PHẦN VII — SUMMARY VÀ LIÊN KẾT

## 7.1 Tóm tắt buổi

| Khái niệm | Hiểu | Cần ôn |
|-----------|------|--------|
| Motivation: fixed-size input problem | ✅ | |
| Attention pooling formula (11.1.1) | ✅ | |
| 4 special cases (nonnegative, convex, hard, uniform) | ✅ | |
| Softmax normalization (11.1.3) | ✅ | |
| QKV roles trong Seq2Seq | ✅ | |
| Database analogy — 4 properties | ✅ | |
| Fig 11.1.1 data flow | ✅ | |
| Convex cone vs convex combination | ✅ | |
| Attention ≠ Softmax attention | ✅ | |

## 7.2 Liên kết — Nadaraya-Watson Estimator

> [!KEY]- Connection quan trọng: Attention = Differentiable Nadaraya-Watson
> D2L Section 11.1.2 nêu rõ: attention mechanism chính là **differentiable version** của **Nadaraya-Watson kernel regression estimator** (Nadaraya 1964, Watson 1964).
>
> Trong regression setting:
> - **Query $\mathbf{q}$** = vị trí cần thực hiện regression
> - **Keys $\mathbf{k}_i$** = vị trí đã quan sát data
> - **Values $\mathbf{v}_i$** = giá trị regression đã quan sát
> - **Attention weights** $\alpha_i$ = kernel similarity (e.g., Gaussian kernel)
>
> Buổi 51 sẽ học chi tiết về Nadaraya-Watson estimator và 4 kernel functions.

## 7.3 Liên kết với các buổi tiếp theo

| Buổi | Chủ đề | Liên kết |
|------|--------|---------|
| **Buổi 51** (11.2) | Attention Pooling by Similarity | Nadaraya-Watson kernel regression, 4 kernel functions |
| **Buổi 52** (11.3) | Attention Scoring Functions | Dot product attention, additive attention, BMM |
| **Buổi 53** (11.4) | Bahdanau Attention | Cross-attention trong Seq2Seq |
| **Buổi 54** (11.5) | Multi-Head Attention | Học đa quan điểm song song |
| **Buổi 55** (11.6) | Self-Attention & Positional Encoding | Q = K = V trong cùng sequence |

## 7.4 Bảng thuật ngữ

| Thuật ngữ | Định nghĩa |
|-----------|-----------|
| Query ($\mathbf{q}$) | Vector đại diện cho "câu hỏi" trong attention |
| Key ($\mathbf{k}_i$) | Vector "định danh" của value thứ $i$ |
| Value ($\mathbf{v}_i$) | Vector chứa thông tin thực sự |
| Attention weight $\alpha_i$ | Trọng số của value thứ $i$ trong weighted sum |
| Compatibility function $a(\mathbf{q}, \mathbf{k}_i)$ | Hàm tính "độ phù hợp" giữa query và key |
| Attention pooling | Weighted sum $\sum_i \alpha_i \mathbf{v}_i$ (D2L Eq. 11.1.1) |
| Softmax attention | Attention với $\alpha_i = \text{softmax}(a(\mathbf{q}, \mathbf{k}_i))$ (D2L Eq. 11.1.3) |
| Hard attention | $\alpha_i \in \{0, 1\}$ — chọn đúng 1 value |
| Soft attention | $\alpha_i \in [0, 1]$, sum-to-1 — convex combination |
| Convex cone | $\{\sum_i \alpha_i \mathbf{v}_i \mid \alpha_i \geq 0\}$ — chỉ nonnegative |
| Convex combination | $\{\sum_i \alpha_i \mathbf{v}_i \mid \sum \alpha_i = 1, \alpha_i \geq 0\}$ — bounded output |
| Nadaraya-Watson estimator | Non-parametric regression = special case của attention |

---

## Active Recall — Câu hỏi về Buổi 50

1. **Cho $\mathcal{D} = \{(k_1, v_1), (k_2, v_2)\}$ với $v_1 = [1, 0]$, $v_2 = [0, 1]$, $q = k_1 = [1, 0]$, dot product attention (softmax). Output?** → $\alpha_1 = \exp(1)/\left(\exp(1) + \exp(0)\right) = e/(e+1) \approx 0.731$, $\alpha_2 = 1/(e+1) \approx 0.269$. Output $= 0.731 \cdot [1,0] + 0.269 \cdot [0,1] \approx [0.731, 0.269]$.

2. **Chứng minh: softmax gradient không vanish?** → $\nabla_{s_i} \text{softmax}_i = \text{softmax}_i (1 - \text{softmax}_i)$ — **không** có term nào tiến về 0 như sigmoid ($\sigma(x)(1-\sigma(x)) \to 0$ khi $x \gg 0$). Softmax gradient luôn positive và proportional với giá trị.

3. **Khi nào dùng hard attention thay vì soft attention?** → Khi cần **hard selection** (chọn chính xác 1 vị trí). Hard attention cần RL training (REINFORCE) vì không differentiable. Ví dụ: image captioning (Xu et al., 2015), "show, attend and tell".

4. **Nếu bỏ softmax, dùng raw scores làm weights — điều gì sai?** → (a) Không sum-to-1 → output scale phụ thuộc score scale; (b) Scores có thể âm → "trừ" information (vô nghĩa về mặt probability); (c) Gradient không bounded → training unstable.

5. **Tại sao attention mechanism gọi là "differentiable"?** → Vì tất cả các phép toán (dot product, softmax, weighted sum) đều khả vi. Gradient flow từ output ngược qua attention weights về query, keys, values → end-to-end training.

6. **4 properties của traditional database mà attention thừa hưởng?** → (1) Query độc lập database size; (2) Cùng query → kết quả khác theo DB; (3) "Code" đơn giản; (4) Không cần nén DB.

---

## Bài tập D2L 11.1.3

1. **Thiết kế approximate (key, query) matches như classical databases — dùng attention function nào?** → Hard attention với $\alpha_i \in \{0, 1\}$ hoặc sharpened softmax (high temperature) gần với argmax.

2. **Cho $a(\mathbf{q}, \mathbf{k}_i) = \mathbf{q}^\top \mathbf{k}_i$ và $\mathbf{k}_i = \mathbf{v}_i$. Chứng minh: $\nabla_{\mathbf{q}} \text{Attention}(\mathbf{q}, \mathcal{D}) = \text{Cov}_{p(\mathbf{k}_i; \mathbf{q})}[\mathbf{k}_i]$.** → Đạo hàm weighted sum: $\nabla_\mathbf{q} \sum_i \alpha_i \mathbf{k}_i$. Với $\alpha_i = \text{softmax}(\mathbf{q}^\top \mathbf{k}_i)$, dùng chain rule và định nghĩa probability $p(\mathbf{k}_i; \mathbf{q})$. Kết quả: $\sum_i p_i \mathbf{k}_i \mathbf{k}_i^\top \mathbf{q} - \left(\sum_i p_i \mathbf{k}_i\right) \left(\sum_i p_i \mathbf{k}_i^\top \mathbf{q}\right) = \text{Cov}$.

3. **Thiết kế differentiable search engine dùng attention mechanism.** → Query = search term, Keys = document embeddings, Values = document content, attention weights = relevance scores. End-to-end trainable với learnable query encoder.

4. **Review Squeeze-and-Excitation Networks (Hu et al., 2018) — interpret qua lens attention.** → SE block: attention weights trên channel dimensions = "which channels to excite". Attention mechanism $\alpha_i$ điều chỉnh importance của từng channel.

---

## TODO

- [ ] Đọc paper gốc Bahdanau (2014) — "Neural Machine Translation by Jointly Learning to Align and Translate"
- [ ] Đọc paper gốc Nadaraya-Watson (1964)
- [ ] Tạo concept note [[Attention Mechanism]] trong 20_Areas/AI/Concepts/
- [ ] Tạo concept note [[Queries Keys Values]] trong 20_Areas/AI/Concepts/
- [ ] Thực hành implement attention pooling từ scratch với PyTorch

---

## Liên kết

- [[Buổi 49 - Tuần 13|Beam Search]]
- [[Buổi 48 - Tuần 13|Sequence-to-Sequence Learning]]
- [[Buổi 47 - Tuần 13|Encoder-Decoder Architecture]]
- [[Softmax Function]]
- [[Tổng ôn RNN]]
- [[Attention Mechanism]] *(concept note — cần tạo)*
- [[Self-Attention]] *(concept note đã có)*
- [[Transformer Architecture]] *(concept note đã có)*
