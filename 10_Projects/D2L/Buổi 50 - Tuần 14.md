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
>
> - Hiểu bản chất QKV — tại sao cần ba thành phần riêng biệt
> - Nắm vững công thức attention pooling (11.1.1) và softmax normalization (11.1.3)
> - Phân biệt bốn trường hợp đặc biệt của attention weights
> - Nắm kết nối từ RNN bottleneck (buổi 48-49) sang attention mechanism
> - Hiểu attention bản chất là phiên bản khả vi của Nadaraya-Watson estimator

---

## Active Recall — Ôn lại Buổi 49

### Câu hỏi truy hồi

1. Beam Search khác Greedy Search ở điểm cốt lõi nào? Khi nào dùng $k=1$ (tương đương greedy)?
2. Tại sao beam search dùng log-probability thay vì xác suất thô? Công thức cụ thể?
3. Length normalization trong beam search giải quyết vấn đề gì? Tại sao dùng $\alpha=0.75$?
4. So sánh độ phức tạp: Greedy ($O(T')$), Exhaustive ($O(V^{T'}$)), Beam ($O(k \cdot V \cdot T')$)?
5. Trong ví dụ beam search với $k=2$, tại sao path B (0.036) thắng path A (0.030) dù bước 1 của B (0.6) nhỏ hơn A (0.8)?

### Tự trả lời

1. **Greedy chỉ giữ một path mỗi bước** → có thể bỏ lỡ điểm tối ưu toàn cục. **Beam giữ $k$ paths** → khám phá song song, khả năng cao tìm được điểm tối ưu toàn cục.
2. **Log-probability tránh numerical underflow**: $\log(a \cdot b) = \log a + \log b$. Với $T'=10$, xác suất thô có thể $10^{-40}$ → tràn số.
3. **Câu ngắn có lợi thế tự nhiên** (ít bước → ít phạt). Length normalization $\frac{1}{L^\alpha} \sum \log P$ phạt câu ngắn. $\alpha=0.75$ là heuristic từ thực nghiệm.
4. Mỗi beam step: $k$ branches × $V$ tokens → $O(kV)$. Giữ top-$k$ → giảm từ $V^{T'}$ xuống $O(kVT')$.
5. Path score là tích/tổng log **toàn bộ sequence**, không phải chỉ bước đầu. Beam chọn dựa trên **điểm tối ưu toàn cục**, không phải tối ưu cục bộ.

### Liên kết cần ôn lại

- [[Buổi 49 - Tuần 13|Beam Search]]
- [[Buổi 48 - Tuần 13|Sequence-to-Sequence Learning]]
- [[Buổi 47 - Tuần 13|Encoder-Decoder Architecture]]

---

# PHẦN I — MOTIVATION: INPUT CÓ KÍCH THƯỚC CỐ ĐỊNH VÀ RNN BOTTLENECK

## 1.1 Vấn đề mà Attention giải quyết

> [!NOTE] ELI5
> CNN và RNN đòi hỏi input có kích thước cố định — ảnh $224 \times 224$, chuỗi xử lý từng token một. Nhưng thế giới thực **không cố định**: câu cần dịch có thể năm từ hoặc năm mươi từ, thông tin quan trọng nằm ở bất kỳ đâu. Attention giúp mô hình **tự chọn** phần nào của input cần tập trung, bất kể độ dài.

**Định nghĩa kỹ thuật:**

- **Đây là gì?** Attention mechanism là kỹ thuật cho phép mô hình "chú ý" có chọn lọc đến các phần khác nhau của input, thay vì xử lý toàn bộ đồng đều.
- **Input/Output là gì?** Input: query vector $\mathbf{q}$ và database $\mathcal{D} = \{(\mathbf{k}_i, \mathbf{v}_i)\}$. Output: tổ hợp có trọng số $\sum_i \alpha_i \mathbf{v}_i$ — vector có trọng số.
- **Giải quyết vấn đề gì?** RNN Seq2Seq không có attention nén **toàn bộ** chuỗi nguồn vào một context vector $c$ cố định. Decoder dùng cùng một $c$ cho mọi bước sinh token — đây chính là nút thắt cổ chai.

**Nút thắt cổ chai cốt lõi của Seq2Seq không có attention:**

Encoder nén toàn bộ chuỗi nguồn vào một context vector $c$ cố định. Ví dụ: câu nguồn mười tokens, hidden size 256:

- Encoder chứa: $10 \times 256 = 2560$ chiều thông tin
- Context vector: $1 \times 256 = 256$ chiều
- **Mất khoảng 90% thông tin!**

> [!question]- Câu hỏi then chốt
> Thay vì nén toàn bộ vào một vector, có cách nào decoder **chủ động nhìn lại** phần nào của chuỗi nguồn cần thiết khi sinh mỗi token không?

**Attention mechanism chính là câu trả lời.**

---

# PHẦN II — ANALOGY CƠ SỞ DỮ LIỆU

## 2.1 Thế giới trước khi có Attention

> [!NOTE] ELI5
> Cơ sở dữ liệu truyền thống giống tra từ điển: hỏi "Li" → trả "Mu". Khớp đúng → có kết quả. Không khớp → không có gì. Máy tính không "suy nghĩ" hay "đoán" — chỉ so khớp chính xác.

**Định nghĩa kỹ thuật:**

Cơ sở dữ liệu truyền thống là tập hợp các cặp **(key, value)**:

$$\mathcal{D} = \{(\text{"Zhang"}, \text{"Aston"}), (\text{"Lipton"}, \text{"Zachary"}), (\text{"Li"}, \text{"Mu"}), (\text{"Smola"}, \text{"Alex"}), (\text{"Hu"}, \text{"Rachel"}), (\text{"Werness"}, \text{"Brent"})\}$$

Query $q$ = "Li" → trả về "Mu". Query $q$ = "Lipton" → trả về "Zachary". Query $q$ = "Lipt" → không có kết quả (exact match).

**Bốn tính chất quan trọng:**

| #   | Tính chất                                    | Ý nghĩa                                               |
| --- | -------------------------------------------- | ----------------------------------------------------- |
| 1   | Query độc lập với database size              | Cùng query hoạt động bất kể cơ sở dữ liệu lớn hay nhỏ |
| 2   | Cùng query → kết quả khác nhau theo database | Query không được hard-coded với output                |
| 3   | "Code" xử lý rất đơn giản                    | Exact/approximate match, không cần học phức tạp       |
| 4   | Không cần nén database                       | Query trực tiếp thao tác trên data                    |

> [!WARNING]- Dấu hiệu nhồi nhét
> Nếu bạn nhớ "attention giống database lookup" mà không hiểu bốn tính chất trên — bạn đang nhồi nhét. Attention **không phải** chỉ là "tra cứu tốt hơn". Nó là **soft selection có khả vi** — có thể học từ data.

**Database truyền thống vs Attention:**

| Khía cạnh   | Traditional DB                    | Attention                            |
| ----------- | --------------------------------- | ------------------------------------ |
| Match type  | Exact (hoặc approximate đơn giản) | Soft, weighted                       |
| Output      | Một value hoặc nothing            | Tổ hợp có trọng số của tất cả values |
| Gradient    | Không cần (không học)             | Khả vi từ đầu đến cuối               |
| Flexibility | Cố định                           | Có thể học được từ data              |

---

# PHẦN III — ATTENTION MECHANISM: CÔNG THỨC CỐT LÕI

## 3.1 Attention Pooling — Công thức (11.1.1)

> [!NOTE] ELI5
> Attention giống như bạn có đội ngũ năm chuyên gia, mỗi người có kiến thức (value) khác nhau. Khi hỏi câu hỏi (query), mỗi chuyên gia cho điểm (attention weight) dựa trên câu hỏi đó. Câu trả lời là **tổ hợp có trọng số** — người giỏi nhất về chủ đề được hỏi có trọng số cao nhất.

**Định nghĩa kỹ thuật (D2L 11.1.1):**

- **Đây là gì?** Attention pooling là phép toán lấy tổ hợp có trọng số (weighted sum) của các values, trong đó trọng số được tính từ độ phù hợp giữa query và keys.
- **Input/Output là gì?** Input: database $\mathcal{D} = \{(\mathbf{k}_1, \mathbf{v}_1), \ldots, (\mathbf{k}_m, \mathbf{v}_m)\}$ với $\mathbf{k}_i, \mathbf{v}_i \in \mathbb{R}^d$, và query $\mathbf{q} \in \mathbb{R}^d$. Output: $\sum_i \alpha(\mathbf{q}, \mathbf{k}_i) \mathbf{v}_i \in \mathbb{R}^d$.
- **Giải quyết vấn đề gì?** Cho phép trích xuất thông tin có chọn lọc từ tập hợp, thay vì phải nén toàn bộ vào một vector cố định.

Cho database $\mathcal{D} \stackrel{\text{def}}{=} \{(\mathbf{k}_1, \mathbf{v}_1), \ldots, (\mathbf{k}_m, \mathbf{v}_m)\}$ gồm $m$ cặp key-value, với $\mathbf{k}_i \in \mathbb{R}^d$ và $\mathbf{v}_i \in \mathbb{R}^d$. Cho query $\mathbf{q} \in \mathbb{R}^d$. Attention pooling:

$$\boxed{\textbf{Attention}(\mathbf{q}, \mathcal{D}) \stackrel{\text{def}}{=} \sum_{i=1}^{m} \alpha(\mathbf{q}, \mathbf{k}_i) \, \mathbf{v}_i}$$

**Từ điển ký hiệu:**

| Ký hiệu                                           | Định nghĩa                                           | Nguồn gốc                                      |
| ------------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------- |
| $\mathbf{q}$                                      | Query vector — "câu hỏi" đặt ra                      | Decoder hidden state hoặc input                |
| $\mathbf{k}_i$                                    | Key vector thứ $i$ — "định danh" của value $i$       | Encoder hidden states                          |
| $\mathbf{v}_i$                                    | Value vector thứ $i$ — "nội dung" cần lấy            | Thường bằng $\mathbf{k}_i$ hoặc $\mathbf{h}_i$ |
| $\alpha(\mathbf{q}, \mathbf{k}_i) \in \mathbb{R}$ | Attention weight — mức độ query quan tâm đến key $i$ | Từ compatibility function                      |
| $m$                                               | Số lượng cặp key-value                               | Độ dài sequence                                |

Output là **linear combination** của các values, với weights $\alpha$ quyết định tỷ lệ đóng góp của mỗi value. Attention chính là **weighted sum**, không phải gì khác.

> [!IMPORTANT]- Tại sao gọi là "Attention"?
> Tên gọi đến từ việc operation này **"pay particular attention"** đến các terms có weight $\alpha$ lớn (significant). Weight càng lớn → attention càng nhiều vào value đó.

---

## 3.2 Bốn Trường hợp Đặc biệt của Attention Weights (D2L 11.1.1)

> [!NOTE] ELI5
> Attention weights có thể hoạt động theo nhiều "phong cách": có thể chỉ chọn đúng một người (hard attention), hoặc phân phối sự chú ý cho nhiều người (soft attention). Có thể cho tất cả bằng nhau (uniform), hoặc dựa trên khoảng cách (kernel-based).

**Định nghĩa kỹ thuật:**

- **Đây là gì?** Bốn ràng buộc toán học khác nhau trên attention weights $\alpha_i$, mỗi cái tạo ra một "phong cách" attention khác nhau.
- **Input/Output là gì?** Input: các ràng buộc trên $\alpha_i$. Output: phân phối attention weights có tính chất khác nhau.
- **Giải quyết vấn đề gì?** Mỗi ràng buộc phù hợp với một tình huống cụ thể — từ database lookup cổ điển đến soft selection có khả vi.

| Trường hợp             | Điều kiện trên $\alpha$                          | Ràng buộc                                                                 | Ý nghĩa                                | Ví dụ dùng                 |
| ---------------------- | ------------------------------------------------ | ------------------------------------------------------------------------- | -------------------------------------- | -------------------------- |
| **Nonnegative**        | $\alpha_i \geq 0$                                | Output nằm trong **hình nón lồi** (convex cone) của values $\mathbf{v}_i$ | Không "trừ" thông tin, chỉ "cộng thêm" | Softmax base case          |
| **Convex combination** | $\sum_i \alpha_i = 1$, $\alpha_i \geq 0$         | Output là **weighted average**                                            | Nội suy giữa các values                | Phổ biến nhất              |
| **Hard attention**     | $\alpha_i \in \{0, 1\}$, đúng một $\alpha_i = 1$ | Chọn đúng một value                                                       | Tra cứu database truyền thống          | Image captioning (Xu 2015) |
| **Uniform**            | $\alpha_i = 1/m$                                 | Equal weights                                                             | Average pooling                        | Baseline                   |

> [!CRITICAL]- Phân biệt "convex cone" vs "convex combination"
>
> - **Convex cone**: $\alpha_i \geq 0$ nhưng **không yêu cầu** $\sum \alpha_i = 1$. Output có thể "phóng đại" — nằm ngoài range của values gốc.
> - **Convex combination**: $\sum \alpha_i = 1$ **VÀ** $\alpha_i \geq 0$. Output **luôn nằm trong** bao lồi (convex hull) của các values — bị chặn.
>
> Trong deep learning, ta thường dùng **convex combination** (softmax đảm bảo cả hai điều kiện).

### Minh họa trực quan: Cone vs Combination trong Attention
![[d2l-fig-11-1-1.png]]
_Hình 1. Luồng Attention: Query + Keys -> scores -> softmax -> attention weights -> weighted sum của Values. Điểm quan trọng: nếu chỉ biết $\alpha_i \ge 0$ thì output thuộc convex cone; nếu thêm ràng buộc $\sum_i \alpha_i = 1$ (softmax) thì output là convex combination._

**Chú giải thuật ngữ trong hình:**

- **Score** $s_i$: mức độ phù hợp giữa query và key ($s_i = a(q, k_i)$).
- **Attention weight** $\alpha_i$: trọng số sau chuẩn hóa; với softmax thì $\alpha_i > 0$ và $\sum_i \alpha_i = 1$.
- **Weighted sum**: $\sum_i \alpha_i v_i$ là output cuối của attention.
- **Hard attention**: trường hợp gần argmax, một $\alpha_i$ áp đảo hoặc bằng 1 (phiên bản rời rạc).

> [!CHECKLIST]- Reader kiểm tra sau khi xem hình
>
> - [ ] Tôi chỉ ra được bước nào trong flow tạo ra ràng buộc $\sum_i \alpha_i = 1$.
> - [ ] Tôi phân biệt được: nonnegative-only -> convex cone, còn softmax -> convex combination.
> - [ ] Tôi giải thích được vì sao hard attention không còn là "trung bình mềm".

> [!CHECKLIST]-
> Reader tự kiểm tra:
>
> - [ ] Tôi giải thích được: attention = tổ hợp có trọng số của values, weights từ similarity giữa q và k
> - [ ] Tôi biết softmax đảm bảo hai điều kiện: nonnegative + sum-to-1
> - [ ] Tôi biết khi nào dùng hard attention (cần hard selection, dùng RL) vs soft attention (có thể khả vi, dùng softmax)
> - [ ] Tôi phân biệt được convex cone (chỉ nonnegative) vs convex combination (nonnegative + sum=1)

---

## 3.3 Softmax Normalization — Công thức (11.1.3)

> [!NOTE] ELI5
> Như chấm điểm thi: mỗi chuyên gia cho điểm (score), softmax biến điểm đó thành phần trăm — tổng bằng 100%, không ai có điểm âm. Ai được điểm cao → trọng số lớn, nhưng vẫn có đóng góp từ người khác (soft selection).

**Định nghĩa kỹ thuật:**

- **Đây là gì?** Softmax là hàm biến đổi raw scores thành phân phối xác suất (attention weights).
- **Input/Output là gì?** Input: $m$ raw scores $s_i = a(\mathbf{q}, \mathbf{k}_i)$. Output: $m$ attention weights $\alpha_i \in (0, 1)$ với $\sum_i \alpha_i = 1$.
- **Giải quyết vấn đề gì?** Biến đổi scores thành convex combination một cách tự động, đảm bảo cả hai điều kiện: nonnegative và sum-to-1.

Bước 1 — Tính raw scores bằng **compatibility function** $a(\mathbf{q}, \mathbf{k}_i)$:

$$s_i = a(\mathbf{q}, \mathbf{k}_i)$$

Bước 2 — Chuẩn hóa bằng softmax (D2L 11.1.3):

$$\alpha(\mathbf{q}, \mathbf{k}_i) = \frac{\exp(a(\mathbf{q}, \mathbf{k}_i))}{\sum_{j=1}^{m} \exp(a(\mathbf{q}, \mathbf{k}_j))}$$

**Tại sao dùng exp? Bốn lý do:**

| Lý do                           | Giải thích                                                                  |
| ------------------------------- | --------------------------------------------------------------------------- |
| **Nonnegative**                 | $\exp(s_i) > 0$ với mọi $s_i$ → đảm bảo $\alpha \geq 0$                     |
| **Sum-to-1**                    | Softmax tự động normalize → $\sum_i \alpha_i = 1$                           |
| **Khả vi ở mọi nơi**            | $\nabla \exp(x) = \exp(x) > 0$ → gradient không biến mất (khác với sigmoid) |
| **Competitive / "soft" argmax** | Scores cạnh tranh với nhau — lớn hơn nhiều → chiếm gần như toàn bộ weight   |

**Edge case quan trọng:** Nếu $s_i$ quá lớn (ví dụ: $\exp(1000)$ tràn số), gradient vẫn ổn nhờ **softmax stability trick** (trừ max trước khi exp). PyTorch và TensorFlow tự xử lý.

**Tại sao dùng softmax thay vì chỉ normalize đơn giản ($\alpha_i = s_i / \sum s_j$)?**

- $s_i$ có thể âm → $\alpha_i$ âm → "trừ" information (vô nghĩa về mặt xác suất)
- $s_i$ có thể rất lớn → numerical instability
- Gradient của $s_i / \sum s_j$ không bị chặn → training unstable

---

## 3.4 Attention ≠ Softmax Attention

> [!NOTE] ELI5
> "Attention" là tên gọi chung cho cả ý tưởng. "Softmax attention" là một cách tính cụ thể — dùng softmax để tạo weights. Giống như "động vật" (chung) vs "mèo" (cụ thể).

**Định nghĩa kỹ thuật:**

- **Đây là gì?** **Attention mechanism** = tổng có trọng số $\sum_i \alpha_i \mathbf{v}_i$ (Eq. 11.1.1). Đây là **định nghĩa tổng quát**.
- **Softmax attention** = cách tính $\alpha_i = \text{softmax}(a(\mathbf{q}, \mathbf{k}_i))$ — cách phổ biến nhất vì có thể khả vi.
- **Giải quyết vấn đề gì?** Attention có thể dùng cách khác: hard attention với argmax, attention dựa trên RL (Mnih 2014), hoặc bất kỳ phương pháp nào sinh ra $\alpha_i$.

---

# PHẦN IV — QKV: TẠI SAO TÁCH RIÊNG BA THÀNH PHẦN?

## 4.1 Trực giác — "Hỏi-Đáp" trong câu

> [!NOTE] ELI5
> Khi đọc "Con mèo ngồi trên bàn":
>
> - **Query**: "Từ nào liên quan đến 'mèo'?" — bạn đang hỏi
> - **Key**: mỗi từ có "nhãn" mô tả nó là gì ("con mèo", "ngồi", "trên bàn")
> - **Value**: bản thân từ đó mang thông tin gì
>
> Attention so sánh query với keys → quyết định "mèo" nên "nghe" từ nào nhiều nhất.

**Định nghĩa kỹ thuật — QKV trong RNN Seq2Seq:**

| Thành phần                | Trong Seq2Seq (Buổi 48)              | Vai trò                                        |
| ------------------------- | ------------------------------------ | ---------------------------------------------- |
| **Query $\mathbf{q}$**    | Decoder hidden state $s_{t'}$        | "Tôi đang dịch từ gì, cần thông tin gì?"       |
| **Keys $\mathbf{k}_i$**   | Encoder hidden states $\mathbf{h}_t$ | "Tôi chứa thông tin gì, tôi nói về chủ đề gì?" |
| **Values $\mathbf{v}_i$** | Encoder hidden states $\mathbf{h}_t$ | "Tôi thực sự chứa thông tin gì để truyền đi?"  |

Trong **self-attention** (Buổi 55-56), cả Q, K, V đến từ cùng một nguồn — mỗi token "hỏi" tất cả tokens khác.

**Tại sao tách Q, K, V thay vì dùng chung một vector?**

1. **Tính linh hoạt**: query cần tìm thứ khác với key chứa → cần hai không gian biểu diễn khác nhau
2. **Có thể học được**: $W_Q, W_K, W_V$ là learnable parameters → model tự học cách query nên "hỏi" gì
3. **Similarity không tầm thường**: nếu Q = K = V, dot product Q·K chỉ đo similarity của cùng vector, không đủ biểu cảm

---

## 4.2 Data Flow — Hình 11.1.1 (D2L)

**Giải thích từng bước:**

1. **Query** → đi vào **Compatibility function** $a(\mathbf{q}, \mathbf{k}_i)$ (tính "độ phù hợp" giữa q và k)
2. **Keys** → đi vào **Compatibility function** (cùng với Query)
3. **Compatibility function** → output scores → đi vào **Softmax** (Eq. 11.1.3)
4. **Softmax** → sinh **Attention Weights** $\alpha_1, \ldots, \alpha_m$ (convex combination)
5. **Keys/Values** → đi vào **Weighted Sum**
6. **Attention Weights + Values** → kết hợp → sinh **Output** $\sum_i \alpha_i \mathbf{v}_i$

---

# PHẦN V — VISUALIZATION: ATTENTION WEIGHTS

## 5.1 Bốn Trường hợp Đặc biệt — Heatmap

![[assets/attachments/d2l-buoi-50/attention-special-cases.png]]
_Hình 2. Bốn trường hợp trọng số attention: nonnegative, convex combination, hard, uniform. Nhìn vào heatmap sẽ thấy convex combination là "phân phối" trên nhiều values, còn hard attention gần như chọn 1 value._

**Phân tích từng heatmap:**

| Hình | Trường hợp      | Pattern                        | Giải thích                                                                                   |
| ---- | --------------- | ------------------------------ | -------------------------------------------------------------------------------------------- |
| (a)  | Hard attention  | Đường chéo = 1, các ô khác = 0 | Query $i$ chỉ attend đúng một key tại vị trí trùng. Giống hệt tra cứu database truyền thống. |
| (b)  | Softmax (sharp) | Một peak rõ, xung quanh gần 0  | Attention tập trung mạnh vào một đến hai keys. Khi similarity score chênh lệch lớn.          |
| (c)  | Softmax (soft)  | Phân bố rộng, peak không rõ    | Attention "lan tỏa" nhiều keys. Scores gần nhau → softmax giữ đóng góp từ nhiều nơi.         |
| (d)  | Uniform         | Mọi ô = 1/m                    | Query nhận đóng góp bằng nhau từ tất cả values. Không có "chọn lọc" gì cả. Baseline.         |

> [!CHECKLIST]-
> Reader tự kiểm tra:
>
> - [ ] Tôi biết identity/hard = "chọn đúng một key" — giống exact match
> - [ ] Tôi biết uniform = "lấy trung bình tất cả" — không có chọn lọc
> - [ ] Tôi biết softmax = "học được từ data" — linh hoạt nhất, là default trong deep learning
> - [ ] Tôi biết "soft" trong softmax không phải soft attention — nó là competitive normalization

---

# PHẦN VI — CODE: VISUALIZATION VÀ SANITY CHECK

## 6.1 Hàm show_heatmaps — Công cụ trực quan hóa attention

```python
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    """Show heatmaps of matrices.

    Args:
        matrices: 4D tensor (num_rows, num_cols, height, width)
                  thường reshape từ attention weights (n_queries, n_keys)
        xlabel: nhãn trục x (Keys)
        ylabel: nhãn trục y (Queries)
    """
    num_rows, num_cols, _, _ = matrices.shape
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize,
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

**Input shape**: `(num_rows, num_cols, n_queries, n_keys)` — cho phép hiển thị mảng các heatmaps.

## 6.2 Sanity check: Ma trận Đơn vị

```python
# Ma trận đơn vị = perfect match (query i chỉ attend đến key i)
attention_weights = torch.eye(10).reshape((1, 1, 10, 10))
show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')
```

Output: đường chéo chính sáng (weight=1), các ô khác tối (weight=0). Nếu attention hoạt động đúng, ta sẽ thấy patterns có ý nghĩa thay vì chỉ đường chéo.

---

# PHẦN VII — NADARAYA-WATSON: TIỀN THÂN NON-PARAMETRIC

## 7.1 Kết nối Quan trọng: Attention = Differentiable Nadaraya-Watson

> [!NOTE] ELI5
> Người xưa đã có cách dự đoán giá trị tại một điểm mới bằng cách lấy trung bình có trọng số của các điểm gần đó. Trọng số được tính bằng "khoảng cách" — điểm càng gần thì trọng số càng lớn. Attention chính là phiên bản **có thể học được** của ý tưởng này.

**Định nghĩa kỹ thuật:**

D2L Section 11.1.2 nêu rõ: attention mechanism chính là **phiên bản khả vi** của **Nadaraya-Watson kernel regression estimator** (Nadaraya 1964, Watson 1964).

Trong regression setting:

- **Query $\mathbf{q}$** = vị trí cần thực hiện regression
- **Keys $\mathbf{k}_i$** = vị trí đã quan sát data
- **Values $\mathbf{v}_i$** = giá trị regression đã quan sát
- **Attention weights** $\alpha_i$ = kernel similarity (ví dụ: Gaussian kernel)

**Sự khác biệt then chốt:**

| Khía cạnh           | Nadaraya-Watson                       | Learned Attention                   |
| ------------------- | ------------------------------------- | ----------------------------------- |
| Similarity function | **Cố định** kernel (ví dụ: Gaussian)  | **Có thể học** dot product hoặc MLP |
| Parameters          | Không có                              | $W_Q, W_K, W_V$ có thể học          |
| Adaptability        | Phụ thuộc vào hyperparameter $\sigma$ | Tự học từ data                      |
| Generalization      | Chỉ nội suy mượt                      | Học các patterns phức tạp           |

> [!KEY]- Key Insight (D2L)
> "Người đọc tinh ý có thể tự hỏi tại sao chúng ta đi sâu vào một phương pháp đã hơn nửa thế kỷ tuổi. Thứ nhất, đây là một trong những tiền thân sớm nhất của attention mechanisms hiện đại. Thứ hai, nó rất tốt cho việc trực quan hóa. Thứ ba, và quan trọng không kém, nó cho thấy **giới hạn của attention mechanisms được thiết kế thủ công**. Một chiến lược tốt hơn nhiều là **học cơ chế này**, bằng cách học các biểu diễn cho queries và keys."

Buổi 51 sẽ học chi tiết về Nadaraya-Watson estimator và bốn kernel functions.

---

# PHẦN VIII — TÓM TẮT VÀ LIÊN KẾT

## 8.1 Tóm tắt buổi

| Khái niệm                                                    | Hiểu | Cần ôn |
| ------------------------------------------------------------ | ---- | ------ |
| Motivation: fixed-size input problem                         |      |        |
| Attention pooling formula (11.1.1)                           |      |        |
| Bốn trường hợp đặc biệt (nonnegative, convex, hard, uniform) |      |        |
| Softmax normalization (11.1.3)                               |      |        |
| QKV roles trong Seq2Seq                                      |      |        |
| Database analogy — bốn properties                            |      |        |
| Convex cone vs convex combination                            |      |        |
| Attention ≠ Softmax attention                                |      |        |
| Nadaraya-Watson = attention precursor                        |      |        |
 
## 8.2 Liên kết với các buổi tiếp theo

| Buổi               | Chủ đề                               | Liên kết                                                |
| ------------------ | ------------------------------------ | ------------------------------------------------------- |
| **Buổi 51** (11.2) | Attention Pooling by Similarity      | Nadaraya-Watson kernel regression, bốn kernel functions |
| **Buổi 52** (11.3) | Attention Scoring Functions          | Dot product attention, additive attention, BMM          |
| **Buổi 53** (11.4) | Bahdanau Attention                   | Cross-attention trong Seq2Seq                           |
| **Buổi 54** (11.5) | Multi-Head Attention                 | Học đa quan điểm song song                              |
| **Buổi 55** (11.6) | Self-Attention & Positional Encoding | Q = K = V trong cùng sequence                           |

## 8.3 Bảng thuật ngữ

| Thuật ngữ                                            | Định nghĩa                                                                                  |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Query ($\mathbf{q}$)                                 | Vector đại diện cho "câu hỏi" trong attention                                               |
| Key ($\mathbf{k}_i$)                                 | Vector "định danh" của value thứ $i$                                                        |
| Value ($\mathbf{v}_i$)                               | Vector chứa thông tin thực sự                                                               |
| Attention weight $\alpha_i$                          | Trọng số của value thứ $i$ trong weighted sum                                               |
| Compatibility function $a(\mathbf{q}, \mathbf{k}_i)$ | Hàm tính "độ phù hợp" giữa query và key                                                     |
| Attention pooling                                    | Weighted sum $\sum_i \alpha_i \mathbf{v}_i$ (D2L Eq. 11.1.1)                                |
| Softmax attention                                    | Attention với $\alpha_i = \text{softmax}(a(\mathbf{q}, \mathbf{k}_i))$ (D2L Eq. 11.1.3)     |
| Hard attention                                       | $\alpha_i \in \{0, 1\}$ — chọn đúng một value                                               |
| Soft attention                                       | $\alpha_i \in [0, 1]$, sum-to-1 — convex combination                                        |
| Convex cone                                          | $\{\sum_i \alpha_i \mathbf{v}_i \mid \alpha_i \geq 0\}$ — chỉ nonnegative                   |
| Convex combination                                   | $\{\sum_i \alpha_i \mathbf{v}_i \mid \sum \alpha_i = 1, \alpha_i \geq 0\}$ — output bị chặn |
| Nadaraya-Watson estimator                            | Non-parametric regression = trường hợp đặc biệt của attention                               |

---

## Active Recall — Câu hỏi về Buổi 50

1. **Cho $\mathcal{D} = \{(k_1, v_1), (k_2, v_2)\}$ với $v_1 = [1, 0]$, $v_2 = [0, 1]$, $q = k_1 = [1, 0]$, dot product attention (softmax). Output?** → $\alpha_1 = \exp(1)/\left(\exp(1) + \exp(0)\right) = e/(e+1) \approx 0.731$, $\alpha_2 = 1/(e+1) \approx 0.269$. Output $= 0.731 \cdot [1,0] + 0.269 \cdot [0,1] \approx [0.731, 0.269]$.

2. **Chứng minh: softmax gradient không biến mất?** → $\nabla_{s_i} \text{softmax}_i = \text{softmax}_i (1 - \text{softmax}_i)$ — **không** có term nào tiến về 0 như sigmoid ($\sigma(x)(1-\sigma(x)) \to 0$ khi $x \gg 0$). Softmax gradient luôn dương và tỷ lệ với giá trị.

3. **Khi nào dùng hard attention thay vì soft attention?** → Khi cần **hard selection** (chọn chính xác một vị trí). Hard attention cần RL training (REINFORCE) vì không khả vi. Ví dụ: image captioning (Xu et al., 2015), "show, attend and tell".

4. **Nếu bỏ softmax, dùng raw scores làm weights — điều gì sai?** → (a) Không sum-to-1 → output scale phụ thuộc score scale; (b) Scores có thể âm → "trừ" information (vô nghĩa về mặt xác suất); (c) Gradient không bị chặn → training unstable.

5. **Tại sao attention mechanism gọi là "khả vi"?** → Vì tất cả các phép toán (dot product, softmax, weighted sum) đều khả vi. Gradient flow từ output ngược qua attention weights về query, keys, values → end-to-end training.

6. **Bốn properties của traditional database mà attention thừa hưởng?** → (1) Query độc lập database size; (2) Cùng query → kết quả khác theo database; (3) "Code" đơn giản; (4) Không cần nén database.

---

## Bài tập D2L 11.1.3

1. **Thiết kế approximate (key, query) matches như classical databases — dùng attention function nào?** → Hard attention với $\alpha_i \in \{0, 1\}$ hoặc sharpened softmax (high temperature) gần với argmax.

2. **Cho $a(\mathbf{q}, \mathbf{k}_i) = \mathbf{q}^\top \mathbf{k}_i$ và $\mathbf{k}_i = \mathbf{v}_i$. Chứng minh: $\nabla_{\mathbf{q}} \text{Attention}(\mathbf{q}, \mathcal{D}) = \text{Cov}_{p(\mathbf{k}_i; \mathbf{q})}[\mathbf{k}_i]$.** → Đạo hàm weighted sum: $\nabla_\mathbf{q} \sum_i \alpha_i \mathbf{k}_i$. Với $\alpha_i = \text{softmax}(\mathbf{q}^\top \mathbf{k}_i)$, dùng chain rule và định nghĩa xác suất $p(\mathbf{k}_i; \mathbf{q})$. Kết quả: $\sum_i p_i \mathbf{k}_i \mathbf{k}_i^\top \mathbf{q} - \left(\sum_i p_i \mathbf{k}_i\right) \left(\sum_i p_i \mathbf{k}_i^\top \mathbf{q}\right) = \text{Cov}$.

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
- [[Attention Mechanism]] _(concept note — cần tạo)_
- [[Self-Attention]] _(concept note đã có)_
- [[Transformer Architecture]] _(concept note đã có)_
