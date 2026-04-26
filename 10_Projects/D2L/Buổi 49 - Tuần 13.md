---
session: "D2L Tuần 13, Buổi 49 — 10.8 Beam Search"
tags: [d2l, deep-learning, seq2seq, beam-search, decoding, nlp]
status: growth
source: "D2L Chapter 10.8 — Sequence Decoding"
created: 2026-04-23
related:
  - "[[Buổi 48 - Tuần 13]]"
  - "[[Buổi 50 - Tuần 14]]"
  - "[[Buổi 47 - Tuần 13]]"
---

# Buổi 49 — 10.8 Beam Search

> [!NOTE] ELI5
> Beam Search giống như bạn đi tìm đường trong rừng. Greedy là đi một mình, luôn chọn hướng có vẻ tốt nhất ngay lúc đó — có thể rẽ nhầm vào ngõ cụt. Exhaustive là thử **mọi** con đường — chắc chắn tìm được đường tốt nhất, nhưng bạn sẽ già trước khi đi xong. Beam Search là đi cùng lúc với **k người bạn**: mỗi người chọn hướng tốt nhất cho mình, rồi tất cả ghép đôi với mọi hướng có thể, chỉ giữ lại **k đường tốt nhất**. Lặp lại cho đến khi đến đích.

---

## Active Recall — Ôn lại Buổi 48

### Câu hỏi truy hồi

1. Encoder-Decoder trong Seq2Seq (chưa attention) nén thông tin câu nguồn vào đâu? Điều gì xảy ra khi câu nguồn dài?
2. Teacher forcing là gì? Tại sao nó giúp gradient ổn định, nhưng lại gây ra exposure bias?
3. Masked loss trong Seq2Seq bỏ qua token nào? Tại sao? Tham số `ignore_index` trong `nn.CrossEntropyLoss` hoạt động như thế nào?
4. Tại sao context vector trong Seq2Seq không attention là "bottleneck"? Minh họa bằng con số cụ thể.
5. BLEU score gồm mấy thành phần? Brevity Penalty phạt trường hợp nào?

### Tự trả lời

1. **Encoder nén vào final hidden state $h_T$ (context vector $c$).** Khi câu nguồn dài, 1 vector cố định không chứa đủ thông tin → mất thông tin → dịch kém. Minh họa: câu 9 tokens, hidden size 256 → nén 9×256 = 2304 dims vào 1×256 = 256 dims → mất ~89% thông tin.
2. **Teacher forcing = cho Decoder ground truth token của bước trước làm input** thay vì predicted token. Ưu: gradient ổn định, hội tụ nhanh. Nhược: exposure bias — training khác inference. Tại step $t$ trong training, Decoder luôn nhận input đúng; ở inference, predicted token có thể sai → sai tích lũy.
3. **Bỏ qua `<pad>` tokens.** Vì padding không mang ý nghĩa — nếu tính loss trên chúng, model học dự đoán `<pad>` → lãng phí gradient. `ignore_index` trong PyTorch: khi ground truth bằng giá trị đó, vị trí đó **không đóng góp vào loss**. Cụ thể: `nn.CrossEntropyLoss(ignore_index=tgt_pad)` → PyTorch tự đặt loss=0 tại mọi vị trí mà `Y == tgt_pad`, không cần mask thủ công. Thay vì tự viết mask, dùng `ignore_index` gọn hơn nhiều.
4. **Bottleneck = dùng chung 1 context vector cho mọi decoder timestep.** Cùng $c$ cho mọi bước → Decoder không thể "nhìn lại" các phần khác nhau của câu nguồn → câu dài → thiếu thông tin cục bộ → dịch thiếu ý.
5. **2 thành phần: precision n-gram và Brevity Penalty (BP).** BP = 1 nếu predicted dài hơn/bằng reference; BP < 1 nếu ngắn hơn → phạt ngắn để tránh "dịch 1 từ luôn đạt precision cao".

### Liên kết cần ôn lại

- [[Buổi 47 - Tuần 13|Encoder-Decoder Architecture]]
- [[Buổi 48 - Tuần 13|Sequence-to-Sequence with Attention]]
- [[Attention Mechanism]]
- [[BLEU Score]]

---

# PHẦN I — VẤN ĐỀ GIẢI MÃ SEQUENCE

## 1.1 Từ Training sang Inference

Trong buổi 48, ta đã xây Seq2Seq model hoàn chỉnh. Ở training: Decoder nhận **ground truth tokens** (teacher forcing) — ta biết trước đáp án. Ở inference: Decoder phải **tự sinh** từng token, mỗi token dựa trên tokens đã sinh trước đó.

Tại mỗi bước $t'$, Decoder xuất ra phân phối xác suất:

$$P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c})$$

cho **mọi token** trong vocabulary $\mathcal{Y}$.

**Câu hỏi cốt lõi:** Làm thế nào chọn được sequence $y_1, \ldots, y_{T'}$ tốt nhất?

## 1.2 Định nghĩa kỹ thuật

- **Đây là gì?** Sequence decoding (hay sequence generation) là quá trình chọn output sequence từ phân phối xác suất mà model sinh ra tại mỗi bước.
- **Input/Output gì?** Input: encoder context vector $\mathbf{c}$ và autoregressive hidden state. Output: sequence các token $y_1, \ldots, y_{T'}$.
- **Giải quyết vấn đề gì?** Ta cần tìm sequence có **xác suất joint cao nhất**:

$$\hat{y} = \operatorname*{argmax}_{y \in \mathcal{Y}^{T'}} \prod_{t'=1}^{T'} P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c})$$

> [!NOTE] ELI5
> Model đưa ra xác suất cho từng từ. Ta cần chọn cả câu sao cho xác suất toàn bộ câu là lớn nhất. Nhưng "câu" = tích xác suất từng từ. Mỗi từ có thể chọn từ 10,000 tokens. Với câu 10 từ → 10,000¹⁰ combinations — không thể thử hết.

- **Không gian tìm kiếm:** $|\mathcal{Y}|^{T'}$ sequences. Với $|\mathcal{Y}|=10{,}000, T'=10$: $10{,}000^{10} = 10^{40}$ — bằng số nguyên tử trong vũ trụ.

## 1.3 Các ký hiệu chính

| Ký hiệu | Ý nghĩa |
|---|---|
| $\mathcal{Y}$ | Output vocabulary (kể cả `<eos>`) |
| $T'$ | Độ dài tối đa của output sequence |
| $y_{t'}$ | Token thứ $t'$ trong output |
| $\mathbf{c}$ | Context vector từ Encoder |
| $k$ | Beam size (số lượng beams/ứng viên) |

---

# PHẦN II — GREEDY SEARCH

## 2.1 Định nghĩa kỹ thuật

- **Đây là gì?** Greedy Search chọn token có xác suất cao nhất tại **mỗi bước** một cách độc lập.
- **Input/Output gì?** Input: phân phối xác suất $P(y_t | \ldots)$. Output: $y_t = \operatorname*{argmax}_y P(y | \ldots)$.
- **Giải quyết vấn đề gì?** Chọn nhanh một sequence mà không cần duyệt không gian lớn.

> [!NOTE] ELI5
> Giống như bạn luôn chọn con đường có vẻ tốt nhất **ngay lúc này**, không cân nhắc về sau. Nhanh, đơn giản — nhưng có thể rẽ vào ngõ cụt, bỏ lỡ con đường tốt hơn ở xa.

**Công thức:**

$$y_{t'} = \operatorname*{argmax}_{y \in \mathcal{Y}} \; P(y \mid y_1, \ldots, y_{t'-1}, \mathbf{c})$$

## 2.2 Minh họa bằng số

Giả sử vocabulary: **A**, **B**, **C**, **\<eos\>**.

| Bước | P(A) | P(B) | P(C) | P(\<eos\>) | Greedy chọn |
|---|---|---|---|---|---|
| 1 | **0.50** | 0.30 | 0.15 | 0.05 | **A** |
| 2 | **0.40** | 0.30 | 0.20 | 0.10 | **B** |
| 3 | **0.40** | 0.30 | 0.20 | 0.10 | **C** |
| 4 | **0.60** | 0.20 | 0.10 | 0.10 | **\<eos\>** |

Greedy output: **A → B → C → \<eos\>** — tích xác suất: $0.5 \times 0.4 \times 0.4 \times 0.6 = 0.048$

![[assets/attachments/d2l-buoi-49/greedy-search.svg]]
_Fig 1: Greedy Search. Mỗi cột là một bước, mỗi hàng là một token. Số màu xanh là P(token|bước trước). Đường xanh là path Greedy chọn._

## 2.3 Tại sao Greedy thất bại?

Hãy thử path khác — chọn **C** ở bước 2 (P=0.3, thấp hơn B):

| Bước | P(A) | P(B) | P(C) | P(\<eos\>) | Chọn |
|---|---|---|---|---|---|
| 1 | **0.50** | 0.30 | 0.15 | 0.05 | A |
| 2 | 0.20 | 0.30 | **0.40** | 0.10 | **C** |
| 3 | 0.20 | **0.60** | 0.10 | 0.10 | **B** |
| 4 | **0.60** | 0.20 | 0.10 | 0.10 | **\<eos\>** |

Path A → C → B → \<eos\> — tích xác suất: $0.5 \times 0.3 \times 0.6 \times 0.6 = 0.054$

$$0.054 > 0.048$$

**Greedy chọn path có tích 0.048, nhưng path tốt nhất có tích 0.054 — Greedy thất bại.**

> [!WARNING] Quan sát then chốt
> Tại bước 2, Greedy chọn **B (P=0.4)** vì 0.4 > 0.3. Nhưng chọn **C** thì bước 3 và 4 sẽ có xác suất cao hơn nhiều. Điều này cho thấy: **quyết định cục bộ tốt nhất có thể dẫn đến kết quả toàn cục tồi hơn** — đây là bản chất của greedy.

## 2.4 Ưu và nhược điểm

| Tiêu chí       | Greedy Search                                |
| -------------- | -------------------------------------------- |
| **Chi phí**    | $\mathcal{O}(\mathcal{Y}\cdot T')$ — rẻ nhất |
| **Chất lượng** | Không tối ưu toàn cục                        |
| **Tốc độ**     | Cực nhanh, inference real-time               |
| **Bộ nhớ**     | Chỉ cần giữ 1 beam                           |
| **Ứng dụng**   | Baseline, các bài toán đơn giản              |

---

# PHẦN III — EXHAUSTIVE SEARCH

## 3.1 Định nghĩa kỹ thuật

- **Đây là gì?** Duyệt **tất cả** $|\mathcal{Y}|^{T'}$ sequences và chọn sequence có tích xác suất lớn nhất.
- **Giải quyết vấn đề gì?** Tìm chính xác sequence tối ưu toàn cục.

> [!NOTE] ELI5
> Thử **mọi** con đường có thể trong rừng, rồi chọn con đường dẫn đến đích đẹp nhất. Chắc chắn tìm được đáp án tốt nhất — nhưng bạn sẽ già trước khi tìm xong.

## 3.2 Phân tích độ phức tạp

| Tham số | Giá trị |
|---|---|
| $\|\mathcal{Y}\|$ | 10,000 |
| $T'$ | 10 |
| Tổng sequences | $\|\mathcal{Y}\|^{T'} = 10{,}000^{10} = 10^{40}$ |
| Greedy cost | $\mathcal{O}(10{,}000 \times 10) = 10^5$ |

$$10^{40} \gg 10^5$$

Không gian tìm kiếm quá lớn — hoàn toàn không khả thi.

![[assets/attachments/d2l-buoi-49/exhaustive-search.svg]]
_Fig 2: Không gian tìm kiếm của Exhaustive Search. Mỗi tầng nhân lên $|\mathcal{Y}|$ lần. Với vocabulary 10,000 tokens và độ dài 10: $10{,}000^{10}$ sequences — không thể tính toán được._

---

# PHẦN IV — BEAM SEARCH

## 4.1 Ý tưởng cốt lõi

> [!NOTE] ELI5
> Thay vì đi một mình (greedy) hay thử mọi đường (exhaustive), bạn đồng thời đi cùng **$k$ người bạn**. Ở mỗi ngã rẽ, mỗi người chọn hướng tốt nhất cho mình. Sau đó, **tất cả $k$ người** ghép đôi với **tất cả** $|\mathcal{Y}|$ hướng → $k \times |\mathcal{Y}|$ khả năng. Chỉ giữ lại $k$ khả năng tốt nhất. Lặp lại!

**Beam Search nằm ở điểm giữa spectrum Greedy ↔ Exhaustive.**

- **Đây là gì?** Giữ $k$ hypotheses (beams) tốt nhất tại mỗi bước, mở rộng tất cả cùng lúc.
- **Input/Output gì?** Input: encoder context, $k$ beams hiện tại. Output: $k$ beams mới (mỗi beam = sequence + log probability tích lũy).
- **Giải quyết vấn đề gì?** Tìm sequence tốt hơn Greedy mà không cần duyệt $|\mathcal{Y}|^{T'}$ sequences.
- **Siêu tham số:** Beam size $k$

| Giá trị $k$                        | Ý nghĩa                             |
| ---------------------------------- | ----------------------------------- |
| $k = 1$                            | ≡ Greedy Search                     |
| $k =\mathcal{Y}\cdot T'$ — rẻ nhất | ≡ Exhaustive Search (không khả thi) |
| $k \in [3, 10]$                    | Thực tế, cân bằng tốt               |

## 4.2 Chi phí tính toán

$$\mathcal{O}(k \cdot |\mathcal{Y}| \cdot T')$$

So với Exhaustive: bỏ đi $|\mathcal{Y}|^{T'-1}$ lần!

## 4.3 Thuật toán từng bước

### Bước 1: Khởi tạo (t = 1)

Từ start state, chọn **$k$ tokens** có xác suất cao nhất:

$$\text{top-}k \; P(y_1 \mid \mathbf{c})$$

Mỗi token này bắt đầu một **beam** (candidate sequence).

### Bước 2: Mở rộng (t = 2, 3, ...)

Với **mỗi beam** hiện có, tính xác suất cho **tất cả** tokens trong vocabulary:

$$P(y_1, \ldots, y_t \mid \mathbf{c}) = P(y_t \mid y_1, \ldots, y_{t-1}, \mathbf{c}) \cdot P(y_1, \ldots, y_{t-1} \mid \mathbf{c})$$

Chọn **$k$ sequences** có xác suất tích lũy cao nhất từ $k \times |\mathcal{Y}|$ ứng viên.

### Bước 3: Kết thúc

- Tất cả $k$ beams kết thúc bằng **\<eos\>**
- Hoặc đạt **độ dài tối đa $T'$**

## 4.4 Minh họa với k = 2

![[assets/attachments/d2l-buoi-49/beam-search.svg]]
_Fig 3 (D2L 10.8.3): Beam Search với k=2. Vocabulary $\mathcal{Y} = \{A, B, C, D, E\}$ (một trong số là \<eos\>). Candidate sequences: A, C, AB, CE, ABD, CED._

### Phân tích chi tiết từng bước

**Timestep 1:**
- Encoder output context vector $\mathbf{c}$
- Decoder tính $P(y_1 \mid \mathbf{c})$ cho mọi $y_1 \in \mathcal{Y}$
- Top-2: **A** (P cao nhất) và **C** (P cao thứ 2)
- Giữ 2 beams: `[A]`, `[C]`

**Timestep 2:**
- Beam `[A]`: tính $P(y_2 \mid A, \mathbf{c})$ cho mọi $y_2$
- Beam `[C]`: tính $P(y_2 \mid C, \mathbf{c})$ cho mọi $y_2$
- Tổng: $2 \times 5 = 10$ ứng viên
- Giả sử top-2: **AB** (từ A) và **CE** (từ C)
- Giữ 2 beams: `[A, B]`, `[C, E]`

Công thức:

$$P(A, B \mid \mathbf{c}) = P(A \mid \mathbf{c}) \cdot P(B \mid A, \mathbf{c})$$

$$P(C, E \mid \mathbf{c}) = P(C \mid \mathbf{c}) \cdot P(E \mid C, \mathbf{c})$$

**Timestep 3:**
- Beam `[A, B]`: tính $P(y_3 \mid A, B, \mathbf{c})$
- Beam `[C, E]`: tính $P(y_3 \mid C, E, \mathbf{c})$
- Tổng: 10 ứng viên
- Giả sử top-2: **ABD** và **CED**

**6 candidate sequences:** A, C, AB, CE, ABD, CED

## 4.5 Length-Normalized Score

> [!NOTE] Tại sao cần normalize theo độ dài?
> Một sequence ngắn có ít tokens → ít phép nhân xác suất → tích số có thể **lớn hơn** so với sequence dài hay. Điều này khiến model **ưu tiên câu ngắn** — không công bằng vì câu ngắn dễ đạt xác suất cao hơn câu dài.

**Beam Search dùng length-normalized log probability:**

$$\text{Score}(y_1, \ldots, y_L) = \frac{1}{L^\alpha} \sum_{t'=1}^{L} \log P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c})$$

**Từ điển ký hiệu:**

| Ký hiệu | Ý nghĩa |
|---|---|
| $L$ | Độ dài sequence |
| $\alpha$ | Hệ số penalty (thường 0.75) |
| $L^\alpha$ | Penalty cho sequence dài — tránh ưu tiên ngắn |
| $\log P$ | Log probability — tránh numerical underflow |

### 4.6 Tại sao dùng $\log$?

> [!NOTE] Anti-cramming check
> **Tại sao dùng $\log$ thay vì raw probability?** Đây là câu hỏi hay bị nhồi nhét mà không hiểu bản chất.

**Vấn đề:** Tích của nhiều xác suất nhỏ → numerical underflow.

$$\prod_{t=1}^{100} 0.001 = 10^{-300}$$

Float64 không biểu diễn được $10^{-300}$ — dưới mức trần (~$10^{-308}$).

**Giải pháp:** Dùng $\log$:

$$\log(a \cdot b) = \log a + \log b$$

Thay vì tích: $10^{-300}$ → Ta có tổng: $-300 + (-300) + \ldots = -3000$ → an toàn, so sánh được.

**Tại sao so sánh được?** $\log$ là hàm **đơn điệu tăng**: nếu $a > b$ thì $\log(a) > \log(b)$. Nên so sánh $\log(P_1 \cdot \ldots)$ ≡ so sánh $P_1 \cdot \ldots$.

## 4.7 Bảng so sánh chiến lược

![[assets/attachments/d2l-buoi-49/beam-search-comparison.svg]]
_Fig 4: So sánh 3 chiến lược giải mã theo chi phí và chất lượng._

| Chiến lược | Chi phí | Chất lượng | Ứng dụng |
|---|---|---|---|
| **Greedy** ($k=1$) | $\mathcal{O}(V \cdot T')$ | Thấp | Baseline, fast inference |
| **Beam Search** ($k \in [3,10]$) | $\mathcal{O}(k \cdot V \cdot T')$ | Cao | Machine translation, summarization |
| **Exhaustive** | $\mathcal{O}(V^{T'})$ | Tối ưu | Không khả thi |

---

# PHẦN V — CODE IMPLEMENTATION

## 5.1 Beam Search — PyTorch

```python
import torch
import torch.nn as nn
from typing import List


def beam_search_decode(
    decoder: nn.Module,
    encoder_outputs: torch.Tensor,
    beam_size: int = 5,
    max_len: int = 50,
    bos_idx: int = 1,
    eos_idx: int = 2,
) -> List[List[int]]:
    """
    Beam Search cho Seq2Seq Decoder.

    Args:
        decoder: RNN Decoder (đã train)
        encoder_outputs: Output của Encoder — shape (1, src_len, hidden)
        beam_size: Số lượng beams (k)
        max_len: Độ dài tối đa output
        bos_idx: Index của BOS token
        eos_idx: Index của EOS token

    Returns:
        List of k best sequences (list of token IDs)
    """
    device = encoder_outputs.device

    # Khởi tạo decoder state từ encoder outputs
    decoder_hidden = decoder.init_hidden(encoder_outputs)

    # Beam structure: (log_prob_sum, sequence, hidden_state)
    beams = [(0.0, [bos_idx], decoder_hidden)]
    completed_sequences = []

    for step in range(max_len):
        all_candidates = []

        for log_prob, seq, hidden in beams:
            # Beam đã kết thúc → không mở rộng nữa
            if seq[-1] == eos_idx:
                completed_sequences.append((log_prob, seq))
                continue

            # Forward một bước decoder
            decoder_input = torch.tensor([[seq[-1]]], device=device)
            decoder_output, hidden = decoder(decoder_input, hidden, encoder_outputs)

            # decoder_output: (1, vocab_size) → log probabilities
            log_probs = torch.log_softmax(decoder_output, dim=-1)

            # Top-k tokens từ distribution hiện tại
            topk_log_probs, topk_indices = log_probs.topk(beam_size, dim=-1)

            for k_idx in range(beam_size):
                token = topk_indices[0, k_idx].item()
                token_log_prob = topk_log_probs[0, k_idx].item()
                new_log_prob = log_prob + token_log_prob
                new_seq = seq + [token]
                all_candidates.append((new_log_prob, new_seq, hidden))

        if not all_candidates:
            break

        # Chọn top-k beams từ tất cả ứng viên
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        beams = all_candidates[:beam_size]

    # Kết hợp completed và in-progress beams
    all_seqs = completed_sequences + [(lp, s) for lp, s, _ in beams]
    all_seqs.sort(key=lambda x: x[0], reverse=True)

    return [seq for _, seq in all_seqs[:beam_size]]
```

### Phân tích từng dòng

**Dòng 30: `torch.log_softmax(decoder_output, dim=-1)`**

- `decoder_output`: (1, vocab_size) — logits cho mọi token
- `log_softmax`: biến logits → log probabilities (đã normalize thành phân phối)
- **Tại sao dùng log_softmax thay vì softmax?** Vì ta cần log probabilities để cộng dồn (tránh underflow). `log_softmax` tính $\log(\text{softmax}(x_i)) = x_i - \log(\sum_j e^{x_j})$ — an toàn về numerical stability.

**Dòng 31: `topk(beam_size, dim=-1)`**

- Trả về `topk_log_probs` (giá trị) và `topk_indices` (chỉ số token)
- Từ $V$ tokens, chỉ giữ lại $k$ tokens có probability cao nhất
- Không dùng `argmax` (greedy) vì ta cần giữ **top-k** beams

**Dòng 35: `log_prob + token_log_prob`**

- Đây là cumulative log probability của sequence mới
- **Cộng vì:** $\log(P_1 \cdot P_2) = \log P_1 + \log P_2$
- Đảm bảo so sánh được giữa sequences có độ dài khác nhau

## 5.2 Length-Normalized Score

```python
def length_normalized_score(
    log_probs: torch.Tensor,
    alpha: float = 0.75,
) -> torch.Tensor:
    """
    Tính length-normalized score.
    Score = (1 / L^alpha) * sum(log P)

    Từ điển ký hiệu:
    - log_probs: Tensor (seq_len,) chứa log P của từng token
    - alpha: Penalty factor — thường 0.75 (Google NMT, D2L)
    - length: Số tokens trong sequence
    - penalty: L^alpha — chia để penalize sequence dài
    """
    length = log_probs.size(0)
    lp = log_probs.sum()       # sum of log probs = log of product
    penalty = (length ** alpha)
    return lp / penalty
```

| $\alpha$ | Hiệu ứng |
|---|---|
| $\alpha = 0$ | Không normalize — ưu tiên câu ngắn |
| $\alpha = 0.5$ | Cân bằng nhẹ |
| $\alpha = 0.75$ | **Phổ biến nhất** (Google NMT, D2L) |
| $\alpha = 1.0$ | Tương đương trung bình log-prob |

## 5.3 Shape Flow

```text
Encoder output:  (1, src_len, hidden)
                       ↓
Decoder hidden:  (num_layers, 1, hidden)
                       ↓
Timestep t:
  decoder_input:    (1, 1)              ← last predicted token
  decoder_output:  (1, vocab_size)     ← logits
  log_probs:       (1, vocab_size)      ← log softmax
                        ↓
  Top-k:           (1, k) giá trị + (1, k) indices
                        ↓
k × vocab_size candidates được tạo từ mỗi beam
  → Chọn top-k → k beams mới
                        ↓
Lặp lại cho đến khi k beams kết thúc
```

---


# PHẦN VI — CHỌN BEAM SIZE

| Beam Size | Ưu điểm | Nhược điểm |
|---|---|---|
| $k = 1$ | Nhanh nhất | Chất lượng thấp |
| $k = 3$ | Cân bằng tốt | Ít diversity |
| $k = 5$ | Thường đủ tốt | Tăng 5× inference time |
| $k = 10$ | Chất lượng cao | Chi phí cao |

> [!TIP] Best practice
> - Bắt đầu với $k=3$ hoặc $k=5$
> - Tăng dần nếu cần chất lượng cao hơn
> - GPU inference: $k \in [5, 10]$ là sweet spot
> - CPU real-time: $k \leq 3$

---

# PHẦN VII — BẢNG TÓM TẮT

| Khái niệm | Giải thích |
|---|---|
| **Greedy Search** | Chọn token có P cao nhất mỗi bước. Rẻ nhưng không tối ưu toàn cục. |
| **Exhaustive Search** | Thử tất cả sequences. Tối ưu nhưng $\mathcal{O}(V^{T'})$ — không khả thi. |
| **Beam Search** | Giữ $k$ beams tốt nhất. Cân bằng $\mathcal{O}(kVT')$ giữa Greedy và Exhaustive. |
| **Length Normalization** | Score = $\frac{1}{L^\alpha}\sum \log P$ → tránh ưu tiên câu ngắn. |
| **Log Probability** | $\log$ biến tích thành tổng → tránh underflow. |
| **Greedy = Beam k=1** | Beam Search là tổng quát hóa của Greedy. |

---

# PHẦN VIII — BÀI TẬP (10.8.5)

## Bài 1: Greedy vs Beam Search

Nếu beam size $k=3$ và vocabulary $|\mathcal{Y}|=10000$:
- Bước 1: Tính bao nhiêu probability? → 10,000
- Bước 2: Tính bao nhiêu? → $3 \times 10{,}000 = 30{,}000$
- Tổng cho $T'=20$? → $10{,}000 + 19 \times 30{,}000 = 580{,}000$

## Bài 2: Tại sao dùng $\log$?

Cho $P_1 = P_2 = \ldots = P_{50} = 0.001$.
- Tích: $\prod_{i=1}^{50} P_i = 0.001^{50} = 10^{-150}$ → underflow (float64 ~ $10^{-308}$)
- Tổng $\log$: $\sum_{i=1}^{50} \log(0.001) = 50 \times (-3) = -150$ → an toàn

## Bài 3: Beam Search trong Transformer

Trong GPT-2/3, beam search thường **không** được dùng. Tại sao?
- Gợi ý: Nghĩ đến sampling strategies, temperature, và sự khác biệt giữa seq2seq (MT) và language modeling.

---

> [!CHECKLIST]-
> Reader tự kiểm tra:
>
> - [x] Tôi có thể giải thích tại sao Greedy có thể thất bại không?
> - [x] Tôi hiểu tại sao $\log$ cần thiết để tránh underflow?
> - [ ] Tôi biết mỗi tham số trong beam_search_decode có ý nghĩa gì?
> - [x] Tôi hiểu tại sao length normalization cần thiết?
> - [x] Tôi biết beam size $k=1$ tương đương với chiến lược nào?

---

> [!NOTE] Buổi trước
> [[Buổi 48 - Tuần 13]] — 10.7 Sequence-to-Sequence Learning (with Attention)

> [!NOTE] Buổi sau
> [[Buổi 50 - Tuần 14]] — 11.1 Queries, Keys, and Values

---

## Thuật ngữ

| Thuật ngữ | Tiếng Anh | Ghi chú |
|---|---|---|
| Tìm kiếm tham lam | Greedy Search | Argmax mỗi bước |
| Tìm kiếm tia | Beam Search | Giữ k beams tốt nhất |
| Kích thước tia | Beam Size | Số beams (k) |
| Chuẩn hóa độ dài | Length Normalization | Score = (1/L^a) * sum log P |
| Log xác suất | Log Probability | Biến tích thành tổng |
| Siêu phạm vi | Underflow | Số quá nhỏ không biểu diễn được |
| Giải mã | Decoding | Từ probability distribution → sequence |
