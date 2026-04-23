---
session: "D2L Tuần 13, Buổi 48 — 10.7 Sequence-to-Sequence Learning"
d2l_chapter: "10.7"
tags:
  - d2l
  - deep-learning
  - rnn
  - seq2seq
  - nlp
  - machine-translation
  - attention
  - teacher-forcing
aliases:
  - Seq2Seq Training
  - Seq2Seq Implementation
  - RNN Seq2Seq
  - Sequence to Sequence
date: 2026-04-22
status: complete
---

# Buổi 48 — 10.7 Sequence-to-Sequence Learning

> **Nguồn:** [d2l.ai — 10.7](https://d2l.ai/chapter_recurrent-modern/seq2seq.html)
> **Buổi trước:** [[Buổi 47 - Tuần 13]] — 10.6 Encoder-Decoder Architecture
> **Buổi sau:** [[Buổi 49 - Tuần 13]] — 10.8 Beam Search

---

## Active Recall — Ôn lại Buổi 47 (Encoder-Decoder Architecture)

### Ôn lại từ gốc: Encoder-Decoder

> [!NOTE] Giải thích thật đơn giản
> Encoder-Decoder giống như một cặp máy thu phát radio: Encoder là máy phát, nén input thành tín hiệu; Decoder là máy thu, giải nén tín hiệu thành output. Trong MT, Encoder đọc câu tiếng Anh, nén thông tin vào hidden state cuối cùng. Decoder nhận hidden state đó, rồi lần lượt sinh từng từ tiếng Pháp. Nhưng Encoder chỉ trả về **một** hidden state cuối — đây chính là "nút thắt cổ chai" mà Buổi 48 sẽ khắc phục.

### Câu hỏi (không nhìn tài liệu)

1. Encoder-Decoder gồm mấy module chính? Mô tả luồng dữ liệu giữa chúng.
2. Tại sao Decoder cần concat context vector vào mỗi bước thời gian?
3. Encoder trả về gì cho Decoder? State cuối hay tất cả hidden states?
4. "Shift right" trong Seq2Seq là gì? Tại sao cần làm điều này?
5. Sự khác nhau giữa `forward()` và `predict()` trong Seq2Seq?
6. Masked loss tính cross-entropy nhưng bỏ qua token nào? Tại sao?
7. Teacher forcing là gì? Nó giúp gì trong training?
8. Tính số tham số của một Encoder GRU với vocab_size=10000, embed_size=256, num_hiddens=256, num_layers=2.
9. Decoder trong Seq2Seq cơ bản (chưa có attention) dùng context vector cố định. Điều gì xảy ra khi câu nguồn dài?
10. Tại sao BLEU score cần Brevity Penalty?

### Tự trả lời

1. **2 module**: Encoder (nén input) và Decoder (sinh output). Luồng: source → Encoder → context/state → Decoder → target.
2. Vì Decoder không có "trí nhớ" về câu nguồn. Nếu không concat, Decoder chỉ biết hidden state cuối cùng — không đủ thông tin để dịch chính xác.
3. Encoder trả về **final hidden state** (tầng cuối cùng) — chính là context vector. Tất cả hidden states (output) có thể được trả về nhưng Decoder cơ bản chỉ dùng state cuối.
4. Shift right = dịch target sequence sang phải 1 bước. Decoder input = [BOS, y_1, y_2, ...], Labels = [y_1, y_2, ..., EOS]. Tránh information leakage — Decoder không được thấy ground truth tại vị trí đang dự đoán.
5. `forward()` = training (teacher forcing, parallel); `predict()` = inference (autoregressive, token-by-token).
6. Bỏ qua `<pad>` tokens. Padding tokens không mang ý nghĩa — nếu tính loss trên chúng, model sẽ học sai.
7. Teacher forcing = cho Decoder thấy **ground truth token** tại bước trước thay vì predicted token. Giúp gradient ổn định, hội tụ nhanh hơn. Nhược điểm: exposure bias.
8. Embedding: 10000×256 = 2.56M. GRU: 3×[(256+256)×256+256]×2 = 791,040. Tổng Encoder ≈ 3.35M params.
9. Bottleneck — thông tin phải nén vào 1 vector cố định → câu dài → mất thông tin → dịch kém.
10. Vì predict ngắn hơn reference dễ đạt precision cao hơn (ít tokens để so sánh). BP phạt ngắn, thưởng dài.

### Ghi chú khái niệm cần ôn lại

- [[Recurrent Neural Network]]
- [[Buổi 44 - Tuần 12|Gated Recurrent Unit]]
- [[20_Areas/AI/Concepts/LSTM|Long Short-Term Memory]]
- [[Buổi 45 - Tuần 12|Deep RNN]]
- [[Buổi 45 - Tuần 12|Bidirectional RNN]]
- [[Buổi 47 - Tuần 13|Encoder-Decoder]]
- [[Buổi 46 - Tuần 13|Machine Translation]]
- [[BLEU Score]]

---

# PHẦN I — TỔNG QUAN: SEQ2SEQ TRONG BỐI CẢNH

---

## 1. Vị trí của 10.7 trong Chương 10

Chương 10 — Modern RNN — đã đi một hành trình dài từ vanilla RNN đến các gating mechanisms:

```
9.4  RNN cơ bản     → H_t = tanh(X_t W_xh + H_{t-1} W_hh)
9.5  RNN from Scratch → Full training loop + gradient clipping
9.6  RNN Concise     → nn.RNN API, cuDNN optimization
10.2 GRU            → 2 cổng: reset + update, giải vanishing gradient
10.3 LSTM           → 3 cổng + cell state, long-term dependencies
10.4 Deep RNN + BiRNN → Stacking layers + bidirectional context
10.5 MT Dataset     → Dữ liệu song ngữ, BLEU metric
10.6 Encoder-Decoder → Abstract framework: nén → representation → sinh
10.7 SEQ2SEQ LEARNING → *Bài này*: Implementation cụ thể Encoder-Decoder
                          cho MT, với attention mechanism
```

> [!NOTE] ELI5 — 10.7 là gì?
> 10.6 cho ta cái khung — như một blueprint cho ngôi nhà. 10.7 là ta **xây thật** ngôi nhà đó: mua gạch (Encoder), trộn vữa (Decoder), đổ móng (training loop). Ta sẽ implement đầy đủ Encoder + Decoder + Teacher Forcing + Masked Loss + BLEU Evaluation — tất cả ghép lại thành một hệ thống dịch máy hoàn chỉnh. Và quan trọng nhất: ta sẽ thấy **tại sao** attention là cần thiết, thông qua việc hiểu rõ hạn chế của Seq2Seq không attention.

**10.7 giải quyết vấn đề gì?** Dùng Encoder-Decoder architecture (đã học ở 10.6) để xây dựng một hệ thống MT hoàn chỉnh: Encoder đọc câu nguồn, Decoder sinh câu đích. Hai paper nền tảng: Cho et al. (2014) và Sutskever et al. (2014).

---

## 2. Hai Paper Nền Tảng

### 2.1 Cho et al. (2014) — Learning Phrase Representations

**"Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"**

Đây là paper đầu tiên đề xuất **RNN Encoder-Decoder** cho MT. Contribution chính:

- Encoder: unidirectional RNN đọc câu nguồn, hidden state cuối là context vector
- Decoder: RNN nhận context vector, sinh câu đích autoregressive
- **Output của Encoder được concat vào input của Decoder tại mọi bước** — đây chính là hạt giống cho attention
- Score function: $p(y_1, ..., y_T | x_1, ..., x_T) = \prod_{t=1}^T p(y_t | y_1, ..., y_{t-1}, c)$

### 2.2 Sutskever et al. (2014) — Sequence to Sequence

**"Sequence to Sequence Learning with Neural Networks"**

- Độc lập với paper của Cho, cùng thời điểm
- Dùng deep LSTM (4 layers) thay vì GRU
- **Chỉ dùng final hidden state** của Encoder để khởi tạo Decoder (không concat tại mỗi bước)
- Reverse source sentence: đảo ngược thứ tự câu nguồn → cải thiện performance đáng kể
- Lý do: thông tin quan trọng gần cuối câu sẽ gần hơn với Decoder

> [!NOTE] Tại sao reverse source?
> "I went to the store yesterday" → reversed thành "yesterday store the to went I". Khi đó, từ quan trọng gần cuối ("yesterday") được đọc **gần nhất** trước khi Decoder bắt đầu sinh. Đây là một trick thực nghiệm — không có lý thuyết hoàn hảo, nhưng hoạt động tốt.

---

# PHẦN II — TEACHER FORCING: CHIẾN LƯỢC TRAINING

---

## 3. Teacher Forcing — Bắt đầu từ đâu?

### 3.1 ELI5 — Teacher Forcing là gì?

> Hãy tưởng tượng bạn dạy con học nói tiếng Pháp. Thay vì để con tự đoán từ tiếp theo (sẽ sai liên tục), bạn **"giúp" con** bằng cách nói trước từ đúng. Con học cách dịch khi được nhìn thấy câu trả lời — rất hiệu quả, nhưng con không bao giờ tự luyện tập việc đoán. Teacher forcing trong Seq2Seq là y hệt: ta cho Decoder thấy ground truth token tại mỗi bước, thay vì predicted token.

### 3.2 Hai chiến lược Training

| Chiến lược          | Mô tả                  | Decoder input tại step $t$ | Ưu điểm                        | Nhược điểm                       |
| ------------------- | ---------------------- | -------------------------- | ------------------------------ | -------------------------------- |
| **Teacher Forcing** | Cho ground truth token | $y_{t-1}^{GT}$             | Gradient ổn định, hội tụ nhanh | Exposure bias                    |
| **Free-running**    | Dùng predicted token   | $\hat{y}_{t-1}$            | Mô phỏng inference thực tế     | Training khó hơn, có thể diverge |

### 3.3 Minh họa chi tiết Teacher Forcing

```
Ground truth:     "I love you"
Target (Labels):  [je, suis, content, .]     (4 tokens)

Decoder input:    [<bos>, je, suis, content]  (shifted right)
Labels:           [je, suis, content, .]       (original)

Timestep 1: input=<bos>,         predict "je"
Timestep 2: input="je" (GT),    predict "suis"
Timestep 3: input="suis" (GT),  predict "content"
Timestep 4: input="content" (GT), predict "."
```

**Điểm quan trọng:** Tại mỗi timestep, Decoder được cho **ground truth token của bước trước** — không phải predicted token. Điều này đảm bảo Decoder luôn nhận input đúng trong training, dù inference sẽ khác.

### 3.4 Exposure Bias là gì?


> [!NOTE] ELI5
> Exposure bias giống như việc dạy con học bơi trong bể nông suốt cả năm, rồi một ngày đột nhiên đưa ra biển. Con chưa bao giờ trải nghiệm sóng thật! Trong training, Decoder luôn nhận ground truth (bể nông). Trong inference, Decoder nhận predicted token (biển thật) — có thể sai → sai tích lũy.


**Các giải pháp cho Exposure Bias:**

1. **Scheduled Sampling** (Bengio et al., 2015): từ từ chuyển từ teacher forcing sang free-running
2. **Curriculum Learning**: bắt đầu với câu ngắn, tăng dần độ dài
3. **DAVE** / **DAGGER**: interactive learning approaches
4. **Sequence Level Training** (RL): tối ưu trực tiếp BLEU thay vì cross-entropy token-level

---

## 4. Shifting Right — Chi tiết Implementation

### 4.1 Tại sao phải shift?

Trong Seq2Seq, ta cần model học **dự đoán token tiếp theo**. Cụ thể:

- Input Decoder tại step $t$: token tại step $t-1$
- Output Decoder tại step $t$: token tại step $t$

```
Target sequence:  [BOS, je, suis, content, ., EOS]

Labels (Y):       [je, suis, content, ., EOS]  ← Decoder phải dự đoán
Decoder input:     [BOS, je, suis, content, .] ← Decoder nhận ground truth

So sánh:
  Step 1: input=BOS,   label=je     → Dự đoán "je" từ "<bos>"
  Step 2: input=je,     label=suis   → Dự đoán "suis" từ "je" (GT)
  Step 3: input=suis,   label=content → Dự đoán "content" từ "suis" (GT)
  Step 4: input=content, label=.      → Dự đoán "." từ "content" (GT)
  Step 5: input=.,     label=EOS    → Dự đoán "<eos>" từ "."
```

### 4.2 Code Implementation

```python
def shift_right(Y, bos_token_id, device):
    """
    Shift target sequence right bằng cách thêm BOS ở đầu và bỏ EOS ở cuối.

    Args:
        Y: (batch_size, seq_len) — target sequence indices
        bos_token_id: int — index của BOS token
        device: torch.device

    Returns:
        dec_input: (batch_size, seq_len) — shifted right input
    """
    batch_size = Y.shape[0]
    bos = torch.full((batch_size, 1), bos_token_id, dtype=Y.dtype, device=device)
    dec_input = torch.cat([bos, Y[:, :-1]], dim=1)  # bỏ token cuối (EOS)
    return dec_input
```

**Phân tích code:**

- `Y[:, :-1]` — bỏ token cuối cùng (thường là `<eos>`)
- `torch.cat([bos, ...], dim=1)` — thêm `<bos>` vào đầu
- Kết quả: mỗi position $t$ nhận ground truth từ position $t-1$

---

# PHẦN III — ENCODER IMPLEMENTATION (10.7.2)

---

## 5. Encoder — Đọc Câu Nguồn

### 5.1 ELI5 — Encoder làm gì?

> Encoder giống như một người đọc sách: đọc từ trang đầu tiên đến trang cuối, ghi nhớ toàn bộ nội dung vào não. Sau khi đọc xong, não của người đó (hidden state) chứa tóm tắt toàn bộ cuốn sách. Đó chính là context — vector nén chứa thông tin về toàn bộ câu nguồn.

### 5.2 Mục đích kỹ thuật

**Encoder làm gì?** Đọc một variable-length sequence $x_1, x_2, ..., x_T$ và biến đổi nó thành **fixed-shape context variable** $\mathbf{c}$.

**Input/Output:**

- **Input**: $x_1, ..., x_T$ — source tokens (variable length)
- **Output**: $\mathbf{c}$ hoặc $H_T$ — context vector (fixed shape)

**Công thức toán:**

Tại mỗi timestep $t$, RNN biến đổi:
$$\mathbf{h}_t = f(\mathbf{x}_t, \mathbf{h}_{t-1})$$

Sau khi đọc toàn bộ sequence, context được tạo bằng:
$$\mathbf{c} = q(\mathbf{h}_1, \ldots, \mathbf{h}_T)$$

Trong kiến trúc cơ bản (Sutskever et al.): $\mathbf{c} = \mathbf{h}_T$ — chỉ hidden state cuối.

Trong paper của Cho et al.: $\mathbf{c}$ có thể được tính bằng cách khác (ví dụ: weighted sum, hoặc dùng hidden state tại các timestep khác).

### 5.3 Encoder: Unidirectional vs Bidirectional

|                              | Unidirectional               | Bidirectional                      |
| ---------------------------- | ---------------------------- | ---------------------------------- |
| Hidden state $h_t$ phụ thuộc | $x_1, ..., x_t$              | $x_1, ..., x_t, x_{t+1}, ..., x_T$ |
| Phù hợp cho                  | Decoder state initialization | NMT, POS tagging                   |
| Thông tin                    | Past context only            | Full context (past + future)       |
| Shape                        | $(T, h)$                     | $(T, 2h)$                          |

> [!NOTE] Trong D2L 10.7
> D2L sử dụng **unidirectional RNN** cho Encoder. Điều này nghĩa là hidden state tại mỗi bước chỉ chứa thông tin từ **đầu câu đến bước đó**. Decoder "không biết" những từ ở phía sau. Trong MT, điều này có nghĩa là Encoder đọc từ trái sang phải.

### 5.4 Implementation — Seq2SeqEncoder

```python
import collections
import math
import torch
from torch import nn
from d2l import torch as d2l


def init_seq2seq(module):
    """Initialize weights cho sequence-to-sequence learning.
    Dùng Xavier uniform cho linear layers và recurrent weights."""
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)
    if type(module) == nn.GRU:
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(module._parameters[param])


class Seq2SeqEncoder(d2l.Encoder):
    """RNN Encoder cho Sequence-to-Sequence Learning.

    Architecture:
        Input tokens → Embedding → GRU → Hidden states + Final state

    Args:
        vocab_size: kích thước vocabulary nguồn
        embed_size: chiều embedding vector
        num_hiddens: chiều hidden state
        num_layers: số lớp GRU stacked
        dropout: tỉ lệ dropout (mặc định 0)
    """
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        # Bước 1: Embedding layer — token index → vector
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Bước 2: Multi-layer GRU
        self.rnn = d2l.GRU(embed_size, num_hiddens, num_layers, dropout)
        self.apply(init_seq2seq)

    def forward(self, X, *args):
        """
        Args:
            X: (batch_size, num_steps) — token indices của source
        Returns:
            outputs: (num_steps, batch_size, num_hiddens) — tất cả hidden states
            state: (num_layers, batch_size, num_hiddens) — hidden state cuối cùng
        """
        # Token → Embedding
        # X: (batch_size, num_steps)
        # embedding: (vocab_size, embed_size)
        # embs: (batch_size, num_steps, embed_size)
        embs = self.embedding(X.t().type(torch.int64))

        # PyTorch GRU expect (seq_len, batch, input_size)
        # embs: (num_steps, batch_size, embed_size)

        # GRU forward
        outputs, state = self.rnn(embs)

        # outputs: (num_steps, batch_size, num_hiddens)
        # state: (num_layers, batch_size, num_hiddens)

        return outputs, state
```

### 5.5 Shape Analysis Chi Tiết

```
Giả sử:
  vocab_size = 10000
  embed_size = 256
  num_hiddens = 256
  num_layers = 2
  batch_size = 128
  num_steps = 50  (độ dài source)

Input:
  X: (128, 50) — 128 câu, mỗi câu 50 token indices

Step 1: Embedding
  self.embedding(X): (128, 50) × (10000, 256) → (128, 50, 256)

Step 2: Transpose (PyTorch RNN convention)
  X_emb.t(): (50, 128, 256)

Step 3: GRU
  output: (50, 128, 256)  — tất cả 50 hidden states
  state:  (2, 128, 256)   — final hidden state, 2 layers

Output:
  outputs: (num_steps=50, batch_size=128, num_hiddens=256)
  state:  (num_layers=2, batch_size=128, num_hiddens=256)
```

### 5.6 Tại sao cần Xavier Initialization?

Xavier uniform: $$U(-\sqrt{\frac {6}{(fan_{in}+fan_{out})}}, \sqrt{\frac{6}{fan_{in}+fan_{out}}})$$
Giúp gradient flow tốt hơn trong deep networks. Với RNN, initialization quan trọng vì:

- Vanishing/exploding gradient nhạy cảm với initial weights
- Xavier giữ variance ổn định qua các layers
- Đặc biệt quan trọng với GRU/LSTM vì có nhiều gates

---

# PHẦN IV — DECODER IMPLEMENTATION (10.7.3)

---

## 6. Decoder — Sinh Câu Đích

### 6.1 ELI5 — Decoder làm gì?

> Decoder giống như một người thông dịch viên: người đó nhận được tóm tắt từ Encoder (một vector nén), rồi lần lượt nói ra từng từ tiếng Pháp. Mỗi lần nói một từ, người đó nhìn lại tóm tắt để nhớ đang nói về chủ đề gì, rồi quyết định từ tiếp theo. Trong Seq2Seq không attention, người đó chỉ có **một** tờ giấy tóm tắt — không thể nhìn lại chi tiết. Trong Seq2Seq có attention (sẽ học ở Chương 11), người đó có thể **nhìn lại toàn bộ** bản gốc mỗi khi cần.

### 6.2 Mục đích kỹ thuật

**Decoder làm gì?** Tại mỗi timestep $t'$, dự đoán phân phối xác suất cho token tiếp theo:

$$P(y_{t'+1} | y_1, ..., y_{t'}, \mathbf{c})$$

**Công thức toán:**

Decoder RNN biến đổi:
$$\mathbf{s}_{t'} = g(y_{t'-1}, \mathbf{c}, \mathbf{s}_{t'-1})$$

Trong đó:

- $y_{t'-1}$: ground truth token (training) hoặc predicted token (inference)
- $\mathbf{c}$: context vector từ Encoder
- $\mathbf{s}_{t'-1}$: hidden state trước đó

Sau đó, output layer + softmax tính:
$$P(y_{t'+1} | ...) = \text{softmax}(\mathbf{s}_{t'} W_{qy} + b_y)$$

### 6.3 Điểm thiết kế quan trọng

**Thiết kế 1: Concatenate context tại mỗi bước**

Theo D2L 10.7, context vector được **concat với target embedding** tại mỗi bước thời gian của Decoder:

```
Input của Decoder tại step t:
  x_t = [embed(y_{t-1}); context]
       = [embed_size + num_hiddens]
```

Điều này khác với Sutskever et al. (2014), nơi Decoder chỉ được khởi tạo bằng context vector tại step 0.

**Thiết kế 2: Encoder và Decoder phải có cùng num_layers và num_hiddens**

Vì Decoder state được khởi tạo từ Encoder final state — chúng phải compatible về shape.

### 6.4 Implementation — Seq2SeqDecoder

```python
class Seq2SeqDecoder(d2l.Decoder):
    """RNN Decoder cho Sequence-to-Sequence Learning.

    Architecture:
        Target token → Embedding → Concat with context → GRU → Dense → Vocab logits

    Args:
        vocab_size: kích thước vocabulary đích
        embed_size: chiều embedding vector
        num_hiddens: chiều hidden state (phải = encoder's num_hiddens)
        num_layers: số lớp GRU (phải = encoder's num_layers)
        dropout: tỉ lệ dropout (mặc định 0)
    """
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        # Embedding cho target tokens
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # GRU: input_size = embed_size + num_hiddens
        #       (target embedding + context vector)
        self.rnn = d2l.GRU(embed_size + num_hiddens, num_hiddens,
                           num_layers, dropout)
        # Output projection: hidden → vocab
        self.dense = nn.LazyLinear(vocab_size)
        self.apply(init_seq2seq)

    def init_state(self, enc_all_outputs, *args):
        """
        Convert encoder outputs → decoder initial state.

        Args:
            enc_all_outputs: tuple of (outputs, state) từ encoder
        Returns:
            enc_all_outputs: truyền nguyên cho decoder
        """
        return enc_all_outputs

    def forward(self, X, state):
        """
        Args:
            X: (batch_size, num_steps) — shifted target tokens
            state: tuple of (enc_outputs, hidden_state)
        Returns:
            outputs: (batch_size, num_steps, vocab_size) — logits
            state: updated state
        """
        # Step 1: Embed target tokens
        # X: (batch_size, num_steps)
        # embs: (num_steps, batch_size, embed_size)
        embs = self.embedding(X.t().type(torch.int32))

        # Step 2: Extract context from encoder
        enc_output, hidden_state = state
        # enc_output: (num_steps, batch_size, num_hiddens)
        # enc_output[-1]: (batch_size, num_hiddens) — final hidden state
        context = enc_output[-1]

        # Step 3: Broadcast context to match sequence length
        # context: (batch_size, num_hiddens)
        # → context: (num_steps, batch_size, num_hiddens)
        context = context.repeat(embs.shape[0], 1, 1)

        # Step 4: Concatenate embeddings và context
        # embs:      (num_steps, batch_size, embed_size)
        # context:   (num_steps, batch_size, num_hiddens)
        # → concat:  (num_steps, batch_size, embed_size + num_hiddens)
        embs_and_context = torch.cat((embs, context), -1)

        # Step 5: GRU forward
        outputs, hidden_state = self.rnn(embs_and_context, hidden_state)
        # outputs:     (num_steps, batch_size, num_hiddens)
        # hidden_state: (num_layers, batch_size, num_hiddens)

        # Step 6: Project to vocabulary
        # outputs: (num_steps, batch_size, num_hiddens)
        # → dense: (num_steps, batch_size, vocab_size)
        outputs = self.dense(outputs).swapaxes(0, 1)
        # outputs: (batch_size, num_steps, vocab_size)

        return outputs, [enc_output, hidden_state]
```

### 6.5 Minh họa Data Flow Chi Tiết

```
Input: "je suis content" (3 tokens đã shifted)
Source: "I am happy" (Encoder đã encode vào enc_output)

Step-by-step trong Decoder:

Token "je" → Embedding → [embed_je; context]
                               ↓
                         GRU → hidden_1 → Dense → P( suis | <bos>, je, context )
                                 ↓
Token "suis" → Embedding → [embed_suis; context]  ← context CỐ ĐỊNH!
                               ↓
                         GRU → hidden_2 → Dense → P( content | <bos>, je, suis, context )
                                 ↓
Token "content" → Embedding → [embed_content; context] ← context CỐ ĐỊNH!
                               ↓
                         GRU → hidden_3 → Dense → P( <eos> | ..., content, context )

Nhận xét: Context vector = enc_output[-1] = h_T
          → Cùng một context vector cho MỌI timestep của Decoder
          → ĐÂY LÀ BOTTLENECK!
```

### 6.6 Shape Analysis Chi Tiết

```
Giả sử:
  vocab_size = 10000
  embed_size = 256
  num_hiddens = 256
  num_layers = 2
  batch_size = 128
  num_steps = 50

Input:
  X: (128, 50) — shifted target tokens

Step 1: Embedding
  self.embedding(X): (128, 50) × (10000, 256) → (128, 50, 256)
  transpose: (50, 128, 256)

Step 2: Context
  enc_output[-1]: (128, 256) — final hidden state
  repeat(50, 1, 1): (50, 128, 256)

Step 3: Concatenate
  torch.cat([embs, context], -1): (50, 128, 512) ← embed + hidden

Step 4: GRU Forward
  GRU(input_size=512, hidden_size=256, num_layers=2)
  → outputs: (50, 128, 256)

Step 5: Dense projection
  Dense(256 → 10000)
  → outputs: (50, 128, 10000)

Step 6: Swap axes
  outputs.swapaxes(0, 1): (128, 50, 10000)
  → Final output: (batch_size, num_steps, vocab_size)
```

---

# PHẦN V — SEQ2SEQ MODEL VÀ LOSS (10.7.4 + 10.7.5)

---

## 7. Seq2Seq Model — Kết Hợp Encoder-Decoder

### 7.1 EncoderDecoder Base Class

D2L định nghĩa `EncoderDecoder` base class để quản lý luồng dữ liệu:

```python
class EncoderDecoder(nn.Module):
    """Base class cho toàn bộ Encoder-Decoder model.

    Quản lý kết nối giữa Encoder và Decoder:
        1. Encoder đọc source → trả về encoded representation
        2. Decoder init_state() convert encoder outputs → decoder state
        3. Decoder forward() sinh predictions
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        """Training forward pass.

        Args:
            enc_X: (batch_size, src_len) — source tokens
            dec_X: (batch_size, tgt_len) — shifted target tokens
        Returns:
            logits: (batch_size * tgt_len, vocab_size)
            state: decoder state
        """
        # Bước 1: Encoder forward
        enc_outputs, enc_state = self.encoder(enc_X)

        # Bước 2: Decoder init state
        dec_state = self.decoder.init_state(enc_outputs, *args)

        # Bước 3: Decoder forward
        dec_outputs, dec_state = self.decoder(dec_X, dec_state)

        return dec_outputs, dec_state
```

### 7.2 Seq2Seq Model — Full Implementation

```python
class Seq2Seq(d2l.EncoderDecoder):
    """RNN Encoder-Decoder cho Sequence-to-Sequence Learning.

    Kết hợp Seq2SeqEncoder + Seq2SeqDecoder + masked loss + Adam optimizer.
    """
    def __init__(self, encoder, decoder, tgt_pad, lr):
        super().__init__(encoder, decoder)
        self.save_hyperparameters()

    def validation_step(self, batch):
        """Validation step — không cần backward."""
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)

    def configure_optimizers(self):
        """Adam optimizer — được khuyến nghị cho Seq2Seq."""
        return torch.optim.Adam(self.parameters(), lr=self.lr)
```

### 7.3 ELI5 — Tại sao dùng Adam thay vì SGD?

> SGD giống như leo núi bằng cách đi từng bước nhỏ theo hướng dốc nhất. Adam giống như có đôi giày leo núi tốt hơn — nó điều chỉnh được kích thước bước dựa trên độ dốc trung bình và phương sai. Trong Seq2Seq, learning rate schedule phức tạp hơn — Adam tự động điều chỉnh per-parameter learning rate.

---

## 8. Masked Loss — Bỏ Qua Padding Tokens

### 8.1 Vấn đề Padding trong Seq2Seq

Trong một batch, các câu có độ dài khác nhau:

```
Câu 1: "je suis"        → [je, suis, <eos>, <pad>, <pad>, ...]  # 6 tokens
Câu 2: "il est calme"   → [il, est, calme, <eos>, <pad>, ...]   # 6 tokens
Câu 3: "merci"         → [merci, <eos>, <pad>, <pad>, <pad>, ...]  # 6 tokens
```

Sau khi padding, tất cả câu có cùng độ dài — nhưng padding tokens `<pad>` **không mang ý nghĩa**. Nếu tính cross-entropy trên chúng:

1. Loss bị "pha loãng" bởi các predictions không quan trọng
2. Model học dự đoán `<pad>` → không tối ưu

### 8.2 Masking Strategy

```python
def masked_loss(Y_hat, Y, tgt_pad):
    """
    Tính cross-entropy loss nhưng bỏ qua padding tokens.

    Args:
        Y_hat: (batch_size * num_steps, vocab_size) — predicted logits
        Y:     (batch_size * num_steps,)             — ground truth indices
        tgt_pad: int                                — index của padding token

    Returns:
        Scalar loss (trung bình trên non-padding tokens)
    """
    # Tính cross-entropy cho TẤT CẢ positions
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    l = loss_fn(Y_hat, Y)  # l: (batch_size * num_steps,)

    # Tạo mask: 1 nơi có token thật, 0 nơi có padding
    mask = (Y.reshape(-1) != tgt_pad).type(torch.float32)
    # mask: (batch_size * num_steps,)

    # Nhân loss với mask → padding positions contribute 0
    # Chia cho tổng mask → lấy trung bình trên non-padding tokens
    return (l * mask).sum() / mask.sum()
```

### 8.3 Ví dụ số

```
Giả sử batch_size = 2, num_steps = 4

Y (labels):     [3, 45, 7, 1,    12, 67, 8, 1]
                ↑ valid         ↑ <pad> (index=1)

Y_hat (logits): predicted logits cho 8 positions

loss_fn(Y_hat, Y): [2.1, 0.3, 1.5, 0.0,    0.2, 0.1, 0.3, 0.0]
                    ↑ valid       ↑ <pad>

mask:            [1.0, 1.0, 1.0, 1.0,    0.0, 0.0, 0.0, 0.0]
                 ↑ valid         ↑ <pad>

(l * mask):      [2.1, 0.3, 1.5, 0.0,    0.0, 0.0, 0.0, 0.0]
sum: 4.0

(l * mask).sum() / mask.sum() = 4.0 / 4.0 = 1.0
```

### 8.4 Tại sao không dùng `ignore_index` trong CrossEntropyLoss?

```python
# Cách 1: Dùng ignore_index
loss_fn = nn.CrossEntropyLoss(ignore_index=tgt_pad)

# Cách 2: Dùng mask (như trên)
loss_fn = nn.CrossEntropyLoss(reduction='none')
masked_loss = (l * mask).sum() / mask.sum()
```

| Tiêu chí              | ignore_index | Masking        |
| --------------------- | ------------ | -------------- |
| Code                  | Đơn giản hơn | Phức tạp hơn   |
| Tính toán             | O(n)         | O(n) + O(mask) |
| Hỗ trợ per-token loss | Không        | Có             |
| Kết hợp với weighting | Khó          | Dễ             |
| Trong D2L             | Không dùng   | Dùng mask      |

---

# PHẦN VI — TRAINING THỰC TẾ (10.7.6)

---

## 9. Training Loop — Từ Data đến Model

### 9.1 Full Training Pipeline

```python
import collections
import math
import torch
from torch import nn
from d2l import torch as d2l


def train_seq2seq(data, net, lr, num_epochs, tgt_vocab, device,
                  tgt_pad=None, grad_clip_val=1):
    """Training loop cho Seq2Seq model.

    Pipeline:
        1. Load batch (source, shifted_target, labels)
        2. Forward pass
        3. Compute masked loss
        4. Backward pass + gradient clipping
        5. Update weights

    Args:
        data: DataLoader chứa MT dataset
        net: Seq2Seq model
        lr: learning rate
        num_epochs: số epochs
        tgt_vocab: target vocabulary (để lấy BOS token)
        device: torch.device
        tgt_pad: padding token index (cho masking)
        grad_clip_val: ngưỡng gradient clipping
    """
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=tgt_pad)

    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs])

    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # loss_sum, num_tokens

        for batch in data:
            X, src_valid_len, Y, tgt_valid_len = batch
            X, Y = X.to(device), Y.to(device)

            # Shift right Y cho Decoder input
            # Y: [y_1, y_2, ..., y_T, <eos>]
            # Y_shifted: [<bos>, y_1, y_2, ..., y_T]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                              device=device).reshape(-1, 1)
            Y_input = torch.cat([bos, Y[:, :-1]], dim=1)

            # Forward
            Y_hat = net(X, Y_input)

            # Loss: flatten predictions và labels
            l = loss_fn(Y_hat, Y.reshape(-1))
            l = l.mean()

            # Backward
            optimizer.zero_grad()
            l.backward()

            # Gradient clipping — bắt buộc cho Seq2Seq
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip_val)

            optimizer.step()

            metric.add(l * Y.numel(), Y.numel())

        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
            print(f"Epoch {epoch+1}, Loss: {metric[0] / metric[1]:.4f}")
```

### 9.2 Gradient Clipping trong Seq2Seq — Tại sao BẮT BUỘC?

```python
torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip_val=1)
```

> [!NOTE] ELI5 — Gradient Clipping
> Gradient clipping giống như khi bạn đang cầm một cốc nước quá đầy. Thay vì làm đổ nước ra sàn, bạn uống bớt một ng sor. Gradient quá lớn (exploding gradient) sẽ làm weights thay đổi quá mạnh → training unstable. Clipping giữ gradient trong ngưỡng an toàn.

**Tại sao Seq2Seq đặc biệt cần clipping?**

1. **Long sequences**: MT sequences có thể dài 50-100 tokens → gradient chain dài → exploding
2. **Cross-attention (attention trong Chương 11)**: gradient từ attention có thể unstable
3. **Decoder LSTM/GRU**: các gates nhạy cảm với gradient magnitude

**So sánh clipping strategies:**

| Strategy                | Công thức                                           | Khi nào                  |     |     |            |                         |
| ----------------------- | --------------------------------------------------- | ------------------------ | --- | --- | ---------- | ----------------------- |
| Gradient Norm Clipping  | $g \leftarrow \min(1, \theta/$                      |                          | g   |     | ) \cdot g$ | Phổ biến nhất, D2L dùng |
| Absolute Value Clipping | $g_i \leftarrow \text{clamp}(g_i, -\theta, \theta)$ | ít dùng                  |     |     |            |                         |
| No Clipping             | —                                                   | Chỉ khi gradient ổn định |     |     |            |                         |

### 9.3 Minh họa Training vs Inference

```
TRAINING (Teacher Forcing):
┌─────────────────────────────────────────────────────────────┐
│  Source: "I love you"                                       │
│  Target: "je t'aime"                                       │
│                                                              │
│  Encoder: I → love → you → h_T                              │
│                                                              │
│  Decoder input:  [<bos>, je, t', aime]                      │
│  Decoder output: [je, t', aime, <eos>]                       │
│                                                              │
│  Loss = CE(predict_je, je) + CE(predict_t', t') + ...     │
└─────────────────────────────────────────────────────────────┘

INFERENCE (Autoregressive):
┌─────────────────────────────────────────────────────────────┐
│  Source: "I love you"                                       │
│                                                              │
│  Encoder: I → love → you → h_T                              │
│                                                              │
│  Step 1: Decoder input=[<bos>]       → predict "je"         │
│  Step 2: Decoder input=[<bos>, je]  → predict "t'"         │
│  Step 3: Decoder input=[<bos>, je, t'] → predict "aime"    │
│  Step 4: Decoder input=[<bos>, je, t', aime] → predict <eos>│
│                                                              │
│  Output: "je t'aime <eos>"                                  │
└─────────────────────────────────────────────────────────────┘
```

---

# PHẦN VII — PREDICTION: AUTOREGRESSIVE GENERATION (10.7.7)

---

## 10. Prediction — Sinh Dịch Thật Sự

### 10.1 ELI5 — Inference khác Training như thế nào?

> Trong training, ta **biết trước đáp án** — cho Decoder thấy từng từ đúng. Trong inference, ta **không biết đáp án** — Decoder phải tự quyết định từng từ, dựa trên những từ đã sinh. Mỗi bước sai → bước tiếp theo sai nhiều hơn (error accumulation).

### 10.2 Greedy Decoding — Chiến lược đơn giản nhất

```python
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, device,
                    num_steps, save_attention_weights=False):
    """Sinh dịch bằng greedy decoding (argmax at each step).

    Args:
        net: trained Seq2Seq model
        src_sentence: câu nguồn (string)
        src_vocab, tgt_vocab: vocabularies
        device: torch.device
        num_steps: số bước tối đa sinh
        save_attention_weights: lưu attention weights (cho Chương 11)

    Returns:
        translation: câu dịch (string)
        attention_weights: (nếu save_attention_weights=True)
    """
    net.eval()

    # Bước 1: Tokenize và encode source
    tokens = src_vocab[src_sentence.lower().split(' ')]
    src_tokens = [src_vocab['<bos>']] + tokens + [src_vocab['<eos>']]

    # Encode source
    src_indices = torch.tensor(src_tokens, device=device).unsqueeze(0)
    enc_outputs, enc_state = net.encoder(src_indices)

    # Bước 2: Khởi tạo decoder state
    dec_state = net.decoder.init_state([enc_outputs, enc_state])

    # Bước 3: Autoregressive generation
    outputs = [tgt_vocab['<bos>']]
    attention_weights = []

    for _ in range(num_steps):
        # Predict next token
        Y = torch.tensor([[outputs[-1]]], device=device)
        Y_hat, dec_state = net.decoder(Y, dec_state)

        # Greedy: chọn token có probability cao nhất
        predicted_token = Y_hat.argmax(2).item()

        # Nếu predict EOS → dừng
        if predicted_token == tgt_vocab['<eos>']:
            break

        outputs.append(predicted_token)

        # Lưu attention weights (cho Chương 11)
        if save_attention_weights:
            attention_weights.append(net.decoder.attention_weights)

    # Bước 4: Convert tokens → words
    translation = tgt_vocab.to_tokens(outputs[1:])  # bỏ <bos>
    return translation, attention_weights
```

### 10.3 Minh họa Greedy Decoding

```
Source: "i love you"

Step 1:
  input: <bos>
  P(je)=0.4, P(il)=0.2, P(merci)=0.1, ... → argmax = "je"
  outputs = [<bos>, je]

Step 2:
  input: je
  P(aime)=0.6, P=suis=0.2, P=sont=0.1, ... → argmax = "aime"
  outputs = [<bos>, je, aime]

Step 3:
  input: aime
  P(<eos>)=0.5, P(.)=0.3, P(les)=0.1, ... → argmax = "<eos>"
  STOP

Output: "je aime"
```

### 10.4 Tại sao Greedy Decoding có hạn chế?

**Vấn đề 1: Local optimal ≠ Global optimal**

```
Greedy chọn "je" vì P(je|<bos>) = 0.9 cao nhất.
Nhưng "je" dẫn đến câu dịch kém: "je suis content" (2.1 BLEU)

Had we chosen "il" (P=0.7), ta có: "il t'aime" (2.8 BLEU)

Greedy không thể nhìn thấy future consequences của mỗi lựa chọn.
```

**Vấn đề 2: Error Propagation**

```
Step 1: predict sai một chút → "je" thay vì "il"
Step 2: context sai → predict tiếp sai hơn
Step 3: sai hơn nữa
...
Step 10: dịch hoàn toàn không liên quan đến nghĩa gốc
```

**Giải pháp: Beam Search**

Beam search giữ **nhiều hypotheses** cùng lúc, không chỉ 1. Sẽ học chi tiết ở Buổi 49.

---

# PHẦN VIII — BLEU EVALUATION THỰC TẾ (10.7.8)

---

## 11. BLEU Score — Đo Chất Lượng Dịch

### 11.1 ELI5 — BLEU là gì?

> BLEU giống như bài kiểm tra chấm điểm dịch: so sánh từng cụm từ (n-gram) trong bản dịch của máy với bản dịch chuẩn. Nếu máy dịch "the black cat" — BLEU kiểm tra xem "the", "black", "cat", "the black", "black cat", "the black cat" có xuất hiện trong reference không. Càng nhiều match → BLEU càng cao. Nhưng nếu máy dịch quá ngắn → bị phạt.

### 11.2 Công thức BLEU — Chi tiết từng thành phần

**Precision $p_n$ cho n-gram bậc $n$:**

$$p_n = \frac{\sum_{\text{n-gram} \in \hat{y}} \min(\text{count}_{\text{n-gram}}, \max_{\text{ref}} \text{count}_{\text{n-gram}})}{\sum_{\text{n-gram} \in \hat{y}} \text{count}_{\text{n-gram}}}$$

**Geometric mean của precisions:**

$$\exp\left(\sum_{n=1}^{k} \frac{1}{k} \log p_n\right) = \prod_{n=1}^{k} p_n^{1/k}$$

D2L dùng công thức hơi khác (weighted by $2^{-n}$):

$$\prod_{n=1}^{k} p_n^{1/2^n}$$

### 11.3 Brevity Penalty — Phạt câu ngắn

$$BP = \begin{cases} 1 & \text{nếu } c \geq r \\ \exp\left(1 - \frac{r}{c}\right) & \text{nếu } c < r \end{cases}$$

Trong đó:

- $c$ = độ dài predicted sequence
- $r$ = độ dài reference sequence

**Ý nghĩa:**

- Predicted dài hơn reference → BP = 1 (không phạt)
- Predicted ngắn hơn reference → BP < 1 (phạt, ngắn hơn nhiều → phạt nặng hơn)

### 11.4 Ví dụ tính BLEU chi tiết

```
Reference: "il est calme ."
Predicted: "elle court ."

Bước 1: Tách tokens
  Reference: [il, est, calme, .]
  Predicted: [elle, court, .]

Bước 2: Tính precisions
  p_1 = matched_1grams / total_1grams = 1/3
         (chỉ "." match)
  p_2 = matched_2grams / total_2grams = 0/2 = 0
  p_3 = matched_3grams / total_3grams = 0/1 = 0
  p_4 = 0/0 = 0

Bước 3: Brevity Penalty
  c = 3, r = 4
  BP = exp(1 - 4/3) = exp(-1/3) ≈ 0.716

Bước 4: BLEU
  BLEU = BP × p_1^(1/2) × p_2^(1/4) × p_3^(1/8) × p_4^(1/16)
       = 0.716 × (1/3)^0.5 × 0^0.25 × 0^0.125 × 0^0.0625
       = 0.716 × 0.577 × 0 × ...
       = 0.0

→ Dịch hoàn toàn sai → BLEU = 0
```

### 11.5 BLEU Implementation

```python
def bleu(pred_seq, label_seq, k):
    """
    Tính BLEU score cho một cặp predicted - reference.

    Args:
        pred_seq: predicted sentence (string)
        label_seq: reference sentence (string)
        k: maximum n-gram order (thường k=2 hoặc k=4)

    Returns:
        BLEU score (0-1, thường nhân 100 để báo cáo %)
    """
    pred_tokens = pred_seq.split(' ')
    label_tokens = label_seq.split(' ')

    len_pred = len(pred_tokens)
    len_label = len(label_tokens)

    # Brevity Penalty
    score = math.exp(min(0, 1 - len_label / len_pred))

    # Geometric mean của precisions
    for n in range(1, min(k, len_pred) + 1):
        num_matches = 0  # số n-grams trong pred mà có trong label
        label_subs = collections.defaultdict(int)

        # Đếm n-grams trong label
        for i in range(len_label - n + 1):
            ngram = ' '.join(label_tokens[i: i + n])
            label_subs[ngram] += 1

        # Đếm matches trong pred (với counting trick để tránh double-count)
        for i in range(len_pred - n + 1):
            ngram = ' '.join(pred_tokens[i: i + n])
            if label_subs[ngram] > 0:
                num_matches += 1
                label_subs[ngram] -= 1  # counting trick

        # Precision cho n-gram bậc n
        p_n = num_matches / (len_pred - n + 1)

        # Cộng vào score với weight = 1/2^n
        score *= math.pow(p_n, math.pow(0.5, n))

    return score
```

### 11.6 Ví dụ BLEU từ D2L

```
engs = ['go .', 'i lost .', "he's calm .", "i'm home ."]
fras = ['va !', "j'ai perdu .", 'il est calme .', 'je suis chez moi .']

Dịch kết quả (PyTorch):
  go .       => ['va', '!'],        BLEU=1.000
  i lost .   => ["j'ai", 'perdu', '.'],  BLEU=1.000
  he's calm . => ['elle', 'court', '.'],  BLEU=0.000  ← SAI!
  i'm home . => ['je', 'suis', 'chez', 'moi', '.'],  BLEU=1.000

"he's calm ." → "elle court ." hoàn toàn sai nghĩa:
  "he's calm" = "anh ấy bình tĩnh" → "elle court" = "cô ấy chạy"
  → BLEU = 0.000

Nhận xét:
  - BLEU = 1.0: dịch gần như hoàn hảo
  - BLEU = 0.0: không có n-gram nào match
  - BLEU không đánh giá SEMANTIC correctness
```

---

# PHẦN IX — HẠN CHẾ VÀ MOTIVATION CHO ATTENTION

---

## 12. Tại sao Seq2Seq cần Attention?

### 12.1 Bottleneck — Vấn đề cốt lõi

```
Seq2Seq không attention:
┌────────────────────────────────────────────────────────────┐
│  Encoder:  h_1 → h_2 → h_3 → ... → h_T                    │
│                                          ↓                 │
│                                    Chỉ một context C = h_T│
│                                          ↓                 │
│  Decoder:    s_1 → s_2 → s_3 → ... → s_T'                │
│              ↑      ↑      ↑                               │
│              C      C      C  ← CỐ ĐỊNH!                 │
└────────────────────────────────────────────────────────────┘

Vấn đề: Decoder phải nhét TOÀN BỘ thông tin câu nguồn
         vào một vector C duy nhất.

         Câu nguồn dài → thông tin bị nén quá mức → MẤT THÔNG TIN!
```

### 12.2 Minh họa bằng số

```
Câu nguồn: "The quick brown fox jumps over the lazy dog"
Độ dài: 9 tokens
Hidden size: 256

Cần nén 9×256 = 2,304 thông tin vào 1×256 = 256 dimensions.
→ Mất ~89% thông tin!

Câu đích ngắn: "Le renard brun"
→ Decoder chỉ cần nhớ: "fox" ~ "renard", "brown" ~ "brun"
→ Nhưng CÓ THỂ đã mất thông tin về "jumps over the lazy dog"
→ Dịch thiếu ý!
```

### 12.3 Attention — Giải pháp

```
Seq2Seq với Attention:
┌────────────────────────────────────────────────────────────┐
│  Encoder:  h_1 → h_2 → h_3 → ... → h_T                    │
│            ↑      ↑      ↑           ↑                     │
│            α_1    α_2    α_3         α_T   ← Attention   │
│            weights weights weights    weights               │
│                  ↓                                         │
│            C_t = Σ α_{t,i} · h_i   ← Context thay đổi!   │
│                  ↓                                         │
│  Decoder:    s_1 → s_2 → s_3 → ... → s_T'                │
│              ↑      ↑      ↑                               │
│            C_1    C_2    C_3  ← MỖI BƯỚC một context MỚI! │
└────────────────────────────────────────────────────────────┘

→ Decoder có thể "nhìn lại" câu nguồn CHỌN LỌC tại mỗi bước
→ Không còn bottleneck!
```

### 12.4 Attention trong Seq2Seq vs Transformer

|                     | Seq2Seq Attention (Chương 11)  | Transformer (Chương 11) |
| ------------------- | ------------------------------ | ----------------------- |
| **Mechanism**       | Bahdanau (additive)            | Scaled dot-product      |
| **Encoder outputs** | All hidden states $h_i$        | All hidden states       |
| **Query**           | Decoder hidden state $s_{t-1}$ | Decoder hidden states   |
| **Key/Value**       | Encoder hidden states $h_i$    | Encoder hidden states   |
| **Complexity**      | $O(T \times T')$               | $O(T^2)$ per layer      |
| **Long sequences**  | Better than no attention       | Best                    |

> [!NOTE] Preview Chương 11
> Attention là cơ chế cho phép Decoder "chú ý" (pay attention) đến các phần khác nhau của câu nguồn tại mỗi bước sinh. Điểm khác biệt với concat context (trong 10.7) là: attention tính **weighted sum có học**, trong đó weights $\alpha_{t,i}$ được tính từ query và keys. Sẽ học chi tiết ở Chương 11.

---

# PHẦN X — LAYER SUMMARY

---

## 13. Tổng Kết Các Layers trong Seq2Seq

![[assets/attachments/d2l-buoi-48/seq2seq-layers.svg]]
_Fig. 10.7.2 (D2L): Layers trong RNN Encoder-Decoder model. Data flow: Source tokens → Embedding → Encoder GRU → Encoder outputs → Decoder init state → Decoder GRU (với context concat) → Dense → Vocab logits → Loss._

### Minh họa kiến trúc tổng thể

![[assets/attachments/d2l-buoi-48/seq2seq_architecture.svg]]
_Fig. 10.7.1 (D2L): Sequence-to-Sequence Learning với RNN Encoder và RNN Decoder. Encoder đọc câu nguồn từ trái sang phải, nén thông tin vào hidden state cuối cùng $H_T$ (context vector). Decoder nhận $H_T$ làm initial state, rồi sinh từng token đích autoregressively với BOS prefix và kết thúc bằng EOS token._

### Minh họa Teacher Forcing

![[assets/attachments/d2l-buoi-48/teacher_forcing.svg]]
_Fig. So sánh Teacher Forcing (trái) vs Free-running (phải). Teacher forcing cho Decoder thấy ground truth token tại mỗi bước; Free-running cho Decoder thấy predicted token. Mũi tên cam chỉ luồng thông tin GT, mũi tên đỏ chỉ luồng loss._

### Minh họa Greedy Decoding

![[assets/attachments/d2l-buoi-48/greedy_decoding.svg]]
_Fig. Quá trình greedy decoding: tại mỗi bước, chọn token có probability cao nhất (argmax). Decoder chỉ có một context vector cố định — đây chính là bottleneck. Hình minh họa câu "I love you" được dịch thành "je aime ." (không chính xác so với reference "il est calme .")._

### Minh họa BLEU Score

![[assets/attachments/d2l-buoi-48/bleu_score.svg]]
_Fig. BLEU score computation: (1) Precision n-grams — đếm số n-grams trong predicted trùng với reference; (2) Brevity Penalty — phạt nếu predicted ngắn hơn reference; (3) Tính tích weighted của precisions. Ví dụ: reference "il est calme ." vs predicted "elle court ." cho BLEU=0.0 vì gần như không có n-gram nào match._

### Minh họa Masked Loss

![[assets/attachments/d2l-buoi-48/masked_loss.svg]]
_Fig. Masked loss: bỏ qua padding tokens trong cross-entropy. Mỗi hàng là một câu trong batch; tokens màu đỏ là `<pad>` được masked ra. Loss chỉ tính trên các valid tokens (màu xanh lá)._

---

# PHẦN XI — BÀI TẬP (10.7.10)

---

## Bài 1: Điều chỉnh Hyperparameters

> _"Điều chỉnh hyperparameters để cải thiện kết quả dịch. Thử nghiệm: embed_size, num_hiddens, num_layers, dropout, lr."_

```python
# Experiment grid
configs = [
    {'embed_size': 128, 'num_hiddens': 128, 'num_layers': 1, 'dropout': 0.0},
    {'embed_size': 256, 'num_hiddens': 256, 'num_layers': 2, 'dropout': 0.1},
    {'embed_size': 512, 'num_hiddens': 512, 'num_layers': 3, 'dropout': 0.2},
    {'embed_size': 256, 'num_hiddens': 512, 'num_layers': 2, 'dropout': 0.1},
]

for config in configs:
    encoder = Seq2SeqEncoder(vocab_size, config['embed_size'],
                             config['num_hiddens'], config['num_layers'],
                             config['dropout'])
    decoder = Seq2SeqDecoder(vocab_size, config['embed_size'],
                             config['num_hiddens'], config['num_layers'],
                             config['dropout'])
    net = Seq2Seq(encoder, decoder, tgt_pad, lr=0.005)
    # train and evaluate
```

**Observations:**

- Embed/hidden lớn hơn → model mạnh hơn nhưng chậm hơn
- Nhiều layers hơn → có thể tốt cho long sequences
- Dropout cao → regularization cho dữ liệu nhỏ
- Learning rate 0.005 với Adam là baseline tốt

## Bài 2: Không có Masking Loss

> _"Chạy experiment không dùng mask trong loss. Quan sát kết quả. Giải thích tại sao."_

```python
# Không mask
loss_fn = nn.CrossEntropyLoss(reduction='mean')  # ignore mask
```

**Expected results:**

- Loss sẽ thấp hơn (vì padding tokens dễ predict → contribute low loss)
- Nhưng actual translation quality không cải thiện
- Model có thể học bias về predicting padding → worse generation

## Bài 3: Encoder và Decoder khác nhau

> _"Nếu Encoder và Decoder có số layers hoặc hidden size khác nhau, làm sao initialize Decoder state?"_

```python
# Trường hợp: Encoder có 2 layers, Decoder có 1 layer
# Encoder state: (2, batch, 256)
# Decoder state: (1, batch, 256)

# Cách 1: Chỉ dùng top layer của Encoder
def init_state(self, enc_outputs, *args):
    enc_output, enc_state = enc_outputs
    # enc_state: (2, batch, h)
    # Lấy top layer
    return enc_state[1:]  # (1, batch, h)

# Cách 2: Project Encoder state sang Decoder dimension
def init_state(self, enc_outputs, *args):
    enc_output, enc_state = enc_outputs
    # Project từ (2, batch, 256) → (1, batch, 512)
    top_layer = enc_state[1]  # (batch, 256)
    projected = self.projection(top_layer)  # (batch, 512)
    return projected.unsqueeze(0)  # (1, batch, 512)
```

## Bài 4: Thay Teacher Forcing bằng Free-running

> _"Trong training, thay teacher forcing bằng feeding predicted token. Quan sát ảnh hưởng."_

```python
def train_free_running(net, data, optimizer, loss_fn, device):
    """Training với free-running (predicted token fed back)."""
    for batch in data:
        X, Y = batch
        X, Y = X.to(device), Y.to(device)

        # Bắt đầu với BOS
        dec_input = torch.full((Y.shape[0], 1), bos_id, device=device)

        # Teacher forcing:
        # for t in range(num_steps):
        #     Y_hat, state = decoder(dec_input, state)
        #     dec_input = Y[:, t:t+1]  # ground truth

        # Free-running:
        for t in range(num_steps):
            Y_hat, state = decoder(dec_input, state)
            predicted = Y_hat.argmax(2)  # greedy
            dec_input = predicted

        # Loss: so sánh với ground truth
        l = loss_fn(Y_hat, Y[:, t])
        ...
```

**Expected:** Training khó hơn nhiều, có thể diverge. Đây là lý do teacher forcing phổ biến.

## Bài 5: Thay GRU bằng LSTM

> _"Chạy experiment thay GRU bằng LSTM. So sánh performance."_

```python
# GRU: 3 gates (reset, update, candidate)
# LSTM: 3 gates (forget, input, output) + cell state

# Cài đặt tương tự, chỉ thay GRU → LSTM
self.rnn = nn.LSTM(embed_size + num_hiddens, num_hiddens,
                   num_layers, dropout=dropout)

# Hidden state becomes (hidden, cell) tuple
# → Cần điều chỉnh decoder init_state
```

**Expected:** LSTM có thêm cell state → tốt hơn cho very long sequences. Nhưng với dataset nhỏ (FR-EN ~40K), khác biệt có thể không đáng kể.

## Bài 6: Thiết kế Output Layer khác

> _"Có cách nào khác để thiết kế output layer của Decoder?"_

```python
# Cách 1: Project hidden + context → vocab (hiện tại)
output = dense(hidden)

# Cách 2: Attention-based scoring
# Dùng attention output (weighted sum của encoder states)
context_t = attention(s_{t-1}, encoder_outputs)
output = dense(concat(hidden, context_t))

# Cách 3: Multi-head output
# Chia hidden thành K parts, mỗi part project riêng
hidden_splits = split(hidden, K)
logits = [dense(h) for h in hidden_splits]

# Cách 4: Share embedding weights với decoder embedding
# Tie weights: output.weight = embedding.weight.T
self.dense.weight = self.embedding.weight.T
```

---

## Tổng kết

| Khía cạnh             | Nội dung                                                  |
| --------------------- | --------------------------------------------------------- | --------------------------------------------------- |
| **Teacher Forcing**   | Cho Decoder thấy GT token tại mỗi step                    | Ổn định gradient, nhanh hơn                         |
| **Encoder**           | Embedding + Multi-layer GRU → hidden states + final state | (T, batch, h) + (L, batch, h)                       |
| **Decoder**           | Embed + Context → GRU → Dense → Vocab                     | Input: embed+context; Output: (batch, steps, vocab) |
| **Context cố định**   | enc_output[-1] được repeat qua mọi step                   | ĐÂY LÀ BOTTLENECK → cần Attention                   |
| **Masked Loss**       | Bỏ qua `<pad>` tokens trong CE                            | Tránh learning padding                              |
| **Gradient Clipping** | $\theta = 1$ norm clipping                                | Ổn định training                                    |
| **Greedy Decoding**   | argmax tại mỗi step                                       | Nhanh nhưng sub-optimal                             |
| **BLEU**              | n-gram precision × Brevity Penalty                        | Đánh giá MT quality                                 |
| **Bottleneck**        | 1 context vector → thông tin mất                          | Motivation cho Attention (Chương 11)                |

---

> **Buổi trước:** [[Buổi 47 - Tuần 13]] — 10.6 Encoder-Decoder Architecture
> **Buổi sau:** [[Buổi 49 - Tuần 13]] — 10.8 Beam Search

---

## Thuật ngữ

| Thuật ngữ             | Tiếng Anh         | Ghi chú                           |
| --------------------- | ----------------- | --------------------------------- |
| Học với thầy          | Teacher Forcing   | Cho GT token làm input            |
| Tự chạy               | Free-running      | Dùng predicted token              |
| Thiên lệch phơi nhiễm | Exposure Bias     | Training ≠ Inference              |
| Mặt nạ                | Masking           | Bỏ qua padding tokens             |
| Cắt gradient          | Gradient Clipping | Giới hạn gradient norm            |
| Dịch tham lam         | Greedy Decoding   | Chọn token tốt nhất tại mỗi bước  |
| Nút thắt cổ chai      | Bottleneck        | 1 context vector cho tất cả steps |
| Trao chiếu            | Attention         | Cơ chế "nhìn lại" câu nguồn       |
| Điểm BLEU             | BLEU Score        | n-gram precision × BP             |

## Liên kết

- [[Recurrent Neural Network]]
- [[Gated Recurrent Unit]]
- [[Long Short-Term Memory]]
- [[Deep Recurrent Neural Networks]]
- [[Bidirectional RNN]]
- [[Encoder-Decoder Architecture]]
- [[Machine Translation]]
- [[BLEU Score]]
- [[Attention Mechanism]]
- [[Gradient Clipping]]
