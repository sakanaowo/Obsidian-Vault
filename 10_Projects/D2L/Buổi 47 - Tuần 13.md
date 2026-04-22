---
session: "D2L Tuần 13, Buổi 47 — 10.6 Encoder-Decoder Architecture"
d2l_chapter: "10.6"
tags:
  - d2l
  - deep-learning
  - rnn
  - encoder-decoder
  - seq2seq
  - nlp
  - module-design
aliases:
  - Encoder-Decoder
  - Seq2Seq Architecture
  - Encoder Decoder
date: 2026-04-21
status: complete
---
	z
# Buổi 47 — 10.6 Encoder-Decoder Architecture

> **Nguồn:** [d2l.ai — 10.6](https://d2l.ai/chapter_recurrent-modern/encoder-decoder.html)
> **Buổi trước:** [[Buổi 46 - Tuần 13]] — 10.5 Machine Translation and the Dataset
> **Buổi sau:** [[Buổi 48 - Tuần 13]] — 10.7 Sequence to Sequence Learning (with Attention)

---

## Active Recall — Ôn lại Buổi 46 (Machine Translation & Dataset)

### Ôn lại từ gốc: Seq2Seq cho Machine Translation

> [!NOTE] Giải thích thật đơn giản
> Dịch máy giống như hai người thư ký: một người (Encoder) nghe toàn bộ câu tiếng Việt và ghi ra ý chính; người kia (Decoder) đọc ý chính đó rồi nói lại bằng tiếng Anh, từ từ, từng câu một. Bottleneck là khi người đầu tiên phải nhét cả bài phát biểu dài vào một tờ giấy nhỏ — rất nhiều thông tin bị mất.

### Câu hỏi (không nhìn tài liệu)

1. Ba paradigm dịch máy theo thứ tự thời gian là gì? Điểm khác biệt cốt lõi?
2. Sequence-to-Sequence gồm mấy thành phần? Mô tả vai trò từng thành phần.
3. Tại sao Seq2Seq cần special tokens `<bos>` và `<eos>`? Chúng được dùng khi nào?
4. Vấn đề "bottleneck" trong Seq2Seq là gì? Tại sao nó xảy ra?
5. Attention giải quyết bottleneck bằng cách nào? Thay đổi gì so với Seq2Seq cơ bản?
6. BLEU score được tính từ những thành phần nào? Công thức tổng quát là gì?
7. Khi nào Brevity Penalty $BP = 1$? Khi nào $BP < 1$?
8. Tại sao MT cần vocabulary riêng cho mỗi ngôn ngữ? Không dùng chung vocab được sao?
9. Collate function trong MT dataloader làm gì? Tại sao cần nó?
10. Out-of-Vocabulary (OOV) là gì? Các giải pháp phổ biến để giảm OOV?

### Tự trả lời

1. Rule-based (từ điển + ngữ pháp viết tay) → Statistical MT (xác suất thống kê) → Neural MT (RNN/Transformer). Khác nhau: mức độ tự động, dữ liệu cần, chất lượng.
2. **2 thành phần**: Encoder (nén câu nguồn) và Decoder (sinh câu đích). Encoder không tạo output trực tiếp mà truyền hidden state cho Decoder.
3. `<bos>` (begin of sequence) = token bắt đầu cho Decoder — Decoder cần một token "khởi đầu" để sinh token đầu tiên. `<eos>` (end of sequence) = báo Decoder dừng sinh — vì độ dài câu đích không cố định.
4. Bottleneck = toàn bộ thông tin câu nguồn bị nén vào **một context vector duy nhất** $C = H_T$. Câu nguồn dài → $C$ phải chứa quá nhiều thông tin → performance giảm mạnh.
5. Attention cho phép Decoder **"nhìn lại"** toàn bộ câu nguồn tại mỗi bước sinh, với weighted sum $\sum_i \alpha_{t,i} H_i$. Mỗi bước có một context vector riêng, không phải 1 vector cố định.
6. BLEU = Brevity Penalty × $\exp(\sum_{n=1}^N w_n \log p_n)$, trong đó $p_n$ là precision n-gram bậc $n$, $w_n = 1/N$.
7. $BP = 1$ khi độ dài output $c$ > độ dài reference $r$. $BP < 1$ khi output ngắn hơn reference.
8. Vì mỗi ngôn ngữ có vocabulary khác nhau: số lượng từ, tần suất, cấu trúc. Dùng chung vocab phí bộ nhớ và tăng OOV rate.
9. Collate function gom các sample riêng lẻ thành một batch: pad các câu ngắn, truncate các câu dài, tính valid lengths để mask padding trong loss.
10. OOV = từ không nằm trong vocabulary → thành `<unk>`. Giải pháp: (a) tăng vocab size; (b) subword tokenization (BPE/WordPiece); (c) character-level modeling.

### Ghi chú khái niệm cần ôn lại

- [[Recurrent Neural Network]]
- [[Gated Recurrent Unit]]
- [[Long Short-Term Memory]]
- [[Deep Recurrent Neural Networks]]
- [[Bidirectional RNN]]
- [[Machine Translation]]

---

# PHẦN I — TỔNG QUAN: THIẾT KẾ KIẾN TRÚC

---

## 1. Tại sao cần kiến trúc Encoder-Decoder?

> [!NOTE] Giải thích đơn giản
> Encoder-Decoder giống như một cặp máy thu phát radio: **Encoder** = máy phát, nhận sóng âm (input), nén thành tín hiệu FM rồi phát đi; **Decoder** = máy thu, nhận tín hiệu FM, giải nén, phát ra âm thanh (output). Channel truyền tín hiệu có thể là dây cáp, sóng vô tuyến, hoặc — trong MT — chính là context vector. Mỗi bộ phát/thu có thể dùng công nghệ khác nhau, nhưng nguyên tắc chung: **nén → truyền → giải nén**.

**Encoder-Decoder là gì?** Encoder-Decoder là một **khung kiến trúc trừu tượng** (abstract architectural pattern) gồm hai module chính:

1. **Encoder**: biến đổi input có cấu trúc tùy ý thành **representation**
2. **Decoder**: biến đổi representation thành output có cấu trúc tùy ý

**Nó giải quyết vấn đề gì?** Cho phép ghép nối bất kỳ encoder nào với bất kỳ decoder nào — tạo ra nhiều ứng dụng khác nhau từ cùng một khung thiết kế.

**Đặc điểm quan trọng:**
- Encoder và Decoder **không chia sẻ trọng số** (trừ pretrained models)
- Thông tin chảy **một chiều**: Encoder → Representation → Decoder
- Có thể dùng attention để cải thiện representation flow

### 1.1 So sánh các biến thể Encoder-Decoder

| Ứng dụng | Encoder | Representation | Decoder |
|---|---|---|---|
| Machine Translation | BiRNN | Hidden states / Attention | RNN autoregressive |
| Image Captioning | CNN backbone | Feature maps / Attention | LSTM autoregressive |
| Speech Recognition | RNN/Transformer | Mel spectrogram | CTC / Attention |
| Video Summarization | 3D CNN | Video features | LSTM autoregressive |
| Question Answering | BERT/Transformer | [CLS] embedding | Generation / Classification |
| Text-to-Speech | Encoder (text) | Linguistic features | Neural vocoder (WaveNet) |

---

## 2. Module Design — Nguyên tắc xây dựng

### 2.1 Nguyên tắc thiết kế module

D2L thiết kế Encoder-Decoder theo kiểu **Object-Oriented**, với base class cho phép subclass override:

```
Encoder (base class)
  └── forward(x) → return encoded representation

Decoder (base class)
  └── init_state(enc_outputs) → convert encoder outputs → decoder state
  └── forward(state, x) → return logits
```

**Tại sao cần base class?**
- Encoder tiền xử lý input (tokenize, embed, positional encoding)
- Decoder tiền xử lý target (tokenize, embed, shift right)
- Logic tiền xử lý khác nhau giữa các ứng dụng → cần interface chung

### 2.2 Hai chiến lược xây dựng Encoder-Decoder

| Chiến lược | Mô tả | Ví dụ |
|---|---|---|
| **Từ đầu (Scratch)** | Tự implement mọi thứ | RNN Seq2Seq trong buổi 46-47 |
| **Tái sử dụng (Composition)** | Ghép nối các module có sẵn | Encoder = pretrained BERT, Decoder = LSTM |

**Trong buổi này:** Tập trung vào chiến lược **từ đầu** — hiểu rõ từng thành phần trước khi dùng pretrained.

---

# PHẦN II — ENCODER: CHI TIẾT IMPLEMENTATION (10.6.1)

---

## 3. Encoder Interface — Thiết kế giao diện

### 3.1 Base class

```python
from torch import nn

class Encoder(nn.Module):
    """Base class cho Encoder. Subclass phải override forward()."""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError("Subclass phải override forward()")
```

### 3.2 RNN Encoder — Cài đặt chi tiết

```python
class RNNEncoder(Encoder):
    """RNN Encoder: đọc sequence, trả về hidden state cuối cùng"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers

    def forward(self, X):
        """
        Args:
            X: (batch_size, num_steps) — token indices
        Returns:
            state: (num_layers, batch_size, num_hiddens) — final hidden state
        """
        # Bước 1: Token → Embedding
        X = self.embedding(X).permute(1, 0, 2)
        # Shape: (batch_size, num_steps) → (batch_size, num_steps, embed_size)
        #       → permute → (num_steps, batch_size, embed_size)

        # Bước 2: RNN forward
        output, state = self.rnn(X)
        # output: (num_steps, batch_size, num_hiddens)
        # state: (num_layers, batch_size, num_hiddens)

        return state
```

### 3.3 Phân tích từng bước

**Bước 1: Embedding**

```
Input:  X = (batch_size, num_steps) = (256, 50)
         Mỗi phần tử = token index (VD: [2, 45, 78, ...])

Embedding layer:
  W_embed = (vocab_size, embed_size) = (10000, 256)
  Output: (batch_size, num_steps, embed_size) = (256, 50, 256)

Permute:
  Output: (num_steps, batch_size, embed_size) = (50, 256, 256)
  Lý do: RNN trong PyTorch expect (seq_len, batch, feature)
```

**Bước 2: RNN Forward**

```
Input:  (num_steps, batch_size, embed_size) = (50, 256, 256)

RNN:
  rnn = nn.GRU(input_size=256, hidden_size=256, num_layers=2)
  Output shape:  (num_steps, batch_size, hidden_size) = (50, 256, 256)
  State shape:  (num_layers, batch_size, hidden_size) = (2, 256, 256)
```

### 3.4 Shape analysis chi tiết

```
Input batch: 256 câu, mỗi câu 50 tokens

Step 0: X = (256, 50)                               # token indices
Step 1: X_emb = embedding(X) = (256, 50, 256)        # 256-dim embeddings
Step 2: X_emb = permute → (50, 256, 256)           # PyTorch RNN format
Step 3: output, state = rnn(X_emb)
         output = (50, 256, 256)                    # tất cả hidden states
         state = (2, 256, 256)                      # chỉ hidden state cuối
                                                           # num_layers=2

Return state: (num_layers=2, batch_size=256, num_hiddens=256)
  → state[0] = hidden state tầng 1 (tầng dưới cùng)
  → state[1] = hidden state tầng 2 (tầng trên cùng) ← DÙNG LÀM CONTEXT
```

---

## 4. Kết nối Encoder-Decoder — State Transfer

### 4.1 Vấn đề kết nối

Encoder trả về `state` — nhưng Decoder cần state ở format nào?

```
Encoder state: (num_layers, batch_size, num_hiddens)
Decoder state: (num_layers, batch_size, num_hiddens) hoặc (batch_size, num_hiddens)
```

### 4.2 Hai cách khởi tạo Decoder state

**Cách 1: Dùng hidden state cuối cùng trực tiếp**

```python
# Encoder trả về state cuối
encoder_state = encoder(X)  # (num_layers, batch, h)

# Decoder nhận state đó
decoder = RNNDecoder(...)
decoder_output, decoder_state = decoder(Y, encoder_state)
```

**Cách 2: Khởi tạo Decoder state từ context**

```python
# Dùng context vector để khởi tạo
context = encoder_state[-1]  # (batch_size, h) — last layer

decoder_hidden = context  # (batch_size, h)
```

> [!NOTE] Tại sao phải khởi tạo cẩn thận?
> Encoder và Decoder có thể có số tầng khác nhau, hidden size khác nhau. Cần một `Wrapper` để quản lý việc kết nối này một cách sạch sẽ — đó là `EncoderDecoder` base class.

---

# PHẦN III — DECODER: CHI TIẾT IMPLEMENTATION (10.6.2)

---

## 5. Decoder Interface — Giao diện và yêu cầu

### 5.1 Decoder cần làm gì?

Decoder trong Seq2Seq nhận:
1. **Target tokens đã shifted right** — để dự đoán token tiếp theo
2. **Encoder state/context** — thông tin từ câu nguồn

Decoder trả về:
1. **Logits** cho mỗi timestep — để tính cross-entropy loss
2. **Decoder state** — để truyền cho timestep tiếp theo

### 5.2 Base Decoder class

```python
class Decoder(nn.Module):
    """Base class cho Decoder"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        """Convert encoder outputs → decoder initial state"""
        raise NotImplementedError("Subclass phải override init_state()")

    def forward(self, X, state):
        """Forward pass"""
        raise NotImplementedError("Subclass phải override forward()")
```

### 5.3 RNN Decoder — Cài đặt chi tiết

```python
class RNNDecoder(Decoder):
    """RNN Decoder cho Seq2Seq"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens,
                          num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        """Encoder outputs → decoder initial state"""
        # enc_outputs = encoder hidden state (num_layers, batch, h)
        # Decoder state = encoder final hidden state
        return enc_outputs

    def forward(self, X, state):
        """
        Args:
            X: (batch_size, num_steps) — shifted target tokens
            state: decoder hidden state
        Returns:
            output: (batch_size * num_steps, vocab_size)
            state: updated hidden state
        """
        # Bước 1: Embed target tokens
        X = self.embedding(X).permute(1, 0, 2)
        # Shape: (batch_size, num_steps) → (num_steps, batch_size, embed_size)

        # Bước 2: Concatenate với context (repeat state)
        context = state[-1].unsqueeze(0)  # (1, batch_size, num_hiddens)
        context = context.repeat(X.shape[0], 1, 1)  # (num_steps, batch_size, h)

        # X_and_context = (num_steps, batch_size, embed_size + h)
        X_and_context = torch.cat((X, context), dim=-1)

        # Bước 3: RNN forward
        output, state = self.rnn(X_and_context, state)

        # Bước 4: Linear → vocab_size logits
        output = self.dense(output)  # (num_steps, batch_size, vocab_size)

        # Bước 5: Reshape cho loss computation
        output = output.permute(1, 0, 2)  # (batch_size, num_steps, vocab_size)
        output = output.reshape(-1, output.shape[-1])  # (batch_size * num_steps, vocab_size)

        return output, state
```

### 5.4 Phân tích từng bước

**Điểm khác biệt quan trọng với Encoder:**

1. **Context concatenation**: Decoder concat target embeddings với context vector từ Encoder — để Decoder "biết" câu nguồn nói gì
2. **Input size lớn hơn**: `embed_size + num_hiddens` (thay vì `embed_size` như Encoder)
3. **Output là logits**: không phải hidden state, mà là probability distribution over vocabulary

```
Input Decoder:
  Target token: "je" (token index = 5)
  Embedding: (batch_size, embed_size=256)  → (1, 256) sau squeeze
  Context from Encoder: (batch_size, num_hiddens=256) → (1, 256)

  Concat: (1, 512)  → đưa vào RNN

Output Decoder:
  RNN output: (1, 1, 256)  → (batch_size=1, num_steps=1, hidden=256)
  Dense: (1, 1, vocab_size=10000) → logits cho từ tiếp theo
  Reshape: (10000,)  → cross-entropy với label
```

### 5.5 Minh họa context trong Decoder

> [!NOTE] Giải thích đơn giản
> Trong Seq2Seq không attention (chương này), Decoder chỉ nhận **một context vector duy nhất** (hidden state cuối của Encoder). Nhưng ngay cả trong thiết kế này, ta đã concat context vào mỗi bước — đây chính là hạt giống cho cơ chế attention ở chương tiếp theo.

```
Seq2Seq cơ bản (Ch. 10.6):
  Encoder state: (num_layers, batch, h)
  Decoder input timestep t: [embed(y_{t-1}); encoder_state[-1]]
                           ^-target-^   ^-context (cố định)-^

Seq2Seq + Attention (Ch. 10.7):
  Encoder outputs: (num_steps, batch, h) — TẤT CẢ hidden states
  Decoder input timestep t: [embed(y_{t-1}); attention_context_t]
                           ^-target-^   ^-context (thay đổi mỗi step)-^
```

---

# PHẦN IV — SEQ2SEQ MODEL: KẾT HỢP ENCODER-DECODER (10.6.3)

---

## 6. Seq2Seq Model — Tổng hợp toàn bộ

### 6.1 EncoderDecoder base class

```python
class EncoderDecoder(nn.Module):
    """Base class cho toàn bộ Encoder-Decoder model"""
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        """Training forward pass"""
        # Encoder
        enc_state = self.encoder(enc_X)

        # Decoder: khởi tạo state từ encoder outputs
        dec_state = self.decoder.init_state(enc_state, *args)

        # Decoder forward: nhận shifted target
        dec_output, dec_state = self.decoder(dec_X, dec_state)

        return dec_output, dec_state

    def predict(self, prefix, num_steps, vocab, device):
        """Inference: sinh từng token một"""
        self.eval()
        outputs = [vocab[prefix[0]]]  # <bos> token

        # Encode prefix
        X = torch.tensor([[vocab[prefix[0]]]], device=device)
        state = self.encoder(X)

        # Decode từng bước
        for _ in range(num_steps):
            state = self.decoder.init_state(state)
            Y = torch.tensor([[outputs[-1]]], device=device)
            pred, state = self.decoder(Y, state)
            pred = pred.argmax(dim=1).item()
            outputs.append(pred)
            if pred == vocab[EOS_TOKEN]:
                break

        return outputs
```

### 6.2 Seq2Seq Model hoàn chỉnh

```python
class Seq2SeqEncoder(Encoder):
    """Encoder cho Seq2Seq"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, X):
        embeddings = self.embedding(X)  # (batch, steps, embed)
        embeddings = embeddings.permute(1, 0, 2)  # (steps, batch, embed)
        output, state = self.rnn(embeddings)
        # output: (steps, batch, h)
        # state: (layers, batch, h)
        return state  # Chỉ trả về state cuối


class Seq2SeqDecoder(Decoder):
    """Decoder cho Seq2Seq"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens,
                          num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_state, *args):
        # enc_state: (layers, batch, h)
        # Dùng trực tiếp làm decoder state
        return enc_state

    def forward(self, X, state):
        # X: (batch, steps)
        Y = self.embedding(X)  # (batch, steps, embed)
        Y = Y.permute(1, 0, 2)  # (steps, batch, embed)

        # Context: last layer hidden state
        context = state[-1]  # (batch, h)
        context = context.unsqueeze(0).repeat(Y.shape[0], 1, 1)  # (steps, batch, h)

        # Concatenate target embeddings với context
        Y_and_context = torch.cat((Y, context), -1)  # (steps, batch, embed+h)

        # RNN
        output, state = self.rnn(Y_and_context, state)
        # output: (steps, batch, h)
        # state: (layers, batch, h)

        # Linear projection
        output = self.dense(output)  # (steps, batch, vocab)
        output = output.permute(1, 0, 2)  # (batch, steps, vocab)
        output = output.reshape(-1, output.shape[-1])  # (batch*steps, vocab)

        return output, state
```

### 6.3 Seq2Seq Training Loop

```python
def train_seq2seq(data, net, lr, num_epochs, device, tgt_vocab):
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    for epoch in range(num_epochs):
        for batch in data:
            X, X_valid_len, Y, Y_valid_len = batch
            X, Y = X.to(device), Y.to(device)

            # Shift right Y cho Decoder input
            # Decoder input: [BOS, y_1, y_2, ..., y_{T-1}]
            # Labels:       [y_1, y_2, ..., y_T, EOS]
            bos = torch.tensor([tgt_vocab[BOS_TOKEN]] * Y.shape[0],
                              device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], dim=1)  # (batch, steps)

            # Forward
            Y_hat, _ = net(X, dec_input)

            # Loss
            l = loss_fn(Y_hat, Y.reshape(-1))
            l = l.mean()

            optimizer.zero_grad()
            l.backward()
            grad_clip_val = 1
            nn.utils.clip_grad_norm_(net.parameters(), grad_clip_val)
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {l.item():.4f}")


# Khởi tạo
encoder = Seq2SeqEncoder(vocab_size=len(src_vocab),
                          embed_size=256, num_hiddens=256,
                          num_layers=2, dropout=0.1)
decoder = Seq2SeqDecoder(vocab_size=len(tgt_vocab),
                          embed_size=256, num_hiddens=256,
                          num_layers=2, dropout=0.1)
net = EncoderDecoder(encoder, decoder)

train_seq2seq(data, net, lr=0.005, num_epochs=100, device=device,
             tgt_vocab=tgt_vocab)
```

### 6.4 Minh họa Shift-Right cho Decoder

```
Ground truth target:  [je, suis, content, .]         (4 tokens)
Labels (Y):           [je, suis, content, ., <eos>]   (5 tokens)
Decoder input (Y_dec): [<bos>, je, suis, content, .] (5 tokens)

Tại sao phải shift right?
  - Decoder dự đoán y_1 từ <bos>
  - Decoder dự đoán y_2 từ y_1
  - Decoder dự đoán y_3 từ y_1, y_2
  - ...
  → Decoder không được "nhìn thấy" ground truth từ vị trí hiện tại
  → Tránh information leakage
  → Teacher forcing vẫn cho ground truth nhưng ở vị trí TRƯỚC, không phải TẠI vị trí đang dự đoán
```

---

## 7. Evaluation — Đánh giá Seq2Seq

### 7.1 Masked Loss

```python
def masked_loss(Y_hat, Y):
    """Cross-entropy loss nhưng bỏ qua padding tokens"""
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    Y_hat = Y_hat.reshape(-1, Y_hat.shape[-1])
    Y = Y.reshape(-1)
    mask = (Y != PAD_TOKEN).float()  # 1 nơi có token thật, 0 nơi padding
    n_tokens = mask.sum()

    loss = loss_fn(Y_hat, Y)
    masked_loss = (loss * mask).sum() / n_tokens
    return masked_loss
```

### 7.2 BLEU Evaluation

```python
def bleu(pred_seq, label_seq, k):
    """Tính BLEU score cho một cặp dự đoán - nhãn"""
    pred_tokens = pred_seq.split()
    label_tokens = label_seq.split()
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))

    for n in range(1, k + 1):
        num_matches = sum(
            pred_tokens[i:i+n] == label_tokens[i:i+n]
            for i in range(len_pred - n + 1)
        )
        score *= math.pow(num_matches / (len_pred - n + 1 + 1e-10),
                        0.5 ** n)

    return score
```

### 7.3 Inference — Sinh dịch

```python
def translate_seq2seq(net, src_sentence, src_vocab, tgt_vocab, device, num_steps):
    """Dịch một câu bằng trained Seq2Seq model"""
    net.eval()
    tokens = tokenize(src_sentence.lower(), src_vocab)
    src_tokens = [src_vocab[BOS_TOKEN]] + tokens + [src_vocab[EOS_TOKEN]]

    # Encode
    src_indices = torch.tensor(src_tokens, device=device).unsqueeze(0)
    encoder_state = net.encoder(src_indices)

    # Decode
    dec_state = net.decoder.init_state(encoder_state)
    output_tokens = [tgt_vocab[BOS_TOKEN]]
    for _ in range(num_steps):
        Y = torch.tensor([[output_tokens[-1]]], device=device)
        pred, dec_state = net.decoder(Y, dec_state)
        pred_token = pred.argmax(dim=1).item()

        if pred_token == tgt_vocab[EOS_TOKEN]:
            break

        output_tokens.append(pred_token)

    return ' '.join(tgt_vocab.to_tokens(output_tokens[1:]))


# Ví dụ
src_sentence = "i love you"
translation = translate_seq2seq(net, src_sentence, src_vocab, tgt_vocab, device, num_steps=50)
print(f"Input:  {src_sentence}")
print(f"Output: {translation}")
# Input:  i love you
# Output: je t'aime .
```

---

# PHẦN V — KIẾN TRÚC TRONG CÁC ỨNG DỤNG THỰC TẾ

---

## 8. Encoder-Decoder ngoài MT

### 8.1 Image Captioning

```
Input image:  (3, 224, 224)  — RGB image
      ↓ CNN Encoder (pretrained ResNet-50)
Feature map: (2048, 7, 7)    — spatial features
      ↓ Flatten hoặc attention
Context:     (49, 2048)       — 49 spatial locations
      ↓ RNN/Transformer Decoder
Output:      "a cat sitting on a wooden table"
```

### 8.2 Speech Recognition

```
Input audio:  (time_steps, 80)  — mel spectrogram
      ↓ RNN/Transformer Encoder
Hidden states: (time_steps, h)
      ↓ CTC/Attention Decoder
Output:        "hello world"
```

### 8.3 Video Summarization

```
Input video:  (T, C, H, W)  — T frames
      ↓ 3D CNN Encoder
Features:     (T, 2048)
      ↓ LSTM Decoder
Output:        [0, 1, 1, 0, 0, 1, ...]  — keyframe selection
```

---

## 9. So sánh các loại Decoder

### 9.1 Bảng so sánh

| Loại Decoder | Mô tả | Ứng dụng | Ưu điểm | Hạn chế |
|---|---|---|---|---|
| **Autoregressive RNN** | Sinh token-by-token | MT, Captioning | Đơn giản | Chậm, không parallel |
| **Attention-based** | Dùng attention để chọn source | MT (sau 2015) | Chất lượng cao | Cần more compute |
| **Transformer Decoder** | Self-attention + cross-attention | GPT, T5 | Parallel, long-range | Cần nhiều dữ liệu |
| **Non-autoregressive** | Sinh tất cả tokens cùng lúc | Fast MT | Rất nhanh | Chất lượng thấp hơn |
| **CTC Decoder** | Connectionist Temporal Classification | Speech Recognition | Xử lý input dài không aligned | Chỉ cho sequence recognition |

### 9.2 Autoregressive vs Non-autoregressive

```
Autoregressive (slow but accurate):
  y_1 = Decoder(x)
  y_2 = Decoder(x, y_1)
  y_3 = Decoder(x, y_1, y_2)
  y_4 = Decoder(x, y_1, y_2, y_3)
  → 4 sequential steps

Non-autoregressive (fast, parallel):
  y_1, y_2, y_3, y_4 = Decoder(x)
  → 1 parallel step
  → Chất lượng thấp hơn vì không có dependency modeling
```

---

# PHẦN VI — BÀI TẬP (10.6.4)

---

## Bài 1: Encoder-Decoder với vocabulary khác nhau

> _"Encoder có vocab EN=10000, Decoder có vocab FR=12000. Điều gì xảy ra khi Encoder và Decoder có vocab size khác nhau?"_

**Phân tích:**
- Encoder vocab độc lập với Decoder vocab — hoàn toàn OK
- Encoder: word-index → embedding EN → hidden state
- Decoder: hidden state → logits over FR vocabulary → FR token
- Translation là nhiệm vụ của hidden representation (ngôn ngữ-independent)

## Bài 2: Tính số tham số

> _"Tính tổng số tham số của Seq2Seq model với vocab_size_en=10000, vocab_size_fr=12000, embed_size=256, num_hiddens=256, num_layers=2."_

**Phân tích:**

```
Encoder:
  Embedding:     vocab_size_en × embed_size = 10000 × 256 = 2,560,000
  RNN (GRU):     3 × [(embed+h) × h + h] × num_layers
                = 3 × [(256+256) × 256 + 256] × 2
                = 3 × [131,584 + 256] × 2
                = 3 × 131,840 × 2 = 791,040

Decoder:
  Embedding:     vocab_size_fr × embed_size = 12000 × 256 = 3,072,000
  RNN (GRU):     791,040
  Dense:         h × vocab_size_fr = 256 × 12000 = 3,072,000

Tổng: 2,560,000 + 791,040 + 3,072,000 + 791,040 + 3,072,000 = 10,286,080
≈ 10.3 triệu parameters
```

## Bài 3: Teacher forcing vs. Autoregressive inference

> _"Tại sao training dùng teacher forcing nhưng inference dùng autoregressive? Hai chiến lược này khác nhau như thế nào?"_

**Training (teacher forcing):**
```
Mỗi step t của Decoder:
  - Nhận ground truth token y_{t-1} (không phải predicted token)
  - Tính loss với ground truth token y_t
  → Gradient ổn định, hội tụ nhanh
  → Nhưng: có "exposure bias" — model chưa bao giờ thấy output của chính nó
```

**Inference (autoregressive):**
```
Mỗi step t của Decoder:
  - Nhận predicted token y_{t-1} từ step trước
  - Không có ground truth để so sánh
  → Lỗi nhỏ có thể lan truyền (error propagation)
```

## Bài 4: Thiết kế Decoder cho non-autoregressive translation

> _"Thiết kế một Decoder đơn giản cho non-autoregressive MT — sinh tất cả tokens cùng lúc."_

```python
class NATDecoder(Decoder):
    """Non-Autoregressive Decoder"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, num_steps):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers)
        self.dense = nn.Linear(num_hiddens, vocab_size)
        self.num_steps = num_steps

    def forward(self, X, state):
        # X: (batch_size, num_steps) — dummy hoặc all <bos>
        # Tất cả tokens được sinh cùng lúc, không có dependency
        Y = self.embedding(X)  # (batch, steps, embed)
        Y = Y.permute(1, 0, 2)  # (steps, batch, embed)

        output, state = self.rnn(Y, state)
        output = self.dense(output)  # (steps, batch, vocab)

        return output.permute(1, 0, 2).reshape(-1, output.shape[-1]), state
```

---

## Tổng kết

| Khía cạnh | Nội dung |
|---|---|
| **Encoder-Decoder là gì** | Khung kiến trúc: Encoder nén input, Decoder sinh output |
| **Encoder** | RNN đọc input, trả về hidden state cuối (context) |
| **Decoder** | RNN nhận target shifted-right + context, sinh logits |
| **Shift right** | Decoder input = [BOS, y_1, y_2, ...], Labels = [y_1, y_2, ..., EOS] |
| **Kết nối** | `decoder.init_state(encoder_outputs)` — chuyển đổi state |
| **Masked loss** | Bỏ qua PAD tokens khi tính cross-entropy |
| **Training** | Teacher forcing — cho Decoder thấy ground truth |
| **Inference** | Autoregressive — dùng predicted token làm input step tiếp theo |
| **BLEU** | Đánh giá chất lượng dịch |
| **Bottleneck** | Context vector cố định → cần Attention (chương tiếp) |

---

> **Buổi trước:** [[Buổi 46 - Tuần 13]] — 10.5 Machine Translation and the Dataset
> **Buổi sau:** [[Buổi 48 - Tuần 13]] — 10.7 Sequence to Sequence Learning (with Attention)

---

## Thuật ngữ

| Thuật ngữ | Tiếng Anh | Ghi chú |
|---|---|---|
| Mã hóa | Encoder | Biến input → representation |
| Giải mã | Decoder | Biến representation → output |
| Representation | Representation | Biểu diễn trung gian |
| Shift right | Shift right | Dịch target sang phải 1 bước |
| Teacher forcing | Teacher Forcing | Cho Decoder thấy ground truth |
| Autoregressive | Autoregressive | Sinh dựa trên output trước |
| Masked loss | Masked Loss | Cross-entropy không tính PAD |
| Exposure bias | Exposure Bias | Training ≠ Inference |
| BLEU | Bilingual Evaluation Understudy | Metric MT |
| Non-autoregressive | Non-autoregressive | Sinh parallel, không có dependency |
