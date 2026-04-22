---
session: "D2L Tuần 13, Buổi 46 — 10.5 Machine Translation and the Dataset"
d2l_chapter: "10.5"
tags:
  - d2l
  - deep-learning
  - rnn
  - machine-translation
  - encoder-decoder
  - seq2seq
  - nlp
  - nmt
aliases:
  - Machine Translation
  - MT
  - Encoder-Decoder
  - Seq2Seq
  - Sequence to Sequence
date: 2026-04-21
status: complete
---

# Buổi 46 — 10.5 Machine Translation and the Dataset

> **Nguồn:** [d2l.ai — 10.5](https://d2l.ai/chapter_recurrent-modern/seq2seq.html)
> **Buổi trước:** [[Buổi 45 - Tuần 12]] — 10.3 & 10.4 Deep RNN + Bidirectional RNN
> **Buổi sau:** [[Buổi 47 - Tuần 13]] — 10.6 Encoder-Decoder Architecture

---

## Active Recall — Ôn lại Buổi 45 (Deep RNN & BiRNN)

### Câu hỏi (không nhìn tài liệu)

1. Deep RNN có bao nhiêu nguồn thông tin cho mỗi hidden state? Kể tên.
2. Viết công thức hidden state tại tầng $l$ của Deep RNN. Giải thích từng thành phần.
3. Tại sao Deep RNN cần learning rate thấp hơn và gradient clipping bắt buộc?
4. Bidirectional RNN gồm mấy unidirectional RNN? Chúng chạy theo hướng nào?
5. Output tại mỗi bước $t$ của BiRNN là gì? Shape của nó là bao nhiêu?
6. Tại sao Bidirectional RNN **không thể** xử lý real-time?
7. Khi nào nên dùng Deep RNN? Khi nào nên dùng Bidirectional RNN?
8. Deep Bidirectional RNN với $L$ tầng, mỗi tầng bidirectional có shape output là bao nhiêu?
9. Số tham số của Deep RNN ($L$ tầng) gấp bao nhiêu lần RNN 1 tầng?
10. Hai ưu điểm chính của việc tăng chiều rộng ($h$) so với tăng chiều sâu ($L$)?

### Tự trả lời

1. **2 nguồn**: từ chiều **không gian/tầng** ($H_t^{(l-1)}$, từ tầng dưới) và từ chiều **thời gian** ($H_{t-1}^{(l)}$, từ bước trước trong cùng tầng).
2. $H_t^{(l)} = \phi(H_t^{(l-1)} W_{xh}^{(l)} + H_{t-1}^{(l)} W_{hh}^{(l)} + b_h^{(l)})$. Quy ước: $H_t^{(0)} = X_t$.
3. Gradient trong Deep RNN phải truyền qua $T \times L$ bước — nhiều hơn RNN 1 tầng ($T$ bước). Gradient clipping cắt ngắn gradient khi nó quá lớn; learning rate thấp hơn giảm kích thước bước cập nhật.
4. **2 unidirectional**: Forward RNN (đọc $x_1 \to x_T$, trái → phải) và Backward RNN (đọc $x_T \to x_1$, phải → trái).
5. Concatenation $[\overrightarrow{H}_t; \overleftarrow{H}_t]$. Shape: $(n, 2h)$ với $n$ là batch size.
6. Backward RNN cần $\overleftarrow{H}_{t+1}$ — hidden state từ **bước tiếp theo** — nên phải biết toàn bộ chuỗi trước khi xử lý bất kỳ bước nào.
7. Deep RNN: cần biểu diễn phân cấp (từ → cụm từ → câu → đoạn). BiRNN: cần ngữ cảnh hai chiều (POS tagging, NER, masked LM).
8. Tầng 1: input $d$ → output $2h$. Tầng 2: input $2h$ → output $4h$. Tầng $L$: output $2^L \times h$.
9. $L$ lần. RNN 1 tầng: $dh + h^2 + h$. Deep RNN: $L \times (dh + h^2 + h)$.
10. (a) Hiệu quả hơn cho sequence thông thường; (b) ít vấn đề vanishing/exploding gradient hơn; (c) mỗi tầng bidirectional gấp đôi chiều → chi phí tăng nhanh.

### Ghi chú khái niệm cần ôn lại

- [[Gated Recurrent Unit]]
- [[Long Short-Term Memory]]
- [[Recurrent Neural Network]]
- [[Backpropagation Through Time]]
- [[Bidirectional RNN]]

---

# PHẦN I — MACHINE TRANSLATION: BỐI CẢNH VÀ ĐỘNG LỰC

---

## 1. Bài toán Dịch máy là gì?

> [!NOTE] Giải thích đơn giản
> Hãy tưởng tượng bạn có một thư ký rất giỏi ngoại ngữ. Khi bạn nói tiếng Việt, anh ấy nghe, **hiểu ý** rồi nói lại bằng tiếng Anh. Anh ấy không dịch từng từ một cách máy móc — mà hiểu ý nghĩa rồi diễn đạt lại. Dịch máy (Machine Translation) là việc dạy máy tính làm điều tương tự: nghe câu tiếng này, hiểu ý, rồi nói ra câu tiếng kia.

**Machine Translation (MT)** là bài toán **tự động dịch** văn bản hoặc lời nói từ ngôn ngữ này sang ngôn ngữ khác. Đây là một trong những bài toán **kinh điển nhất** của NLP, có lịch sử hơn 70 năm.

**MT là gì?** MT là quá trình tự động chuyển đổi chuỗi tokens từ **ngôn ngữ nguồn** (source language, ví dụ: tiếng Việt) thành chuỗi tokens tương đương về ý nghĩa trong **ngôn ngữ đích** (target language, ví dụ: tiếng Anh).

**Nó giải quyết vấn đề gì?** Cho phép con người giao tiếp xuyên ngôn ngữ một cách tự động — từ dịch tài liệu, website, tin nhắn, đến hội nghị thông dịch.

**Tại sao nó khó?** Ngôn ngữ có tính **đa nghĩa** (polysemy), **ngữ cảnh phụ thuộc** (context-dependent), **cấu trúc khác nhau** (word order), và **tục ngữ, thành ngữ** không thể dịch word-by-word.

### 1.1 Ba paradigm dịch máy

| Paradigm | Mô tả | Ví dụ | Hạn chế |
|---|---|---|---|
| **Rule-based MT** | Dùng từ điển + ngữ pháp viết tay | Babylon, Systran | Cần chuyên gia, không mở rộng được |
| **Statistical MT** | Dùng xác suất thống kê trên dữ liệu song ngữ | Moses, Google Translate (2006-2016) | Cần dữ liệu lớn, khó xử lý từ hiếm |
| **Neural MT (NMT)** | Dùng neural network (RNN → Transformer) | Google Neural MT (2016+), DeepL | Cần GPU, dữ liệu lớn, nhưng chất lượng vượt trội |

> [!IMPORTANT] Neural MT — Cuộc cách mạng 2016
> Năm 2016, Google thay thế Statistical MT bằng Neural MT dựa trên **sequence-to-sequence (seq2seq)** với attention, và chất lượng dịch tăng vượt bậc. Từ đó, hầu hết các hệ thống dịch máy hiện đại đều dựa trên neural networks.

### 1.2 Minh họa bài toán

```
Nguồn (Tiếng Việt):  "Hôm nay tôi đi học"
Đích (Tiếng Anh):     "Today I go to school"
```

**Điểm quan trọng:** Đây là bài toán **sequence-to-sequence** — cả input và output đều là chuỗi có độ dài thay đổi, và không nhất thiết bằng nhau. "Hôm nay" (2 tokens) dịch thành "Today" (1 token) — chứng tỏ dịch không đơn giản là word-by-word.

---

## 2. Sequence-to-Sequence (Seq2Seq) — Khung kiến trúc tổng quát

> [!NOTE] Giải thích đơn giản
> Seq2seq giống như một cặp thư ký: người đầu tiên (Encoder) nghe toàn bộ câu nguồn vào đầu rồi thì thầm một câu tóm tắt vào tai người thứ hai; người thứ hai (Decoder) nhận được câu tóm tắt đó rồi lần lượt nói ra từng từ của câu đích. Encoder không biết Decoder sẽ nói gì, Decoder chỉ biết câu tóm tắt — chúng giao tiếp qua đúng **một câu tóm tắt duy nhất**. Đây chính là điểm yếu cần cải thiện.

**Sequence-to-Sequence (Seq2Seq)** là khung kiến trúc gồm hai thành phần chính:

1. **Encoder**: đọc chuỗi đầu vào và nén thông tin thành một **context vector** (hoặc tập hợp state vectors)
2. **Decoder**: đọc context vector và lần lượt sinh ra chuỗi đầu ra token-by-token

**Đặc điểm quan trọng:**
- Độ dài input $\neq$ độ dài output
- Cả encoder và decoder đều là các mô hình sequence (thường là RNN/LSTM/GRU)
- Decoder sinh token **từ trái sang phải**, mỗi token phụ thuộc vào các token trước đó

### 2.1 Minh họa kiến trúc Seq2Seq

![[assets/attachments/d2l-buoi-46/seq2seq.svg]]
_Fig. 10.5.1 (D2L): Kiến trúc Sequence-to-Sequence. Encoder (phía trên) đọc câu nguồn từ trái sang phải, nén thông tin vào hidden state cuối cùng $H_T$. Decoder (phía dưới) nhận $H_T$ làm initial state, rồi sinh từng token đích autoregressively._

**Đọc sơ đồ — từng bước:**

1. **Encoder** (phía trên):
   - Đọc $x_1, x_2, \ldots, x_T$ (câu nguồn)
   - Hidden state: $H_1 \to H_2 \to \ldots \to H_T$
   - Hidden state cuối cùng $H_T$ đóng vai trò **context vector** $C = H_T$

2. **Context vector** $C$:
   - Vector duy nhất chứa **toàn bộ thông tin** của câu nguồn
   - Đây là "nút thắt cổ chai" (bottleneck) — thông tin phải được nén vào một vector duy nhất

3. **Decoder** (phía dưới):
   - Nhận $C$ (và token `<bos>` — begin of sequence) làm initial state
   - Sinh $y_1$ → dùng $y_1$ sinh $y_2$ → ... → sinh $y_{T'}$ → token `<eos>` (end of sequence)

### 2.2 Quy trình sinh autoregressive

```
C = Encoder(x_1, ..., x_T)

y_1 = Decoder(<bos>, C)           # Lần lượt sinh
y_2 = Decoder(y_1, C)
y_3 = Decoder(y_1, y_2, C)
...
y_T' = Decoder(y_1, ..., y_{T'-1}, C)
output = [y_1, y_2, ..., y_T']
```

**Đặc điểm:**
- Mỗi token được sinh dựa trên **tất cả** token đã sinh trước đó (autoregressive)
- Tại test time, ta không biết độ dài output trước → dùng `<eos>` để báo hiệu kết thúc
- Training: dùng **teacher forcing** — cho Decoder thấy ground truth token trước đó

### 2.3 Sơ đồ chi tiết Encoder và Decoder

![[assets/attachments/d2l-buoi-46/encoder.svg]]
_Fig. (D2L): Chi tiết Encoder trong Seq2Seq. Mỗi timestep đọc một token nguồn $x_t$, cập nhật hidden state $h_t$. Cuối cùng, hidden state tại bước cuối $h_T$ được truyền cho Decoder._

![[assets/attachments/d2l-buoi-46/decoder.svg]]
_Fig. (D2L): Chi tiết Decoder trong Seq2Seq. Tại mỗi bước, nhận token đích trước đó $y_{t-1}$ (hoặc `<bos>`), cập nhật hidden state $s_t$, rồi dùng $s_t$ để dự đoán token tiếp theo $\hat{y}_t$. State cuối của Encoder được dùng làm initial state cho Decoder._

---

## 3. Vấn đề "Bottleneck" — Tại sao Seq2Seq cần Attention?

### 3.1 Context Vector là "nút thắt cổ chai"

> [!NOTE] Giải thích đơn giản
> Nén toàn bộ câu dài vào một vector duy nhất giống như cố nhét cả một cuốn tiểu thuyết vào một câu tweet. Rất nhiều thông tin bị mất!

Trong kiến trúc Seq2Seq cơ bản (không có attention), Encoder nén **toàn bộ** thông tin câu nguồn vào **một context vector duy nhất** $C = H_T$. Điều này tạo ra hai vấn đề nghiêm trọng:

**Vấn đề 1 — Information Overload:**
- Câu nguồn dài → rất nhiều thông tin phải nén vào $C$
- $C$ có kích thước cố định ($h$) → không thể chứa đủ thông tin
- Kết quả: **performance giảm mạnh** khi câu nguồn dài hơn ~15-20 tokens

**Vấn đề 2 — Gradient Bottleneck:**
- Toàn bộ thông tin phải đi qua $C$ khi Decoder sinh mỗi token
- Decoder không thể "nhìn lại" câu nguồn một cách chọn lọc
- Kết quả: khó học các phụ thuộc **dài** và **chọn lọc**

### 3.2 Attention — Giải pháp cho Bottleneck

**Attention** (sẽ học chi tiết ở Chương 11) là cơ chế cho phép Decoder **"nhìn lại"** toàn bộ câu nguồn **tại mỗi bước sinh**, và tự động **weighting** những phần quan trọng nhất.

Thay vì một context vector duy nhất:

| Phiên bản | Context vector |
|---|---|
| **Không có attention** | $C = H_T$ (1 vector cố định) |
| **Có attention** | $C_t = \sum_{i} \alpha_{t,i} H_i$ (weighted sum, khác nhau mỗi bước) |

Mỗi bước sinh của Decoder có một context vector riêng, được tính dựa trên:
- Hidden state hiện tại của Decoder ($s_{t-1}$)
- Toàn bộ hidden states của Encoder ($H_1, H_2, \ldots, H_T$)

### 3.3 Minh họa Encoder-Decoder với Attention

![[assets/attachments/d2l-buoi-46/mt-seq2seq.svg]]
_Fig. (D2L): Encoder-Decoder với Attention. Tại mỗi bước, Decoder không chỉ dùng một context vector duy nhất mà tính weighted sum của tất cả encoder hidden states — nhìn lại câu nguồn một cách chọn lọc._

### 3.4 So sánh Seq2Seq cơ bản và với Attention

| Khía cạnh | Seq2Seq cơ bản | Seq2Seq + Attention |
|---|---|---|
| Context vector | 1 vector duy nhất $C = H_T$ | $T$ vectors khác nhau $C_t = \sum \alpha_{t,i} H_i$ |
| Decoder access | Chỉ biết $C$ | "Nhìn lại" toàn bộ source |
| Câu dài | Performance giảm nhanh | Ổn định |
| Tham số | Ít hơn | Thêm attention weights |
| Bước sinh 1 của Decoder | $s_1 = f(s_0, C, y_0)$ | $s_1 = f(s_0, C_1, y_0)$ với $C_1$ riêng |
| Năm phát minh | 2014 (Sutskever et al.) | 2015 (Bahdanau et al.) |

---

# PHẦN II — BỘ DỮ LIỆU MACHINE TRANSLATION (10.5.1)

---

## 4. Tại sao cần Dataset đặc biệt cho MT?

### 4.1 Đặc điểm của dữ liệu MT

> [!NOTE] Giải thích đơn giản
> Để dạy máy dịch, ta cần **cặp câu** — cùng một ý được nói bằng hai ngôn ngữ. Ví dụ: "Hôm nay trời đẹp" ↔ "The weather is nice today". Càng nhiều cặp, máy học càng tốt. Nhưng chất lượng cặp câu rất quan trọng — câu dịch sai sẽ làm máy học sai, như dạy học sinh từ sách giáo khoa có lỗi.

Dữ liệu MT khác với các bài toán NLP khác ở chỗ:

1. **Yêu cầu cặp song ngữ** (parallel/bilingual corpus) — không phải text đơn ngữ
2. **Alignment cấp câu** (sentence alignment) — cần biết câu nào trong ngôn ngữ A tương ứng với câu nào trong ngôn ngữ B
3. **Alignment cấp từ** (word alignment) — cần biết từ nào ở nguồn tương ứng với từ nào ở đích (cho attention visualization)
4. **Ngôn ngữ có nhiều biến thể** — tiếng Anh Anh vs tiếng Anh Mỹ

### 4.2 Các bộ dữ liệu phổ biến

| Dataset | Ngôn ngữ | Kích thước | Đặc điểm |
|---|---|---|---|
| **WMT Dataset** | Đa ngôn ngữ | Rất lớn (triệu cặp câu) | Standard cho research, annual competitions |
| **Europarl** | EN↔DE, EN↔FR | ~2M câu | European Parliament proceedings |
| **NIST MT Dataset** | ZH↔EN, AR↔EN | ~4M câu | Used by US government |
| **TED Talks** | 100+ ngôn ngữ | ~3K-200K/cặp | Spoken, diverse topics |
| **Multi30k** | EN↔DE, EN↔FR | 30K cặp câu | Images + captions, nhỏ, cho quick experiments |

> [!NOTE] D2L dùng dataset nào?
> D2L sử dụng bộ dữ liệu **English-French** từ tập dữ liệu WMT nhỏ gọn (~40K cặp câu), gọi là **FR-EN**. Đủ nhỏ để huấn luyện trên CPU/GPU cá nhân trong thời gian hợp lý, nhưng đủ lớn để minh họa các kỹ thuật quan trọng.

### 4.3 Dataset structure trong D2L

```python
from d2l import torch as d2l

# Tải dataset English-French
# Nguồn: WMT English-French translation dataset (version of 2014)
# Format: danh sách các cặp (sentence_en, sentence_fr)
d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip', '946ef45a8f1d1e06dff0c8115c8e4a9b5047b838')
```

**Dataset gồm:**
- ~40,000 cặp câu song ngữ Anh-Pháp
- Mỗi cặp: `(english_sentence, french_sentence)`
- Kích thước nhỏ → phù hợp cho thực hành

---

## 5. Tiền xử lý dữ liệu MT — Chi tiết từng bước

### 5.1 Các bước tiền xử lý

> [!NOTE] Giải thích đơn giản
> Dữ liệu thô (văn bản thường) phải được chuẩn bị kỹ trước khi đưa vào mô hình. Quá trình này giống như chuẩn bị nguyên liệu trước khi nấu ăn: rửa rau, cắt thịt, ướp gia vị — mỗi bước đều quan trọng. Nếu rau bẩn thì món ăn sẽ dở, nếu text không chuẩn hóa thì model sẽ học sai.

**Bước 1: Chuẩn hóa văn bản (Text Normalization)**

```python
def preprocess_raw(text):
    # Thay thế non-breaking space bằng space
    text = text.replace('\u202f', ' ').replace('\xa0', ' ')

    # Lowercase (hoặc không, tùy ngôn ngữ)
    text = text.lower()

    # Thêm space giữa word và punctuation
    out = []
    for i, char in enumerate(text):
        if i > 0 and re.match(r'[!,?\.']+', char):
            out.append(' ' + char)
        else:
            out.append(char)
    text = ''.join(out)

    return text
```

**Bước 2: Tokenization**

```python
# Word-level tokenization
tokens = text.split(' ')

# Hoặc character-level
tokens = list(text)
```

**Bước 3: Xây dựng Vocabulary**

```python
class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []

        counter = count_corpus(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)

        # Index cho unknown token
        self.unk_token = UNK = '<unk>'
        self.pad_token = PAD = '<pad>'
        self.bos_token = BOS = '<bos>'
        self.eos_token = EOS = '<eos>'

        # Xây dựng index → token
        idx_to_token = [unk, pad, bos, eos] + reserved_tokens
        for token, freq in self.token_freqs:
            if freq < min_freq and token not in reserved_tokens:
                continue
            idx_to_token.append(token)

        self.idx_to_token = idx_to_token
        self.token_to_idx = {token: idx for idx, token in
                             enumerate(self.token_to_token)}
```

**Bước 4: Padding và Truncation**

```python
def pad(lines, pad_val):
    max_len = max(len(line) for line in lines)
    result = []
    for line in lines:
        if len(line) < max_len:
            line = line + [pad_val] * (max_len - len(line))
        result.append(line)
    return result

def truncate_pad(line, num_steps, padding_token):
    """Cắt câu dài hoặc pad câu ngắn"""
    if len(line) > num_steps:
        return line[:num_steps]
    else:
        return line + [padding_token] * (num_steps - len(line))
```

### 5.2 Special Tokens trong MT

| Token | Ý nghĩa | Ví dụ |
|---|---|---|
| `<unk>` | Unknown — từ không có trong vocabulary | Từ hiếm gặp |
| `<pad>` | Padding — đệm cho batch có độ dài đồng nhất | Câu ngắn |
| `<bos>` | Begin of Sequence — bắt đầu câu | Decoder input đầu tiên |
| `<eos>` | End of Sequence — kết thúc câu | Báo hiệu Decoder dừng |

### 5.3 Minh họa dòng dữ liệu

```
Raw text (English):   "i'm sorry !"
                         ↓ preprocess
Normalized:          "i ' m sorry !"
                         ↓ tokenize
Tokens:              ['i', "'", 'm', 'sorry', '!']
                         ↓ vocab lookup
Token indices:       [2, 30, 4, 105, 7]
                         ↓ pad (max_len=10)
Final:               [2, 30, 4, 105, 7, 1, 1, 1, 1, 1]
                                                    ↑ PAD
```

---

## 6. Dataset Iterator — Tải và sử dụng (10.5.2)

### 6.1 Tải dataset

```python
class MTFraEng(d2l.DataModule):
    def _download(self):
        d2l.extract(d2l.download(d2l.DATA_HUB['fra-eng'], self.root,
                                '946ef45a8f1d1e06dff0c8115c8e4a9b5047b838'),
                    self.root + '/fra-eng/')
        self.raw_train_df = pd.read_csv(self.root + '/fra-eng/fra.txt',
                                         sep='\t', header=None,
                                         names=['en', 'fr', 'cc'])
        self.raw_val_df = self.raw_train_df.sample(frac=0.1, random_state=42)
        self.raw_train_df = self.raw_train_df.drop(self.raw_val_df.index)
```

### 6.2 Tokenization và Vocabulary riêng

```python
def build_array(self, lines, vocab, num_steps, is_source=True):
    """Chuyển lines → token indices → tensor"""
    lines = [vocab[l] for l in lines]  # token → index
    if not is_source:
        lines = [[vocab.bos_token] + l + [vocab.eos_token]
                 for l in lines]  # thêm BOS/EOS cho target

    # Truncate hoặc pad
    array = torch.tensor([truncate_pad(l, num_steps, vocab.pad_token)
                          for l in lines])

    # Tính valid lengths (không tính PAD)
    valid_len = (array != vocab.pad_token).sum(dim=1)

    return array, valid_len
```

### 6.3 Shape của dữ liệu

```
Source (English):     (batch_size, num_steps)    → Encoder input
Target (French):     (batch_size, num_steps)    → Decoder input (với BOS prefix)
Labels (French):      (batch_size, num_steps)    → Decoder output (với EOS suffix)
Source valid_len:    (batch_size,)              → Độ dài thực (không tính PAD)
Target valid_len:    (batch_size,)              → Độ dài thực (không tính PAD)
```

**Ví dụ cụ thể:**

```
Batch size = 32, num_steps = 50
Source shape:     torch.Size([32, 50])    → 32 câu Anh, mỗi câu 50 tokens
Target shape:    torch.Size([32, 50])    → 32 câu Pháp tương ứng
Labels shape:    torch.Size([32, 50])    → Target shifted right
Valid lens:      torch.Size([32])        → [45, 38, 42, ...]
```

---

## 7. Tầm quan trọng của Dataset trong MT

### 7.1 Dữ liệu quyết định chất lượng

| Yếu tố | Ảnh hưởng | Ví dụ |
|---|---|---|
| **Kích thước** | Nhiều hơn → tốt hơn | Google dịch dùng hàng tỷ cặp câu |
| **Chất lượng** | Nhiễu → học sai | Câu dịch không chính xác |
| **Đa dạng** | Phủ nhiều domain → tốt hơn | Y tế ≠ Kỹ thuật |
| **Độ dài** | Câu quá dài → khó dịch | > 100 tokens: performance giảm |
| **Alignment** | Chính xác → học tốt | Word alignment cho attention |

### 7.2 Các vấn đề thực tế

**1. Out-of-Vocabulary (OOV):**
- Từ hiếm gặp không có trong vocabulary → `<unk>`
- Giải pháp: Subword tokenization (BPE/WordPiece)

**2. Tokenization cho đa ngôn ngữ:**
- Tiếng Anh: dễ tokenize (spaces)
- Tiếng Trung: không có spaces → cần word segmentation
- Tiếng Việt: spaces nhưng cần word segmentation để tốt hơn

**3. Data augmentation:**
- Back-translation: dịch từ đích → nguồn để tăng dữ liệu
- Noise injection: thêm noise để tăng robustness

---

# PHẦN III — DATALOADER VÀ COLLATE FUNCTION (10.5.2)

---

## 8. Collate Function — Xử lý Batch không đồng nhất

### 8.1 Vấn đề

Trong MT, mỗi câu có độ dài khác nhau:

```
Câu 1: "i love you"               → 3 tokens
Câu 2: "the weather is nice today" → 5 tokens
Câu 3: "once upon a time in a far away kingdom" → 9 tokens
```

Khi đưa vào batch, ta cần tất cả có cùng độ dài → cần **padding**.

### 8.2 Batch collation cho MT

```python
def collate_fn_mt(batch):
    """Collate function cho MT batches"""
    # batch: list of (src_line, tgt_line) tuples
    src_lines, tgt_lines = zip(*batch)

    # Tokenize
    src_tokens = [tokenize(src, vocab_src) for src in src_lines]
    tgt_tokens = [tokenize(tgt, vocab_tgt) for tgt in tgt_lines]

    # Thêm BOS cho target, BOS + EOS cho labels
    tgt_inputs = [[BOS] + t for t in tgt_tokens]
    tgt_labels = [t + [EOS] for t in tgt_tokens]

    # Pad
    src_padded = pad(src_tokens, PAD)
    tgt_inputs_padded = pad(tgt_inputs, PAD)
    tgt_labels_padded = pad(tgt_labels, PAD)

    # Valid lengths
    src_valid_len = [len(s) for s in src_tokens]
    tgt_valid_len = [len(t) for t in tgt_tokens]

    return (torch.tensor(src_padded),
            torch.tensor(tgt_inputs_padded),
            torch.tensor(tgt_labels_padded),
            torch.tensor(src_valid_len),
            torch.tensor(tgt_valid_len))
```

### 8.3 Minibatch example

```python
dataloader = DataLoader(dataset, batch_size=256, collate_fn=collate_fn_mt)

for src, tgt_in, tgt_out, src_len, tgt_len in dataloader:
    print(f"Source batch:      {src.shape}")       # (256, max_src_len)
    print(f"Target input:     {tgt_in.shape}")    # (256, max_tgt_len)
    print(f"Target labels:    {tgt_out.shape}")   # (256, max_tgt_len)
    print(f"Source valid lens: {src_len.shape}")   # (256,)
    print(f"Target valid lens: {tgt_len.shape}")  # (256,)
    break
```

---

# PHẦN IV — EVALUATION METRICS CHO MT

---

## 9. BLEU Score — Đo chất lượng dịch

### 9.1 Giới thiệu

> [!NOTE] Giải thích đơn giản
> Khi máy dịch một câu, làm sao biết dịch đúng hay sai? Nếu máy dịch "con mèo đen" thành "the black cat" — có bao nhiêu phần đúng? BLEU đếm xem có bao nhiêu n-grams trong dịch của máy xuất hiện trong dịch chuẩn (reference). Ví dụ: "the black cat" có cả "the", "black", "cat" đúng → precision cao.

**BLEU (Bilingual Evaluation Understudy)** là metric phổ biến nhất để đánh giá chất lượng MT, được Papineni et al. (2002) đề xuất.

### 9.2 Công thức chi tiết

**Precision $p_n$** — tỉ lệ n-grams trong output có trong reference:

$$p_n = \frac{\sum_{\text{n-gram} \in \hat{y}} \min(\text{count}_{\text{n-gram}}, \max_{\text{ref}} \text{count}_{\text{n-gram}})}{\sum_{\text{n-gram} \in \hat{y}} \text{count}_{\text{n-gram}}}$$

**Brevity Penalty $BP$** — phạt nếu output ngắn hơn reference:

$$BP = \begin{cases} 1 & \text{nếu } c > r \\ e^{(1 - r/c)} & \text{nếu } c \leq r \end{cases}$$

Trong đó $c$ = độ dài output, $r$ = độ dài reference.

**BLEU cuối cùng:**

$$\text{BLEU} = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

Thường dùng $N=4$ và $w_n = 1/N$.

### 9.3 Ví dụ tính BLEU

```
Reference:   "the black cat is on the mat"
Hypothesis: "the cat is on the mat"

1-gram precision: 4/5 = 0.80  ("the", "cat", "on", "the", "mat")
2-gram precision: 3/4 = 0.75  ("the cat", "cat is", "is on", "on the")
3-gram precision: 2/3 = 0.67  ("the cat is", "cat is on", "is on the")
4-gram precision: 1/2 = 0.50  ("the cat is on", "cat is on the")

c = 5, r = 8
BP = e^(1 - 8/5) = e^(-0.6) ≈ 0.55

BLEU = 0.55 × exp((0.8 + 0.75 + 0.67 + 0.5)/4) = 0.55 × exp(0.68) ≈ 1.08
Normalized BLEU (thường báo cáo) = 0.68 × 100 = 68
```

### 9.4 Ưu điểm và hạn chế của BLEU

**Ưu điểm:**
- Nhanh, rẻ, có thể tính tự động
- Correlated well với human judgment (trong phạm vi)
- Phổ biến → so sánh được với literature

**Hạn chế:**
- Chỉ đánh giá **surface-level similarity**, không đánh giá **ý nghĩa**
- Không đánh giá **fluency** của output
- Có thể bị đánh lừa bởi "overlap cao nhưng sai nghĩa"
- Không phải metric tốt nhất cho modern NMT

### 9.5 Các metrics khác

| Metric | Mô tả | Ưu điểm | Hạn chế |
|---|---|---|---|
| **BLEU** | n-gram overlap | Nhanh, phổ biến | Không đánh giá nghĩa |
| **ROUGE** | Recall-oriented | Tốt cho summarization | Ít dùng cho MT |
| **METEOR** | Alignment-based | Đánh giá synonym | Cần word alignment |
| **chrF** | Character n-gram | Tốt cho agglutinative languages | Ít phổ biến |
| **BERTScore** | Contextual embeddings | Đánh giá semantic similarity | Cần pretrained BERT |
| **COMET** | Neural metric | State-of-the-art, aligned with humans | Cần GPU, pretrained model |

---

# PHẦN V — MINH HỌA DÒNG DỮ LIỆU TOÀN BỘ

---

## 10. Từ Raw Text đến Training Batch

### 10.1 Pipeline hoàn chỉnh

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. RAW DATA                                                     │
│    "I'm sorry !\tJe suis désolé !\tCC"                           │
│                                              (tab-separated)      │
└──────────────────────────┬──────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. PREPROCESS                                                   │
│    English: "i ' m sorry !"                                       │
│    French:  "je suis désolé !"                                    │
└──────────────────────────┬──────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. TOKENIZE                                                     │
│    EN: ['i', "'", 'm', 'sorry', '!']                             │
│    FR: ['je', 'suis', 'désolé', '!']                             │
└──────────────────────────┬──────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. VOCABULARY LOOKUP                                            │
│    EN: [2, 30, 4, 105, 7]                                        │
│    FR: [3, 12, 45, 67, 1, 1, ...]  ← thêm BOS/EOS, PAD          │
└──────────────────────────┬──────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. BATCHING & PADDING                                           │
│    Batch: tensor (batch_size, max_len)                           │
│    Pad all sentences in batch to same length                     │
└──────────────────────────┬──────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. ENCODER-DECODER TRAINING                                     │
│    Source → Encoder → Context Vector                              │
│    Target (shifted) → Decoder → Predictions                      │
│    Loss = CrossEntropy(predictions, labels)                      │
└─────────────────────────────────────────────────────────────────┘
```

### 10.2 Minh họa bằng số cụ thể

```python
# Giả sử max_len = 10, batch_size = 3
src_sentences = [
    "i love you",
    "the weather is nice",
    "once upon a time"
]

tgt_sentences = [
    "je t'aime",
    "le temps est beau",
    "il était une fois"
]

# Sau tokenize (giả sử vocabulary indexes)
src_tokens = [
    [2, 45, 67, 89],              # "i love you" → 4 tokens
    [4, 78, 56, 12, 34],         # "the weather is nice" → 5 tokens
    [23, 90, 3, 5, 78, 91],      # "once upon a time" → 6 tokens
]

# Thêm BOS/EOS cho target
tgt_inputs = [
    [1, 5, 20, 89, 0],           # [<bos>, je, t', aime, <eos>] → pad 1 token
    [1, 12, 45, 67, 8, 0, 0],   # [<bos>, le, temps, est, beau, <eos>] → pad 2 tokens
    [1, 34, 56, 78, 90, 12, 0], # [<bos>, il, était, une, fois, <eos>] → pad 1 token
]

# Pad thành batch
src_batch = [[2, 45, 67, 89, 1, 1, 1, 1, 1, 1],  # pad 6
             [4, 78, 56, 12, 34, 1, 1, 1, 1, 1],  # pad 5
             [23, 90, 3, 5, 78, 91, 1, 1, 1, 1]]   # pad 4

# Shape: (3, 10)
src_batch = torch.tensor(src_batch)  # torch.Size([3, 10])
```

### 10.3 Visualize dataset sample

```python
# Xem một vài mẫu từ dataset
data = MTFraEng()
for src, tgt in zip(data.train_df['en'][:5], data.train_df['fr'][:5]):
    print(f"EN: {src}")
    print(f"FR: {tgt}")
    print("---")
```

Output:
```
EN: go .
FR: va !
---
EN: hi .
FR: salut !
---
EN: run !
FR: cours !
---
EN: who ?
FR: qui ?
---
EN: i see .
FR: je vois .
```

---

## 11. So sánh với các bài toán đã học

### 11.1 MT vs Language Modeling

| Khía cạnh | Language Modeling | Machine Translation |
|---|---|---|
| Input | prefix (prefix tự nhiên) | câu nguồn hoàn chỉnh |
| Output | token tiếp theo | câu đích hoàn chỉnh |
| Độ dài output | cố định (sinh vô hạn) | biến đổi, kết thúc bằng `<eos>` |
| Encoder | Không | Có (đọc câu nguồn) |
| Decoder | Unidirectional | Autoregressive |
| Loss | CrossEntropy(next_token) | CrossEntropy(từng token đích) |
| Evaluation | Perplexity | BLEU, chrF |

### 11.2 MT vs Image Captioning

Cùng là **Seq2Seq** — điểm khác biệt chính:

| Khía cạnh | Image Captioning | Machine Translation |
|---|---|---|
| Input | Image (2D array) | Text (1D sequence) |
| Encoder input processing | CNN backbone | Embedding + RNN |
| Multi-modality | Image → Text | Text → Text |

---

## 12. Các kiến trúc Seq2Seq trong lịch sử

### 12.1 Timeline

```
2014: Seq2Seq (Sutskever et al.)      → RNN Encoder-Decoder, 1 context vector
2015: Attention (Bahdanau et al.)      → Bahdanau attention, dynamic context
2015: Luong Attention (Luong et al.)   → Global + Local attention
2016: Google Neural MT                → Production, replaced statistical MT
2017: Transformer (Vaswani et al.)      → Attention-only, no RNN
2018: BERT (Devlin et al.)            → Pretrained encoder, bidirectional
2019: T5 (Raffel et al.)              → Text-to-Text framework
2020+: Large-scale NMT                → Massive models, multilingual
```

### 12.2 Từ RNN Seq2Seq đến Transformer

![[assets/attachments/d2l-buoi-46/mt-transformer.svg]]
_Fig. (D2L): Minh họa sự tiến hóa từ RNN Seq2Seq (trái) sang Transformer (phải). Transformer thay RNN encoder bằng self-attention layers, và RNN decoder bằng self-attention + cross-attention. Cơ chế attention cho phép mô hình capture dependencies dài mà không cần recurrence._

---

# PHẦN VI — BÀI TẬP (10.5.3)

---

## Bài 1: Phân tích dữ liệu

> _"Phân tích phân bố độ dài câu trong dataset. Có bao nhiêu câu ngắn, trung bình, dài? Tỉ lệ max_len nào là phù hợp?"_

```python
import matplotlib.pyplot as plt

# Phân bố độ dài
src_lens = [len(s.split()) for s in data.train_df['en']]
tgt_lens = [len(s.split()) for s in data.train_df['fr']]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(src_lens, bins=50)
axes[0].set_title('English sentence lengths')
axes[0].set_xlabel('Tokens')
axes[0].set_ylabel('Frequency')

axes[1].hist(tgt_lens, bins=50)
axes[1].set_title('French sentence lengths')
axes[1].set_xlabel('Tokens')
axes[1].set_ylabel('Frequency')
plt.show()

print(f"EN: mean={np.mean(src_lens):.1f}, max={max(src_lens)}, p95={np.percentile(src_lens, 95):.0f}")
print(f"FR: mean={np.mean(tgt_lens):.1f}, max={max(tgt_lens)}, p95={np.percentile(tgt_lens, 95):.0f}")
```

**Phân tích:**
- Câu quá ngắn (< 3 tokens): có thể là noise
- Câu quá dài (> p95 percentile): nên truncate
- max_len tối ưu: p95-99 của phân bố độ dài

## Bài 2: Thiết kế vocabulary

> _"Thiết kế vocabulary cho EN-FR MT. Vocab size bao nhiêu là phù hợp? min_freq nào?"_

```python
# Experiment với vocab_size
for vocab_size in [5000, 10000, 20000, 50000]:
    vocab = Vocab(tokens, min_freq=2, reserved_tokens=[PAD, BOS, EOS, UNK])
    oov_rate = sum(1 for s in val_df['en'] if UNK in s) / len(val_df)
    print(f"Vocab size: {vocab_size}, OOV rate: {oov_rate:.4f}")
```

**Nguyên tắc:**
- Vocab size lớn → OOV rate thấp nhưng model lớn hơn
- Vocab size nhỏ → model nhỏ hơn nhưng nhiều `<unk>`
- Trade-off: thường chọn vocab_size = 10K-50K

## Bài 3: BLEU Score

> _"Tính BLEU score cho hai bản dịch giả định và phân tích."_

```python
reference = "the black cat is on the mat".split()
hypothesis_1 = "the cat is on the mat".split()        # thiếu "black"
hypothesis_2 = "a black cat sits on the mat".split() # "sits" khác

print(f"BLEU-4 (hyp1): {bleu(reference, hypothesis_1):.4f}")  # 0.68
print(f"BLEU-4 (hyp2): {bleu(reference, hypothesis_2):.4f}")  # thấp hơn
```

**Phân tích:**
- hypothesis_1 có 4/5 unigrams đúng, nhưng thiếu "black" → BLEU 0.68
- hypothesis_2 có nhiều words đúng nhưng có thêm "a", "sits" → vẫn cao
- BLEU không phạt "sits" sai vì nó có trong output nhiều lần

## Bài 4: Encoder-Decoder không attention

> _"Mô tả flow dữ liệu trong Encoder-Decoder cơ bản (không attention) khi dịch câu ngắn."_

```
Input: "I love you" (EN)
Output: "Je t'aime" (FR)

Encoder:
  step 1: x_1="I"     → Embed → RNN → h_1
  step 2: x_2="love" → RNN(h_1) → h_2
  step 3: x_3="you"  → RNN(h_2) → h_3 = C (context vector)

Decoder:
  step 1: y_0="<bos>", C → RNN → s_1 → predict "Je"
  step 2: y_1="Je", C → RNN → s_2 → predict "t'"
  step 3: y_2="t'", C → RNN → s_3 → predict "aime"
  step 4: y_3="aime", C → RNN → s_4 → predict "<eos>"
```

**Nhận xét:** Decoder không có "cửa sổ nhìn lại" — nó chỉ biết context vector $C$ duy nhất. Nếu câu nguồn dài, $C$ phải nén quá nhiều thông tin.

---

## Tổng kết

| Khía cạnh | Nội dung |
|---|---|
| **MT là gì** | Bài toán tự động dịch từ ngôn ngữ này sang ngôn ngữ khác |
| **Seq2Seq** | Encoder-Decoder — nén source vào context, sinh target token-by-token |
| **Bottleneck** | 1 context vector duy nhất → cần Attention |
| **Dataset đặc biệt** | Song ngữ, alignment, special tokens (BOS/EOS/PAD/UNK) |
| **Tokenization** | Lowercase, split, vocab lookup, pad/truncate |
| **BLEU Score** | n-gram precision × brevity penalty |
| **Hạn chế Seq2Seq** | Bottleneck, không real-time, không capture long-range dependencies |
| **Kế thừa** | Attention (Ch.11), Transformer (Ch.11), pretrained models |

---

> **Buổi trước:** [[Buổi 45 - Tuần 12]] — 10.3 & 10.4 Deep RNN + Bidirectional RNN
> **Buổi sau:** [[Buổi 47 - Tuần 13]] — 10.6 Encoder-Decoder Architecture

---

## Thuật ngữ

| Thuật ngữ | Tiếng Anh | Ghi chú |
|---|---|---|
| Dịch máy | Machine Translation (MT) | |
| Sequence-to-Sequence | Seq2Seq | Encoder-Decoder framework |
| Encoder | Encoder | Nén câu nguồn |
| Decoder | Decoder | Sinh câu đích |
| Context vector | Context vector | Thông tin nguồn nén |
| Bottleneck | Bottleneck | Nút thắt cổ chai thông tin |
| N-gram | N-gram | Chuỗi N tokens liên tiếp |
| BLEU | Bilingual Evaluation Understudy | Metric đánh giá MT |
| Teacher forcing | Teacher Forcing | Cho Decoder thấy ground truth |
| Autoregressive | Autoregressive | Mỗi step phụ thuộc step trước |
| Special tokens | Special tokens | PAD, UNK, BOS, EOS |
| Padding | Padding | Đệm cho batch đồng nhất |
| Truncation | Truncation | Cắt câu dài |
| Out-of-vocabulary | OOV | Từ không có trong vocab |
| Parallel corpus | Parallel corpus | Dữ liệu song ngữ |
