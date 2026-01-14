---
type: paper
title: Masked Autoencoders Are Scalable Vision Learners
venue: CVPR
year: 2022
authors:
  - Kaiming He
  - Xinlei Chen
  - Saining Xie
  - Yanghao Li
  - Piotr Dollar
  - Ross Girshick
tags:
  - papers
  - computer-vision
  - self-supervised-learning
  - autoencoders
  - transformers
source_pdf: assets/Library/7. He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.pdf
assets_dir: assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022
original_text: 30_Resources/Papers/Masked Autoencoders Are Scalable Vision Learners (Original).md
aliases:
  - MAE Paper
  - Masked Autoencoders
---

> [!NOTE] Phạm vi & nguyên tắc trình bày
> Ghi chú này **giải thích từng đoạn** trong bài báo gốc, đối chiếu trực tiếp với văn bản tiếng Anh. Mỗi phần bao gồm: (1) **Trích dẫn gốc** (quote block), (2) **Dịch nghĩa**, và (3) **Giải thích sâu** theo chuẩn First Principles. Các khái niệm nền được liên kết tới [[Masked Autoencoders (MAE)]], [[Vision Transformers (ViT)]], [[Self-Supervised Learning (Computer Vision)]], [[Autoencoders]], [[Transformer Architecture]].

---

## 0. Định vị bài báo trong hệ thống tri thức

**Masked Autoencoders (MAE)** là phương pháp **self-supervised learning** cho thị giác máy tính, lấy cảm hứng từ thành công của **BERT** trong NLP. Ý tưởng cốt lõi: che (**mask**) một lượng lớn patch trong ảnh (75%+), yêu cầu mô hình tái tạo pixel bị thiếu. Điểm đặc biệt của MAE so với các phương pháp trước:

1. **Asymmetric encoder-decoder**: Encoder chỉ xử lý patch nhìn thấy (không có mask token), decoder nhẹ chịu trách nhiệm tái tạo.
2. **High masking ratio**: 75% patch bị loại bỏ, tạo bài toán khó buộc mô hình học cấu trúc toàn cục.
3. **Pixel reconstruction**: Không cần tokenizer phức tạp (như dVAE trong BEiT), chỉ dùng MSE loss trên pixel.

---

## 1. Abstract — Tóm tắt

### 1.1 Văn bản gốc

> *"This paper shows that masked autoencoders (MAE) are scalable self-supervised learners for computer vision. Our MAE approach is simple: we mask random patches of the input image and reconstruct the missing pixels. It is based on two core designs. First, we develop an asymmetric encoder-decoder architecture, with an encoder that operates only on the visible subset of patches (without mask tokens), along with a lightweight decoder that reconstructs the original image from the latent representation and mask tokens. Second, we find that masking a high proportion of the input image, e.g., 75%, yields a nontrivial and meaningful self-supervisory task. Coupling these two designs enables us to train large models efficiently and effectively: we accelerate training (by 3× or more) and improve accuracy. Our scalable approach allows for learning high-capacity models that generalize well: e.g., a vanilla ViT-Huge model achieves the best accuracy (87.8%) among methods that use only ImageNet-1K data. Transfer performance in downstream tasks outperforms supervised pre-training and shows promising scaling behavior."*

### 1.2 Dịch nghĩa

Bài báo chứng minh rằng **masked autoencoders (MAE)** là phương pháp học tự giám sát có khả năng mở rộng (**scalable**) cho thị giác máy tính. Cách tiếp cận MAE đơn giản: che ngẫu nhiên các **patch** của ảnh đầu vào và tái tạo lại các pixel bị thiếu. Thiết kế dựa trên hai ý chính:

1. **Kiến trúc encoder-decoder bất đối xứng**: encoder chỉ hoạt động trên tập patch nhìn thấy (không dùng **mask token**), còn decoder nhẹ chịu trách nhiệm tái tạo ảnh gốc từ biểu diễn ẩn và các mask token.
2. **Tỉ lệ che cao**: che 75% ảnh đầu vào tạo ra nhiệm vụ tự giám sát vừa khó vừa có ý nghĩa.

Kết hợp hai thiết kế này cho phép huấn luyện mô hình lớn hiệu quả: tăng tốc huấn luyện (≥3×) và cải thiện độ chính xác. ViT-Huge "vanilla" đạt **87.8%** trên ImageNet-1K (tốt nhất trong nhóm chỉ dùng dữ liệu ImageNet-1K). Hiệu năng transfer learning vượt supervised pre-training và cho thấy xu hướng scale hứa hẹn.

### 1.3 Giải thích sâu

**Tại sao "scalable" là đặc tính quan trọng?**

Từ "scalable" ở đây có hai chiều nghĩa:
- **Model scaling**: Khi tăng kích thước mô hình (nhiều layer, chiều ẩn lớn hơn), hiệu năng vẫn tăng chứ không bão hòa.
- **Compute efficiency**: Dù mô hình lớn, ta vẫn huấn luyện được với chi phí hợp lý.

MAE đạt được cả hai nhờ cơ chế **encoder thưa** (sparse encoder):
$$
\text{Compute}_{\text{attention}} \propto n^2 \cdot d
$$

Với $n$ là số token, $d$ là hidden dimension. Nếu masking ratio $r = 0.75$, số token vào encoder chỉ còn $(1-r) \cdot N = 0.25 \cdot N$. Chi phí attention giảm còn:
$$
\frac{(0.25N)^2}{N^2} = \frac{1}{16}
$$

Đây là lý do MAE có thể **tăng tốc 3–4×** trong thực tế (bao gồm cả MLP, I/O, decoder).

---

## 2. Introduction — Giới thiệu

### 2.1 Bối cảnh: Sự bùng nổ kiến trúc và nhu cầu dữ liệu

#### Văn bản gốc

> *"Deep learning has witnessed an explosion of architectures of continuously growing capability and capacity [33, 25, 57]. Aided by the rapid gains in hardware, models today can easily overfit one million images [13] and begin to demand hundreds of millions of—often publicly inaccessible—labeled images [16]."*

#### Dịch nghĩa

Deep learning đã chứng kiến sự bùng nổ các kiến trúc có năng lực và dung lượng ngày càng tăng. Nhờ tiến bộ phần cứng, các mô hình ngày nay dễ dàng overfit trên một triệu ảnh và bắt đầu đòi hỏi hàng trăm triệu ảnh có nhãn — thường không công khai.

#### Giải thích sâu

Đây là **paradox của deep learning hiện đại**: mô hình càng lớn, càng mạnh, nhưng cũng càng cần nhiều dữ liệu gán nhãn. Các tập dữ liệu như **JFT-300M** (Google, 300 triệu ảnh có nhãn) không công khai, tạo ra bất bình đẳng trong nghiên cứu. MAE giải quyết vấn đề này bằng cách học từ **ảnh không nhãn**.

---

### 2.2 NLP đã giải quyết vấn đề này như thế nào

#### Văn bản gốc

> *"This appetite for data has been successfully addressed in natural language processing (NLP) by self-supervised pre-training. The solutions, based on autoregressive language modeling in GPT [47, 48, 4] and masked autoencoding in BERT [14], are conceptually simple: they remove a portion of the data and learn to predict the removed content. These methods now enable training of generalizable NLP models containing over one hundred billion parameters [4]."*

#### Dịch nghĩa

Nhu cầu dữ liệu này đã được giải quyết thành công trong NLP bằng **pre-training tự giám sát**. Các giải pháp dựa trên **mô hình ngôn ngữ tự hồi quy** (GPT) và **masked autoencoding** (BERT) có ý tưởng đơn giản: loại bỏ một phần dữ liệu và học dự đoán phần bị loại. Các phương pháp này cho phép huấn luyện mô hình NLP tổng quát với hơn 100 tỉ tham số.

#### Giải thích sâu

Cả GPT và BERT đều dựa trên nguyên lý **"giấu rồi dự đoán"** (hide and predict):
- **GPT** (autoregressive): Dự đoán token tiếp theo dựa trên các token trước.
- **BERT** (masked language model): Che 15% token, dự đoán token bị che.

Câu hỏi đặt ra: **Tại sao cách tiếp cận tương tự chưa thành công trong vision?**

---

### 2.3 Ba khác biệt giữa Vision và Language

#### 2.3.1 Khác biệt kiến trúc

##### Văn bản gốc

> *"(i) Until recently, architectures were different. In vision, convolutional networks [34] were dominant over the last decade [33]. Convolutions typically operate on regular grids and it is not straightforward to integrate 'indicators' such as mask tokens [14] or positional embeddings [57] into convolutional networks. This architectural gap, however, has been addressed with the introduction of Vision Transformers (ViT) [16] and should no longer present an obstacle."*

##### Dịch nghĩa

Cho đến gần đây, kiến trúc trong vision khác với NLP. Trong thập kỷ qua, CNN thống trị. CNN hoạt động trên lưới đều và không dễ tích hợp các "chỉ báo" như **mask token** hay **positional embedding**. Tuy nhiên, rào cản này đã được giải quyết với sự xuất hiện của **Vision Transformers (ViT)**.

##### Giải thích sâu

CNN có **inductive bias cố định** (locality, translation equivariance) được thiết kế vào kiến trúc. Transformer không có bias này, nhưng linh hoạt hơn — có thể thêm mask token, positional embedding như trong NLP.

```
CNN: input → conv layers → output
     (không có khái niệm "token" rõ ràng)

ViT: input → patch embedding → [CLS] token + patches → Transformer → output
     (có thể thêm [MASK] token tự nhiên)
```

---

#### 2.3.2 Khác biệt mật độ thông tin

##### Văn bản gốc

> *"(ii) Information density is different between language and vision. Languages are human-generated signals that are highly semantic and information-dense. When training a model to predict only a few missing words per sentence, this task appears to induce sophisticated language understanding. Images, on the contrary, are natural signals with heavy spatial redundancy—e.g., a missing patch can be recovered from neighboring patches with little high-level understanding of parts, objects, and scenes."*

##### Dịch nghĩa

Mật độ thông tin khác nhau giữa ngôn ngữ và hình ảnh:
- **Ngôn ngữ**: Do con người tạo ra, giàu ngữ nghĩa và nén thông tin. Che vài từ trong câu đã tạo nhiệm vụ đòi hỏi hiểu ngữ nghĩa sâu.
- **Hình ảnh**: Tín hiệu tự nhiên với dư thừa không gian lớn. Một patch bị che có thể được phục hồi từ patch lân cận mà không cần hiểu về đối tượng/cảnh.

##### Giải thích sâu

Đây là **insight quan trọng nhất** của bài báo. Nếu coi ảnh như biến ngẫu nhiên $X$:
$$
p(X_{\text{mask}} \mid X_{\text{vis}})
$$

- **Masking ratio thấp** (như BERT 15%): $X_{\text{mask}}$ gần như được quyết định bởi $X_{\text{vis}}$ qua **interpolation cục bộ**. Mô hình có thể đạt loss thấp bằng heuristic như "điền texture lân cận".
- **Masking ratio cao** (MAE 75%): $X_{\text{mask}}$ trở nên **thiếu thông tin thực sự**. Mô hình buộc phải học cấu trúc toàn cục (gestalt) để dự đoán hợp lý.

> [!IMPORTANT] Nguyên lý cốt lõi
> **Masking ratio cao là cách "ép" mô hình học ngữ nghĩa thông qua bài toán tái tạo pixel.** Pixel tự thân không phải đơn vị ngữ nghĩa, nhưng việc dự đoán chúng dưới che phủ lớn đòi hỏi hiểu về đối tượng/cảnh.

---

#### 2.3.3 Vai trò decoder khác nhau

##### Văn bản gốc

> *"(iii) The autoencoder's decoder, which maps the latent representation back to the input, plays a different role between reconstructing text and images. In vision, the decoder reconstructs pixels, hence its output is of a lower semantic level than common recognition tasks. This is in contrast to language, where the decoder predicts missing words that contain rich semantic information. While in BERT the decoder can be trivial (an MLP) [14], we found that for images, the decoder design plays a key role in determining the semantic level of the learned latent representations."*

##### Dịch nghĩa

Decoder có vai trò khác nhau giữa văn bản và hình ảnh:
- **Vision**: Decoder tái tạo pixel (mức ngữ nghĩa thấp hơn recognition tasks).
- **Language**: Decoder dự đoán từ bị thiếu (giàu ngữ nghĩa).

Trong BERT, decoder có thể đơn giản (MLP). Nhưng trong ảnh, **thiết kế decoder quyết định mức ngữ nghĩa của latent representation**.

##### Giải thích sâu

Nếu decoder quá yếu, encoder buộc phải gánh việc tái tạo pixel → representation thiên về chi tiết thấp tầng. Nếu decoder đủ mạnh, nó hấp thụ tính chuyên biệt tái tạo → encoder có thể học representation trừu tượng hơn.

```
Decoder yếu: Encoder → [latent chứa cả low-level details] → Decoder → pixels
Decoder mạnh: Encoder → [latent semantic] → Decoder (gánh low-level) → pixels
```

---

## 3. Approach — Phương pháp

### 3.1 Masking Strategy

#### Văn bản gốc

> *"Following ViT [16], we divide an image into regular non-overlapping patches. Then we sample a subset of patches and mask (i.e., remove) the remaining ones. Our sampling strategy is straightforward: we sample random patches without replacement, following a uniform distribution. We simply refer to this as 'random sampling'."*

> *"Random sampling with a high masking ratio (i.e., the ratio of removed patches) largely eliminates redundancy, thus creating a task that cannot be easily solved by extrapolation from visible neighboring patches."*

#### Dịch nghĩa

Theo ViT, chia ảnh thành các patch không chồng lấp. Lấy mẫu một tập con patch và che (loại bỏ) các patch còn lại. Chiến lược lấy mẫu: **random sampling không hoàn lại** theo phân phối đều.

Random sampling với masking ratio cao loại bỏ phần lớn dư thừa, tạo bài toán không thể giải đơn giản bằng ngoại suy từ patch lân cận.

#### Giải thích sâu + Hình minh họa

![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/img-001.png]]
*Figure 1 (tái tạo): Kiến trúc MAE. Trong pre-training, 75% patch bị che. Encoder chỉ xử lý patch nhìn thấy. Mask token được thêm sau encoder, và toàn bộ set được xử lý bởi decoder nhẹ để tái tạo ảnh gốc.*

**Tại sao random sampling?**
- **Block masking** (che vùng liên tục): Mô hình có thể inpaint theo statistics vùng lân cận.
- **Grid masking** (giữ lại theo pattern đều): Bài toán dễ hơn, loss thấp hơn nhưng representation yếu hơn.
- **Random masking**: Phân tán vùng thiếu trên toàn ảnh → đòi hỏi suy luận toàn cục.

---

### 3.2 MAE Encoder

#### Văn bản gốc

> *"Our encoder is a ViT [16] but applied only on visible, unmasked patches. Just as in a standard ViT, our encoder embeds patches by a linear projection with added positional embeddings, and then processes the resulting set via a series of Transformer blocks. However, our encoder only operates on a small subset (e.g., 25%) of the full set. Masked patches are removed; no mask tokens are used. This allows us to train very large encoders with only a fraction of compute and memory."*

#### Dịch nghĩa

Encoder là ViT nhưng chỉ áp dụng trên patch nhìn thấy. Như ViT chuẩn, encoder nhúng patch bằng linear projection + positional embedding, rồi xử lý qua các Transformer block. Tuy nhiên, encoder chỉ xử lý 25% patch. **Patch bị che bị loại bỏ hoàn toàn; không dùng mask token.**

#### Giải thích sâu

> [!IMPORTANT] Tại sao không đưa mask token vào encoder?
> Nếu encoder thấy [MASK] token trong pre-training nhưng downstream thấy ảnh đầy đủ patch → **distribution mismatch** làm giảm chất lượng transfer. MAE cố tình "ép" encoder chỉ nhìn patch thật.

Về mặt compute:
- ViT-L có 24 block, mỗi block có self-attention $O(n^2)$.
- Với 75% masking, $n$ giảm 4×, attention giảm 16×.
- Thực tế speedup ~3-4× vì còn MLP, I/O.

---

### 3.3 MAE Decoder

#### Văn bản gốc

> *"The input to the MAE decoder is the full set of tokens consisting of (i) encoded visible patches, and (ii) mask tokens. Each mask token [14] is a shared, learned vector that indicates the presence of a missing patch to be predicted. We add positional embeddings to all tokens in this full set; without this, mask tokens would have no information about their location in the image. The decoder has another series of Transformer blocks."*

> *"The MAE decoder is only used during pre-training to perform the image reconstruction task (only the encoder is used to produce image representations for recognition). Therefore, the decoder architecture can be flexibly designed in a manner that is independent of the encoder design."*

#### Dịch nghĩa

Input của decoder là **toàn bộ token**: (i) patch đã encode, (ii) mask token (là vector học được, chỉ ra vị trí patch cần dự đoán). Positional embedding được thêm cho tất cả token — nếu không, mask token không biết vị trí trong ảnh.

**Decoder chỉ dùng trong pre-training.** Khi fine-tune, chỉ giữ encoder → decoder có thể thiết kế độc lập, nhẹ hơn encoder.

#### Giải thích sâu

Decoder mặc định có:
- **8 block** (encoder ViT-L có 24 block)
- **512-d** width (encoder 1024-d)
- Chỉ chiếm **<10% FLOPs/token** so với encoder

```
Pre-training:
  Encoder (ViT-L, 24 blocks) → latent [visible patches]
  → Insert mask tokens → Decoder (8 blocks, 512-d) → reconstructed pixels

Fine-tuning:
  Encoder (ViT-L, 24 blocks) → latent → Classification head
  (Decoder bị bỏ đi)
```

---

### 3.4 Reconstruction Target

#### Văn bản gốc

> *"Our MAE reconstructs the input by predicting the pixel values for each masked patch. Each element in the decoder's output is a vector of pixel values representing a patch. The last layer of the decoder is a linear projection whose number of output channels equals the number of pixel values in a patch. The decoder's output is reshaped to form a reconstructed image. Our loss function computes the mean squared error (MSE) between the reconstructed and original images in the pixel space. We compute the loss only on masked patches, similar to BERT [14]."*

#### Dịch nghĩa

MAE tái tạo ảnh bằng cách dự đoán pixel cho mỗi patch bị che. Output của decoder là vector pixel, được reshape thành ảnh. **Loss function: MSE giữa ảnh tái tạo và ảnh gốc, chỉ tính trên patch bị che** (như BERT chỉ tính loss trên token bị mask).

#### Giải thích sâu

$$
\mathcal{L} = \frac{1}{|M|} \sum_{i \in M} \| \hat{x}_i - x_i \|^2
$$

Trong đó:
- $M$: tập chỉ số các patch bị che
- $\hat{x}_i$: pixel được dự đoán
- $x_i$: pixel ground truth

**Tại sao chỉ tính loss trên patch bị che?**
Paper cho biết tính loss trên toàn bộ pixel làm giảm accuracy ~0.5%. Trực giác: loss trên patch nhìn thấy là "nhiễu" không mang thông tin hữu ích — encoder đã thấy chúng rồi.

---

### 3.5 Simple Implementation

#### Văn bản gốc

> *"Our MAE pre-training can be implemented efficiently, and importantly, does not require any specialized sparse operations. First we generate a token for every input patch (by linear projection with an added positional embedding). Next we randomly shuffle the list of tokens and remove the last portion of the list, based on the masking ratio. This process produces a small subset of tokens for the encoder and is equivalent to sampling patches without replacement. After encoding, we append a list of mask tokens to the list of encoded patches, and unshuffle this full list (inverting the random shuffle operation) to align all tokens with their targets. The decoder is applied to this full list (with positional embeddings added)."*

#### Dịch nghĩa

MAE pre-training có thể triển khai hiệu quả, **không cần sparse operations phức tạp**:

1. Tạo token cho mỗi patch (linear projection + positional embedding)
2. **Shuffle ngẫu nhiên** danh sách token
3. **Bỏ phần cuối** theo masking ratio → tập con nhỏ cho encoder
4. Sau encoding, thêm mask token và **unshuffle** để align với vị trí gốc
5. Decoder xử lý full list

#### Giải thích sâu

Đây là **engineering trick quan trọng**:
- Không cần sparse attention (như Longformer)
- Shuffle + truncate là thao tác rẻ (O(N))
- Tương thích với các deep learning framework tiêu chuẩn

---

## 4. Experiments — Thực nghiệm

### 4.1 Masking Ratio

#### Văn bản gốc

> *"The optimal ratios are surprisingly high. The ratio of 75% is good for both linear probing and fine-tuning. This behavior is in contrast with BERT [14], whose typical masking ratio is 15%. Our masking ratios are also much higher than those in related works [6, 16, 2] in computer vision (20% to 50%)."*

#### Dịch nghĩa

Tỉ lệ mask tối ưu cao bất ngờ: **75%** tốt cho cả linear probing và fine-tuning. Khác với BERT (15%) và các công trình vision trước (20-50%).

#### Giải thích sâu + Kết quả

| Masking Ratio | Fine-tuning Acc | Linear Probing Acc |
|---------------|-----------------|-------------------|
| 20%           | 83.4%           | 58.9%             |
| 50%           | 84.9%           | 69.9%             |
| 75%           | **84.9%**       | **73.5%**         |
| 90%           | 84.5%           | 66.1%             |

**Quan sát**: 
- Linear probing nhạy với masking ratio hơn fine-tuning.
- 75% là sweet spot: bài toán đủ khó để học semantic, nhưng không khó đến mức không học được.

---

### 4.2 Decoder Design

#### Văn bản gốc

> *"A sufficiently deep decoder is important for linear probing. This can be explained by the gap between a pixel reconstruction task and a recognition task: the last several layers in an autoencoder are more specialized for reconstruction, but are less relevant for recognition. A reasonably deep decoder can account for the reconstruction specialization, leaving the latent representations at a more abstract level."*

#### Dịch nghĩa

Decoder đủ sâu quan trọng cho linear probing. Giải thích: các layer cuối autoencoder chuyên biệt cho tái tạo, không phù hợp recognition. Decoder sâu "hấp thụ" tính chuyên biệt này, để latent representation trừu tượng hơn.

#### Kết quả Ablation

| Decoder Blocks | Fine-tuning | Linear Probing |
|----------------|-------------|----------------|
| 1              | 84.8%       | 65.5%          |
| 4              | 84.9%       | 71.9%          |
| 8 (default)    | 84.9%       | **73.5%**      |
| 12             | 84.4%       | 73.3%          |

> [!TIP] Insight
> Fine-tuning ít nhạy với decoder depth vì encoder được update theo downstream objective. Linear probing cố định encoder → chất lượng feature phụ thuộc vào việc encoder có bị "kéo về pixel" hay không.

---

### 4.3 Mask Token in Encoder

#### Văn bản gốc

> *"If the encoder uses mask tokens, it performs worse: its accuracy drops by 14% in linear probing. In this case, there is a gap between pre-training and deploying: this encoder has a large portion of mask tokens in its input in pre-training, which does not exist in uncorrupted images. This gap may degrade accuracy in deployment."*

#### Dịch nghĩa

Nếu encoder dùng mask token, accuracy giảm 14% trong linear probing. Nguyên nhân: **gap giữa pre-training và deployment** — encoder thấy [MASK] trong training nhưng không thấy trong inference.

#### Kết quả

| Configuration      | Fine-tuning | Linear Probing | FLOPs  |
|-------------------|-------------|----------------|--------|
| Encoder w/ [M]    | 84.2%       | 59.6%          | 3.3×   |
| Encoder w/o [M]   | **84.9%**   | **73.5%**      | 1×     |

---

### 4.4 Reconstruction Target

#### Văn bản gốc

> *"Using pixels with normalization improves accuracy. This per-patch normalization enhances the contrast locally... Both experiments suggest that the high-frequency components are useful in our method."*

> *"We also compare an MAE variant that predicts tokens, the target used in BEiT [2]... This tokenization improves fine-tuning accuracy by 0.4% vs. unnormalized pixels, but has no advantage vs. normalized pixels."*

#### Dịch nghĩa

Dùng pixel với **per-patch normalization** cải thiện accuracy. Normalization tăng contrast cục bộ và high-frequency components hữu ích.

So sánh với dVAE token (như BEiT): token chỉ tốt hơn unnormalized pixel 0.4%, không lợi thế so với normalized pixel. **Tokenization không cần thiết cho MAE.**

#### Kết quả

| Target              | Fine-tuning | Linear Probing |
|--------------------|-------------|----------------|
| Pixel (w/o norm)   | 84.9%       | 73.5%          |
| Pixel (w/ norm)    | **85.4%**   | **73.9%**      |
| dVAE token         | 85.3%       | 71.6%          |

---

### 4.5 Data Augmentation

#### Văn bản gốc

> *"Our MAE works well using cropping-only augmentation, either fixed-size or random-size (both having random horizontal flipping). Adding color jittering degrades the results... Surprisingly, our MAE behaves decently even if using no data augmentation (only center-crop, no flipping). This property is dramatically different from contrastive learning."*

#### Dịch nghĩa

MAE hoạt động tốt với **augmentation tối giản** (chỉ crop). Color jittering làm giảm kết quả. MAE thậm chí hoạt động tốt **không có augmentation** — khác biệt lớn với contrastive learning (cần augmentation mạnh).

#### Giải thích sâu

Trong contrastive learning (SimCLR, MoCo), augmentation tạo "positive pairs" để học invariance. Nếu không có augmentation, hai view giống hệt → trivial solution.

Trong MAE, **random masking đóng vai trò augmentation**: mỗi iteration có mask khác nhau, tạo training sample mới. Bài toán khó nhờ mask, không cần augmentation nặng.

---

### 4.6 Mask Sampling Strategy

#### Văn bản gốc

> *"The block-wise masking strategy, proposed in [2], tends to remove large blocks. Our MAE with block-wise masking works reasonably well at a ratio of 50%, but degrades at a ratio of 75%... We also study grid-wise sampling, which regularly keeps one of every four patches. This is an easier task and has lower training loss. The reconstruction is sharper. However, the representation quality is lower."*

#### Dịch nghĩa

- **Block masking**: Che vùng lớn. Hoạt động OK ở 50%, kém ở 75%.
- **Grid masking**: Giữ 1/4 patch đều đặn. Task dễ hơn, loss thấp hơn, nhưng representation yếu hơn.
- **Random masking**: Tốt nhất, cho phép masking ratio cao và representation mạnh.

#### Hình minh họa

![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-006-054.jpg]] ![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-006-055.jpg]] ![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-006-056.jpg]]
*Từ trái sang: Random 75%, Block 50%, Grid 75%. Random tạo reconstruction hợp nghĩa nhất.*

---

### 4.7 Comparison with Previous Results

#### Văn bản gốc

> *"For ViT-B, all methods perform closely. For ViT-L, the gaps among methods are bigger, suggesting that a challenge for bigger models is to reduce overfitting. Our MAE can scale up easily and has shown steady improvement from bigger models. We obtain 86.9% accuracy using ViT-H (224 size). By fine-tuning with a 448 size, we achieve 87.8% accuracy, using only IN1K data."*

#### Kết quả so sánh

| Method       | Pre-train Data | ViT-B  | ViT-L  | ViT-H  | ViT-H448 |
|--------------|---------------|--------|--------|--------|----------|
| DINO         | IN1K          | 82.8%  | -      | -      | -        |
| MoCo v3      | IN1K          | 83.2%  | 84.1%  | -      | -        |
| BEiT         | IN1K+DALLE    | 83.2%  | 85.2%  | -      | -        |
| **MAE**      | IN1K          | **83.6%** | **85.9%** | **86.9%** | **87.8%** |

> [!IMPORTANT] Kết quả then chốt
> MAE đạt **87.8%** trên ImageNet-1K với ViT-H448 — tốt nhất trong các phương pháp chỉ dùng IN1K data. Vượt qua cả BEiT (cần DALLE 250M images cho tokenizer) và supervised pre-training.

---

### 4.8 Partial Fine-tuning

#### Văn bản gốc

> *"Notably, fine-tuning only one Transformer block boosts the accuracy significantly from 73.5% to 81.0%. Moreover, if we fine-tune only 'half' of the last block (i.e., its MLP sub-block), we can get 79.1%, much better than linear probing."*

#### Dịch nghĩa

Fine-tune chỉ 1 block tăng từ 73.5% lên 81.0%. Fine-tune nửa block cuối (MLP sub-block) đạt 79.1% — tốt hơn nhiều so với linear probing.

#### Giải thích sâu

![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/img-000.png]]
*Figure 9: Partial fine-tuning. MAE representation mạnh hơn MoCo v3 khi tune ≥1 block, dù linear probing yếu hơn.*

**Insight**: Linear separability không phải thước đo duy nhất cho representation quality. MAE học **non-linear features** mạnh hơn contrastive methods.

---

## 5. Transfer Learning

### 5.1 Object Detection & Segmentation (COCO)

#### Kết quả

| Method       | ViT-B APbox | ViT-L APbox | ViT-B APmask | ViT-L APmask |
|--------------|-------------|-------------|--------------|--------------|
| Supervised   | 47.9        | 49.3        | 42.9         | 43.9         |
| MoCo v3      | 47.9        | 49.3        | 42.7         | 44.0         |
| BEiT         | 49.8        | 53.3        | 44.4         | 47.1         |
| **MAE**      | **50.3**    | **53.3**    | **44.9**     | **47.2**     |

MAE outperforms supervised pre-training by **4.0 points** với ViT-L (53.3 vs 49.3).

---

### 5.2 Semantic Segmentation (ADE20K)

| Method       | ViT-B mIoU | ViT-L mIoU |
|--------------|------------|------------|
| Supervised   | 47.4       | 49.9       |
| MoCo v3      | 47.3       | 49.1       |
| BEiT         | 47.1       | 53.3       |
| **MAE**      | **48.1**   | **53.6**   |

MAE outperforms supervised pre-training by **3.7 points** với ViT-L.

---

### 5.3 Classification Transfer

| Dataset   | ViT-B  | ViT-L  | ViT-H  | ViT-H448 | Previous Best |
|-----------|--------|--------|--------|----------|---------------|
| iNat 2017 | 70.5%  | 75.7%  | 79.3%  | **83.4%**| 75.4%*        |
| iNat 2018 | 75.4%  | 80.1%  | 83.0%  | **86.8%**| 81.2%*        |
| Places205 | 63.9%  | 65.8%  | 65.9%  | **66.8%**| 66.0%†        |
| Places365 | 57.9%  | 59.4%  | 59.8%  | **60.3%**| 58.0%‡        |

*: pre-trained on different data | †: pre-trained on 1B images | ‡: pre-trained on 3.5B images

> [!TIP] Scaling Behavior
> MAE cho thấy **strong scaling**: accuracy tăng đáng kể với model lớn hơn. Trên iNaturalist, MAE vượt previous best (trained on much more data) by large margins.

---

## 6. Discussion & Conclusion

### 6.1 Văn bản gốc

> *"Simple algorithms that scale well are the core of deep learning. In NLP, simple self-supervised learning methods (e.g., [47, 14, 48, 4]) enable benefits from exponentially scaling models. In computer vision, practical pre-training paradigms are dominantly supervised despite progress in self-supervised learning. In this study, we observe on ImageNet and in transfer learning that an autoencoder—a simple self-supervised method similar to techniques in NLP—provides scalable benefits."*

> *"On the other hand, we note that images and languages are signals of a different nature and this difference must be addressed carefully. Images are merely recorded light without a semantic decomposition into the visual analogue of words. Instead of attempting to remove objects, we remove random patches that most likely do not form a semantic segment. Likewise, our MAE reconstructs pixels, which are not semantic entities. Nevertheless, we observe... that our MAE infers complex, holistic reconstructions, suggesting it has learned numerous visual concepts, i.e., semantics."*

### 6.2 Dịch nghĩa

**Thuật toán đơn giản nhưng scale tốt là cốt lõi của deep learning.** Trong NLP, các phương pháp self-supervised đơn giản (GPT, BERT) mang lại lợi ích từ việc scale mô hình theo hàm mũ. Trong vision, pre-training vẫn bị supervised thống trị. Nghiên cứu này cho thấy autoencoder — phương pháp self-supervised đơn giản tương tự NLP — mang lại **scalable benefits** cho vision.

Tuy nhiên, ảnh và ngôn ngữ có bản chất khác nhau. Ảnh không có phân rã ngữ nghĩa như từ. MAE che patch ngẫu nhiên (không cố xóa vật thể) và tái tạo pixel (không phải entity ngữ nghĩa). Dù vậy, MAE **suy luận được reconstruction phức tạp, toàn thể**, cho thấy nó đã học nhiều khái niệm thị giác — tức là ngữ nghĩa.

### 6.3 Giải thích sâu (Suy luận thêm — đánh dấu rõ)

> [!NOTE] Suy luận của tác giả ghi chú
> MAE không cần trở thành generative model hoàn chỉnh. Mục tiêu là học latent đủ tốt để giảm lỗi có điều kiện trên patch bị che. Nhưng chính việc "điền vào thiếu" dưới che phủ cao tạo áp lực học cấu trúc **nguyên nhân–hệ quả** (bố cục/đối tượng) hơn là tương quan cục bộ.
>
> Đây có thể giải thích tại sao MAE tạo representation mạnh cho downstream, ngay cả khi chất lượng tái tạo pixel không hoàn toàn khớp ground truth.

---

## 7. Ví dụ Reconstruction

### 7.1 ImageNet Validation

![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-002-010.jpg]] ![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-002-011.jpg]] ![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-002-012.jpg]]
*Masked (left) → MAE Reconstruction (middle) → Ground Truth (right). Masking ratio 80%.*

![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-002-022.jpg]] ![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-002-023.jpg]] ![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-002-024.jpg]]
*Reconstruction có thể khác ground truth nhưng "hợp nghĩa" (semantically plausible).*

### 7.2 COCO Validation (Zero-shot transfer)

![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-003-034.jpg]] ![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-003-035.jpg]] ![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-003-036.jpg]]
*MAE trained on ImageNet, applied to COCO. Model generalizes to out-of-domain images.*

---

## 8. Tóm tắt các Contributions

1. **High masking ratio (75%)**: Khác biệt quan trọng với BERT (15%), giải quyết vấn đề redundancy của ảnh.

2. **Asymmetric encoder-decoder**: Encoder không thấy mask token → tránh distribution mismatch, tăng tốc 3-4×.

3. **Pixel reconstruction works**: Không cần tokenizer phức tạp (dVAE), normalized pixel đủ tốt.

4. **Minimal augmentation**: Random masking đóng vai trò augmentation, không cần color jitter.

5. **Strong scaling behavior**: ViT-H448 đạt 87.8% trên ImageNet-1K, vượt all previous methods using only IN1K data.

6. **Superior transfer learning**: Outperforms supervised pre-training on COCO (+4.0 APbox) and ADE20K (+3.7 mIoU).

---

## 9. Liên kết tới các Concept Notes

- [[Masked Autoencoders (MAE)]] — Concept note tổng hợp về MAE
- [[Vision Transformers (ViT)]] — Kiến trúc encoder của MAE
- [[Self-Supervised Learning (Computer Vision)]] — Bối cảnh rộng hơn
- [[Autoencoders]] — Họ phương pháp gốc
- [[BERT]] — Nguồn cảm hứng từ NLP
- [[Contrastive Learning]] — Nhánh SSL khác (để so sánh)

---

## 10. Nguyên văn bài báo (English)

![[30_Resources/Papers/Masked Autoencoders Are Scalable Vision Learners (Original)]]
