---
title: "Masked Autoencoders Are Scalable Vision Learners"
aliases:
  - MAE Paper
  - He et al. 2022
type: source-note
source: CVPR
year: 2022
authors:
  - Kaiming He
  - Xinlei Chen
  - Saining Xie
  - Yanghao Li
  - Piotr Dollár
  - Ross Girshick
pdf: assets/Library/7. He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.pdf
assets_dir: assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022
original_text: 30_Resources/Papers/Masked Autoencoders Are Scalable Vision Learners (Original).md
tags:
  - paper
  - computer-vision
  - self-supervised-learning
  - autoencoders
  - transformers
---

# Tóm tắt (Abstract)

Bài báo này chứng minh rằng **masked autoencoders (MAE)** là phương pháp học tự giám sát có khả năng mở rộng (**scalable**) cho thị giác máy tính. Cách tiếp cận MAE rất đơn giản: che ngẫu nhiên các **patch** của ảnh đầu vào và yêu cầu mô hình tái tạo lại các pixel bị thiếu.

Thiết kế dựa trên hai ý chính. Thứ nhất, tác giả xây dựng kiến trúc **encoder–decoder bất đối xứng**, trong đó encoder chỉ hoạt động trên tập patch nhìn thấy (không đưa mask token vào encoder), còn decoder nhẹ chịu trách nhiệm tái tạo ảnh gốc từ biểu diễn ẩn và các mask token. Thứ hai, việc che một tỉ lệ rất lớn ảnh đầu vào (ví dụ 75%) tạo ra một nhiệm vụ tự giám sát vừa khó vừa có ý nghĩa. Kết hợp hai thiết kế này cho phép huấn luyện mô hình lớn hiệu quả: tăng tốc huấn luyện (≥3×) và cải thiện độ chính xác. ViT-Huge "vanilla" đạt **87.8%** trên [[ImageNet]]-1K — tốt nhất trong nhóm chỉ dùng dữ liệu ImageNet-1K. Hiệu năng transfer learning vượt supervised pre-training.

> [!TIP] ELI5 — Giải thích như cho trẻ 5 tuổi
> Hãy tưởng tượng bạn đang chơi trò ghép hình, nhưng thay vì ghép từ từ, ai đó giấu đi 3/4 số mảnh ghép và hỏi bạn: "Bức tranh hoàn chỉnh trông như thế nào?". Để trả lời đúng, bạn phải hiểu bức tranh đang vẽ cái gì — có phải là con mèo, ngôi nhà, hay bầu trời? Bạn không thể chỉ nhìn một mảnh nhỏ và đoán màu của mảnh bên cạnh. MAE dạy máy tính học theo cách này: giấu phần lớn ảnh, bắt máy đoán phần còn lại.

---

# 1. Giới thiệu (Introduction)

## 1.1 Bối cảnh: Sự bùng nổ kiến trúc và nhu cầu dữ liệu

Deep learning đã chứng kiến sự bùng nổ của các kiến trúc có năng lực và dung lượng ngày càng tăng. Nhờ tiến bộ phần cứng nhanh chóng, các mô hình ngày nay dễ dàng overfit trên một triệu ảnh và bắt đầu đòi hỏi hàng trăm triệu ảnh có nhãn — thường không công khai. Nhu cầu dữ liệu này đã được giải quyết thành công trong NLP bằng **pre-training tự giám sát**. Các giải pháp dựa trên mô hình ngôn ngữ tự hồi quy như [[GPT]] và masked autoencoding như [[BERT]] có ý tưởng đơn giản: loại bỏ một phần dữ liệu và học dự đoán phần bị loại. Các phương pháp này cho phép huấn luyện mô hình NLP tổng quát với hơn 100 tỉ tham số.

Ý tưởng masked autoencoders, một dạng [[Denoising Autoencoders]] tổng quát hơn, rất tự nhiên và có thể áp dụng trong thị giác máy tính. Thực tế, các nghiên cứu liên quan trong vision đã có trước [[BERT]]. Tuy nhiên, dù có sự quan tâm đáng kể sau thành công của BERT, tiến bộ của các phương pháp autoencoding trong vision vẫn tụt hậu so với NLP.

Tác giả đặt câu hỏi: **Điều gì khiến masked autoencoding khác nhau giữa vision và language?** Họ trả lời theo ba góc nhìn.

## 1.2 Ba khác biệt giữa Vision và Language

### (i) Khác biệt kiến trúc

Cho đến gần đây, kiến trúc trong vision khác với NLP. Mạng tích chập (CNN) thống trị trong thập kỷ qua. CNN thường hoạt động trên lưới đều và không dễ tích hợp các "chỉ báo" như mask token hay [[Positional Embedding]]. Tuy nhiên, rào cản kiến trúc này đã được giải quyết với sự xuất hiện của [[Vision Transformers (ViT)]] và không còn là trở ngại.

> [!TIP] ELI5
> CNN giống như đọc sách bằng cách nhìn từng ô vuông nhỏ — nó không có khái niệm "vị trí trong câu". Transformer giống như đọc cả câu cùng lúc và biết từ nào ở đâu. Vì vậy, "giấu một từ" dễ làm với Transformer hơn CNN.

### (ii) Khác biệt mật độ thông tin

Ngôn ngữ là tín hiệu do con người tạo ra, **giàu ngữ nghĩa và nén thông tin**. Khi huấn luyện mô hình dự đoán chỉ vài từ bị thiếu trong một câu, nhiệm vụ này dường như kích thích việc hiểu ngôn ngữ sâu sắc. Ngược lại, hình ảnh là tín hiệu tự nhiên với **dư thừa không gian lớn** — ví dụ, một patch bị thiếu có thể được phục hồi từ các patch lân cận với ít sự hiểu biết về đối tượng hay cảnh. Để khắc phục điều này và khuyến khích học các đặc trưng hữu ích, tác giả cho thấy một chiến lược đơn giản hoạt động tốt: **che một tỉ lệ rất cao các patch ngẫu nhiên**.

> [!TIP] ELI5
> Trong một câu tiếng Việt, nếu giấu từ "mèo" trong "Con ___ đang ngủ", bạn cần hiểu ngữ cảnh để đoán. Nhưng trong ảnh, nếu giấu một mảnh nhỏ bầu trời xanh, bạn chỉ cần nhìn mảnh bên cạnh (cũng màu xanh) và tô tiếp — không cần hiểu gì cả! Vì vậy MAE giấu **75%** ảnh để máy không thể "tô màu theo lân cận".

### (iii) Vai trò decoder khác nhau

Trong vision, decoder tái tạo pixel, do đó đầu ra của nó có mức ngữ nghĩa thấp hơn các tác vụ nhận dạng thông thường. Điều này trái ngược với language, nơi decoder dự đoán các từ bị thiếu chứa thông tin ngữ nghĩa phong phú. Trong khi decoder của BERT có thể đơn giản (một MLP), tác giả phát hiện rằng trong ảnh, **thiết kế decoder đóng vai trò quan trọng** trong việc xác định mức ngữ nghĩa của biểu diễn ẩn học được.

> [!TIP] ELI5
> Trong tiếng Anh, đoán từ "cat" là đoán một khái niệm có nghĩa. Trong ảnh, đoán pixel (255, 128, 64) chỉ là đoán một con số — không mang nghĩa gì cả! Vì vậy phần "dịch ngược" (decoder) trong ảnh phải được thiết kế cẩn thận để không kéo phần học chính (encoder) xuống mức "tô màu".

---

# 2. Các công trình liên quan (Related Work)

## 2.1 Masked Language Modeling và GPT

Masked language modeling (BERT) và các đối tác tự hồi quy (GPT) là các phương pháp pre-training rất thành công trong NLP. Các phương pháp này giữ lại một phần chuỗi đầu vào và huấn luyện mô hình dự đoán nội dung bị thiếu. Chúng đã cho thấy khả năng scale tuyệt vời và bằng chứng dồi dào cho thấy các biểu diễn pre-trained này tổng quát tốt cho nhiều downstream tasks.

## 2.2 Autoencoding

[[Autoencoders]] là phương pháp cổ điển để học biểu diễn. Nó có một encoder ánh xạ đầu vào sang biểu diễn ẩn và một decoder tái tạo đầu vào. Ví dụ, PCA và k-means là autoencoders. [[Denoising Autoencoders]] (DAE) là một lớp autoencoders làm hỏng tín hiệu đầu vào và học tái tạo tín hiệu gốc không bị hỏng. Một loạt phương pháp có thể được coi là DAE tổng quát với các loại corruption khác nhau: che pixel, loại bỏ kênh màu, v.v. MAE là một dạng denoising autoencoding, nhưng khác với DAE cổ điển theo nhiều cách.

## 2.3 Masked Image Encoding

Các phương pháp masked image encoding học biểu diễn từ ảnh bị làm hỏng bằng masking. Công trình tiên phong trình bày masking như một loại nhiễu trong DAE. Context Encoder inpaint các vùng lớn bị thiếu bằng CNN. Các phương pháp gần đây dựa trên Transformer bao gồm iGPT (hoạt động trên chuỗi pixel và dự đoán pixel chưa biết), ViT paper (nghiên cứu masked patch prediction), và [[BEiT]] (dự đoán discrete tokens). Các phương pháp [[Contrastive Learning]] (SimCLR, MoCo) theo hướng khái niệm khác — mô hình hóa sự tương đồng giữa các views của ảnh.

> [!NOTE] Suy luận thêm
> MAE đại diện cho nhánh "generative/reconstruction" trong self-supervised learning, đối lập với nhánh "contrastive". Contrastive phụ thuộc mạnh vào [[Data Augmentation]] để tạo positive pairs; MAE có thể hoạt động với augmentation tối giản.

---

# 3. Phương pháp (Approach)

![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/img-001.png]]
*Figure 1: Kiến trúc MAE. Trong pre-training, 75% patch bị che. Encoder chỉ xử lý patch nhìn thấy. Mask token được thêm sau encoder, và toàn bộ set được xử lý bởi decoder nhẹ để tái tạo ảnh gốc. Sau pre-training, decoder bị bỏ đi và encoder được áp dụng cho ảnh đầy đủ.*

[[Masked Autoencoders (MAE)]] là một phương pháp autoencoding đơn giản tái tạo tín hiệu gốc từ quan sát bộ phận. Như tất cả autoencoders, MAE có encoder ánh xạ tín hiệu quan sát được sang biểu diễn ẩn, và decoder tái tạo tín hiệu gốc. Khác với autoencoders cổ điển, MAE sử dụng **thiết kế bất đối xứng** cho phép encoder chỉ hoạt động trên tín hiệu bộ phận đã quan sát (không có mask token) và decoder nhẹ tái tạo toàn bộ tín hiệu.

## 3.1 Masking

Theo ViT, ảnh được chia thành các patch không chồng lấp đều đặn. Sau đó lấy mẫu một tập con patch và che (loại bỏ) các patch còn lại. Chiến lược lấy mẫu rất đơn giản: **lấy mẫu ngẫu nhiên các patch không hoàn lại**, theo phân phối đều — gọi là "random sampling".

Random sampling với **masking ratio cao** (tỉ lệ patch bị loại) loại bỏ phần lớn dư thừa, tạo ra nhiệm vụ không thể giải quyết dễ dàng bằng ngoại suy từ các patch lân cận. Phân phối đều ngăn chặn bias tiềm năng về tâm ảnh (nhiều patch bị che gần tâm hơn). Cuối cùng, đầu vào thưa cao tạo cơ hội thiết kế encoder hiệu quả.

> [!TIP] ELI5
> Bạn có một ảnh ghép 196 mảnh (14×14). MAE giấu ngẫu nhiên 147 mảnh (75%), chỉ để lại 49 mảnh. Các mảnh được chọn ngẫu nhiên, không theo pattern — nên máy không thể "gian lận" bằng cách nhớ pattern.

## 3.2 MAE Encoder

Encoder là một [[Vision Transformers (ViT)]] nhưng chỉ áp dụng trên các patch nhìn thấy, không bị che. Giống ViT chuẩn, encoder nhúng các patch bằng linear projection với [[Positional Embedding]], sau đó xử lý qua các Transformer block. Tuy nhiên, encoder chỉ hoạt động trên tập con nhỏ (ví dụ 25%) của toàn bộ set. **Các patch bị che bị loại bỏ; không sử dụng mask token.** Điều này cho phép huấn luyện encoder rất lớn với chỉ một phần compute và memory.

> [!TIP] ELI5
> Encoder chỉ nhìn 49 mảnh ghép thay vì 196 mảnh. Vì [[Self-Attention]] có chi phí tăng theo bình phương số mảnh, nhìn ít hơn 4 lần nghĩa là nhanh hơn khoảng 16 lần! Đây là lý do MAE huấn luyện nhanh.

## 3.3 MAE Decoder

Đầu vào của decoder là **toàn bộ set tokens** gồm: (i) các patch đã encode, và (ii) mask tokens. Mỗi mask token là một vector học được, chia sẻ, chỉ ra sự hiện diện của patch cần dự đoán. [[Positional Embedding]] được thêm vào tất cả tokens trong set đầy đủ — nếu không, mask tokens sẽ không có thông tin về vị trí trong ảnh. Decoder có một loạt Transformer block khác.

Decoder chỉ được sử dụng trong pre-training để thực hiện tái tạo ảnh — chỉ encoder được dùng để tạo biểu diễn ảnh cho recognition. Do đó, kiến trúc decoder có thể được thiết kế linh hoạt, độc lập với encoder. Thực nghiệm với decoder rất nhỏ, hẹp và nông hơn encoder. Ví dụ, decoder mặc định có **<10% computation/token** so với encoder. Với thiết kế bất đối xứng này, toàn bộ tokens chỉ được xử lý bởi decoder nhẹ, giảm đáng kể thời gian pre-training.

> [!TIP] ELI5
> Encoder là "bộ não chính" — lớn và mạnh. Decoder là "bộ vẽ" — nhỏ và chỉ dùng để tô màu. Sau khi học xong, ta vứt bộ vẽ đi, chỉ giữ bộ não để làm việc thực sự (nhận dạng ảnh).

## 3.4 Mục tiêu tái tạo (Reconstruction Target)

MAE tái tạo đầu vào bằng cách dự đoán **giá trị pixel cho mỗi patch bị che**. Mỗi phần tử trong output của decoder là một vector giá trị pixel đại diện cho một patch. Layer cuối của decoder là linear projection với số kênh output bằng số giá trị pixel trong một patch. Output được reshape thành ảnh tái tạo. Hàm loss tính [[Mean Squared Error]] (MSE) giữa ảnh tái tạo và ảnh gốc trong không gian pixel. **Loss chỉ được tính trên các patch bị che**, tương tự BERT.

Tác giả cũng nghiên cứu biến thể với mục tiêu là **giá trị pixel đã chuẩn hóa** của mỗi patch bị che. Cụ thể, tính mean và std của tất cả pixel trong patch và dùng chúng để chuẩn hóa patch. Sử dụng pixel đã chuẩn hóa làm mục tiêu cải thiện chất lượng biểu diễn.

## 3.5 Triển khai đơn giản

MAE pre-training có thể triển khai hiệu quả và không yêu cầu bất kỳ sparse operation chuyên biệt nào:

1. Tạo token cho mỗi patch đầu vào (linear projection + positional embedding)
2. **Shuffle ngẫu nhiên** danh sách tokens
3. **Loại bỏ phần cuối** của danh sách theo masking ratio → tập con nhỏ cho encoder
4. Sau encoding, thêm mask tokens và **unshuffle** để align với targets
5. Decoder xử lý danh sách đầy đủ

Không cần sparse operations. Triển khai đơn giản này tạo overhead không đáng kể vì shuffle/unshuffle rất nhanh.

---

# 4. Thực nghiệm trên ImageNet

Tác giả thực hiện **self-supervised pre-training** trên ImageNet-1K training set. Sau đó đánh giá biểu diễn với (i) **fine-tuning end-to-end** hoặc (ii) **[[Linear Probing]]**. Báo cáo top-1 validation accuracy với single 224×224 crop.

**Baseline: ViT-Large.** ViT-L được dùng làm backbone trong ablation study. ViT-L rất lớn (lớn hơn ResNet-50 một bậc độ lớn) và có xu hướng overfit.

| Training | Accuracy |
|----------|----------|
| Scratch, original [16] | 76.5% |
| Scratch, our impl. | 82.5% |
| **Baseline MAE** | **84.9%** |

Huấn luyện supervised ViT-L từ scratch không đơn giản và cần recipe tốt với regularization mạnh. Dù vậy, MAE pre-training vẫn đóng góp cải thiện lớn. Fine-tuning chỉ 50 epochs (vs 200 từ scratch).

## 4.1 Các đặc tính chính

### Masking Ratio

![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/img-000.png]]
*Figure 5: Masking ratio. Tỉ lệ cao (75%) hoạt động tốt cho cả fine-tuning và linear probing.*

Tỉ lệ tối ưu **cao bất ngờ — 75%** tốt cho cả linear probing và fine-tuning. Hành vi này trái ngược với BERT (masking ratio điển hình 15%) và các công trình liên quan trong vision (20-50%).

| Masking Ratio | Fine-tuning | Linear Probing |
|---------------|-------------|----------------|
| 20% | 83.4% | 58.9% |
| 50% | 85.0% | 69.9% |
| **75%** | **84.9%** | **73.5%** |
| 90% | 84.5% | 66.1% |

Mô hình suy luận các patch bị thiếu để tạo ra output khác nhau nhưng **hợp lý (plausible)**. Nó hiểu gestalt của đối tượng và cảnh — không thể đơn giản hoàn thành bằng cách kéo dài đường thẳng hoặc texture.

> [!TIP] ELI5
> Nếu giấu ít (20%), máy chỉ cần nhìn mảnh bên cạnh và tô tiếp — học được rất ít. Nếu giấu nhiều (75%), máy phải "tưởng tượng" cả bức tranh mới đoán được — học được nhiều hơn!

### Decoder Design

**Decoder depth** (số Transformer block): Decoder đủ sâu quan trọng cho linear probing. Các layer cuối trong autoencoder chuyên biệt hơn cho reconstruction, ít liên quan đến recognition. Decoder đủ sâu có thể "hấp thụ" tính chuyên biệt này, để biểu diễn ẩn ở mức trừu tượng hơn.

| Decoder Blocks | Fine-tuning | Linear Probing |
|----------------|-------------|----------------|
| 1 | 84.8% | 65.5% |
| 4 | 84.9% | 71.9% |
| **8 (default)** | **84.9%** | **73.5%** |
| 12 | 84.4% | 73.3% |

**Decoder width** (số channels): 512-d là mặc định, hoạt động tốt. Decoder hẹp hơn cũng hoạt động tốt với fine-tuning. Decoder mặc định chỉ có **9% FLOPs/token** so với ViT-L.

### Mask Token

Một thiết kế quan trọng của MAE là **bỏ mask token trong encoder** và chỉ dùng trong decoder nhẹ.

| Configuration | Fine-tuning | Linear Probing | FLOPs |
|--------------|-------------|----------------|-------|
| Encoder w/ [MASK] | 84.2% | 59.6% | 3.3× |
| **Encoder w/o [MASK]** | **84.9%** | **73.5%** | **1×** |

Nếu encoder sử dụng mask tokens, accuracy giảm **14% trong linear probing**. Nguyên nhân: có gap giữa pre-training (encoder thấy nhiều mask token) và deployment (encoder thấy ảnh đầy đủ không có mask). Bằng cách loại bỏ mask token khỏi encoder, ta ràng buộc encoder luôn thấy patch thật.

Hơn nữa, bỏ mask token trong encoder **giảm FLOPs training 3.3×**, dẫn đến speedup wall-clock **2.8–4.1×**.

> [!TIP] ELI5
> Nếu encoder học với nhiều "ô trống" [MASK], khi làm việc thật nó sẽ bối rối vì không còn ô trống nữa. MAE cho encoder chỉ nhìn ô thật, nên khi làm việc nó không bị "ngạc nhiên".

### Reconstruction Target

| Target | Fine-tuning | Linear Probing |
|--------|-------------|----------------|
| Pixel (w/o norm) | 84.9% | 73.5% |
| **Pixel (w/ norm)** | **85.4%** | **73.9%** |
| PCA | 84.6% | 72.3% |
| dVAE token | 85.3% | 71.6% |

Sử dụng **pixel đã chuẩn hóa** (per-patch normalization) cải thiện accuracy. Normalization tăng contrast cục bộ. Dùng token từ dVAE (như BEiT) chỉ tốt hơn unnormalized pixel 0.4%, không có lợi thế so với normalized pixel. **Tokenization không cần thiết** — và dVAE cần pre-training riêng trên 250M ảnh, thêm overhead đáng kể.

### Data Augmentation

| Augmentation | Fine-tuning | Linear Probing |
|--------------|-------------|----------------|
| None (center-crop only) | 84.0% | 65.7% |
| Crop, fixed size | 84.7% | 73.1% |
| **Crop, random size** | **84.9%** | **73.5%** |
| Crop + color jitter | 84.3% | 71.9% |

MAE hoạt động tốt với **augmentation tối giản** (chỉ crop). Color jittering **làm giảm** kết quả. Đáng ngạc nhiên, MAE vẫn hoạt động tốt **không có augmentation** — khác biệt lớn với [[Contrastive Learning]] (giảm 13-28% khi chỉ dùng crop).

Trong MAE, vai trò của data augmentation chủ yếu được thực hiện bởi **random masking**: mask khác nhau mỗi iteration, tạo training sample mới bất kể augmentation.

### Mask Sampling Strategy

![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-006-054.jpg]] ![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-006-055.jpg]] ![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-006-056.jpg]]
*Figure 6: Mask sampling strategies. Trái: Random 75% (default). Giữa: Block 50%. Phải: Grid 75%.*

| Strategy | Ratio | Fine-tuning | Linear Probing |
|----------|-------|-------------|----------------|
| **Random** | 75% | **84.9%** | **73.5%** |
| Block | 50% | 83.9% | 72.3% |
| Block | 75% | 82.8% | 63.9% |
| Grid | 75% | 84.0% | 66.0% |

**Random sampling hoạt động tốt nhất.** Block masking (che vùng lớn liên tục) khó hơn, reconstruction mờ hơn. Grid masking (giữ 1/4 patch đều) dễ hơn, reconstruction sắc hơn nhưng representation yếu hơn.

### Training Schedule

| Epochs | Fine-tuning | Linear Probing |
|--------|-------------|----------------|
| 100 | 82.3% | 57.3% |
| 200 | 83.3% | 64.4% |
| 400 | 84.3% | 69.7% |
| 800 | 84.9% | 73.5% |
| 1600 | 85.1% | 75.1% |

*Figure 7: Training schedules. Accuracy cải thiện đều với huấn luyện dài hơn.*

Linear probing chưa bão hòa ngay cả ở 1600 epochs — khác với contrastive methods (MoCo v3 bão hòa ở 300 epochs). Lưu ý rằng MAE encoder chỉ thấy 25% patches/epoch, trong khi contrastive learning thấy 200% (two-crop) hoặc hơn.

---

# 5. So sánh với các phương pháp trước

## 5.1 So sánh với Self-Supervised Methods

| Method | Pre-train Data | ViT-B | ViT-L | ViT-H | ViT-H448 |
|--------|---------------|-------|-------|-------|----------|
| Scratch | - | 82.3% | 82.6% | 83.1% | - |
| DINO | IN1K | 82.8% | - | - | - |
| MoCo v3 | IN1K | 83.2% | 84.1% | - | - |
| BEiT | IN1K+DALLE | 83.2% | 85.2% | - | - |
| **MAE** | IN1K | **83.6%** | **85.9%** | **86.9%** | **87.8%** |

MAE có thể **scale up dễ dàng** và cho thấy cải thiện đều từ model lớn hơn. Đạt **86.9%** với ViT-H (224), và **87.8%** khi fine-tune với 448 size — **state-of-the-art** trong các phương pháp chỉ dùng IN1K data.

So với [[BEiT]], MAE chính xác hơn trong khi đơn giản và nhanh hơn (3.5× per epoch). MAE tái tạo pixel, trong khi BEiT dự đoán token — BEiT báo cáo giảm 1.8% khi tái tạo pixel với ViT-B.

## 5.2 So sánh với Supervised Pre-training

| Model | Params | MAE (IN1K) | Supervised (IN1K) | Supervised (JFT-300M) |
|-------|--------|------------|-------------------|------------------------|
| ViT-B/16 | 86M | 83.6% | 82.3% | ~85% |
| ViT-L/16 | 307M | 85.9% | 82.6% | 87.1% |
| ViT-H/14 | 632M | 86.9% | 83.1% | 88.5% |

*Figure 8: MAE pre-training vs supervised pre-training, evaluated by fine-tuning in ImageNet-1K. MAE có thể generalize tốt hơn: gain so với training from scratch lớn hơn với model có dung lượng cao hơn.*

Xu hướng này tương tự JFT-300M supervised pre-training. Điều này cho thấy MAE có thể giúp **scale up model sizes**.

## 5.3 Partial Fine-tuning

Linear probing và fine-tuning results phần lớn **không tương quan**. Linear probing bỏ lỡ cơ hội khai thác các đặc trưng mạnh nhưng phi tuyến — vốn là thế mạnh của deep learning.

| # Blocks Fine-tuned | MAE | MoCo v3 |
|---------------------|------|--------|
| 0 (linear probing) | 73.5% | 77.6% |
| 1 | 81.0% | 79.9% |
| 4 | 84.2% | 81.6% |
| 12 | 84.7% | 83.8% |
| 24 (full) | 84.9% | 84.1% |

*Figure 9: Partial fine-tuning results of ViT-L. Tuning 0 blocks = linear probing; 24 = full fine-tuning. MAE representations ít linearly separable, nhưng consistently better khi ≥1 blocks được tuned.*

Đặc biệt, fine-tune chỉ **một Transformer block** tăng accuracy từ 73.5% lên **81.0%**. Fine-tune "nửa" block cuối (MLP sub-block) đạt 79.1% — tốt hơn nhiều linear probing.

So với MoCo v3: MoCo v3 có linear probing cao hơn, nhưng **tất cả partial fine-tuning results của MoCo v3 đều thua MAE**. MAE representations ít linearly separable hơn, nhưng là **non-linear features mạnh hơn**.

> [!NOTE] Suy luận thêm
> Linear separability không phải thước đo duy nhất cho representation quality. Linear probing không tương quan tốt với transfer learning performance.

---

# 6. Transfer Learning

## 6.1 Object Detection & Segmentation (COCO)

Fine-tune Mask R-CNN end-to-end trên COCO với ViT backbone + FPN.

| Method | Pre-train Data | ViT-B APbox | ViT-L APbox | ViT-B APmask | ViT-L APmask |
|--------|---------------|-------------|-------------|--------------|--------------|
| Supervised | IN1K w/ labels | 47.9 | 49.3 | 42.9 | 43.9 |
| MoCo v3 | IN1K | 47.9 | 49.3 | 42.7 | 44.0 |
| BEiT | IN1K+DALLE | 49.8 | 53.3 | 44.4 | 47.1 |
| **MAE** | IN1K | **50.3** | **53.3** | **44.9** | **47.2** |

MAE outperforms supervised pre-training **4.0 points** với ViT-L (53.3 vs 49.3). Pixel-based MAE tốt hơn hoặc ngang token-based BEiT, trong khi đơn giản và nhanh hơn nhiều.

## 6.2 Semantic Segmentation (ADE20K)

UperNet trên ADE20K.

| Method | ViT-B mIoU | ViT-L mIoU |
|--------|------------|------------|
| Supervised | 47.4 | 49.9 |
| MoCo v3 | 47.3 | 49.1 |
| BEiT | 47.1 | 53.3 |
| **MAE** | **48.1** | **53.6** |

MAE outperforms supervised pre-training **3.7 points** với ViT-L.

## 6.3 Classification Tasks

| Dataset | ViT-B | ViT-L | ViT-H | ViT-H448 | Previous Best |
|---------|-------|-------|-------|----------|---------------|
| iNat 2017 | 70.5% | 75.7% | 79.3% | **83.4%** | 75.4%* |
| iNat 2018 | 75.4% | 80.1% | 83.0% | **86.8%** | 81.2%* |
| Places205 | 63.9% | 65.8% | 65.9% | **66.8%** | 66.0%† |
| Places365 | 57.9% | 59.4% | 59.8% | **60.3%** | 58.0%‡ |

*: previous best | †: pre-trained on 1B images | ‡: pre-trained on 3.5B images

MAE cho thấy **strong scaling behavior** trên iNaturalists. Kết quả vượt previous best (được pre-train trên hàng tỉ ảnh) bằng margins lớn.

## 6.4 Pixels vs Tokens

| Target | IN1K (ViT-B/L/H) | COCO (ViT-B/L) | ADE20K (ViT-B/L) |
|--------|------------------|----------------|------------------|
| Pixel (w/ norm) | 83.6/85.9/86.9 | 50.3/53.3 | 48.1/53.6 |
| dVAE token | 83.6/85.7/86.9 | 50.3/53.2 | 48.1/53.4 |
| **Δ (token - pixel)** | 0.0/-0.2/0.0 | 0.0/-0.1 | 0.0/-0.2 |

**Tokenization không cần thiết** cho MAE — sự khác biệt thống kê không đáng kể.

---

# 7. Thảo luận và Kết luận (Discussion and Conclusion)

**Thuật toán đơn giản nhưng scale tốt là cốt lõi của deep learning.** Trong NLP, các phương pháp self-supervised đơn giản (GPT, BERT) mang lại lợi ích từ việc scale model theo hàm mũ. Trong vision, pre-training thực tế vẫn chủ yếu là supervised dù có tiến bộ trong self-supervised learning. Trong nghiên cứu này, tác giả quan sát trên ImageNet và transfer learning rằng autoencoder — một phương pháp self-supervised đơn giản tương tự NLP — **cung cấp lợi ích scalable**. Self-supervised learning trong vision có thể đang bắt đầu đi theo quỹ đạo tương tự NLP.

Mặt khác, ảnh và ngôn ngữ là tín hiệu có bản chất khác nhau và sự khác biệt này phải được giải quyết cẩn thận. Ảnh chỉ là ánh sáng được ghi lại, không có phân rã ngữ nghĩa thành từ. Thay vì cố gắng loại bỏ đối tượng, ta loại bỏ các patch ngẫu nhiên có thể không tạo thành phân đoạn ngữ nghĩa. Tương tự, MAE tái tạo pixel — không phải thực thể ngữ nghĩa. Tuy nhiên, ta quan sát rằng MAE suy luận **các reconstruction phức tạp, toàn thể**, gợi ý rằng nó đã học nhiều khái niệm thị giác, tức là ngữ nghĩa. Tác giả giả thuyết rằng hành vi này xảy ra thông qua **biểu diễn ẩn phong phú** bên trong MAE.

> [!NOTE] Suy luận thêm (quan điểm của tác giả ghi chú)
> MAE không cần trở thành mô hình sinh ảnh hoàn chỉnh. Mục tiêu là học latent đủ tốt để giảm lỗi có điều kiện trên patch bị che. Nhưng chính việc "điền vào thiếu" dưới che phủ cao tạo áp lực học cấu trúc **nguyên nhân–hệ quả** (bố cục/đối tượng) hơn là tương quan cục bộ. Vì vậy MAE tạo representation mạnh cho downstream, ngay cả khi reconstruction pixel không hoàn toàn khớp ground truth.

---

# 8. Ví dụ Reconstruction

## 8.1 ImageNet Validation — Figure 2 (Masking 80%)

> [!NOTE] Figure 2 Caption (Paper)
> Example results on ImageNet validation images. For each triplet, we show the masked image (left), MAE reconstruction (middle), and ground-truth (right). The masking ratio is 80%, leaving only 39 out of 196 patches.

**Example 1: Bird**
![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-002-010.jpg]] ![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-002-011.jpg]] ![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-002-012.jpg]]

**Example 2: Dog**
![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-002-013.jpg]] ![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-002-014.jpg]] ![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-002-015.jpg]]

**Example 3: Lighthouse**
![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-002-016.jpg]] ![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-002-017.jpg]] ![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-002-018.jpg]]

**Example 4: Orchid**
![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-002-019.jpg]] ![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-002-020.jpg]] ![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-002-021.jpg]]

**Example 5: Portrait**
![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-002-022.jpg]] ![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-002-023.jpg]] ![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-002-024.jpg]]

**Example 6: Elephant**
![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-002-025.jpg]] ![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-002-026.jpg]] ![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-002-027.jpg]]

**Example 7: Lizard**
![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-002-028.jpg]] ![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-002-029.jpg]] ![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-002-030.jpg]]

**Example 8: Outdoor Scene**
![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-002-031.jpg]] ![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-002-032.jpg]] ![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-002-033.jpg]]

*Note: As no loss is computed on visible patches, the model output on visible patches is qualitatively worse. One can simply overlay the output with the visible patches to improve visual quality.*

## 8.2 COCO Validation — Figure 3 (Zero-shot transfer)

> [!NOTE] Figure 3 Caption (Paper)
> Example results on COCO validation images, using an MAE trained on ImageNet (the same model weights as in Figure 2). Observe the reconstructions on the right-most examples, which, although different from the ground truth, are semantically plausible.

**Example 1**
![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-003-034.jpg]] ![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-003-035.jpg]] ![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-003-036.jpg]]

**Example 2**
![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-003-043.jpg]] ![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-003-044.jpg]] ![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-003-045.jpg]]

**Example 3**
![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-003-046.jpg]] ![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-003-047.jpg]] ![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-003-048.jpg]]

**Example 4 (Semantically Plausible)**
![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-003-049.jpg]] ![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-003-050.jpg]] ![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-003-051.jpg]]

**Example 5 (Semantically Plausible)**
![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-003-052.jpg]] ![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-003-053.jpg]]

*Reconstruction khác với ground truth nhưng "hợp nghĩa" (semantically plausible) — cho thấy MAE đã học semantics, không chỉ copy pixels.*

## 8.3 Higher Masking Ratios — Figure 4

> [!NOTE] Figure 4 Caption (Paper)
> Reconstructions of ImageNet validation images using an MAE pre-trained with a masking ratio of 75% but applied on inputs with higher masking ratios. The predictions differ plausibly from the original images, showing that the method can generalize.

**Row 1**
| Original | Mask 75% | Mask 85% | Mask 95% |
|----------|----------|----------|----------|
| ![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-003-037.jpg]] | ![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-003-038.jpg]] | ![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-003-039.jpg]] | ![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-003-040.jpg]] |

**Row 2**
| Original | Mask 75% | Mask 85% | Mask 95% |
|----------|----------|----------|----------|
| ![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-003-041.jpg]] | ![[assets/attachments/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022/fig-003-042.jpg]] | - | - |

*Khi masking ratio tăng từ 75% → 95%, predictions vẫn plausible dù khác original — chứng minh MAE học holistic understanding, không chỉ local interpolation.*

---

# 9. Tóm tắt các Contributions

1. **High masking ratio (75%)**: Khác biệt quan trọng với BERT (15%), giải quyết vấn đề redundancy của ảnh.

2. **Asymmetric encoder-decoder**: Encoder không thấy mask token → tránh distribution mismatch, tăng tốc 3-4×.

3. **Pixel reconstruction works**: Không cần tokenizer phức tạp (dVAE), normalized pixel đủ tốt.

4. **Minimal augmentation**: Random masking đóng vai trò augmentation, không cần color jitter.

5. **Strong scaling behavior**: ViT-H448 đạt 87.8% trên ImageNet-1K, vượt all previous methods using only IN1K data.

6. **Superior transfer learning**: Outperforms supervised pre-training on COCO (+4.0 APbox) and ADE20K (+3.7 mIoU).

---

# 10. Liên kết tới các Concept Notes

- [[Masked Autoencoders (MAE)]] — Concept note tổng hợp về MAE
- [[Vision Transformers (ViT)]] — Kiến trúc encoder
- [[Self-Supervised Learning (Computer Vision)]] — Bối cảnh phương pháp
- [[Autoencoders]] — Họ phương pháp gốc
- [[Denoising Autoencoders]] — Ancestor trực tiếp
- [[BERT]] — Nguồn cảm hứng từ NLP
- [[GPT]] — Phương pháp autoregressive trong NLP
- [[Contrastive Learning]] — Nhánh SSL khác (để so sánh)
- [[BEiT]] — Phương pháp masked prediction dùng tokens
- [[Linear Probing]] — Cách đánh giá representation
- [[Fine-Tuning (Transfer Learning)]] — Cách đánh giá khác
- [[Positional Embedding]] — Thành phần kỹ thuật quan trọng
- [[Mean Squared Error]] — Loss function
- [[ImageNet]] — Dataset và benchmark
- [[Data Augmentation]] — So sánh với contrastive methods
- [[Transfer Learning]] — Kết quả downstream tasks

---

# 11. Nguyên văn bài báo (English)

![[30_Resources/Papers/Masked Autoencoders Are Scalable Vision Learners (Original)]]
