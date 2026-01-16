---
type: concept
title: Positional Embedding
aliases:
  - Positional Encoding
  - Position Embedding
tags:
  - ai
  - transformers
  - nlp
  - computer-vision
---

**Positional Embedding** là cơ chế để đưa thông tin **vị trí** vào các mô hình không có inductive bias về thứ tự, đặc biệt là [[Transformer Architecture]]. Vì self-attention xử lý các token như một **tập hợp** (set) chứ không phải chuỗi có thứ tự, nếu không có positional embedding, mô hình không thể phân biệt "The cat ate the fish" với "The fish ate the cat".

**Các loại Positional Embedding**

1. **Sinusoidal (fixed)**: Đề xuất trong paper gốc "Attention is All You Need" (Vaswani et al., 2017)
   $$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
   $$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$
   
   Ưu điểm: generalizes tốt cho độ dài không thấy trong training.

2. **Learned**: Học một embedding table cho mỗi vị trí. Sử dụng trong BERT, GPT, ViT.
   - ViT: Học một embedding cho mỗi vị trí patch (0, 1, ..., N-1)
   - Ưu điểm: Linh hoạt hơn, có thể học patterns phức tạp
   - Nhược điểm: Không generalizes cho độ dài vượt quá training

3. **Relative positional**: Encode khoảng cách giữa các token thay vì vị trí tuyệt đối. Sử dụng trong Transformer-XL, T5.

4. **RoPE (Rotary Position Embedding)**: Kết hợp absolute và relative bằng cách xoay (rotate) embedding vectors. Sử dụng trong LLaMA, GPT-NeoX.

**Positional Embedding trong Vision**

Trong [[Vision Transformers (ViT)]], ảnh được chia thành patches và mỗi patch được gán một positional embedding để mô hình biết "patch này ở đâu trong ảnh". Điều này quan trọng vì self-attention không có locality bias như CNN — nó có thể attend đến bất kỳ patch nào với chi phí như nhau.

**Positional Embedding trong MAE**

Trong [[Masked Autoencoders (MAE)]], positional embedding đóng vai trò đặc biệt quan trọng:

1. **Trong encoder**: Positional embedding được cộng vào patch embedding của **chỉ các patch nhìn thấy**. Vì encoder không thấy mask token, positional embedding giúp encoder biết "patch này ở vị trí nào trong ảnh gốc" dù thứ tự trong chuỗi đã bị xáo trộn (shuffle).

2. **Trong decoder**: Positional embedding được cộng vào **cả encoded patches và mask tokens**. Paper nhấn mạnh: "without this, mask tokens would have no information about their location in the image". Mask token là shared vector giống nhau cho tất cả vị trí — positional embedding là **nguồn duy nhất** để decoder biết "cần tái tạo pixel ở đâu".

**Tại sao cần positional embedding cho mask token?**

Nếu không có positional embedding, decoder nhận được:
- Encoded patches (có thông tin nội dung + vị trí)
- Mask tokens (shared vector, **không có thông tin vị trí**)

Decoder sẽ không biết mask token nào tương ứng với vùng nào của ảnh → không thể tái tạo đúng vị trí. Với positional embedding, mask token ở vị trí (3, 5) sẽ khác với mask token ở vị trí (7, 2), cho phép decoder dự đoán pixel phù hợp.
