## 1. Khối VGG16 ban đầu (CNN backbone)
- 2 block đầu từ VGG16 pretrained on ImageNet
- Freeze || fine-tune
## 2. Khối Inception (tự tạo thủ công)
- Thiết kế các nhánh song song 
	- Conv2D 1x1
	- Conv2D (3x1) -> Conv2D (1x3)
	- Conv2D (2x1) -> Conv2D (1x5)
	- MaxPooling(3x3) -> Conv2D(1x1)
- Kết hợp các nhánh bằng `Concatenate`
## 3. Patch Embedding
- Chia output thành các **patch 5×5** (non-overlapping).
- Flatten + Dense projection → vector 16 chiều
- Kết quả: tensor `121×16` (nếu ảnh đầu ra là `56×56×512`).
## 4. 4 Transformer Encoder Blocks
- Mỗi Block:
	- LayerNormalization
	- Multihead Attention 
	- Skip Connection
	- MLP (2 Dense với GELU)
	- Skip Connection
(Có thể dùng ViT-Keras)
