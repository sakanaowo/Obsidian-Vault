# 🎯 Hướng Dẫn Tái Tạo Kiến Trúc MAE

> **Mục tiêu**: Tự tay implement Masked Autoencoder từ paper "Masked Autoencoders Are Scalable Vision Learners"

---

## Bước 1: Setup Environment

```bash
# 1. Tạo project folder
mkdir -p ~/Code/mae-reproduction && cd ~/Code/mae-reproduction

# 2. Tạo virtual environment
python -m venv venv
source venv/bin/activate

# 3. Cài dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install timm==0.3.2 tensorboard matplotlib numpy

# 4. Tạo cấu trúc project
mkdir -p mae data configs scripts tests
touch mae/__init__.py
```

---

## Bước 2: Clone Official Code (Reference)

```bash
# Clone để tham khảo
git clone https://github.com/facebookresearch/mae.git official_mae

# Download checkpoint để test
mkdir -p checkpoints
wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth -O checkpoints/mae_vit_base.pth
```

---

## Bước 3: Implement Patch Embedding

**File**: `mae/patch_embed.py`

```python
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """Chia ảnh thành patches và embed thành vectors."""
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Conv2d kernel=stride=patch_size = linear projection per patch
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # (B, 3, 224, 224) -> (B, 768, 14, 14) -> (B, 196, 768)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
```

**Test**:
```python
pe = PatchEmbed()
x = torch.randn(2, 3, 224, 224)
out = pe(x)
assert out.shape == (2, 196, 768), f"Expected (2, 196, 768), got {out.shape}"
print("✓ PatchEmbed OK")
```

---

## Bước 4: Implement Positional Embedding

**File**: `mae/pos_embed.py`

```python
import numpy as np
import torch

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """Tạo 2D sinusoidal positional embedding (fixed, không học)."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    
    emb_h = get_1d_sincos(embed_dim // 2, grid[0].reshape(-1))
    emb_w = get_1d_sincos(embed_dim // 2, grid[1].reshape(-1))
    emb = np.concatenate([emb_h, emb_w], axis=1)
    
    if cls_token:
        emb = np.concatenate([np.zeros([1, embed_dim]), emb], axis=0)
    return emb

def get_1d_sincos(embed_dim, pos):
    omega = np.arange(embed_dim // 2, dtype=np.float32) / (embed_dim / 2.)
    omega = 1. / 10000**omega
    out = np.outer(pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)
```

---

## Bước 5: Implement Random Masking

**File**: `mae/masking.py`

```python
import torch

def random_masking(x, mask_ratio=0.75):
    """
    Mask 75% patches ngẫu nhiên.
    
    Returns:
        x_masked: Chỉ visible patches (B, L*0.25, D)
        mask: Binary mask (B, L), 1=masked
        ids_restore: Để unshuffle về thứ tự gốc
    """
    B, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))
    
    # Random shuffle
    noise = torch.rand(B, L, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    
    # Keep first len_keep
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
    
    # Create binary mask
    mask = torch.ones([B, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, 1, ids_restore)
    
    return x_masked, mask, ids_restore
```

---

## Bước 6: Implement Transformer Block

**File**: `mae/transformer.py`

```python
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
```

---

## Bước 7: Implement Full MAE Model

**File**: `mae/model.py`

```python
import torch
import torch.nn as nn
from .patch_embed import PatchEmbed
from .pos_embed import get_2d_sincos_pos_embed
from .masking import random_masking
from .transformer import TransformerBlock

class MAE(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, depth=12, 
                 num_heads=12, decoder_dim=512, decoder_depth=8):
        super().__init__()
        
        # Encoder
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        
        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_dim), requires_grad=False)
        self.decoder_blocks = nn.ModuleList([TransformerBlock(decoder_dim, 16) for _ in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, patch_size**2 * 3)
        
        self.patch_size = patch_size
        self._init_weights()
    
    def _init_weights(self):
        pos = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**0.5), True)
        self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))
        dec_pos = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**0.5), True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(dec_pos).float().unsqueeze(0))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
    
    def forward(self, imgs, mask_ratio=0.75):
        # Encoder (visible patches only)
        x = self.patch_embed(imgs) + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = random_masking(x, mask_ratio)
        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        # Decoder (full sequence)
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1) + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        pred = self.decoder_pred(self.decoder_norm(x))[:, 1:, :]
        
        # Loss (masked patches only)
        target = self.patchify(imgs)
        loss = ((pred - target) ** 2).mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss, pred, mask
    
    def patchify(self, imgs):
        p = self.patch_size
        return imgs.reshape(imgs.shape[0], 3, imgs.shape[2]//p, p, imgs.shape[3]//p, p).permute(0,2,4,3,5,1).reshape(imgs.shape[0], -1, p*p*3)
```

---

## Bước 8: Training Script

**File**: `train.py`

```python
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from mae.model import MAE

# Config
EPOCHS = 100
BATCH_SIZE = 256
LR = 1.5e-4
MASK_RATIO = 0.75

# Data
transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# TODO: Thay bằng dataset của bạn
dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# Model
model = MAE().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)

# Train
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for imgs, _ in loader:
        imgs = imgs.cuda()
        loss, _, _ = model(imgs, MASK_RATIO)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}")
    
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f'checkpoints/mae_epoch{epoch+1}.pth')
```

---

## Bước 9: Visualization Script

**File**: `visualize.py`

```python
import torch
import matplotlib.pyplot as plt
from mae.model import MAE

def visualize_reconstruction(model, img):
    model.eval()
    with torch.no_grad():
        loss, pred, mask = model(img.unsqueeze(0).cuda())
    
    # Unpatchify
    p = model.patch_size
    pred = pred.reshape(1, 14, 14, p, p, 3).permute(0,5,1,3,2,4).reshape(1,3,224,224)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img.permute(1,2,0).cpu() * 0.225 + 0.45)
    axes[0].set_title('Original')
    axes[1].imshow((pred[0].permute(1,2,0).cpu() * 0.225 + 0.45).clamp(0,1))
    axes[1].set_title('Reconstruction')
    axes[2].imshow(mask[0].reshape(14,14).cpu(), cmap='gray')
    axes[2].set_title('Mask (75%)')
    plt.savefig('reconstruction.png')
    plt.show()
```

---

## Bước 10: Verify Implementation

```bash
# 1. Test forward pass
python -c "
import torch
from mae.model import MAE
model = MAE()
x = torch.randn(2, 3, 224, 224)
loss, pred, mask = model(x)
print(f'Loss: {loss.item():.4f}')
print(f'Pred shape: {pred.shape}')
print(f'Mask ratio: {mask.sum()/mask.numel():.2%}')
print('✓ All tests passed!')
"

# 2. Count parameters
python -c "
from mae.model import MAE
model = MAE()
params = sum(p.numel() for p in model.parameters())
print(f'Total params: {params/1e6:.1f}M')
"

# 3. Run training
python train.py
```

---

## Model Configurations

| Config | Patch | Embed Dim | Depth | Heads | Params |
|--------|-------|-----------|-------|-------|--------|
| ViT-Base/16 | 16 | 768 | 12 | 12 | ~86M |
| ViT-Large/16 | 16 | 1024 | 24 | 16 | ~307M |
| ViT-Huge/14 | 14 | 1280 | 32 | 16 | ~632M |

---

## Key Design Decisions

1. **Masking ratio 75%** — Cao hơn BERT (15%) vì ảnh có redundancy cao
2. **Asymmetric encoder-decoder** — Encoder lớn, decoder nhẹ (512-d, 8 blocks)
3. **No mask token in encoder** — Tránh distribution mismatch, tăng tốc 3-4×
4. **Pixel reconstruction** — Không cần tokenizer phức tạp
5. **Minimal augmentation** — Random crop + flip là đủ

---

## Tham Khảo

- **Official repo**: https://github.com/facebookresearch/mae
- **Paper**: https://arxiv.org/abs/2111.06377
- **Colab demo**: https://colab.research.google.com/github/facebookresearch/mae/blob/main/demo/mae_visualize.ipynb
