# 🎯 MAE Implementation Guide - Cấu Trúc Chuẩn

> Hướng dẫn chi tiết tự tái tạo Masked Autoencoder với cấu trúc project chuẩn

---

## 📁 Cấu Trúc Project Đề Xuất

```
mae-project/
├── configs/                    # Cấu hình training
│   ├── pretrain_base.yaml
│   ├── pretrain_large.yaml
│   └── finetune.yaml
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── patch_embed.py      # Patch Embedding
│   │   ├── pos_embed.py        # Positional Embedding
│   │   ├── attention.py        # Multi-Head Attention
│   │   ├── transformer.py      # Transformer Block
│   │   ├── encoder.py          # MAE Encoder (ViT)
│   │   ├── decoder.py          # Lightweight Decoder
│   │   └── mae.py              # Full MAE Model
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py          # Dataset classes
│   │   └── transforms.py       # Data augmentation
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py          # Training loop
│   │   ├── scheduler.py        # LR schedulers
│   │   └── loss.py             # Loss functions
│   └── utils/
│       ├── __init__.py
│       ├── checkpoint.py       # Save/load models
│       ├── logging.py          # TensorBoard logging
│       └── visualization.py    # Reconstruction viz
├── scripts/
│   ├── pretrain.py             # Pre-training entry
│   ├── finetune.py             # Fine-tuning entry
│   ├── evaluate.py             # Evaluation
│   └── visualize.py            # Demo visualization
├── tests/
│   ├── test_patch_embed.py
│   ├── test_masking.py
│   └── test_model.py
├── notebooks/
│   └── demo.ipynb              # Interactive demo
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🔧 Bước 1: Setup Project

```bash
# Tạo project
mkdir -p mae-project && cd mae-project

# Tạo virtual environment
python -m venv venv && source venv/bin/activate

# Cài dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install einops timm tensorboard pyyaml matplotlib

# Tạo structure
mkdir -p src/{models,data,training,utils} configs scripts tests notebooks
touch src/__init__.py src/models/__init__.py src/data/__init__.py
touch src/training/__init__.py src/utils/__init__.py
```

---

## 🧩 Bước 2: Implement Từng Component

### 2.1 Patch Embedding (`src/models/patch_embed.py`)

```python
"""Patch Embedding - Chia ảnh thành patches và embed."""
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    
    Công thức: (B, C, H, W) -> (B, num_patches, embed_dim)
    Với H=W=224, patch_size=16: num_patches = 14*14 = 196
    """
    def __init__(
        self, 
        img_size: int = 224, 
        patch_size: int = 16, 
        in_chans: int = 3, 
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Conv2d với kernel=stride=patch_size = linear projection per patch
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input images
        Returns:
            (B, num_patches, embed_dim) patch embeddings
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input size ({H}x{W}) doesn't match model ({self.img_size}x{self.img_size})"
        
        # (B, C, H, W) -> (B, embed_dim, H/P, W/P) -> (B, embed_dim, num_patches)
        x = self.proj(x)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


# Test
if __name__ == "__main__":
    pe = PatchEmbed(img_size=224, patch_size=16, embed_dim=768)
    x = torch.randn(2, 3, 224, 224)
    out = pe(x)
    assert out.shape == (2, 196, 768)
    print(f"✓ PatchEmbed: {x.shape} -> {out.shape}")
```

---

### 2.2 Positional Embedding (`src/models/pos_embed.py`)

```python
"""Sinusoidal 2D Positional Embedding - Fixed, không học."""
import numpy as np
import torch

def get_2d_sincos_pos_embed(
    embed_dim: int, 
    grid_size: int, 
    cls_token: bool = False
) -> np.ndarray:
    """
    Tạo 2D sinusoidal positional embedding.
    
    Args:
        embed_dim: Embedding dimension (phải chia hết cho 2)
        grid_size: Số patches mỗi chiều (14 cho 224/16)
        cls_token: Thêm position cho CLS token hay không
    
    Returns:
        pos_embed: (grid_size^2, embed_dim) hoặc (1+grid_size^2, embed_dim)
    """
    # Tạo grid 2D
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # (2, grid_size, grid_size)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    
    # Embed mỗi chiều với sin-cos
    pos_embed = _get_2d_sincos_from_grid(embed_dim, grid)
    
    if cls_token:
        # CLS token có position embedding = 0
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    
    return pos_embed


def _get_2d_sincos_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """Embed 2D grid với sin-cos."""
    assert embed_dim % 2 == 0
    
    # Nửa dimension cho mỗi chiều
    emb_h = _get_1d_sincos(embed_dim // 2, grid[0].flatten())
    emb_w = _get_1d_sincos(embed_dim // 2, grid[1].flatten())
    
    return np.concatenate([emb_h, emb_w], axis=1)


def _get_1d_sincos(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """1D sinusoidal embedding."""
    assert embed_dim % 2 == 0
    
    # Frequencies: 1/10000^(2i/d) for i in [0, d/2)
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= (embed_dim / 2.)
    omega = 1. / (10000 ** omega)
    
    # Outer product: pos x omega
    out = np.outer(pos, omega)
    
    # Sin, Cos
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


# Test
if __name__ == "__main__":
    pos = get_2d_sincos_pos_embed(embed_dim=768, grid_size=14, cls_token=True)
    assert pos.shape == (197, 768)
    print(f"✓ PositionalEmbed: shape = {pos.shape}")
```

---

### 2.3 Attention (`src/models/attention.py`)

```python
"""Multi-Head Self-Attention."""
import torch
import torch.nn as nn

class Attention(nn.Module):
    """Multi-Head Self-Attention với qkv_bias."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Fused QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) input tokens
        Returns:
            (B, N, C) output tokens
        """
        B, N, C = x.shape
        
        # QKV projection: (B, N, 3*C) -> (B, N, 3, heads, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)
        
        # Attention: softmax(Q @ K^T / sqrt(d)) @ V
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Output projection
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


# Test
if __name__ == "__main__":
    attn = Attention(dim=768, num_heads=12)
    x = torch.randn(2, 197, 768)
    out = attn(x)
    assert out.shape == x.shape
    print(f"✓ Attention: {x.shape} -> {out.shape}")
```

---

### 2.4 Transformer Block (`src/models/transformer.py`)

```python
"""Transformer Block với Pre-Norm (LayerNorm trước Attention/MLP)."""
import torch
import torch.nn as nn
from .attention import Attention

class MLP(nn.Module):
    """MLP với GELU activation."""
    
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Pre-Norm Transformer Block."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop: float = 0.,
        attn_drop: float = 0.
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-Norm: LayerNorm trước mỗi sub-layer
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# Test
if __name__ == "__main__":
    block = TransformerBlock(dim=768, num_heads=12)
    x = torch.randn(2, 197, 768)
    out = block(x)
    assert out.shape == x.shape
    print(f"✓ TransformerBlock: {x.shape} -> {out.shape}")
```

---

### 2.5 Random Masking (`src/models/masking.py`)

```python
"""Random Masking Strategy cho MAE."""
import torch
from typing import Tuple

def random_masking(
    x: torch.Tensor, 
    mask_ratio: float = 0.75
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Random mask patches bằng shuffle + truncate.
    
    Thuật toán:
    1. Tạo random noise cho mỗi sample
    2. Argsort để lấy random permutation
    3. Giữ lại (1-mask_ratio) patches đầu tiên
    4. Lưu ids_restore để unshuffle sau
    
    Args:
        x: (B, L, D) patch embeddings
        mask_ratio: Tỷ lệ patches bị mask (default: 75%)
    
    Returns:
        x_masked: (B, L*(1-mask_ratio), D) visible patches
        mask: (B, L) binary mask, 1=masked, 0=visible
        ids_restore: (B, L) indices để unshuffle
    """
    B, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))
    
    # Random noise [0, 1] cho mỗi sample
    noise = torch.rand(B, L, device=x.device)
    
    # Sort: giá trị nhỏ -> giữ, giá trị lớn -> mask
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    
    # Giữ len_keep patches đầu tiên
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
    
    # Tạo binary mask (trong shuffled order, rồi unshuffle)
    mask = torch.ones([B, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    
    return x_masked, mask, ids_restore


# Test
if __name__ == "__main__":
    x = torch.randn(2, 196, 768)
    x_masked, mask, ids_restore = random_masking(x, mask_ratio=0.75)
    
    assert x_masked.shape == (2, 49, 768)  # 25% của 196
    assert mask.shape == (2, 196)
    assert mask.sum(dim=1).mean() == 147  # 75% masked
    print(f"✓ Masking: {x.shape} -> {x_masked.shape}, mask_ratio={mask.mean():.2%}")
```

---

### 2.6 Full MAE Model (`src/models/mae.py`)

```python
"""Masked Autoencoder - Full Model."""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

from .patch_embed import PatchEmbed
from .pos_embed import get_2d_sincos_pos_embed
from .transformer import TransformerBlock
from .masking import random_masking


class MaskedAutoencoder(nn.Module):
    """
    Masked Autoencoder với ViT backbone.
    
    Architecture:
        Encoder: ViT xử lý CHỈ visible patches (25%)
        Decoder: Lightweight Transformer xử lý FULL sequence
    
    Key Design:
        - Asymmetric: Encoder lớn, Decoder nhỏ
        - No mask token in encoder: Tránh distribution mismatch
        - Per-patch normalized pixel loss: Cải thiện representation
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        # Encoder config
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        # Decoder config (lightweight)
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        # Common
        mlp_ratio: float = 4.,
        norm_pix_loss: bool = True
    ):
        super().__init__()
        self.patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss
        
        # ===================== ENCODER =====================
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # CLS token (learnable)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embedding (fixed sinusoidal)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim),
            requires_grad=False
        )
        
        # Encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # ===================== DECODER =====================
        # Project encoder output -> decoder dimension
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        
        # Mask token (learnable, shared)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # Decoder positional embedding
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim),
            requires_grad=False
        )
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim, eps=1e-6)
        
        # Prediction head: predict pixels
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            patch_size ** 2 * in_chans
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        # Positional embeddings (sinusoidal, fixed)
        grid_size = int(self.patch_embed.num_patches ** 0.5)
        
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], grid_size, cls_token=True
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )
        
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], grid_size, cls_token=True
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )
        
        # Patch embedding: Xavier init
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        
        # Tokens: normal init
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        
        # Other layers
        self.apply(self._init_layer)
    
    def _init_layer(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Chuyển images thành patches.
        (B, C, H, W) -> (B, L, patch_size^2 * C)
        """
        p = self.patch_size
        B, C, H, W = imgs.shape
        assert H == W and H % p == 0
        
        h = w = H // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)  # (B, h, w, p, p, C)
        x = x.reshape(B, h * w, p * p * C)
        return x
    
    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Chuyển patches thành images.
        (B, L, patch_size^2 * C) -> (B, C, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        
        x = x.reshape(x.shape[0], h, w, p, p, 3)
        x = x.permute(0, 5, 1, 3, 2, 4)  # (B, C, h, p, w, p)
        imgs = x.reshape(x.shape[0], 3, h * p, w * p)
        return imgs
    
    def forward_encoder(
        self, x: torch.Tensor, mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encoder forward - chỉ xử lý visible patches.
        
        Returns:
            latent: (B, 1 + len_keep, embed_dim)
            mask: (B, num_patches)
            ids_restore: (B, num_patches)
        """
        # Patch embedding + positional embedding
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]  # Không có CLS position
        
        # Random masking
        x, mask, ids_restore = random_masking(x, mask_ratio)
        
        # Thêm CLS token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Encoder blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(
        self, x: torch.Tensor, ids_restore: torch.Tensor
    ) -> torch.Tensor:
        """
        Decoder forward - xử lý full sequence với mask tokens.
        
        Returns:
            pred: (B, num_patches, patch_size^2 * 3)
        """
        # Project to decoder dimension
        x = self.decoder_embed(x)
        
        # Append mask tokens
        num_mask = ids_restore.shape[1] + 1 - x.shape[1]
        mask_tokens = self.mask_token.repeat(x.shape[0], num_mask, 1)
        
        # Unshuffle: đưa mask tokens về đúng vị trí
        x_no_cls = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_no_cls = torch.gather(
            x_no_cls, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2])
        )
        x = torch.cat([x[:, :1, :], x_no_cls], dim=1)  # Add CLS back
        
        # Add decoder positional embedding
        x = x + self.decoder_pos_embed
        
        # Decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # Predict pixels
        x = self.decoder_pred(x)
        
        # Remove CLS token
        x = x[:, 1:, :]
        
        return x
    
    def forward_loss(
        self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MSE loss trên masked patches only.
        
        Args:
            imgs: (B, C, H, W) original images
            pred: (B, L, patch_size^2 * C) predicted patches
            mask: (B, L) binary mask, 1=masked
        
        Returns:
            loss: scalar
        """
        target = self.patchify(imgs)
        
        if self.norm_pix_loss:
            # Per-patch normalization (recommended)
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5
        
        # MSE loss
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # (B, L)
        
        # Only on masked patches
        loss = (loss * mask).sum() / mask.sum()
        
        return loss
    
    def forward(
        self, imgs: torch.Tensor, mask_ratio: float = 0.75
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.
        
        Returns:
            loss: reconstruction loss
            pred: predicted patches
            mask: binary mask
        """
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


# ==================== Model Configs ====================

def mae_vit_base_patch16(**kwargs):
    """ViT-Base: 86M params"""
    return MaskedAutoencoder(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        **kwargs
    )

def mae_vit_large_patch16(**kwargs):
    """ViT-Large: 307M params"""
    return MaskedAutoencoder(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        **kwargs
    )

def mae_vit_huge_patch14(**kwargs):
    """ViT-Huge: 632M params"""
    return MaskedAutoencoder(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        **kwargs
    )


# Test
if __name__ == "__main__":
    model = mae_vit_base_patch16()
    x = torch.randn(2, 3, 224, 224)
    loss, pred, mask = model(x)
    
    print(f"✓ MAE Forward Pass:")
    print(f"  Input: {x.shape}")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Pred: {pred.shape}")
    print(f"  Mask ratio: {mask.mean():.2%}")
    print(f"  Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
```

---

## 📝 Bước 3: Training Script (`scripts/pretrain.py`)

```python
"""MAE Pre-training Script."""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
sys.path.append('..')
from src.models.mae import mae_vit_base_patch16


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1.5e-4)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--output_dir', type=str, default='./output')
    return parser.parse_args()


def main(args):
    # Data
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    
    # Model
    model = mae_vit_base_patch16(norm_pix_loss=True).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    
    # Training
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for imgs, _ in loader:
            imgs = imgs.cuda()
            
            with torch.cuda.amp.autocast():
                loss, _, _ = model(imgs, mask_ratio=args.mask_ratio)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, f"{args.output_dir}/checkpoint_{epoch+1}.pth")


if __name__ == "__main__":
    main(get_args())
```

---

## 🎨 Bước 4: Visualization (`scripts/visualize.py`)

```python
"""MAE Reconstruction Visualization."""
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

import sys
sys.path.append('..')
from src.models.mae import mae_vit_base_patch16


def visualize(model, img_path, mask_ratio=0.75):
    # Load image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = transform(Image.open(img_path).convert('RGB')).unsqueeze(0).cuda()
    
    # Forward
    model.eval()
    with torch.no_grad():
        loss, pred, mask = model(img, mask_ratio)
    
    # Unpatchify
    pred_img = model.unpatchify(pred)
    
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
    img = img * std + mean
    pred_img = pred_img * std + mean
    
    # Create masked image
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_size**2 * 3)
    mask = model.unpatchify(mask)
    masked_img = img * (1 - mask)
    
    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img[0].permute(1, 2, 0).cpu().clamp(0, 1))
    axes[0].set_title('Original')
    axes[1].imshow(masked_img[0].permute(1, 2, 0).cpu().clamp(0, 1))
    axes[1].set_title(f'Masked ({mask_ratio:.0%})')
    axes[2].imshow(pred_img[0].permute(1, 2, 0).cpu().clamp(0, 1))
    axes[2].set_title('Reconstruction')
    axes[3].imshow(mask[0, 0].cpu(), cmap='gray')
    axes[3].set_title('Mask')
    
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('reconstruction.png', dpi=150)
    plt.show()
    print(f"Loss: {loss.item():.4f}")


if __name__ == "__main__":
    import sys
    model = mae_vit_base_patch16().cuda()
    # Load checkpoint nếu có
    # model.load_state_dict(torch.load('checkpoint.pth')['model'])
    visualize(model, sys.argv[1])
```

---

## ✅ Bước 5: Testing

```bash
# Test từng component
cd mae-project

# 1. Test PatchEmbed
python -c "from src.models.patch_embed import PatchEmbed; PatchEmbed()"

# 2. Test full model
python -c "
from src.models.mae import mae_vit_base_patch16
import torch
model = mae_vit_base_patch16()
x = torch.randn(2, 3, 224, 224)
loss, pred, mask = model(x)
print(f'Loss: {loss.item():.4f}')
print(f'Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M')
"

# 3. Run training (demo với CIFAR-10)
python scripts/pretrain.py --data_path /path/to/data --epochs 10 --batch_size 64
```

---

## 📊 Model Configurations

| Model | Embed | Depth | Heads | Decoder | Params |
|-------|-------|-------|-------|---------|--------|
| ViT-Base/16 | 768 | 12 | 12 | 512-d, 8 blocks | ~86M |
| ViT-Large/16 | 1024 | 24 | 16 | 512-d, 8 blocks | ~307M |
| ViT-Huge/14 | 1280 | 32 | 16 | 512-d, 8 blocks | ~632M |

---

## 🔗 References

- [Official MAE Repo](https://github.com/facebookresearch/mae)
- [Paper](https://arxiv.org/abs/2111.06377)
- [Colab Demo](https://colab.research.google.com/github/facebookresearch/mae/blob/main/demo/mae_visualize.ipynb)
