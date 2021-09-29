import torch
from torch import nn

from collections import OrderedDict

class AttentionBlock(nn.Module):
  def __init__(self, dim: int, n_heads: int, mask: torch.Tensor = None):
    super().__init__()
    self.mask = mask
    self.attention = nn.MultiheadAttention(dim, n_heads)
    self.mlp = nn.Sequential(*[nn.Linear(dim, 4 * dim),
                               nn.ReLU(),
                               nn.Linear(4 * dim, dim)
                              ])

    self.layer_norm1 = nn.LayerNorm(dim) # paper: https://arxiv.org/pdf/1607.06450.pdf
    self.layer_norm2 = nn.LayerNorm(dim)

  def forward(self, x):
    self.mask = self.mask.to(x.device) if self.mask is not None else None
    x = self.layer_norm1(x)
    x = x + self.attention(x, x, x, need_weight=False, attn_mask=self.mask)
    x = self.mlp(self.layer_norm2(x))
    return x

class Transformer(nn.Module):
  def __init__(self, dim: int, layers: int, attn_heads: int, mask: torch.Tensor = None):
    super().__init__()
    self.blocks = nn.Sequential(*[AttentionBlock(dim, attn_heads, mask) for _ in range(layers)])

  def forward(self, x):
    return self.blocks(x)

# Contrastive Scene-Fragment Pre-training
class CSFP(nn.Module):
  def __init__(self, embed_dim):
    super().__init__()
    self.embed_dim = embed_dim
  def forward(self, pre_fragment, post_fragment, fragment):
    '''
      pre_fragment - script before fragment
      post_fragment - script after fragment
      fragment - the fragment
    '''
    return 0
