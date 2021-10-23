import torch
from torch import nn


class ResAttentionBlock(nn.Module):
  def __init__(self, dim: int, n_heads: int, attention_mask: torch.Tensor = None):
    super().__init__()
    self.attention_mask = attention_mask
    self.multihead_attention = nn.MultiheadAttention(dim, n_heads)
    self.mlp = nn.Sequential(*[nn.Linear(dim, 4 * dim),
                               nn.ReLU(),
                               nn.Linear(4 * dim, dim)
                              ])

    self.layer_norm1 = nn.LayerNorm(dim) # paper: https://arxiv.org/pdf/1607.06450.pdf
    self.layer_norm2 = nn.LayerNorm(dim)

  def attention(self, x):
    self.attention_mask = self.attention_mask.to(dtype=x.dtype, device=x.device) if self.attention_mask is not None else None
    return self.multihead_attention(x, x, x, need_weights=False, attn_mask=self.attention_mask)[0]

  def forward(self, x):
    #TODO: figure out why layer_norm isn't after mlp and attention instead of before
    x = x + self.attention(self.layer_norm1(x))
    x = x + self.mlp(self.layer_norm2(x))
    return x

class Transformer(nn.Module):
  def __init__(self, dim: int, layers: int, attn_heads: int, mask: torch.Tensor = None):
    super().__init__()
    self.blocks = nn.Sequential(*[ResAttentionBlock(dim, attn_heads, mask) for _ in range(layers)])

  def forward(self, x: torch.Tensor):
    return self.blocks(x)

    # return self.blocks(x)

# Contrastive Scene-Fragment Pre-training
class CSFP(nn.Module):
  def __init__(self,
               embed_dim: int,
               vocab_size: int,
               transformer_width: int,
               transformer_heads: int,
               transformer_layers: int
               ):
    super().__init__()

    '''
      Initial structure:
        pre_transformer(pre_fragment) = encoded_pre_fragment
        post_transformer(post_fragment) = encoded_post_fragment
        frag_transformer(fragment) = encoded_fragment

        context = concat(encoded_pre_fragment, encoded_post_fragment)
        encoded_context = SOME_MODEL(context) # need to figure out what model

        cosine_similarity = encoded_context @ encoded_fragment
    '''

    '''
    #TODO: maybe diff params for pre/post and frag?
    self.pre_transformer = Transformer(
      dim=transformer_width,
      layers=transformer_layers,
      attn_heads=transformer_heads,
      mask=None # temporary
    )
    self.post_transformer = Transformer(
      dim=transformer_width,
      layers=transformer_layers,
      attn_heads=transformer_heads,
      mask=None # temporary
    )
    self.frag_transformer = Transformer(
      dim=transformer_width,
      layers=transformer_layers,
      attn_heads=transformer_heads,
      mask=None # temporary
    )

    # TODO: figure out better model
    self.some_model = nn.Sequential(*[
      nn.Linear(2 * embed_dim, 4 * embed_dim),
      nn.ReLU(),
      nn.Linear(4 * embed_dim, embed_dim)
    ])
    '''
    self.embed_dim = embed_dim

    attn_mask = None
    self.transformer = Transformer(transformer_width, transformer_heads, transformer_layers, attn_mask)

    self.embedding = nn.Embedding(vocab_size, transformer_width)

    # TODO: figure out where to use this (+more) LayerNorm in encode func
    self.layer_norm_final = nn.LayerNorm(transformer_width)

  def encode(self, pre_fragment, post_fragment, fragment):
    '''
    # TODO: tokenization here
    tokenized_pre_fragment = pre_fragment
    tokenized_post_fragment = post_fragment
    tokenized_fragment = fragment

    encoded_pre_fragment = self.pre_transformer(tokenized_pre_fragment)
    encoded_post_fragment = self.post_transformer(tokenized_post_fragment)
    encoded_fragment = self.frag_transformer(tokenized_fragment)

    # TODO: axis might be wrong
    context = torch.cat((encoded_pre_fragment, encoded_post_fragment), 0)
    encoded_context = self.some_model(context)
    return encoded_context, encoded_fragment
    '''

  def temp_encode(self, text_tokens):

    x = self.embedding(text_tokens)
    #TODO: positional encodings

    x = self.transformer(x)

    x = self.layer_norm_final(x)

    return x


    
    

  def forward(self, pre_fragment, post_fragment, fragment):
    '''
      pre_fragment - script before fragment
      post_fragment - script after fragment
      fragment - the fragment
    '''

