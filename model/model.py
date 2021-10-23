import torch
import numpy as np
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
               context_len: int,
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

    self.context_len = context_len

    # TODO: figure out attention mask
    attn_mask = None

    # for now all transformers have same params
    self.fragment_transformer = Transformer(transformer_width, transformer_heads, transformer_layers, attn_mask)
    self.pre_fragment_transformer = Transformer(transformer_width, transformer_heads, transformer_layers, attn_mask)
    self.post_fragment_transformer = Transformer(transformer_width, transformer_heads, transformer_layers, attn_mask)

    self.simple_mlp = nn.Sequential(*[
      nn.Linear(2 * embed_dim, 4 * embed_dim),
      nn.ReLU(),
      nn.Linear(4 * embed_dim, embed_dim)
    ])

    # Assuming Embedding and Text Projection should be the same for all transformers
    self.embedding = nn.Embedding(vocab_size, transformer_width)

    # TODO: figure out where to use this (+more) LayerNorm in encode func
    self.layer_norm_frag = nn.LayerNorm(transformer_width)
    self.layer_norm_pre = nn.LayerNorm(transformer_width)
    self.layer_norm_post = nn.LayerNorm(transformer_width)

    self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))
    self.text_proj = nn.Parameter(torch.empty(transformer_width, embed_dim))
    nn.init.normal_(self.text_proj, std=transformer_width ** (-0.5))
    self.positional_embedding = nn.Parameter(torch.empty(context_len, transformer_width))
    nn.init.normal_(self.positional_embedding, std=0.01)

  def encode_text(self, text_tokens, transformer, layer_norm):
    '''
      Running the assumption that embedding and text projection should be the same for all transformers
    '''
    x = self.embedding(text_tokens)
    x = x + self.positional_embedding

    x = x.permute(1, 0, 2)
    x = transformer(x)
    x = x.permute(1, 0, 2)
    x = layer_norm(x)

    # TODO: why do we do this?
    x = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)] @ self.text_proj
    return x

  def encode_context(self, pre_tokens, post_tokens):
    '''
      Very simple model for mixing pre fragment info with post fragment info into the context:

      # 1. concatenate
      context = torch.cat(pre_embedding, post_embedding) 
      # 2. mix up with some linear model
      embedded_context = some_linear_model(context)
    '''

    pre_embedding = self.encode_text(pre_tokens, self.pre_fragment_transformer, self.layer_norm_pre)
    post_embedding = self.encode_text(post_tokens, self.post_fragment_transformer, self.layer_norm_post)

    context = torch.cat((pre_embedding, post_embedding), dim=1)
    embedded_context = self.simple_mlp(context)

    return embedded_context

  def encode_fragment(self, fragment_tokens):
    return self.encode_text(fragment_tokens, self.fragment_transformer, self.layer_norm_frag)

  def forward(self, pre_fragment, fragment, post_fragment):
    '''
      pre_fragment - script before fragment
      post_fragment - script after fragment
      fragment - the fragment
    '''

    frag_features = self.encode_fragment(fragment)
    context_features = self.encode_context(pre_fragment, post_fragment)

    frag_features = frag_features / frag_features.norm(dim=-1, keepdim=True)
    context_features = context_features / context_features.norm(dim=-1, keepdim=True)

    logit_scale = self.logit_scale.exp()
    logits_per_context = logit_scale * context_features @ frag_features.t()
    logits_per_fragment = logits_per_context.t()

    return logits_per_context, logits_per_fragment
