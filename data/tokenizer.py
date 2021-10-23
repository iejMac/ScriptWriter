import os
import torch
import random
from torch.nn.functional import one_hot
from tokenizers import BertWordPieceTokenizer

class ScriptTokenizer:
  def __init__(self,
               vocab_file="bert-base-uncased-vocab.txt",
               pre_length=100,
               frag_length=100,
               post_length=100, 
               ):


    self.tokenizer = BertWordPieceTokenizer(vocab_file, lowercase=True)
    self.tokenizer.add_special_tokens(["|<TAB>|", "|<NL>|"])

    self.pre_length = pre_length
    self.frag_length = frag_length
    self.post_length = post_length

  def get_vocab_size(self):
    return self.tokenizer.get_vocab_size()

  def encode(self, texts, start_seq=0):
    '''
      Encodes raw text into indices for data

      texts: List(str) - list of Scripts in str form
             str - singular script in str form


      Potential pre-processing steps:
      data = data.replace("    ", "\t") # 4 space tabs
      data = data.replace("  ", "\t") # 4 space tabs
    '''

    texts = [texts] if isinstance(texts, str) else texts

    tokenized_pres = []
    tokenized_frags = []
    tokenized_posts = []

    for text in texts:

      ids = self.get_ids(text)
      start_ind = start_seq if start_seq is not None else random.randint(0, len(ids))

      # TODO: this gets rid of start/end tokens, maybe this is a problem????
      pre_ids = ids[start_ind:start_ind + self.pre_length]
      frag_ids = ids[start_ind + self.pre_length: start_ind + self.pre_length + self.frag_length]
      post_ids = ids[start_ind + self.pre_length + self.frag_length : start_ind + self.pre_length + self.frag_length + self.post_length]
      
      for prefix in ["pre", "frag", "post"]:
        while len(eval(prefix + "_ids")) < eval("self." + prefix + "_length"):
          eval(prefix + "_ids").append(0)
        eval("tokenized_" + prefix + "s").append(eval(prefix + "_ids"))

    return torch.tensor(tokenized_pres), torch.tensor(tokenized_frags), torch.tensor(tokenized_posts)

  def get_ids(self, text: str):
    text = text.replace("\t", "|<TAB>|") # Tab structure gives a lot of information
    text = text.replace("\n\n", "|<NL>|") # Empty lines in script
    return self.tokenizer.encode(text).ids
