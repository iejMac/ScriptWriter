import os
import torch
from torch.nn.functional import one_hot
from tokenizers import BertWordPieceTokenizer

class ScriptTokenizer:
  def __init__(self,
               vocab_file="bert-base-uncased-vocab.txt",
               sequence_length=100):


    self.tokenizer = BertWordPieceTokenizer(vocab_file, lowercase=True)
    self.tokenizer.add_special_tokens(["|<TAB>|", "|<NL>|"])
    self.sequence_length = sequence_length

  def get_vocab_size(self):
    return self.tokenizer.get_vocab_size()

  def encode(self, texts):
    '''
      Encodes raw text into indices for data

      data = data.replace("    ", "\t") # 4 space tabs
      data = data.replace("  ", "\t") # 4 space tabs
    '''

    texts = [texts] if isinstance(texts, str) else texts
    for i in range(len(texts)):
      texts[i] = texts[i].replace("\t", "|<TAB>|") # Tab structure gives a lot of information
      texts[i] = texts[i].replace("\n\n", "|<NL>|") # Empty lines in script

    tokenized_texts = []

    for text in texts:
      tokenized = self.tokenizer.encode(text)
      ids = tokenized.ids[:self.sequence_length] # cap sequence length at sequence_length tokens
      while len(ids) < self.sequence_length:
        ids.append(0)
      tokenized_texts.append(ids)
    return torch.tensor(tokenized_texts)

if __name__ == "__main__":
  data_dir = "./out_dir"
  test_script_name = os.path.join(data_dir, os.listdir(data_dir)[1])

  with open(test_script_name, "r") as f:
    test = f.read()

  tok = ScriptTokenizer()

  test = test[500:2000]
  test = test.replace("    ", "\t")

  encoded = tok.encode(test)
  print(encoded.shape)
