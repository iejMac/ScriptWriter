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

  def encode(self, text):
    '''
      Encodes raw text into indices for data

      data = data.replace("    ", "\t") # 4 space tabs
      data = data.replace("  ", "\t") # 4 space tabs
    '''
    text = text.replace("\t", "|<TAB>|") # Tab structure gives a lot of information
    text = text.replace("\n\n", "|<NL>|") # Empty lines in script

    tokenized = self.tokenizer.encode(text)
    ids = tokenized.ids[:self.sequence_length] # cap sequence length at 100 tokens
    while len(ids) < self.sequence_length:
      ids.append(0)

    return torch.tensor(ids)

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
