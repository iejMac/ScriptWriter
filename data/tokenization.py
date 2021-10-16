import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers import BertWordPieceTokenizer
from tokenizers.trainers import BpeTrainer
from tokenizers import pre_tokenizers


class ScriptTokenizer:
  def __init__(self,
               vocab_file="bert-base-uncased-vocab.txt"):


    self.tokenizer = BertWordPieceTokenizer(vocab_file, lowercase=True)
    self.tokenizer.add_special_tokens(["|<TAB>|", "|<NL>|"])

  def encode(self, text):
    '''
      Encodes raw text into indices for data

      Maybe some options for Karan and Pau:
      data = data.replace("    ", "\t") # 4 space tabs
      data = data.replace("  ", "\t") # 4 space tabs
    '''
    text = text.replace("\t", "|<TAB>|") # Tab structure gives a lot of information
    text = text.replace("\n\n", "|<NL>|") # Prounounced breaks in script structure

    tokenized = self.tokenizer.encode(text)
    return tokenized.ids



if __name__ == "__main__":
  data_dir = "./out_dir"
  test_script_name = os.path.join(data_dir, os.listdir(data_dir)[0])
  # test_script_name = os.path.join(data_dir, os.listdir(data_dir)[1])
  print(test_script_name)

  with open(test_script_name, "r") as f:
    test = f.read()

  test = test[500:1000]

  tok = ScriptTokenizer()

  print(tok.encode(test))


  






