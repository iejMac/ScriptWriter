import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers import BertWordPieceTokenizer
from tokenizers.trainers import BpeTrainer
from tokenizers import pre_tokenizers

data_dir = "./out_dir"
test_script_name = os.path.join(data_dir, os.listdir(data_dir)[0])
# test_script_name = os.path.join(data_dir, os.listdir(data_dir)[1])
print(test_script_name)

with open(test_script_name, "r") as f:
  test = f.read()

# data = [test[500:2000]]
# data = [test[500:200000]]
data = [test, test, test, test, test, test, test, test, test, test]
# data = data[0]
# data = data.replace("    ", "\t") # 4 space tabs
# data = data.replace("  ", "\t") # 4 space tabs
data = data[0].replace("\t", "|<TAB>|")
data = data[0].replace("\n", "|<NL>|")
# print(repr(data))

'''
tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(vocab_size=1000, special_tokens=["|<TAB>|", "|<NL>|"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet())

tokenizer.train_from_iterator(data, trainer=trainer)
'''

tokenizer = BertWordPieceTokenizer("bert-base-uncased-vocab.txt", lowercase=True)
tokenizer.add_special_tokens(["|<TAB>|", "|<NL>|"])

to_encode = test[500:1000]
to_encode = to_encode.replace("\t", "|<TAB>|")
print(to_encode)

# test = tokenizer.encode("|<TAB>||<NL>| cringe what when why, the")
test = tokenizer.encode(to_encode)
print(test.tokens)






