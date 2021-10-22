import os
import sys
sys.path.append("../data")

import model
import torch
from tokenizer import ScriptTokenizer

'''
mod = model.Transformer(5, 2, 1)
x = torch.randn((1, 2, 5))

out = mod(x)
print(out)
'''

data_dir = "../data/out_dir"
test_script_name = os.path.join(data_dir, os.listdir(data_dir)[1])

with open(test_script_name, "r") as f:
	test = f.read()

tok = ScriptTokenizer("../data/bert-base-uncased-vocab.txt")

test = test[500:2000]
test = test.replace("    ", "\t")

encoded = tok.encode(test)


TRANSFORMER_WIDTH = 512
emb = torch.nn.Embedding(tok.tokenizer.get_vocab_size(), TRANSFORMER_WIDTH)

print(len(encoded))

embedded = emb(encoded)
print(embedded[0])

print(embedded.shape)







