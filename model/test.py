import os
import sys
sys.path.append("../data")

import model
import torch
from tokenizer import ScriptTokenizer

'''
	ASSUMPTIONS:
		
		1. len(pre_frag) == len(frag) == len(post_frag)


'''

data_dir = "../data/out_dir"
test_script_name = os.path.join(data_dir, os.listdir(data_dir)[0])

with open(test_script_name, "r") as f:
	test = f.read()


PRE_LEN = 10
FRAG_LEN = 10
POST_LEN = 10

tok = ScriptTokenizer("../data/bert-base-uncased-vocab.txt", pre_length=PRE_LEN, frag_length=FRAG_LEN, post_length=POST_LEN)

test = test.replace("    ", "\t")
texts = [test, test]
encoded_pres, encoded_frags, encoded_posts = tok.encode(texts, None)

mod = model.CSFP(784, tok.get_vocab_size(), 10, 512, 2, 4)
lpc, lpf = mod(encoded_pres, encoded_frags, encoded_posts)

print(lpc)
print(lpf)








