import os
import sys
sys.path.append("../data")

import model
import torch
from torch import nn
from torch import optim
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

BATCH_SIZE = 10
EPOCHS = 50

tok = ScriptTokenizer("../data/bert-base-uncased-vocab.txt", pre_length=PRE_LEN, frag_length=FRAG_LEN, post_length=POST_LEN)

test = test.replace("    ", "\t")
texts = [test] * BATCH_SIZE

mod = model.CSFP(800, tok.get_vocab_size(), 10, 500, 10, 10)
opt = optim.Adam(mod.parameters(), lr=3e-4,betas=(0.9,0.98),eps=1e-6,weight_decay=0.0)
loss_context = torch.nn.CrossEntropyLoss()
loss_fragment = torch.nn.CrossEntropyLoss()

for e in range(EPOCHS):
	opt.zero_grad()
	encoded_pres, encoded_frags, encoded_posts = tok.encode(texts, None)

	lpc, lpf = mod(encoded_pres, encoded_frags, encoded_posts)
	ground_truth = torch.arange(BATCH_SIZE)

	loss = (loss_context(lpc, ground_truth) + loss_fragment(lpf, ground_truth))/2
	print(loss)
	loss.backward()
	opt.step()










