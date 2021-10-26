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

data_dir = "../data/data_dir"

VOCAB_SIZE = 30526
BATCH_SIZE = 500
EPOCHS = 20
SEQ_LEN = 50

mod = model.CSFP(800, VOCAB_SIZE, SEQ_LEN, 500, 10, 10)
opt = optim.Adam(mod.parameters(), lr=3e-4,betas=(0.9,0.98),eps=1e-6,weight_decay=0.0)
loss_context = torch.nn.CrossEntropyLoss()
loss_fragment = torch.nn.CrossEntropyLoss()

encoded_pres = torch.load(os.path.join(data_dir, "pres", "pre1.pt"))
encoded_frags = torch.load(os.path.join(data_dir, "frags", "frag1.pt"))
encoded_posts = torch.load(os.path.join(data_dir, "posts", "post1.pt"))

for e in range(EPOCHS):
	opt.zero_grad()

	lpc, lpf = mod(encoded_pres, encoded_frags, encoded_posts)
	ground_truth = torch.arange(BATCH_SIZE)

	loss = (loss_context(lpc, ground_truth) + loss_fragment(lpf, ground_truth))/2
	print(loss)
	loss.backward()
	opt.step()










