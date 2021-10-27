import os
import sys
sys.path.append("../data")

import math
import torch
from torch import nn
from torch import optim
from tokenizer import ScriptTokenizer

import model

'''
	ASSUMPTIONS:
		
		1. len(pre_frag) == len(frag) == len(post_frag)


'''

if __name__ == "__main__":
  # TODO: also make this nice with argparse
  if len(sys.argv) != 2:
    print("Usage: python train.py data_dir")

  data_dir = sys.argv[1]
  data_chunks = len(os.listdir(os.path.join(data_dir, "frags")))

  VOCAB_SIZE = 30526
  BATCH_SIZE = 10
  EPOCHS = 20
  SEQ_LEN = 10

  mod = model.CSFP(800, VOCAB_SIZE, SEQ_LEN, 500, 10, 10)
  opt = optim.Adam(mod.parameters(), lr=3e-4,betas=(0.9,0.98),eps=1e-6,weight_decay=0.0)
  loss_context = torch.nn.CrossEntropyLoss()
  loss_fragment = torch.nn.CrossEntropyLoss()

  for e in range(EPOCHS):
    for chunk in range(data_chunks):
      encoded_pres = torch.load(os.path.join(data_dir, "pres", f"pre{chunk}.pt"))
      encoded_frags = torch.load(os.path.join(data_dir, "frags", f"frag{chunk}.pt"))
      encoded_posts = torch.load(os.path.join(data_dir, "posts", f"post{chunk}.pt"))

      batch_count = math.ceil(len(encoded_pres) / BATCH_SIZE)

      for batch in range(batch_count):
        b_pre = encoded_pres[batch * BATCH_SIZE : (batch+1) * BATCH_SIZE]
        b_frag = encoded_frags[batch * BATCH_SIZE : (batch+1) * BATCH_SIZE]
        b_post = encoded_posts[batch * BATCH_SIZE : (batch+1) * BATCH_SIZE]

        opt.zero_grad()

        lpc, lpf = mod(b_pre, b_frag, b_post)
        ground_truth = torch.arange(BATCH_SIZE)

        loss = (loss_context(lpc, ground_truth) + loss_fragment(lpf, ground_truth))/2
        print(loss)
        loss.backward()
        opt.step()
