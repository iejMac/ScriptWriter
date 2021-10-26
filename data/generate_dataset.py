import os
import sys
import torch

from tokenizer import ScriptTokenizer

def get_next_data_num(data_dir):
  nums = [int(name[4:-3]) for name in os.listdir(os.path.join(data_dir, "frags"))]
  return 0 if len(nums) == 0 else max(nums) + 1
  
def generate_dataset(text_dir, data_dir, samples_per_script, fragment_length):
  '''
    Temporarily assuming pre_len == frag_len == pos_len


    TODO: need to make a 
    1. uniform samples_per_script option where it takes the fragment_length, script_length
    and calculates a samples_per_script that should give uniform coverage of the entire script
    2. complete samples_per_script option where it just divides the script into the appriopriate amount of
    samples one by one (no overlap)
  '''

  tok = ScriptTokenizer("bert-base-uncased-vocab.txt", pre_length=fragment_length, frag_length=fragment_length, post_length=fragment_length)

  script_names = os.listdir(text_dir)

  pre_dat = torch.zeros(len(script_names), samples_per_script, fragment_length).type(torch.int)
  frag_dat = torch.zeros(len(script_names), samples_per_script, fragment_length).type(torch.int)
  post_dat = torch.zeros(len(script_names), samples_per_script, fragment_length).type(torch.int)

  for i, script_name in enumerate(script_names):
    with open(os.path.join(text_dir, script_name), "r") as f:
      script = [f.read()] * samples_per_script

    encoded_pres, encoded_frags, encoded_posts = tok.encode(script, None) # encode with random location sampling
    pre_dat[i] = encoded_pres
    frag_dat[i] = encoded_frags
    post_dat[i] = encoded_posts
  
  pre_dat = pre_dat.reshape(-1, fragment_length)
  frag_dat = frag_dat.reshape(-1, fragment_length)
  post_dat = post_dat.reshape(-1, fragment_length)

  new_id = get_next_data_num(data_dir)
  torch.save(pre_dat, os.path.join(data_dir, "pres", "pre" + str(new_id) + ".pt"))
  torch.save(frag_dat, os.path.join(data_dir, "frags", "frag" + str(new_id) + ".pt"))
  torch.save(post_dat, os.path.join(data_dir, "posts", "post" + str(new_id) + ".pt"))

if __name__ == "__main__":
  # TODO: make this nice with argparse
  if len(sys.argv) == 5:
    generate_dataset(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
  else:
    print("Usage: python generate_dataset.py text_dir data_dir samples_per_script fragment_length")
