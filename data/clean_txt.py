import os

def get_txt(script_name):
  with open(os.path.join(TXT_DIR, script_name), "r") as f:
    st = f.read()
  return st
  


TXT_DIR = "txt_dir"

scripts = os.listdir(TXT_DIR)
scripts = [get_txt(script_name) for script_name in scripts]

for script in scripts:
  print(len(script))












