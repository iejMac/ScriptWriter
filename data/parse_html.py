import os
from bs4 import BeautifulSoup

data_dir = "test"

scripts = os.listdir(data_dir)
script_path = os.path.join(data_dir, scripts[5])

with open(script_path, "r") as html:
  soup = BeautifulSoup(html, "html.parser")

pre = soup.find("pre")

string = str(pre)
print(len(string))








