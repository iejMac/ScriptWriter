import os
from bs4 import BeautifulSoup

data_dir = "test"

scripts = os.listdir(data_dir)
script_path = os.path.join(data_dir, scripts[2])
print(scripts[2])

with open(script_path, "r") as html:
  soup = BeautifulSoup(html, "html.parser")

string = soup.get_text()

with open("text.txt", "w+") as f:
  f.write(string)

