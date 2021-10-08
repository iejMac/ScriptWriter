import os
import sys
from bs4 import BeautifulSoup

'''
  Separating getting HTML and parsing it for now since I'm not certain this is how we want to do it in the end
  Lack of a unified format might cause problems later
'''

def parse_html(data_dir, out_dir):
  all_html = os.listdir(data_dir)
  for html_name in all_html:
    with open(os.path.join(data_dir, html_name), "r") as html:
      soup = BeautifulSoup(html, "html.parser")

    no_tag = soup.get_text()
    
    with open(os.path.join(out_dir, html_name.split(".")[0] + ".txt"), "w+") as f:
      f.write(no_tag)

if __name__ == "__main__":
  if 4 > len(sys.argv) > 2:
    parse_html(sys.argv[1], sys.argv[2])
  else:
    print("Usage: python get_html.py html_dir out_dir")
