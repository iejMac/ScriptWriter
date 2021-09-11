import os
import sys
from bs4 import BeautifulSoup
from urllib.parse import quote
from urllib.request import Request, urlopen, urlretrieve

def get_script_data(data_dir):
  r = "https://imsdb.com"
  all_scripts = "https://imsdb.com/all-scripts.html"

  req = Request(all_scripts)
  html_page = urlopen(req)
  soup = BeautifulSoup(html_page, 'html.parser')
  movie_links = []
  for link in soup.findAll('a'):
    l = quote(link.get('href'))
    if "Script" in l:
      movie_links.append(r+l)

  movie_links = movie_links[5:] # first 5 are from little sidebar recommenation thing (will repeat)
  movie_count = 0
  error_count = 0
  scripts = []

  for movie_link in movie_links:
    movie_count += 1
    req = Request(movie_link)
    html_page = urlopen(req)
    soup = BeautifulSoup(html_page, 'html.parser')
    script_links = []
    for link in soup.findAll('a'):
      l = link.get('href')
      if l is not None and "scripts" in l and "all-scripts" not in l:
        script_links.append(r+l)

    if len(script_links) == 0:
      continue

    script_link = script_links[0]
    name = script_link.split("/")[-1]
    try:
      urlretrieve(script_link, os.path.join(data_dir, name))
    except:
      print(f"Link failed: {script_link}")
      error_count += 1
      print(f"{error_count}/{movie_count} links failed")

if __name__ == "__main__":
  if len(sys.argv) > 1:
    get_script_data(sys.argv[1])
  else:
    print("Usage: python get_html.py data_dir")
