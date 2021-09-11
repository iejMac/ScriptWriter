import re
from bs4 import BeautifulSoup
from urllib.parse import quote
from urllib.request import Request, urlopen

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

links = movie_links[5:] # first 5 are from little sidebar recommenation thing (will repeat)
scripts = []

for movie_link in movie_links:
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
  req = Request(script_link)
  html_page = urlopen(req)
  soup = BeautifulSoup(html_page, 'html.parser')

  test = soup.findAll('pre')
  if len(test) == 0:
    print(script_link)
    quit()



