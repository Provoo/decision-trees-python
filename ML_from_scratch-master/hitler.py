"""
A simple implementation of k steps to hitler
Have planned to use breadth first search as the algoritjhm
x
"""
import bs4
from urllib import request
import re 

text = request.urlopen("https://en.wikipedia.org/wiki/Satan").read()
m_patt = r'^\/wiki\/.+\.(?!jpg|svg$)[^.]+$'
bs_text = bs4.BeautifulSoup(text, "lxml")

url_list = []

for link in bs_text.find_all('a'):
	url_list.append(str(link.get('href')))

#cleaning the urls 
url_list = [link for link in url_list if re.match(m_patt, link)]

print (url_list)
