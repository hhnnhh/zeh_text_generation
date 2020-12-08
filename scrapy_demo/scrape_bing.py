
# be careful when scraping BING, it might not be allowed
# code found here: https://stackoverflow.com/questions/63119210/no-results-in-scraping-bing-search
# 8/12/2020
# pip install bs4 (beautifulsoup latest release)
import requests
from bs4 import BeautifulSoup

term = 'Anschreiben+Bewerbung+Mustertexte'
url = 'https://www.bing.com/search?q={}&setlang=en-us'.format(term)
headers = {'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}
response = requests.get(url,headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')
print(soup.prettify())

soup = BeautifulSoup(response.text, 'lxml')

print(f'HTML: {soup.h2}, name: {soup.h2.name}, text: {soup.h2.text}')
print(soup.h2.text)
