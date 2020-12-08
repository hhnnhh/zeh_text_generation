#https://www.imagescape.com/blog/2018/08/20/scraping-pdf-doc-and-docx-scrapy/
#mkdir scrapy_demo
#cd scrapy_demo
# mkdir spiders

#pip install scrapy
#import scrapy
from scrapy.linkextractors import LinkExtractor
le = LinkExtractor()
le.deny_extensions
