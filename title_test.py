import re
import urllib 
import requests
from urllib.request import Request,urlopen

link = 'https://seekingalpha.com/api/sa/combined/AMZN.xml'
session = requests.Session() #发送请求
html = session.get(link).content.decode('utf-8') 
print(html)

web_list = []
web_list_ori = []


web_start1 = [ta.start() for ta in re.finditer('<link>',html)]
web_end1 = [tm.start() for tm in re.finditer('</link>',html)] #前缀后缀必须唯一
num = len(web_start1)
for i in range(num):#循环查找多个跟上面前后缀唯一 
    content1 = html[web_start1[i]:web_end1[i]]# content 就是 起始中间的内容
    # print(content1)
    link = re.findall("article",content1)
    if link:
        content_ori = re.sub(r"<link>","",content1)
        content = re.sub(r"<link>https://seekingalpha.com/","",content1)
        content = "https://seekingalpha.com/amp/"+content
        web_list.append(content)
        web_list_ori.append(content_ori)######得到网站原网页 展示在界面上

html_n = session.get(web_list[0]).content.decode('utf-8')
web_start1 = [ta.start() for ta in re.finditer('<p>',html_n)]
web_end1 = [tm.start() for tm in re.finditer('</p>',html_n)]     #前缀后缀必须唯一
num = len(web_start1)
news_content=""
for i in range(2,num):
    con_n = html_n[web_start1[i]:web_end1[i]]
    con_n = re.sub(r"<.*?>","",con_n)
    news_content+=con_n+"\n"

print(web_list_ori,news_content)