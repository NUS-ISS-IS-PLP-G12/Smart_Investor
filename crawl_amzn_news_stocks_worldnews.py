import re
import urllib 
import requests
from urllib.request import Request,urlopen

"""
news_list 亚马逊新闻的前两个标题 string      略
web_list_ori 所有analysis的链接地址 list    get_news_link_content return web_list_ori,news_content
news_content analysis内容 string          get_news_link_content   
open high low volume_amzn   float   get_stock_ohlv
world_news 世界新闻10个 string              略
text 亚马逊新闻的前两个标题+世界新闻10个 string get_txt
"""

def get_news_link_content():    
 
    link = 'https://seekingalpha.com/api/sa/combined/AMZN.xml'
    session = requests.Session() #发送请求
    html = session.get(link).content.decode('utf-8') 

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
        news_content = str(news_content)

    return web_list_ori,news_content

##################

def get_txt():

    link = 'https://seekingalpha.com/api/sa/combined/AMZN.xml'
    session = requests.Session() #发送请求
    html = session.get(link).content.decode('utf-8') 

    news_list = ""

    web_start1 = [ta.start() for ta in re.finditer('<title>',html)]
    web_end1 = [tm.start() for tm in re.finditer('</title>',html)] #前缀后缀必须唯一
    num = len(web_start1)
    for i in range(1,2):#循环查找多个跟上面前后缀唯一 
        content1 = html[web_start1[i]:web_end1[i]]# content 就是 起始中间的内容
        content_ori = re.sub(r"<.*?>","",content1)
        news_list+=str(content_ori)# 得到新闻标题

    world_news = ""

    link_a="https://www.reuters.com/news/archive/worldNews"
    html_amzn = session.get(link_a).content.decode('utf-8')

    with open('news_html.txt','w') as f:    #设置文件对象
        for i in html_amzn:
            f.write(i)                 #将字符串写入文件中

    web_start1 = [ta.start() for ta in re.finditer('\<h3 class\=\"story\-title\"\>',html_amzn)]
    # print(len(web_start1))
    web_end1 = [tm.start() for tm in re.finditer('</h3>',html_amzn)]
    # print(len(web_end1))

    num = len(web_start1)
    for i in range(num):#循环查找多个跟上面前后缀唯一 
        content_amzn = html_amzn[web_start1[i]:web_end1[i]]# content 就是 起始中间的内容
        # print(content_amzn)
        # link = re.findall("article",content1)
        # if link:
        content_amzn = re.sub(r"<.*?>|\s\s","",content_amzn)
        world_news += str(content_amzn)
    # print(world_news)
    text = news_list + world_news  

    return text
##################

def get_stock_ohlv():
    ##################
    session = requests.Session() #发送请求
    link_a="https://sg.finance.yahoo.com/quote/AMZN/history?p=AMZN"
    html_amzn = session.get(link_a).content.decode('utf-8')

    with open('data_html.txt','w') as f:    #设置文件对象
        for i in html_amzn:
            f.write(i)                 #将字符串写入文件中

    web_start1 = [ta.start() for ta in re.finditer('\<span data-reactid\=\"55\"\>',html_amzn)]
    # print(web_start1)
    web_end1 = [tm.start() for tm in re.finditer('\<td class\=\"Py\(10px\) Pstart\(10px\)\" data-reactid\=\"56\"\>',html_amzn)]

    content_amzn = html_amzn[web_start1[0]:web_end1[0]]
    content_amzn = re.sub(r"<.*?>|,","",content_amzn)
    open_amzn = float(content_amzn)
    ###################
    web_start1 = [ta.start() for ta in re.finditer('\<span data\-reactid\=\"57\"\>',html_amzn)]
    # print(web_start1)
    web_end1 = [tm.start() for tm in re.finditer('\<td class\=\"Py\(10px\) Pstart\(10px\)\" data\-reactid\=\"58\"\>',html_amzn)]

    content_amzn = html_amzn[web_start1[1]:web_end1[0]]
    content_amzn = re.sub(r"<.*?>|,","",content_amzn)
    high_amzn = float(content_amzn)

    ###################
    web_start1 = [ta.start() for ta in re.finditer('\<span data\-reactid\=\"59\"\>',html_amzn)]
    # print(web_start1)
    web_end1 = [tm.start() for tm in re.finditer('\<td class\=\"Py\(10px\) Pstart\(10px\)\" data-reactid\=\"60\"\>',html_amzn)]

    content_amzn = html_amzn[web_start1[1]:web_end1[0]]
    content_amzn = re.sub(r"<.*?>|,","",content_amzn)
    low_amzn = float(content_amzn)
    # print(low_amzn)
    ###################
    web_start1 = [ta.start() for ta in re.finditer('\<span data\-reactid\=\"63"\>',html_amzn)]

    web_end1 = [tm.start() for tm in re.finditer('\<td class\=\"Py\(10px\) Pstart\(10px\)\" data-reactid\=\"64\"\>',html_amzn)]

    content_amzn = html_amzn[web_start1[0]:web_end1[0]]
    content_amzn = re.sub(r"<.*?>|,","",content_amzn)
    volume_amzn = float(content_amzn)

    return open_amzn,high_amzn,low_amzn,volume_amzn

###################
###################

"""
news_list 亚马逊新闻的前两个标题 string, low_amzn
web_list_ori 所有analysis的链接地址 list
news_content analysis内容 string
open high low volume_amzn   float
world_news 世界新闻10个 string
text 亚马逊新闻的前两个标题+世界新闻10个 string
"""