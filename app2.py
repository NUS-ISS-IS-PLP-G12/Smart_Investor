from flask import Flask,render_template
import re

from predict_ensemble import get_final ##得到 （1 buyin，0 不买）（list：概率[1,0]）
from sumarization import get_sum ##得到 string 第一条新闻 里的 sum-内容
from crawl_amzn_news_stocks_worldnews import get_news_link_content
###(list：新闻标题地址列表)（string：第一篇新闻原内容）
import re

app = Flask(__name__,template_folder='template')

f_result, possi = get_final()#f_result=0/1;possi置信度
#####possi(all posts) 保留位数百分数##########



if f_result==1:
	all_posts = possi[1]
	advice = "Increase"
else:
    	
	all_posts = possi[0]
	advice = "Decrease"

all_posts = round(all_posts, 4) 

all_posts = all_posts*100
all_posts = str(all_posts)+"%"

if advice == "Increase":
	advice2 = "Buy In"
else:
	advice2 = "Sell Out"

news_link_l,news = get_news_link_content()
summer = ""
summer = get_sum()#summerazation
title_l = []#5titles report list

for i in news_link_l:
    i = re.sub(r"https://seekingalpha.com/article/+\d+\-","",i)
    i = re.sub(r"\-"," ",i)
    i = re.sub(r"\?.*","",i)
    i = i.upper()
    title_l.append(i)

##0 decrease放到advice(post2)里
##1 increase
##if advice=increase下面advice2(post14)是buyin

for i in range(6):
	names = globals()
	names['link' + str(i) ] = ""

	a_names = globals()
	a_names['title' + str(i)] = ""

num = len(news_link_l)

for i in range(len(news_link_l)):
	names = globals()
	names['link' + str(i) ] = str(news_link_l[i])

	a_names = globals()
	a_names['title' + str(i)] = str(title_l[i])

# a=['abc','def','ghi']
# b=str(a[0])
# print(b,type(b))

summerazation=str(summer)
##summer news_link_l[0]  title_l[0] type###
##############################################################################Web##########################
@app.route('/')

def hello_world():
	return render_template('index.html',posts="65.25%",post2=advice,post3=title0,post4=summerazation,post5=link0,post6=link1,post7=title1,
	post8=link2,post9=title2,post10=link3,post11=title3,post12=link4,post13=title4,post14=advice2)




if __name__=='__main__':
	app.run()