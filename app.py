from flask import Flask,render_template

from predict_ensemble import get_final ##得到 （1 buyin，0 不买）（list：概率[1,0]）
from sumarization import get_sum ##得到 string 第一条新闻 里的 sum-内容
from crawl_amzn_news_stocks_worldnews import get_news_link_content
###(list：新闻标题地址列表)（string：第一篇新闻原内容）
import re

app = Flask(__name__,template_folder='template')

f_result, possi = get_final()

if f_result==1:
	all_posts = str(possi[1])
	advice = "Buy In"
else:
	all_posts = str(possi[0])
	advice = "Sale Out"

news_link_l,news = get_news_link_content()
summer = get_sum()
title_l = [] ## contenttext

for i in news_link_l:
    i = re.sub(r"https://seekingalpha.com/article/+\d+\-","",i)
    i = re.sub(r"\-"," ",i)
    i = re.sub(r"\?.*","",i)
    i = i.upper()
    title_l.append(i)

title=title_l[0]

# all_posts = "test"
# advice = "test"


# sum_tet = get_sum()
# news_link_l, ori_news = get_news_link_content()

# all_posts = [
# 	{
# 		'title':'95% Buy in',
# 		'content':'This is the content of p1',
# 		'author':'oo'
# 	},


# ]






@app.route('/')

def hello_world():
	return render_template('index.html',posts=all_posts,post2=advice,post3=title_l[0],post4=summer)

@app.route('/posts')

def posts():
	return render_template('post.html',posts=all_posts)



if __name__=='__main__':
	app.run()