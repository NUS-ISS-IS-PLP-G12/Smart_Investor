##################################
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

from crawl_amzn_news_stocks_worldnews import get_news_link_content

def get_sum():

    # news_text = "Encoder contains the input words that want to be transformed (translate, generate summary), and each word is a vector that go through forward and backward activation with bi-directional RNN. Then calculate the attention value for each words in encoder reflects its importance in a sentence. Decoder generates the output word one at a time, by taking dot product of the feature vector and their corresponding attention for each timestamp."
    new_list,news_text = get_news_link_content()

    LANGUAGE = "english"
    SENTENCES_COUNT = 3

    print(news_text)


    if __name__ == "__main__":
        # url = "https://en.wikipedia.org/wiki/Automatic_summarization"
        # parser = HtmlParser.from_url(url, Tokenizer(LANGUAGE))
        # or for plain text files
        # parser = PlaintextParser.from_file("document.txt", Tokenizer(LANGUAGE))
        parser = PlaintextParser.from_string(news_text, Tokenizer(LANGUAGE))
        stemmer = Stemmer(LANGUAGE)

        summarizer = Summarizer(stemmer)
        summarizer.stop_words = get_stop_words(LANGUAGE)
        
        sum_newss = ""

        for sentence in summarizer(parser.document, SENTENCES_COUNT):
            sum_newss += str(sentence)
    
    return sum_newss
# a=""
a=get_sum()
print(a)

    