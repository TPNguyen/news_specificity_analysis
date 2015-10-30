#-*- coding:utf-8 -*-
import nltk
import string
import jieba
import codecs
import numpy as np
from nltk.stem.porter import *
from nltk.corpus import stopwords

jieba.initialize()

pro_dict = dict((ord(char), None) for char in string.punctuation)
stemmer = PorterStemmer()

punct = set(u''':!),.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳﹐､﹒
﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠
々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻
︽︿﹁﹃﹙﹛﹝（｛“‘-—_…''')

filterpunt = lambda s: ''.join(filter(lambda x: x not in punct, s))
filterpuntl = lambda l: list(filter(lambda x: x not in punct, l))

def read_file(path):
    lines = []
    with codecs.open(path, 'r', 'utf-8') as f:
        lines = f.readlines()
    return lines

import os
stop_words_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'stopwords.txt')
stopwords_list = [x.strip() for x in read_file(stop_words_file)]
remove_stopwords = lambda l: list(filter(lambda x: x not in stopwords_list, l))

def en_str_process(s):
    text = s.decode('utf-8').lower()
    text_no_punctuation = ''
    if type(text)==unicode:
        text_no_punctuation = text.translate(pro_dict)
    if type(text)==str:
        text_no_punctuation = text.translate(None, string.punctuation)        
    tokens = nltk.word_tokenize(text_no_punctuation)
    filter_stop_text = [w for w in tokens if not w in stopwords.words('english')]
    stemmed = []
    for item in filter_stop_text:
        t = (stemmer.stem(item)).strip()
        stemmed.append(t)
    return ' '.join(stemmed)

def ch_str_process(s):
    seg_list = jieba.cut(s.decode('utf-8'), cut_all=False)
    seg_list = filterpuntl(seg_list)
    seg_list = [x.strip() for x in remove_stopwords(seg_list)]    
    return ' '.join(seg_list)
    
if __name__=='__main__':
    print en_str_process('I have eaten!. Nice to meet you eating? 你好 么')
    print ch_str_process('我惹你了吗？')
    print ch_str_process('我惹你了吗？')    
    