#-*- coding:utf-8 -*-
import os
import logging
import MySQLdb
import MySQLdb.cursors
import nltk.tokenize as nltoken


def write_file(name, content):
    f = open(name, 'w')
    f.write(content.encode('utf-8'))
    f.close()

    
class DataGenerator(object):
    def __init__(self, news_table_name, comments_table_name, outpath):
        self.news_table_name = news_table_name
        self.comments_table_name = comments_table_name
        self.conn = None
        self.outpath = outpath

    def _connect(self,):
        self.conn = MySQLdb.connect(host='seis10.se.cuhk.edu.hk', port=3306, user='bshi', passwd='20141031shib', db='bshi', charset='utf8', cursorclass=MySQLdb.cursors.DictCursor)

    def _close(self,):
        if not self.conn:
            self.conn.close()

    def generate_news_comments(self):
        self._connect()
        sql_news = 'select id, content from %s where comment_num > 10 and length(content) > 10' % (self.news_table_name,)
        cur = self.conn.cursor()
        cur.execute(sql_news)
        r_news = cur.fetchall()
        for index, news in enumerate(r_news):
            id = news['id']
            content = news['content']
            sent_tokenize_list = [x.strip() for x in nltoken.sent_tokenize(content) if x.strip()]
            per_news_content = '\n'.join(sent_tokenize_list)
            per_news_name = os.path.join(self.outpath, 'news_{0}.txt'.format(index,))
            per_comment_name = os.path.join(self.outpath, 'comment_{0}.txt'.format(index,))
            
            comments_list = []
            sql_comment = 'select content from %s where news_id = %s' % (self.comments_table_name, id)
            cur.execute(sql_comment)
            r_comment = cur.fetchall()
            for comment in r_comment:
                content = comment['content'].strip()
                l_content = content.split()
                if len(l_content) > 5:
                    comments_list.append(' '.join(l_content))
            per_comment_content = '\n'.join(comments_list)
            write_file(per_news_name, per_news_content)
            write_file(per_comment_name, per_comment_content)
            print '{0}th news and comments are generated!'.format(index,)
        cur.close()
        self._close()

if __name__=='__main__':
    dg = DataGenerator('news_mh370', 'comment_mh370', './data/mh370')
    dg.generate_news_comments()