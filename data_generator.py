#-*- coding:utf-8 -*-
import MySQLdb
import MySQLdb.cursors
import nltk.tokenize as nltoken


class DataGenerator(Object):
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
        sql_news = 'select id, content from %s where comment_num > 10' % (self.news_table_name,)
        cur = self.conn.cursor()
        cur.execute(sql_news)
        r_news = cur.fetchall()
        for news, index in r_news, xrange(len(r_news)):
            id = news['id']
            content = news['content']
            sent_tokenize_list = [x.strip() for x in nltoken.sent_tokenize(content) if x.strip()]
            post_sent_tokenize_list = []
            per_news_content = '\n'.join(sent_tokenize_list)
            per_news_name = 'news_{0}.txt'.format(index,)
            per_comment_name = 'comment_{0}.txt'.format(index,)
            comments_content = []
            sql_comment = 'select content from %s where news_id = %s' % (self.comments_table_name, id)
            cur.execute(sql_comment)
            r_comment = cur.fetchall()
            for comment in r_comment:
                content = comment['content']
        cur.close()
        self._close()
