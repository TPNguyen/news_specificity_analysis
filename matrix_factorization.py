#-*- coding:utf-8 -*-


class MatrixFactorization(object):
    def __init__(self, event_matrix, news_matrix, comment_matrix, k):
        self.event_matrix = event_matrix
        self.news_matrix = news_matrix
        self.comment_matrix = comment_matrix
        self.k = k

    def factorize(self):
        raise NotImplementedError()