#-*- coding:utf-8 -*-
import numpy as np

from matrix_factorization import MatrixFactorization


class NMF(MatrixFactorization):
    def __init__(self, event_matrix, news_matrix, comment_matrix, k, lambda=0.1, epsilon=05, maxiter=100, verbose=True):
        MatrixFactorization.__init__(self, event_matrix, news_matrix, comment_matrix, k)

    def factorize(self):
        self.nfm_factorize(self.event_matrix)
        self.nfm_factorize(self.news_matrix)
        self.nfm_factorize(self.comment_matrix)

    '''
    X = Y * Z
    X: document - term matrix
    Y: document - topic matrix
    Z: topic -term matrix
    k: topic number
    '''
    def nfm_factorize(self, X):
        pass
        
        
if __name__=='__main__':
    pass