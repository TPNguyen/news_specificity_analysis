#-*- coding:utf-8 -*-
import numpy as np
from matrix_factorization import MatrixFactorization


class NMF(MatrixFactorization):
    def __init__(self, event_matrix, news_matrix, comment_matrix, k=5, lam=0.1, eps=0.01, iter_eps=0.01, max_iter_num=0.1, verbose=True):
        MatrixFactorization.__init__(self, event_matrix, news_matrix, comment_matrix, k)
        self.lam = lam
        self.eps = eps
        self.iter_eps = iter_eps
        self.max_iter_num = max_iter_num
        self.verbose = verbose

    def __init__(self, k=5, lam=10000, eps=0.01, iter_eps=0.01, max_iter_num=10000, verbose=True):
        self.k = 5
        self.lam = lam
        self.eps = eps
        self.iter_eps = iter_eps
        self.max_iter_num = max_iter_num
        self.verbose = verbose
        

    def factorize(self):
        self.nmf_factorize(self.event_matrix)
        self.nmf_factorize(self.news_matrix)
        self.nmf_factorize(self.comment_matrix)

    '''
    X = W * H
    X: document - term matrix
    W: document - topic matrix
    H: topic -term matrix
    k: topic number
    '''
    def nmf_factorize(self, X):
        d_num, v_num = X.shape
        self.XX = np.sum(X*X)
        W = np.random.rand(d_num, self.k)
        H = np.random.rand(self.k, v_num)
        iter_num = 0
        pre_obj_value = 100000.0
        obj_value = pre_obj_value * 2
        
        while((np.fabs(obj_value - pre_obj_value)>self.iter_eps) and (iter_num < self.max_iter_num)):
            W = W * (X.dot(H.transpose()) / np.maximum(W.dot(H).dot(H.transpose()) + self.lam, self.eps))
            WtW = W.transpose().dot(W)
            WtX = W.transpose().dot(X)
            H = H * (WtX / np.maximum(WtW.dot(H) + self.lam, self.eps))
            iter_num += 1
            pre_obj_value = obj_value
            obj_value = self.compute_loss(X, W, H)
            delta = np.fabs(obj_value - pre_obj_value)
            print delta
            if self.verbose and iter_num%100==0:
                print 'It:{0} \t obj:{1} \t delta:{2} '.format(iter_num, obj_value, delta)
        print 'W:', W
        print 'H:', H

    def compute_loss(self, X, W, H):
        WH = W.dot(H)
        tr1 = self.XX - 2 * np.sum(X*WH) + np.sum(WH*WH)
        tr2 = self.lam * (np.sum(H) + np.sum(W))
        return tr1 + tr2
        
if __name__=='__main__':
    X = np.random.random((500,20000))
    print X
    nmf = NMF()
    nmf.nmf_factorize(X)
    