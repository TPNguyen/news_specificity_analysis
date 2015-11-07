#-*- coding:utf-8 -*-
import numpy as np
from matrix_factorization import MatrixFactorization


class NER_MF(MatrixFactorization):
    def __init__(self, event_matrix, news_matrix, comment_matrix, k=10, lam=0.1, eps=0.1, beta = 0.1, delta=0.1, iter_eps=0.05, max_iter_num=500, verbose=True):
        MatrixFactorization.__init__(self, event_matrix, news_matrix, comment_matrix, k)
        self.lam = lam
        self.eps = eps
        self.beta = beta
        self.delta = delta
        self.iter_eps = iter_eps
        self.max_iter_num = max_iter_num
        
        self.lam = 0.01
        self.eps = 0.001
        self.beta = beta
        self.delta = 10000
        self.beta1 = 0.1
        self.beta2 = 0.1
        self.mu1 = 5000
        self.mu2 = 3000
        self.mu3 = 200

        # self.lam = 0.01
        # self.eps = 0.001
        # self.beta = beta
        # self.delta = 0
        # self.beta1 = 0
        # self.beta2 = 0
        # self.mu1 = 0.8
        # self.mu2 = 0.2
        # self.mu3 = 0
        
        self.verbose = verbose


    def factorize(self):
        Xe = self.event_matrix
        Xd = self.news_matrix
        Xc = self.comment_matrix
        #print np.count_nonzero(Xe)
        #print np.count_nonzero(Xd)
        #print np.count_nonzero(Xc)
        print Xe
        print Xd
        print Xc
        self.XeXe = np.sum(Xe * Xe)
        self.XdXd = np.sum(Xd * Xd)
        self.XcXc = np.sum(Xc * Xc)        
        e_num, v1_num = Xe.shape
        d_num, v2_num = Xd.shape
        c_num, v3_num = Xc.shape
        assert ((v1_num==v2_num) and (v2_num==v3_num))==True
        v_num = v1_num
        I = np.identity(self.k)

        We = np.random.rand(e_num, self.k)
        He = np.random.rand(self.k, v_num)
        Me = np.random.rand(self.k, self.k)
        Wd = np.random.rand(d_num, self.k)
        Hd = np.random.rand(self.k, v_num)
        Md = np.random.rand(self.k, self.k)
        Wc = np.random.rand(c_num, self.k)
        Hc = np.random.rand(self.k, v_num)

        pre_obj_value = 100000.0
        obj_value = pre_obj_value * 2
        iter_num = 0

        while((np.fabs(obj_value-pre_obj_value) > self.iter_eps) and (iter_num < self.max_iter_num)):

            WctWc = Wc.transpose().dot(Wc)
            WctXc = Wc.transpose().dot(Xc)

            Wd = Wd * Xd.dot(Hd.transpose()) * self.mu1 / np.maximum(Wd.dot(Hd).dot(Hd.transpose()) * self.mu1 + self.lam, self.eps)
            We = We * Xe.dot(He.transpose()) * self.mu2 / np.maximum(We.dot(He).dot(He.transpose()) * self.mu2 + self.lam, self.eps)

            HcHct = Hc.dot(Hc.transpose())
            Hd = Hd * (Wd.transpose().dot(Xd) * self.mu1 + Md.transpose().dot(Wc.transpose()).dot(Xc) * self.mu3) / np.maximum((Wd.transpose().dot(Wd).dot(Hd) * self.mu1 + self.lam + self.delta * He.dot(He.transpose()).dot(Hd) + self.delta * HcHct.dot(Hd) + Md.transpose().dot(WctWc).dot(Md.dot(Hd) + Me.dot(He) + Hc) * self.mu3), self.eps)
            HdHdt = Hd.dot(Hd.transpose())
            He = He * (We.transpose().dot(Xe) * self.mu2 + Me.transpose().dot(WctXc) * self.mu3) / np.maximum((We.transpose().dot(We).dot(He) * self.mu2 + self.lam + self.delta * HdHdt.dot(He) + self.delta * HcHct.dot(He) + Me.transpose().dot(WctWc).dot(Md.dot(Hd)+Me.dot(He)+Hc) * self.mu3), self.eps)

            temp_sum = Md.dot(Hd) + Me.dot(He) + Hc
            Wc = Wc * Xc.dot(temp_sum.transpose()) * self.mu3 / np.maximum(self.lam + Wc.dot(temp_sum).dot(temp_sum.transpose() * self.mu3), self.eps)
            WctWc = Wc.transpose().dot(Wc)
            WctXc = Wc.transpose().dot(Xc)
            Hc = Hc * WctXc *self.mu3 / np.maximum(self.lam + self.delta * ((HdHdt) + He.dot(He.transpose())).dot(Hc) + WctWc.dot(temp_sum) * self.mu3, self.eps)
            temp_sum = Md.dot(Hd) + Me.dot(He) + Hc            
            Md = Md * (self.beta1 * I + WctXc.dot(Hd.transpose()) * self.mu3) / np.maximum((self.beta1 * Md.transpose() + WctWc.dot(temp_sum).dot(Hd.transpose()) * self.mu3 + self.lam), self.eps)
            temp_sum = Md.dot(Hd) + Me.dot(He) + Hc            
            Me = Me * (self.beta2 * I + WctXc.dot(He.transpose()) * self.mu3) / np.maximum((self.beta2 * Me.transpose() + WctWc.dot(temp_sum).dot(He.transpose()) * self.mu3 + self.lam), self.eps)

            iter_num += 1
            pre_obj_value = obj_value
            obj_value = self.compute_loss(Xd, Wd, Hd, Md, Xe, We, He, Me, Xc, Wc, Hc, I)
            if pre_obj_value < obj_value:
                print iter_num, obj_value

            delta = np.fabs(obj_value - pre_obj_value)
            if (self.verbose and iter_num%100==0):
                print 'It:{0} \t obj:{1} \t delta:{2} '.format(iter_num, obj_value, delta)
            '''print 'He:', He
            print 'Hd:', Hd
            print 'Hc:', Hc
            print 'delta',delta'''
        return Wd, Hd, Md, We, He, Me, Wc, Hc

    def compute_loss(self, Xd, Wd, Hd, Md, Xe, We, He, Me, Xc, Wc, Hc, I):
        WdHd = Wd.dot(Hd)
        WeHe = We.dot(He)
        WcHc = Wc.dot(Hc)
        HdtHe = Hd.transpose().dot(He)
        HetHc = He.transpose().dot(Hc)
        HctHd = Hc.transpose().dot(Hd)
        
        tr1 = self.XdXd - 2 * np.sum(Xd * WdHd) + np.sum(WdHd * WdHd)
        tr2 = self.XeXe - 2 * np.sum(Xe * WeHe) + np.sum(WeHe * WeHe)
        t_sum = Wc.dot(Md).dot(Hd) + Wc.dot(Me).dot(He) + WcHc
        tr3 = self.XcXc - 2 * np.sum(Xc * t_sum) + np.sum(t_sum * t_sum)
        tr4 = self.lam * (np.sum(Wd) + np.sum(Hd) + np.sum(We) + np.sum(He) + np.sum(Wc) + np.sum(Hc))
        #tr4 = self.lam * (np.sum(Wd) + np.sum(Hd) + np.sum(We) + np.sum(He))        
        tr51 = self.beta1 * (np.sum(Md * Md) - 2 * np.trace(Md) + np.trace(I))
        tr52 = self.beta2 * (np.sum(Me * Me) - 2 * np.trace(Me) + np.trace(I))
        tr5 = tr51 + tr52
        tr6 = self.delta * (np.sum(HdtHe * HdtHe) + np.sum(HetHc * HetHc) + np.sum(HctHd * HctHd))
        obj = tr1 * self.mu1 + tr2 * self.mu2 + tr3*self.mu3 + tr4 + tr5 + tr6
        print tr1 * self.mu1, tr2 * self.mu2, tr3 * self.mu3, tr4, tr51, tr52, tr6
        
        return tr1 * self.mu1 + tr2 * self.mu2 + tr3*self.mu3 + tr4 + tr5 + tr6

if __name__=='__main__':
    e_num = 100
    d_num = 100
    c_num = 100
    v_num = 2000
    
    e_matrix = np.random.random((e_num, v_num))
    d_matrix = np.random.random((d_num, v_num))
    c_matrix = np.random.random((c_num, v_num))
    my_mf = NER_MF(e_matrix, d_matrix, c_matrix)
    my_mf.factorize()