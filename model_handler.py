#-*- coding:utf-8 -*-
import os
import time
import cPickle as pickle
import numpy as np
import ner_mf
import matrix_factorization

def read_file(name):
    f = open(name)
    lines = f.readlines()
    return lines

class ModelHandler(object):
    def __init__(self, input_path):
        self.input_path = input_path

    def parse_data(self):
        f_list = [os.path.join(self.input_path, x) for x in os.listdir(self.input_path) if 'proc' in x]
        assert len(f_list)%2 == 0
        post_file_dict = {}
        self._word_count = {}
        for f_item in f_list:
            print f_item
            lines = read_file(f_item)
            post_lines = []
            for post_line in lines:
                for word in post_line.split():
                    if word not in self._word_count:
                        self._word_count[word] = 1
                    else:
                        self._word_count[word] = self._word_count[word] + 1
                post_lines.append(post_line)
            post_file_dict[f_item] = post_lines
            
        self._temp_dict = {k:v for k, v in self._word_count.items() if v > 5}
        word_type_list = self._temp_dict.keys()
        self._type_to_index = {}
        self._index_to_type = {}
        for index, word in enumerate(word_type_list):
            self._type_to_index[word] = index
            self._index_to_type[index] = word
        assert len(self._type_to_index)==len(self._index_to_type)
        self._type_num = len(self._type_to_index)
        print 'Parse dictionary successfully.'

        news_num = len(f_list) / 2
        comments_num = news_num
        self.event_matrix = np.zeros((news_num, self._type_num))
        self.news_matrix_list = []
        self.comments_matrix_list = []
        self.raw_comments_list = []
        for index in xrange(news_num):
            news_name = os.path.join(self.input_path, 'news_proc_{0}.txt'.format(index,))
            comments_name = os.path.join(self.input_path, 'comment_proc_{0}.txt'.format(index,))
            raw_comments_name = os.path.join(self.input_path, 'comment_{0}.txt'.format(index,))
            news_line = post_file_dict[news_name]
            comments_line = post_file_dict[comments_name]
            temp_news_matrix = np.zeros((len(news_line), self._type_num))
            temp_comments_matrix = np.zeros((len(comments_line), self._type_num))
            f = open(raw_comments_name, 'r')
            raw_comments = np.array(f.readlines())
            f.close()
            for r_index, line in enumerate(news_line):
                word_list = line.split()
                for word in word_list:
                    if word in self._type_to_index:
                        self.event_matrix[index, self._type_to_index[word]] += 1.0
                        temp_news_matrix[r_index, self._type_to_index[word]] += 1.0
            temp_news_matrix = np.concatenate((temp_news_matrix, self.window_stack(temp_news_matrix)), axis=0)
            self.news_matrix_list.append(self.normalize(temp_news_matrix))
            for r_index, line in enumerate(comments_line):
                word_list = line.split()
                for word in word_list:
                    if word in self._type_to_index:
                        temp_comments_matrix[r_index, self._type_to_index[word]] += 1.0
            assert len(raw_comments) == temp_comments_matrix.shape[0]
            self.raw_comments_list.append(raw_comments[~np.all(temp_comments_matrix==0, axis=1)])
            self.comments_matrix_list.append(self.normalize(temp_comments_matrix))
        self.event_matrix = self.normalize(self.event_matrix)
        print 'Parse data successfully!'

    '''
    tf * idf normalization
    '''
    def normalize(self, X):
        X = X[~np.all(X == 0, axis=1)]
        idf = np.log(X.shape[0] / ((X!=0).sum(0) + 1))
        X = X / np.amax(X, axis=1)[:, np.newaxis]
        X = X * idf[np.newaxis,:]
        X_leng = np.sqrt(np.sum(X * X, axis=1))
        square_sum_row = np.where(X_leng != 0, 1.0 / (X_leng+0.000000001), 0)
        return X * square_sum_row[:, np.newaxis]

    def window_stack(self, a, width=3):
        l = a.shape[0]
        if width >= l:
            return a
        b = [np.sum(a[i:i+width], axis=0) for i in xrange(0,a.shape[0]-width+1)]
        return np.array(b)
        
    def inference(self,):
        #e_matrix = np.delete(self.event_matrix, [80,:], axis=0)
        e_matrix = self.event_matrix
        d_matrix = self.news_matrix_list[60]
        c_matrix = self.comments_matrix_list[60]
        mf = ner_mf.NER_MF(e_matrix, d_matrix, c_matrix)
        Wd, Hd, Md, We, He, Me, Wc, Hc = mf.factorize()
        return Wd, Hd, Md, We, He, Me, Wc, Hc

    def export_topic(self, H, out_path, f_index, f_name, top_display=100):
        topic_num = H.shape[0]
        type_num = H.shape[1]
        file_name = 'topic_{0}_{1}.txt'.format(f_index, f_name)
        output = open(os.path.join(out_path, file_name), 'w')
        for topic_index in xrange(topic_num):
            word_value = H[topic_index, :]
            i = 0
            output.write('-----------------\t{0}\t-------------------\n'.format(topic_index))
            for type_index in reversed(np.argsort(word_value)):
                i += 1
                output.write("%s\t%g\n" % (self._index_to_type[type_index], word_value[type_index]))
                if top_display > 0 and i >= top_display:
                    break
        output.close()

    def export_comments(self, Wd, Hd, Md, We, He, Me, Wc, Hc, out_path, f_index, f_name, top_display=10):
        # d_value = np.sum(Wc.dot(Md).dot(Hd), axis=1)
        # e_value = np.sum(Wc.dot(Me).dot(He), axis=1)
        # c_value = np.sum(Wc.dot(Hc), axis=1)
        # sum_value = d_value + e_value + c_value
        # d_value = d_value / sum_value
        # e_value = e_value / sum_value
        # c_value = c_value / sum_value

        d_value = Wc.dot(Md)
        norm_d_value = np.sqrt(np.sum(d_value * d_value, axis=1) + 0.000001)[:, np.newaxis]
        d_value = np.where(norm_d_value!=0.0, d_value/norm_d_value, 0.)
        Wd = Wd / np.sqrt(np.sum(Wd * Wd, axis=1) + 0.0000001)[:, np.newaxis]
        d_sim_matrix = d_value.dot(Wd.transpose())
        d_value = np.mean(d_sim_matrix, axis=1)
        
        e_value = Wc.dot(Me)
        norm_e_value = np.sqrt(np.sum(e_value * e_value, axis=1) + 0.000001)[:, np.newaxis]
        e_value = np.where(norm_e_value!=0.0, e_value/norm_e_value, 0.)
        We = We / np.sqrt(np.sum(We * We, axis=1) + 0.0000001)[:, np.newaxis]
        e_sim_matrix = e_value.dot(We.transpose())
        e_value = np.mean(e_sim_matrix, axis=1)

        c_value = e_value * d_value
        
        file_name = 'comments_specificity_{0}_{1}.txt'.format(f_index, f_name)
        output = open(os.path.join(out_path, file_name), 'w')
        i = 0
        output.write('-------------------\t{0}\t-------------------\n'.format('news specific'))
        for c_index in reversed(np.argsort(d_value)):
            i += 1
            output.write(self.raw_comments_list[f_index][c_index] + '\n')
            if top_display > 0 and i >= top_display:
                break
        i = 0
        output.write('-------------------\t{0}\t-------------------\n'.format('events specific'))
        for c_index in reversed(np.argsort(e_value)):
            i += 1
            output.write(self.raw_comments_list[f_index][c_index] + '\n')
            if top_display > 0 and i >= top_display:
                break
        i = 0
        output.write('-------------------\t{0}\t-------------------\n'.format('comments specific')) 
        for c_index in reversed(np.argsort(c_value)[::-1]):
            i += 1
            output.write(self.raw_comments_list[f_index][c_index] + '\n')
            if top_display > 0 and i >= top_display:
                break
        i = 0
               
        output.close()

if __name__=='__main__':
    mh = ModelHandler('./data/mh370')
    mh.parse_data()
    Wd, Hd, Md, We, He, Me, Wc, Hc = mh.inference()
    mh.export_topic(Hd, './data/mh370/output', 80, 'hd')
    mh.export_topic(He, './data/mh370/output', 80, 'he')
    mh.export_topic(Hc, './data/mh370/output', 80, 'hc')
    mh.export_comments(Wd, Hd, Md, We, He, Me, Wc, Hc, './data/mh370/output', 80, 'test')