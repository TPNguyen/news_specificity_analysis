#-*- coding:utf-8 -*-
import os
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
        for index in xrange(news_num):
            news_name = os.path.join(self.input_path, 'news_proc_{0}.txt'.format(index,))
            comments_name = os.path.join(self.input_path, 'comment_proc_{0}.txt'.format(index,))
            news_line = post_file_dict[news_name]
            comments_line = post_file_dict[comments_name]
            temp_news_matrix = np.zeros((len(news_line), self._type_num))
            temp_comments_matrix = np.zeros((len(comments_line), self._type_num))
            for r_index, line in enumerate(news_line):
                word_list = line.split()
                for word in word_list:
                    if word in self._type_to_index:
                        self.event_matrix[index, self._type_to_index[word]] += 1.0
                        temp_news_matrix[r_index, self._type_to_index[word]] += 1.0
            self.news_matrix_list.append(self.normalize(temp_news_matrix))
            for r_index, line in enumerate(comments_line):
                word_list = line.split()
                for word in word_list:
                    if word in self._type_to_index:
                        temp_comments_matrix[r_index, self._type_to_index[word]] += 1.0
            self.comments_matrix_list.append(self.normalize(temp_comments_matrix))
        self.event_matrix = self.normalize(self.event_matrix)
        print 'Parse data successfully!'

    '''
    tf * idf normalization
    '''
    def normalize(self, X):
        X = X[~np.all(X == 0, axis=1)]
        X = 0.01 + 0.99 * X / np.amax(X, axis=1)[:, np.newaxis]
        idf = np.log(X.shape[0] / (np.sum(X, axis=0) + 0.001))
        X = X * idf[np.newaxis,:]
        square_sum_row = 1.0 / np.sqrt(np.sum(X * X, axis=1) + 0.001)
        return X * square_sum_row[:, np.newaxis]
        
    def inference(self,):
        e_matrix = self.event_matrix
        d_matrix = self.news_matrix_list[2]
        c_matrix = self.comments_matrix_list[2]
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

    def export_comments(self, top_display=10):
        pass

if __name__=='__main__':
    mh = ModelHandler('./data/mh370')
    mh.parse_data()
    Wd, Hd, Md, We, He, Me, Wc, Hc = mh.inference()
    mh.export_topic(Hd, './data/mh370/output', 2, 'hd')
    mh.export_topic(He, './data/mh370/output', 2, 'he')
    mh.export_topic(Hc, './data/mh370/output', 2, 'hc')