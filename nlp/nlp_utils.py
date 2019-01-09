#coding=utf-8
import numpy as np
import codecs
import os
import sys
import time
from functools import reduce

'''
load dict data which generated by trans_fastText
'''
def load_fastText_dict(dict_path):
    dict = np.load(dict_path)
    index = np.where(dict==u"甲肝")
    index = np.where(dict == u"乙肝")
    index = np.where(dict == u"丙肝")
    index = np.where(dict == u"肝炎")
    dict = np.concatenate((dict,[u"甲肝"],[u"乙肝"],[u"丙肝"],[u"<NULL>"]))
    return dict,index[0]

'''
load word embedding data which generated by trans_fastText
'''
def load_fastText_word_embeadding(path,index=None):
    we = np.load(path)
    if index is not None:
        d = we[index]
        we = np.concatenate((we,d,d,d,np.zeros(shape=[1,300])))
    return we

'''
load the dict file and word embedding data file, which generated by trans_fastText
'''
def load_fastTextByFile(dict_path,word_embeadding_path):
    dict,index = load_fastText_dict(dict_path)
    we = load_fastText_word_embeadding(word_embeadding_path,index)
    assert np.shape(dict)[0]==np.shape(we)[0]
    return dict,we

'''
load the dict file and word embedding data file, which generated by trans_fastText
'''
def load_fastTextByDir(dir_path):
    return load_fastTextByFile(os.path.join(dir_path,"dict.bin.npy"),os.path.join(dir_path,"wordembeadding.bin.npy"))

'''
Trans fast text word embedding data to two binary file: word embedding data and dict data
'''
def trans_fastText(file_path,save_dir="./"):
    file = codecs.open(file_path, "r", "utf-8")
    dict = []
    file.readline()
    we = []
    nr = 332647
    tmp = range(300)
    count = 0
    begin_t = time.time()
    while True:
        line = file.readline()
        if not line:
            break
        data = line.split(u" ")
        if len(data) == 300:
            data = [' '] + data
        elif len(data) < 300:
            continue
        dict.append(data[0])
        if count == 73144:
            print("A")
        for i in range(300):
            tmp[i] = (float(data[i + 1]))
        we.append(np.array([tmp], dtype=np.float32))
        if count % 100 == 0:
            sys.stdout.write('\r>> Converting image %d/%d' % (len(dict), nr))
            sys.stdout.flush()
        count = count + 1
    print("\n")
    print("total time=%f" % (time.time() - begin_t))
    # index = dict.index(u"甲肝")
    # index = dict.index(u"乙肝")
    # index = dict.index(u"丙炎")
    # index = dict.index(u"")
    we = np.concatenate(we)
    # we = np.concatenate([we,[we[0]],[we[0]],[we[0]]])
    np_dict = np.array(dict)
    np.save(os.path.join(save_dir,"wordembeadding.bin"), we)
    np.save(os.path.join(save_dir,"dict.bin"), np_dict)

'''
将文本进行分词并返回在词典中的索引
'''
def tokenize(text,thul,dict):
    text = text.encode("utf-8")
    thul_token = thul.cut(text)
    res = []
    token=[]
    for t in thul_token:
        word = t[0]
        u_word = word.decode("utf-8")
        index = np.where(dict == u_word)
        shape = np.shape(index[0])
        if shape[0] == 0:
            words = tokenize_word(u_word,dict)
            token.extend(words)
            res.extend(indexs_of_words(words,dict))
        else:
            res.append(index[0][0])
            token.append(u_word)
    return res,token

def tokenize_word(word,dict):
    if len(word)<=1:
        return [word]
    if len(word)==2:
        return [word[0],word[1]]
    begin_word = word[:2]
    index = np.where(dict==begin_word)

    if np.shape(index[0])[0] ==0:
        return [begin_word[0],begin_word[1]]+tokenize_word(word[2:],dict)
    else:
        return [begin_word]+tokenize_word(word[2:],dict)

def indexs_of_words(words,dict):
    res = []

    for word in words:
        index = np.where(dict == word)
        shape = np.shape(index[0])
        if shape[0] == 0:
            res.append(0)
        else:
            res.append(index[0][0])

    return res

'''
词/字典表
'''
class VocabTable(object):
    def __init__(self,vocab,default_word=None,default_index=None):
        '''
        vocab: a list of word
        '''
        self.size = len(vocab)
        x = range(len(vocab))
        self.vocab_to_id = dict(zip(vocab,x))
        self.vocab = vocab
        if default_word is not None:
            self.default_word = default_word
            self.default_index = self.vocab_to_id[default_word]
        elif default_index is not None:
            self.default_index = default_index
            self.default_word = self.vocab[default_index]
        else:
            self.default_word = "UNK"
            self.default_index = 0

    def get_id(self,word):
        '''
        get the index(id) of a word
        '''
        return self.vocab_to_id.get(word,self.default_index)

    def get_word(self,id):
        '''
        :param id: index(id) of a word
        :return: word string
        '''
        if id<0 or id>=len(self.vocab):
            return self.default_word
        return self.vocab[id]

    def get_id_of_string(self,string):
        '''
        :param string: a word string splited by ' '
        :return: id list of words
        '''
        words = string.strip().split(' ')
        res = []
        for w in words:
            res.append(self.get_id(w))
        return res

    def get_string_of_ids(self,ids,spliter=" "):
        '''
        :param ids: a list of ids
        :return: a word string splited by ' '
        '''
        words = ""
        for id in ids:
            words+= spliter+self.get_word(id)
        return words

    def get_vocab(self):
        return self.vocab

    def vocab_size(self):
        return self.size

def load_glove_data(dir_path):
    embedding_path = os.path.join(dir_path,"glove_embd.bin"+".npy")
    vocab_path = os.path.join(dir_path,"glove_vocab.bin")
    file = open(vocab_path,'r')
    vocab = []
    for s in file.readlines():
        vocab.append(s.strip())
    file.close()
    embedding = np.load(embedding_path)
    return embedding,VocabTable(vocab)

def load_default_dict(filepath):
    file = open(filepath,'r')
    vocab = []
    for s in file.readlines():
        vocab.append(s.strip())
    file.close()
    
    return vocab
    

def load_default_embedding_data(dir_path,embedding_name="word_embedding.bin"+".npy",vocab_name="vocab.bin"):
    embedding_path = os.path.join(dir_path,embedding_name)
    vocab_path = os.path.join(dir_path,vocab_name)
    vocab = load_default_dict(vocab_path)
    embedding = np.load(embedding_path)

    return embedding,VocabTable(vocab)

def merge_words(words:list,user_voc,max_merge_nr=4):
    res_words = []
    def words_to_word(v:list):
        return reduce(lambda lhv,rhv: lhv+rhv,v)
    i = 0
    while i<len(words):
        is_in = False
        for j in range(max_merge_nr,1,-1):
            if i+j>len(words):
                continue
            word = words_to_word(words[i:i+j])
            if word in user_voc:
                res_words.append(word)
                i += j
                is_in = True
                break
        if not is_in:
            res_words.append(words[i])
            i += 1

    return res_words