#coding=utf-8
import tensorflow as tf
import numpy as np
import os
import wml_utils as wmlu

tf.app.flags.DEFINE_string("file_path","../../../mldata/GLOVE/glove.6B.200d.txt","file path")
tf.app.flags.DEFINE_string("save_dir","","save path")
tf.app.flags.DEFINE_integer("emb_size",200,"emb_size")
FLAGS = tf.app.flags.FLAGS

def load_glove(file_path,emb_size):
    vocab = []
    embd = []
    vocab.append('unk')
    embd.append([0]*emb_size)
    file = open(file_path,'r')
    lines = file.readlines()
    for line in lines:
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    file.close()
    return vocab,embd

def main(_):
    vocab,embd = load_glove(FLAGS.file_path,FLAGS.emb_size)
    vocab_size = len(vocab)
    embedding_dim = len(embd[0])
    embedding = np.array(embd,np.float32)

    save_dir = FLAGS.save_dir if FLAGS.save_dir!="" else os.path.dirname(FLAGS.file_path)

    vocab_file_path = os.path.join(save_dir,"glove_vocab.bin")
    embd_file_path = os.path.join(save_dir,"glove_embd.bin")

    file = open(vocab_file_path,'w')
    for s in vocab:
        file.write(s+"\n")
    file.close()
    np.save(embd_file_path,embedding)

if __name__ == '__main__':
    tf.app.run()
