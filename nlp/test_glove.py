#coding=utf-8
import tensorflow as tf
import numpy as np
import nlp.nlp_utils as nlpu
from tensorflow.contrib import learn
tf.app.flags.DEFINE_string("data_dir","../../../mldata/GLOVE","data dir")
FLAGS = tf.app.flags.FLAGS

def main(_):
    embedding,vocab_table = nlpu.load_glove_data(FLAGS.data_dir)
    embedding = embedding.astype(np.float32)
    tf_embedding = tf.constant(embedding,tf.float32)
    x = tf.placeholder(dtype=tf.int32)
    y = tf.placeholder(dtype=tf.int32)
    x_res = tf.nn.embedding_lookup(tf_embedding,x)
    y_res = tf.nn.embedding_lookup(tf_embedding,y)
    distance = tf.nn.l2_loss(x_res-y_res)
    test_data = ["man","woman","king","queen","blue","gray","red","white","dog","cat","car","trunk"]
    sess = tf.Session()
    size = len(test_data)
    sess.run(tf.global_variables_initializer())
    dis_dict = []
    for i in range(size):
        for j in range(i+1,size):
            x_id = vocab_table.get_id(test_data[i])
            y_id = vocab_table.get_id(test_data[j])
            print(test_data[i]," ",test_data[j])
            print("x_id:",x_id)
            print("y_id:",y_id)
            dis,xres,yres = sess.run([distance,x_res,y_res],feed_dict={x:x_id,y:y_id})
            #print("x vector:",xres)
            #print("y vector:",yres)
            dis_dict.append([test_data[i]+" "+test_data[j],dis])
            print("distance:",dis)
    dis_dict.sort(key=lambda x:x[1])
    for x in dis_dict:
        print(x)

if __name__ == '__main__':
    tf.app.run()