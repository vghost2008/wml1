#coding=utf-8
import tensorflow as tf
import wml_tfutils as wmlt

def subsample_labels(labels, num_samples, positive_fraction):
    '''
    大于0为正样本，等于零为背景，小于0为忽略
    :param labels:
    :param num_samples:
    :param positive_fraction:
    :return:
    '''

    with tf.variable_scope("subsample_labels"):
        labels = tf.reshape(labels, shape=[-1])

        pmask = tf.greater(labels,0)
        nmask = tf.equal(labels,0)
        psize = num_samples*positive_fraction
        nsize = num_samples*(1-positive_fraction)
        select_pmask = wmlt.subsample_indicator(pmask,psize)
        select_nmask = wmlt.subsample_indicator(nmask,nsize)

        return select_pmask,select_nmask

