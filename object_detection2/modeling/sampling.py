#coding=utf-8
import tensorflow as tf
import wml_tfutils as wmlt
from object_detection2.config.config import global_cfg

def subsample_labels(labels, num_samples, positive_fraction):
    '''
    大于0为正样本，等于零为背景，小于0为忽略
    :param labels:[...]
    :param num_samples:
    :param positive_fraction:
    :return:
    [N],[N] (N=num_elements(labels)
    '''

    with tf.variable_scope("subsample_labels"):
        labels = tf.reshape(labels, shape=[-1])

        pmask = tf.greater(labels,0)
        psize = int(num_samples*positive_fraction)
        select_pmask = wmlt.subsample_indicator(pmask,psize)

        nmask = tf.equal(labels,0)
        psize = tf.reduce_sum(tf.cast(select_pmask, tf.int32))
        nsize = num_samples-psize
        select_nmask = wmlt.subsample_indicator(nmask,nsize)
        if global_cfg.GLOBAL.DEBUG:
            tf.summary.scalar("psize",psize)
            tf.summary.scalar("nsize",nsize)

        return select_pmask,select_nmask

def subsample_labels_by_negative_loss(labels, num_samples, probability,num_classes,positive_fraction):
    '''
    大于0为正样本，等于零为背景，小于0为忽略
    SSD选择样本的方法，最指定百分比的正样本，负样本按损失大小选
    :param probability: 必须为相应的概率，不然计算不准确
    :param labels:
    :param num_samples:
    :param positive_fraction:
    :return:
    [N],[N] (N=num_elements(labels)
    '''

    with tf.variable_scope("subsample_labels_negative_loss"):
        labels = tf.reshape(labels, shape=[-1])
        probability = tf.reshape(probability, [-1, num_classes+1])
        pmask = tf.greater(labels,0)

        psize = int(num_samples*positive_fraction)
        selected_pmask = wmlt.subsample_indicator(pmask,psize)

        nmask = tf.equal(labels,0)
        psize = tf.reduce_sum(tf.cast(selected_pmask, tf.int32))
        nsize = num_samples-psize

        fnmask = tf.cast(nmask, tf.float32)
        # 负样本的概率为模型预测为负样本的概率，正样本的地方设置为1
        nclass_prediction = tf.where(nmask, probability[:, 0], 1.0 - fnmask)
        # 取预测的最不好的nsize个负样本
        selected_nmask = tf.logical_and(nmask, wmlt.bottom_k_mask(nclass_prediction, k=nsize))

        if global_cfg.GLOBAL.DEBUG:
            tf.summary.scalar("psize",psize)
            tf.summary.scalar("nsize",nsize)

        return selected_pmask,selected_nmask


