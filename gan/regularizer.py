#coding=utf-8
import tensorflow as tf
import wml_tfutils as wml
import numpy as np

'''
Improved Training of Wasserstein GANs
'''
def wgan_gp(images_c,images_g,D_fn,gp_lambda=0.1,scope=None):
    print("Discriminative gradient penalty loss")
    with tf.name_scope(scope,"GradientPenalty"):
        alpha = tf.random_uniform(shape=(), minval=0, maxval=1)
        interpolates = (images_c-images_g)* alpha + images_g
        a_logits_t = D_fn(images=interpolates, reuse=True)
        if isinstance(a_logits_t, tuple):
            with tf.name_scope("MergeLogits"):
                _al_list = []
                for l in a_logits_t:
                    _al_list.append(tf.reduce_mean(l))
                a_logits_t = tf.add_n(_al_list)
        gradients = tf.gradients(a_logits_t, [interpolates])[0]
        slopes = tf.norm(tf.layers.flatten(gradients),axis=1)
        #slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        a_loss = gp_lambda * gradient_penalty
        wml.variable_summaries_v2(a_loss,"gp_loss")
        wml.variable_summaries_v2(slopes,"gp_slopes")
        return a_loss

'''
Which Training Methods for GANs do actually Converge?
'''
def r1_regularizer(images,D_fn,r1_lambda=10,scope=None):
    print("Discriminative R1 regularizer loss.")
    with tf.name_scope(scope,"R1regularizer"):
        a_logits_t = D_fn(images=images)
        if not isinstance(images,list):
            images = [images]
        gradients = tf.gradients(a_logits_t, images,stop_gradients=images)
        res = []
        for gradient in gradients:
            axis = np.arange(1, gradient.get_shape().ndims) if gradient.get_shape().ndims is not 1 else None
            l2_squqred_grads = tf.reduce_sum(tf.square(gradient),axis=axis)
            res.append(l2_squqred_grads)
        if len(res) == 1:
            res = res[0]
        else:
            res = tf.add_n(res)
        #slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
        gradient_penalty = res*0.5
        a_loss = r1_lambda * gradient_penalty
        wml.variable_summaries_v2(a_loss,"r1_loss")
        return a_loss
