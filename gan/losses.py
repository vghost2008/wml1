#coding=utf-8
import tensorflow as tf

def get_hinge_v1_losses_fn():
    def d_loss_fn(*,r_logits, f_logits):
        r_loss = tf.reduce_mean(tf.maximum(1 - r_logits, 0))
        f_loss = tf.reduce_mean(tf.maximum(1 + f_logits, 0))
        return r_loss, f_loss

    def g_loss_fn(f_logits):
        f_loss = tf.reduce_mean(tf.maximum(1 - f_logits, 0))
        return f_loss

    return d_loss_fn, g_loss_fn


def get_hinge_v2_losses_fn():
    def d_loss_fn(*,r_logits, f_logits):
        r_loss = tf.reduce_mean(tf.maximum(1 - r_logits, 0))
        f_loss = tf.reduce_mean(tf.maximum(1 + f_logits, 0))
        return r_loss, f_loss

    def g_loss_fn(f_logits):
        f_loss = tf.reduce_mean(- f_logits)
        return f_loss

    return d_loss_fn, g_loss_fn