#coding=utf-8
import tensorflow as tf

def get_gan_losses_fn():
    def d_loss_fn(*,r_logits, f_logits):
        r_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.ones_like(r_logits), r_logits,loss_collection=None))
        f_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.zeros_like(f_logits), f_logits,loss_collection=None))
        return r_loss, f_loss

    def g_loss_fn(f_logits):
        f_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.ones_like(f_logits), f_logits,loss_collection=None))
        return f_loss

    return d_loss_fn, g_loss_fn

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

def get_wgan_losses_fn():
    def d_loss_fn(*,r_logits, f_logits):
        r_loss = - tf.reduce_mean(r_logits)
        f_loss = tf.reduce_mean(f_logits)
        return r_loss, f_loss

    def g_loss_fn(f_logits):
        f_loss = - tf.reduce_mean(f_logits)
        return f_loss
    return d_loss_fn, g_loss_fn