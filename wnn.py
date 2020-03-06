#coding=utf-8
import tensorflow as tf
import numpy as np
import logging
import wml_tfutils as wmlt
from wml_tfutils import *
import wml_utils as wmlu
from tensorflow.python.training import moving_averages
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from wtfop.wtfop_ops import slide_batch
import scipy
import img_utils
import re
import eval_toolkit as evt
import os
import copy
import functools
from collections import Iterable

slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS
g_bn_variables={}

def str2optimizer(name="Adam",learning_rate=None,**kwargs):
    print("Optimizer {}".format(name))
    opt = None
    if name== "Adam":
        opt = tf.train.AdamOptimizer(learning_rate,**kwargs)
    elif name== "GD":
        opt =  tf.train.GradientDescentOptimizer(learning_rate,**kwargs)
    elif name== "Momentum":
        opt = tf.train.MomentumOptimizer(learning_rate,**kwargs)
    elif name== "RMSProp":
        opt = tf.train.RMSPropOptimizer(learning_rate,**kwargs)
    else:
        raise ValueError("error optimizer")

    return opt

def piecewise_lr(initial_lr,step,steps,decay):
    with tf.name_scope("build_piecewise_lr"):
        steps = list(steps)
        begin_steps = [0]+steps[:-1]
        steps_pair = list(zip(begin_steps,steps))
        funs = {}
        lr = initial_lr
        def func(lr):
            return lr
        for i,(l,h) in enumerate(steps_pair):
            funs[tf.logical_and(tf.greater_equal(step,l),tf.less(step,h))] = functools.partial(func,lr)
            lr = lr*decay
        return tf.case(funs,default=functools.partial(func,lr))


def build_learning_rate(initial_lr,
                        global_step,
                        steps_per_epoch=None,
                        lr_decay_type='exponential',
                        decay_factor=0.97,
                        decay_epochs=2.4,
                        total_steps=None,
                        steps=None,
                        warmup_steps=1):
  with tf.name_scope("build_learning_rate"):
      if lr_decay_type == 'exponential':
        assert steps_per_epoch is not None
        decay_steps = steps_per_epoch * decay_epochs
        lr = tf.train.exponential_decay(
            initial_lr, global_step, decay_steps, decay_factor, staircase=True)
      elif lr_decay_type == 'cosine':
        assert total_steps is not None
        lr = 0.5 * initial_lr * (
            1 + tf.cos(np.pi * tf.cast(global_step, tf.float32) / total_steps))
      elif lr_decay_type == 'constant':
        lr = initial_lr
      elif lr_decay_type == "piecewise":
          assert steps is not None
          lr = piecewise_lr(initial_lr=initial_lr,step=global_step,steps=steps,decay=decay_factor)
      else:
        assert False, 'Unknown lr_decay_type : %s' % lr_decay_type

      if warmup_steps>2:
        logging.info('Learning rate warmup steps: %d', warmup_steps)
        warmup_steps = int(warmup_steps)
        warmup_lr = (
            initial_lr * tf.cast(global_step, tf.float32) / tf.cast(
                warmup_steps, tf.float32))
        lr = tf.cond(global_step < warmup_steps, lambda: warmup_lr, lambda: lr)

      return lr

def get_train_op(global_step,batch_size=32,learning_rate=1E-3,scopes=None,scopes_pattern=None,clip_norm=None,loss=None,
                 colocate_gradients_with_ops=False,optimizer="Adam",scope=None,num_epochs_per_decay=None):
    with tf.name_scope(name=scope,default_name="train_op"):
        num_batches_per_epoch=float(FLAGS.example_size)/batch_size
        if num_epochs_per_decay is None:
            num_epochs_per_decay=FLAGS.num_epochs_per_decay
        learn_rate_decay_factor=FLAGS.learn_rate_decay_factor
        min_learn_rate = FLAGS.min_learn_rate
        steps_per_epoch = num_batches_per_epoch
        lr = build_learning_rate(learning_rate,
                        global_step,
                        steps_per_epoch=steps_per_epoch,
                        lr_decay_type='exponential',
                        decay_factor=learn_rate_decay_factor,
                        decay_epochs=num_epochs_per_decay,
                        total_steps=None,
                        warmup_steps=1000)
        lr = tf.maximum(min_learn_rate,lr)
        tf.summary.scalar("lr",lr)
        #opt = tf.train.GradientDescentOptimizer(lr)
        variables_to_train = get_variables_to_train(scopes,scopes_pattern)
        show_values(variables_to_train,"variables_to_train",fn=logging.info)
        logging.info("Total train variables num %d."%(parameterNum(variables_to_train)))
        variables_not_to_train = get_variables_not_to_train(variables_to_train)
        show_values(variables_not_to_train,"variables_not_to_train",fn=logging.info)
        logging.info("Total not train variables num %d."%(parameterNum(variables_not_to_train)))
        opt = str2optimizer(optimizer,lr)

        if loss is not None:
            total_loss = loss
        else:
            loss_wr = get_regularization_losses(scopes=scopes,re_pattern=scopes_pattern)
            if loss_wr is not None:
                tf.losses.add_loss(loss_wr)
                wmlt.variable_summaries_v2(loss_wr,"wr_loss")
            losses = tf.losses.get_losses()
            show_values(losses,"Losses")
            total_loss = tf.add_n(losses, "total_loss")
            total_loss = tf.reduce_sum(total_loss)

        if clip_norm is not None:
            grads, global_norm = tf.clip_by_global_norm(tf.gradients(total_loss, variables_to_train,colocate_gradients_with_ops=colocate_gradients_with_ops),
                                                        clip_norm)
            apply_gradient_op = opt.apply_gradients(zip(grads, variables_to_train), global_step=global_step)
        else:
            grads = opt.compute_gradients(total_loss, variables_to_train,colocate_gradients_with_ops=colocate_gradients_with_ops)
            apply_gradient_op = opt.apply_gradients(grads,global_step=global_step)

        slim_batch_norm_ops = get_batch_norm_ops(scopes=scopes,re_pattern=scopes_pattern)
        train_op = tf.group(apply_gradient_op,slim_batch_norm_ops)
        if clip_norm:
            tf.summary.scalar("global_norm", global_norm)
        return train_op,total_loss,variables_to_train

def nget_train_op(global_step,lr=None,scopes=None,scopes_pattern=None,clip_norm=None,loss=None,
                 colocate_gradients_with_ops=False,optimizer="Adam",scope=None):
    with tf.name_scope(name=scope,default_name="train_op"):
        variables_to_train = get_variables_to_train(scopes,scopes_pattern)
        show_values(variables_to_train,"variables_to_train",fn=logging.info)
        logging.info("Total train variables num %d."%(parameterNum(variables_to_train)))
        variables_not_to_train = get_variables_not_to_train(variables_to_train)
        show_values(variables_not_to_train,"variables_not_to_train",fn=logging.info)
        logging.info("Total not train variables num %d."%(parameterNum(variables_not_to_train)))

        if lr is not None:
            tf.summary.scalar("lr",lr)

        if isinstance(optimizer,str):
            opt = str2optimizer(optimizer,lr)
        else:
            opt = optimizer

        if loss is not None:
            total_loss = loss
        else:
            loss_wr = get_regularization_losses(scopes=scopes,re_pattern=scopes_pattern)
            if loss_wr is not None:
                tf.losses.add_loss(loss_wr)
                wmlt.variable_summaries_v2(loss_wr,"wr_loss")
            losses = tf.losses.get_losses()
            show_values(losses,"Losses")
            total_loss = tf.add_n(losses, "total_loss")
            total_loss = tf.reduce_sum(total_loss)

        if clip_norm is not None:
            grads, global_norm = tf.clip_by_global_norm(tf.gradients(total_loss, variables_to_train,colocate_gradients_with_ops=colocate_gradients_with_ops),
                                                        clip_norm)
            apply_gradient_op = opt.apply_gradients(zip(grads, variables_to_train), global_step=global_step)
        else:
            grads = opt.compute_gradients(total_loss, variables_to_train,colocate_gradients_with_ops=colocate_gradients_with_ops)
            apply_gradient_op = opt.apply_gradients(grads,global_step=global_step)

        slim_batch_norm_ops = get_batch_norm_ops(scopes=scopes,re_pattern=scopes_pattern)
        train_op = tf.group(apply_gradient_op,slim_batch_norm_ops)
        if clip_norm:
            tf.summary.scalar("global_norm", global_norm)
        return train_op,total_loss,variables_to_train

def get_batch_norm_ops(scopes=None,re_pattern=None):
    bn_ops = get_variables_of_collection(key=tf.GraphKeys.UPDATE_OPS,scopes=scopes,re_pattern=re_pattern)
    show_values_name(bn_ops,"batch_norm_ops",fn=logging.info)
    return bn_ops

def get_update_batch_norm_ops(global_step,scopes=None,re_pattern=None):
    bn_ops = get_variables_of_collection(key=tf.GraphKeys.UPDATE_OPS,scopes=scopes,re_pattern=re_pattern)
    values = get_variables_of_collection(key=tf.GraphKeys.GLOBAL_VARIABLES,scopes=scopes,re_pattern=".*moving_.*")
    show_values_name(bn_ops,"batch_norm_ops",fn=logging.info)
    global_step = tf.assign_add(global_step,tf.constant(1,dtype=global_step.dtype))
    losses = tf.losses.get_losses()
    show_values(losses,"Losses")
    total_loss = tf.add_n(losses, "total_loss")
    total_loss = tf.reduce_sum(total_loss)
    op = tf.group(global_step,bn_ops)
    return op,total_loss,values

'''
主要用于在不同时间在同一个GPU上训练同一个网络 
'''
def get_train_opv2(global_step,batch_size=32,learning_rate=1E-3,scopes=None,clip_norm=None,loss=None):
    '''
    train_op,apply_grads_op,grads,grads_holder,_,_ = get_train_opv2(global_step=global_step,batch_size=1,learning_rate=.5)
    pgs = pure_grads(grads)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        gs = []
        for j in range(3):
            g,_ = sess.run([pgs,train_op])
            gs.append(g)
        avg_grads = average_npgrads(gs)
        feed_dict = dict(zip(grads_holder,avg_grads))
        _,p_x,p_y = sess.run([apply_grads_op,x,y],feed_dict=feed_dict)
        print(p_x,p_y)
    '''

    with tf.name_scope("train_op"):
        num_batches_per_epoch=float(FLAGS.example_size)/batch_size
        num_epochs_per_decay=FLAGS.num_epochs_per_decay
        learn_rate_decay_factor=FLAGS.learn_rate_decay_factor
        min_learn_rate = FLAGS.min_learn_rate

        decay_step = int(num_batches_per_epoch*num_epochs_per_decay)
        lr = tf.train.exponential_decay(learning_rate,global_step,decay_step,learn_rate_decay_factor,staircase=True)
        lr = tf.maximum(min_learn_rate,lr)
        tf.summary.scalar("lr",lr)
        variables_to_train = get_variables_to_train(scopes)
        show_values(variables_to_train,"variables_to_train",fn=logging.info)
        logging.info("Total train variables num %d."%(parameterNum(variables_to_train)))
        opt = tf.train.AdamOptimizer(lr)

        if loss is not None:
            total_loss = loss
        else:
            loss_wr = get_regularization_losses(scopes=scopes)
            if loss_wr is not None and len(loss_wr)>0:
                tf.losses.add_loss(loss_wr)
            total_loss = tf.add_n(tf.losses.get_losses(), "total_loss")
            total_loss = tf.reduce_sum(total_loss)

        grads = opt.compute_gradients(total_loss, variables_to_train)
        slim_batch_norm_ops = get_batch_norm_ops(scopes=scopes)
        train_op = slim_batch_norm_ops
        apply_grad_op,grads_holder = apply_gradients(grads,global_step,opt,clip_norm=clip_norm)
        return train_op,apply_grad_op,grads,grads_holder,total_loss,variables_to_train

def get_optimizer(global_step,learning_rate=1E-3,batch_size=32,optimizer="Adam",num_epochs_per_decay=None,
                  example_size=None,
                  learn_rate_decay_factor=None,min_learn_rate=None):
    if example_size is None:
        example_size = FLAGS.example_size
    if num_epochs_per_decay is None:
        num_epochs_per_decay = FLAGS.num_epochs_per_decay
    if learn_rate_decay_factor is None:
        learn_rate_decay_factor =FLAGS.learn_rate_decay_factor
    if min_learn_rate is None:
        min_learn_rate = FLAGS.min_learn_rate
    num_batches_per_epoch=float(example_size)/batch_size

    decay_step = int(num_batches_per_epoch*num_epochs_per_decay)
    lr = tf.train.exponential_decay(learning_rate,global_step,decay_step,learn_rate_decay_factor,staircase=True)
    lr = tf.maximum(min_learn_rate,lr)
    tf.summary.scalar("lr",lr)
    opt = str2optimizer(optimizer,lr)

    return opt

def get_optimizerv2(lr=1E-3,optimizer="Adam"):
    tf.summary.scalar("lr",lr)
    opt = str2optimizer(optimizer,lr)
    return opt

'''
主要用于同时在不同的GPU上训练同一个网络
'''
def get_train_opv3(optimizer,scopes=None,re_pattern=None,loss=None):
    '''
    opt = get_optimizer(global_step,learning_rate=.5,batch_size=1)
    for i in range(2):
        with tf.device("/cpu:{}".format(0)):
            with tf.name_scope("cpu_{}".format(i)):
                loss = tf.pow(x-10.0,2)+9.0+tf.pow(y-5.,2)
                tf.losses.add_loss(tf.reduce_sum(loss))

                grads,_,_ = get_train_opv3(optimizer=opt,loss=loss)
                tower_grads.append(grads)

    avg_grads = average_grads(tower_grads)
    opt0 = apply_gradientsv3(avg_grads,global_step,opt)
    opt1 = get_batch_norm_ops()
    train_op = tf.group(opt0,opt1)
    :param optimizer:
    :param scopes:
    :param loss:
    :return:
    '''
    with tf.name_scope("train_op"):
        variables_to_train = get_variables_to_train(scopes,re_pattern=re_pattern)
        show_values(variables_to_train,"variables_to_train",fn=logging.info)
        logging.info("Total train variables num %d."%(parameterNum(variables_to_train)))

        if loss is not None:
            total_loss = loss
        else:
            loss_wr = get_regularization_losses(scopes=scopes,re_pattern=re_pattern)
            tf.losses.add_loss(loss_wr)
            total_loss = tf.add_n(tf.losses.get_losses(), "total_loss")
            total_loss = tf.reduce_sum(total_loss)

        grads = optimizer.compute_gradients(total_loss, variables_to_train)
        return grads,total_loss,variables_to_train

'''
主要用于同时在不同的GPU上训练同一个网络
'''
def nget_train_opv3(optimizer,scopes=None,re_pattern=None,loss=None):
    '''
    opt = get_optimizer(global_step,learning_rate=.5,batch_size=1)
    for i in range(2):
        with tf.device("/cpu:{}".format(0)):
            with tf.name_scope("cpu_{}".format(i)):
                loss = tf.pow(x-10.0,2)+9.0+tf.pow(y-5.,2)
                tf.losses.add_loss(tf.reduce_sum(loss))

                grads,_,_ = get_train_opv3(optimizer=opt,loss=loss)
                tower_grads.append(grads)

    avg_grads = average_grads(tower_grads)
    opt0 = apply_gradientsv3(avg_grads,global_step,opt)
    opt1 = get_batch_norm_ops()
    train_op = tf.group(opt0,opt1)
    :param optimizer:
    :param scopes:
    :param loss:
    :return:
    '''
    with tf.name_scope("train_op"):
        variables_to_train = get_variables_to_train(scopes,re_pattern=re_pattern)
        show_values(variables_to_train,"variables_to_train",fn=logging.info)
        logging.info("Total train variables num %d."%(parameterNum(variables_to_train)))

        if not isinstance(loss,Iterable):
            loss = [loss]
        total_loss = list(loss)
        loss_wr = get_regularization_losses(scopes=scopes,re_pattern=re_pattern,reduction=None)
        if loss_wr is not None:
            total_loss = total_loss+loss_wr
        total_loss = tf.add_n(total_loss)

        grads = optimizer.compute_gradients(total_loss, variables_to_train)
        return grads,total_loss,variables_to_train

def pure_grads(grads):
    pg=[]
    for g in grads:
        pg.append(g[0])
    return pg

def apply_gradients(grads,global_step,optimizer,clip_norm=None):
    '''
    grads:
    '''
    with tf.name_scope("train_op"):
        grads_holder = []
        vars = []
        for g in grads:
            grads_holder.append(tf.placeholder(dtype=tf.float32))
            vars.append(g[1])
        if clip_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads_holder, clip_norm)
            op = optimizer.apply_gradients(zip(grads,vars),global_step=global_step)
        else:
            op = optimizer.apply_gradients(zip(grads_holder,vars),global_step=global_step)
        return op,grads_holder

def apply_gradientsv3(grads,global_step,optimizer,clip_norm=None):
    with tf.name_scope("train_op"):
        pg=[]
        vars=[]
        for g in grads:
            pg.append(g[0])
            vars.append(g[1])

        if clip_norm is not None:
            grads, _ = tf.clip_by_global_norm(pg, clip_norm)
            op = optimizer.apply_gradients(zip(grads,vars),global_step=global_step)
        else:
            op = optimizer.apply_gradients(grads,global_step=global_step)
        return op

def average_grads(tower_grads,clip_norm=None):
    with tf.name_scope("average_grads"):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g,_v in grad_and_vars:
                if g is None:
                    print(_v,"have no grads.")
                    continue
                expanded_g = tf.expand_dims(g,0)
                grads.append(expanded_g)
            if len(grads)>0:
                grad = tf.concat(values=grads,axis=0)
                grad = tf.reduce_mean(grad,0)
            else:
                grad = None

            v = grad_and_vars[0][1]
            grad_and_var = (grad,v)
            average_grads.append(grad_and_var)

        if clip_norm is not None:
            grads,vars = zip(*average_grads)
            grads, global_norm = tf.clip_by_global_norm(grads, clip_norm)
            tf.summary.scalar("global_norm", global_norm)
            average_grads = list(zip(grads,vars))

        return average_grads

def average_npgrads(grads_list):
    with tf.name_scope("train_op"):
        grads_avg = []
        for gs in zip(*grads_list):
            gs = np.stack(gs,axis=0)
            grads_avg.append(np.mean(gs,axis=0))
        return grads_avg

'''def get_train_opv3(global_step,total_num_replicas,batch_size=32,learning_rate=1E-3,scopes=None,clip_norm=None,loss=None):
    num_batches_per_epoch=float(FLAGS.example_size)/batch_size
    num_epochs_per_decay=FLAGS.num_epochs_per_decay
    learn_rate_decay_factor=FLAGS.learn_rate_decay_factor
    min_learn_rate = FLAGS.min_learn_rate

    decay_step = int(num_batches_per_epoch*num_epochs_per_decay)
    lr = tf.train.exponential_decay(learning_rate,global_step,decay_step,learn_rate_decay_factor,staircase=True)
    lr = tf.maximum(min_learn_rate,lr)
    tf.summary.scalar("lr",lr)
    #opt = tf.train.GradientDescentOptimizer(lr)
    variables_to_train = get_variables_to_train(scopes)
    show_values(variables_to_train,"variables_to_train")
    print("Total train variables num %d."%(parameterNum(variables_to_train)))
    opt = tf.train.AdamOptimizer(lr)

    if loss is not None:
        total_loss = loss
    else:
        total_loss = tf.add_n(tf.losses.get_losses(), "total_loss")
        total_loss = tf.reduce_sum(total_loss)

    if clip_norm is not None:
        grads, global_norm = tf.clip_by_global_norm(tf.gradients(total_loss, variables_to_train), clip_norm)
        rep_op = tf.train.SyncReplicasOptimizer(opt,
                                                replicas_to_aggregate=total_num_replicas,
                                                total_num_replicas=total_num_replicas,
                                                use_locking=False)
        apply_gradient_op = opt.apply_gradients(zip(grads, variables_to_train), global_step=global_step)
    else:
        grads = opt.compute_gradients(total_loss, variables_to_train)
        rep_op = tf.train.SyncReplicasOptimizer(opt,
                                                replicas_to_aggregate=total_num_replicas,
                                                total_num_replicas=total_num_replicas,
                                                use_locking=False)
        apply_gradient_op = opt.apply_gradients(grads,global_step=global_step)
    init_token_op = rep_op.get_init_tokens_op()
    chief_queue_runner = rep_op.get_chief_queue_runner()

    slim_batch_norm_ops = get_batch_norm_ops(scopes=scopes)
    train_op = tf.group(apply_gradient_op,slim_batch_norm_ops)
    if clip_norm:
        tf.summary.scalar("global_norm", global_norm)
    return train_op,total_loss,variables_to_train'''

def get_variables_to_train(trainable_scopes,re_pattern=None):
    return get_variables_of_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scopes=trainable_scopes,re_pattern=re_pattern)

def get_variables_not_to_train(train_variables):
    variables = tf.trainable_variables()
    _variables = list(variables)
    for v in _variables:
        if v in train_variables:
            variables.remove(v)

    return variables


def get_variables_exclude(exclude=None,only_scope=None,key=None):
    if key is None:
        key = tf.GraphKeys.TRAINABLE_VARIABLES
    if exclude is None and only_scope is None:
        return tf.get_collection(key)
    if only_scope is not None:
        res_variables = []
        scopes = [scope.strip() for scope in only_scope.split(',')]
        for scope in scopes:
            variables = tf.get_collection(key,scope)
            res_variables.extend(variables)
    else:
        res_variables = tf.get_collection(key)

    if exclude is not None:
        if isinstance(exclude,str):
            scopes = [scope.strip() for scope in exclude.split(',')]
            variables_to_exclude=[]
            for scope in scopes:
                variables = tf.get_collection(key,scope)
                variables_to_exclude.extend(variables)
            for v in variables_to_exclude:
                if v in res_variables:
                    res_variables.remove(v)
        elif callable(exclude):
            clone_res = [x.name for x in res_variables]
            remove_names = []
            for name in clone_res:
                if exclude(name):
                    remove_names.append(name)

            res_variables = list(filter(lambda x:x.name not in remove_names,res_variables))
    return res_variables

def restore_variables_by_key(sess,file_path, exclude=None,only_scope=None,key=None,name=None,silent=False,value_key=None):
    if key is None:
        key = tf.GraphKeys.TRAINABLE_VARIABLES
    variables_to_restore = get_variables_exclude(exclude,only_scope,key)
    if len(variables_to_restore) == 0:
        return []
    if not silent:
        show_values(variables_to_restore, name+"_variables_to_restore",fn=logging.info)
    if value_key is not None:
        _variables_to_restore = variables_to_restore
        variables_to_restore = {}
        for v in _variables_to_restore:
            variables_to_restore[value_key(v)] = v

    if not silent:
        logging.info(name+"_variables_to_restore:"+str(parameterNum(variables_to_restore)))

    restorer = tf.train.Saver(variables_to_restore)

    if file_path is not None:
        logging.info("Restore values from"+file_path)
        restorer.restore(sess, file_path)
        return file_path,variables_to_restore
    return []

def restore_variables_by_var_list(sess,file_path,vars):
    if not isinstance(vars,list):
        vars = [vars]
    restorer = tf.train.Saver(vars)

    if file_path is not None:
        logging.info("Restore values from"+file_path)
        restorer.restore(sess, file_path)
        return file_path,vars
    return []

'''
value_key: a callable object trans variable(tf.Tensor) to it's save or restore name, for example 
def func(v)
    if v.name == "v1":
       return "model/v1"
'''
def restore_variables(sess,path,exclude=None,only_scope=None,silent=False,restore_evckp=True,value_key=None,exclude_var=None,extend_vars=None,global_scopes=None,verbose=False):
    #if restore_evckp and os.path.isdir(path):
    #    evt.WEvalModel.restore_ckp(FLAGS.check_point_dir)
    if exclude is None and exclude_var is not None:
        exclude = exclude_var
    file_path = wmlt.get_ckpt_file_path(path)
    if file_path is None:
        return False
    print(f"resotre from {file_path}.")
    for v in tf.global_variables():
        if "moving_mean" in v.name or "moving_variance" in v.name:
            tf.add_to_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES,v)
    variables = []
    variables0 = restore_variables_by_key(sess,file_path,exclude,only_scope,key=tf.GraphKeys.TRAINABLE_VARIABLES,name="train",silent=silent,value_key=value_key)

    if len(variables0)>1:
        if isinstance(variables0[1],list):
            for v in variables0[1]:
                variables.append(v.name)
        else:
            for v in variables0[1]:
                variables.append(v)
    variables1 = restore_variables_by_key(sess,file_path,exclude,only_scope,key=tf.GraphKeys.MOVING_AVERAGE_VARIABLES,name='moving',silent=silent,value_key=value_key)
    if len(variables1)>1:
        if isinstance(variables1[1],list):
            for v in variables1[1]:
                variables.append(v.name)
        else:
            for v in variables1[1]:
                variables.append(v)

    if global_scopes is not None:
        variables2 = restore_variables_by_key(sess,file_path,None,global_scopes,key=tf.GraphKeys.GLOBAL_VARIABLES,name='global_variables',silent=silent,value_key=value_key)
        if len(variables2)>1:
            if isinstance(variables2[1],list):
                for v in variables2[1]:
                    variables.append(v.name)
            else:
                for v in variables2[1]:
                    variables.append(v)

    if extend_vars is not None:
        variables2 = restore_variables_by_var_list(sess,file_path,extend_vars)
        if len(variables2)>1:
            if isinstance(variables2[1],list):
                for v in variables2[1]:
                    variables.append(v.name)
            else:
                for v in variables2[1]:
                    variables.append(v)

    for i,v in enumerate(variables):
        index = v.find(':')
        if index>0:
            variables[i] = variables[i][:index]
    if not silent:
        unrestored_variables = wmlt.get_variables_unrestored(variables,file_path,exclude_var="Adam")
        unrestored_variables0 = wmlt.get_variables_unrestoredv1(variables,exclude_var="Adam")
        if not verbose:
            def v_filter(x:str):
                #return (not x.endswith("ExponentialMovingAverage")) and (not x.endswith("/u"))
                return (not x.endswith("ExponentialMovingAverage")) and (not x.endswith("Momentum"))
            unrestored_variables = filter(v_filter,unrestored_variables)
            unrestored_variables0 = filter(v_filter,unrestored_variables0)
        show_values(unrestored_variables, "Unrestored variables in ckpt")
        show_values(unrestored_variables0, "Unrestored variables of global variables")
    return True

def accuracy_ratio(logits,labels):
    with tf.name_scope("accuracy_ratio"):
        _,actually_get = tf.nn.top_k(logits,1)
        dim = len(actually_get.get_shape().as_list())-1
        actually_get = tf.squeeze(actually_get,axis=[dim])
        if labels.dtype != actually_get.dtype:
            actually_get = tf.cast(actually_get,labels.dtype)
        correct_num = tf.reduce_sum(tf.cast(tf.equal(actually_get,labels),tf.float32))
        all_num = tf.cast(wmlt.num_elements(labels),tf.float32)
        return correct_num*100./all_num

def accuracy_num(logits,labels):
    with tf.name_scope("accuracy_num"):
        _,actually_get = tf.nn.top_k(logits,1)
        dim = len(actually_get.get_shape().as_list())-1
        actually_get = tf.squeeze(actually_get,axis=[dim])
        if labels.dtype != actually_get.dtype:
            actually_get = tf.cast(actually_get,labels.dtype)
        correct_num = tf.reduce_sum(tf.cast(tf.equal(actually_get,labels),tf.float32))

    return correct_num,tf.reduce_prod(tf.shape(labels))

def get_regularization_losses(scopes=None,re_pattern=None,reduction="mean"):
    with tf.name_scope("regularization_losses"):
        col = get_variables_of_collection(tf.GraphKeys.REGULARIZATION_LOSSES,scopes=scopes,re_pattern=re_pattern)
        if len(col)>0:
            print("wr_loss")
            wmlu.show_list(col)
            if reduction == "mean":
                return tf.reduce_mean(col)
            elif reduction == "sum":
                return tf.reduce_sum(col)
            else:
                return col
        else:
            return None

'''
for multi classifier problem
logits:[batch_size,num_classes]
labels:[batch_size,num_classes]
'''
def iou_of_multi_label(logits,labels,threshold=0.5):
    expected = tf.greater(labels,threshold)
    predicted = tf.greater(tf.nn.sigmoid(logits),threshold)
    union = tf.logical_or(expected,predicted)
    union = tf.reduce_sum(tf.cast(union,tf.float32),axis=1)
    intersection = tf.logical_and(expected,predicted)
    intersection = tf.reduce_sum(tf.cast(intersection,tf.float32),axis=1)
    def fn(u,i):
        return tf.cond(tf.greater(u,0.5),lambda:i/u,lambda:1.0)
    eval_results = tf.map_fn(lambda x:fn(x[0],x[1]),(union,intersection),dtype=tf.float32,back_prop=False)
    return eval_results

def miou_of_multi_label(logits,labels,threshold=0.5):
    return tf.reduce_mean(iou_of_multi_label(logits,labels,threshold))


#每个小项单独计算正确率
def sigmoid_accuracy_ratio(logits,labels,threshold=0.5):
    with tf.name_scope("accuracy_ratio"):
        prob = tf.sigmoid(logits)
        actually_get = prob>threshold
        actually_get = tf.cast(actually_get,labels.dtype)
        correct_num = tf.reduce_sum(tf.cast(tf.equal(actually_get,labels),tf.float32))
        all_num = tf.cast(wmlt.num_elements(labels),tf.float32)
        return correct_num*100./all_num


#每一个小项都正确才算正确
def sigmoid_accuracy_ratiov2(logits,labels,threshold=0.5):
    with tf.name_scope("accuracy_ratio"):
        prob = tf.sigmoid(logits)
        if prob.get_shape().ndims>1:
            classes_nr = prob.get_shape().as_list()[1]
        if labels.dtype==tf.float32:
            labels = labels>threshold
            labels = tf.cast(labels,tf.int32)
        batch_size = tf.cast(tf.shape(prob)[0],tf.float32)
        actually_get = prob>threshold
        actually_get = tf.cast(actually_get,labels.dtype)
        if prob.get_shape().ndims>1:
            correct_num = tf.reduce_sum(tf.cast(tf.equal(actually_get,labels),tf.float32),axis=1)
            correct_num = tf.reduce_sum(tf.cast(tf.equal(correct_num,classes_nr),tf.float32))
        else:
            correct_num = tf.reduce_sum(tf.cast(tf.equal(actually_get,labels),tf.float32))

        return correct_num*100./batch_size

def get_prediction(logits):
    with tf.name_scope("prediction"):
        probabilitys = tf.nn.softmax(logits)
        #return probabilitys
        probability,indices = tf.nn.top_k(probabilitys,1)
        return probability,indices

class WHotGraph:
    def __init__(self,evaluator,batch_size,mask,sess,strides=None):
        self.image = tf.placeholder(tf.float32,shape=[None,None,3])
        self.image_batchs = slide_batch(data=self.image,filter=tf.zeros(mask),strides=strides,padding="SAME")
        self.evaluator = evaluator
        self.batch_size = batch_size
        self.sess = sess

    def __call__(self, img):
        img = self.evaluator.preprocess(img)
        res = self.evaluator(np.expand_dims(img_utils.npgray_to_rgb(img),axis=0))
        if res[0]< 0.5:
            print("Current image is negative.")
            return None,None

        image_batchs = self.sess.run(self.image_batchs,feed_dict={self.image:img})
        shape = list(image_batchs.shape)
        image_shape = list(img.shape)
        image_batchs = np.reshape(image_batchs,[-1]+shape[-3:])
        total_nr = image_batchs.shape[0]
        processed_nr = 0
        hot_value = None
        batch_size = self.batch_size

        while processed_nr<total_nr:
            s = total_nr-processed_nr if total_nr-processed_nr<=batch_size else batch_size
            res = self.evaluator(image_batchs[processed_nr:processed_nr+s,:,:,:])
            processed_nr += s

            if hot_value is None:
                hot_value = res
            else:
                hot_value = np.concatenate([hot_value,res],axis=0)
        #hot_value = np.array(range(shape[0]*shape[1]),dtype=np.float32)/(shape[0]*shape[1])

        hot_value = np.reshape(hot_value,[shape[0],shape[1]])
        hot_value = scipy.ndimage.zoom(hot_value,[float(image_shape[0])/shape[0],float(image_shape[1])/shape[1]])
        '''for i in range(image_shape[0]):
            for j in range(image_shape[1]):
                hot_value[i][j] = float(j)/image_shape[1]'''

        return img_utils.npgray_to_rgbv2(hot_value),img_utils.merge_hotgraph_image(img,hot_value,alpha=0.4)


'''
每一个stage layer的shape都会减小1/2
output_shape约为input_shape/2^stage
'''
def get_layer_shape(input_shape,stage):
    output_shape = input_shape
    for _ in range(stage):
        output_shape = [int((x+1)/2) for x in output_shape]
    return output_shape

'''
用于计算分级的soft_max损失
N:第一层的类别数
Y:第二层的最大类别数
logits:[batch_size,N,Y], 用于表示每一个batch的每个第一层类别下第二层对应类别的logits
labels:[batch_size,2], 用于表示每一个batch的第一层类别及第二层类别
num_classes:[Y],用于表示每一个子类的类别数
return:
[batch_size]
'''
def hierarchical_sparse_softmax_cross_entropy(logits,labels,num_classes,scope="hierarchical_sparse_softmax_cross_entropy"):
    with tf.name_scope(scope):
        batch_size = tf.shape(logits)[0]
        loss = tf.cond(batch_size>0,lambda:_hierarchical_sparse_softmax_cross_entropy(logits,labels,num_classes),lambda:0.)
        return tf.reshape(loss,shape=[batch_size])

def _hierarchical_sparse_softmax_cross_entropy(logits,labels,num_classes):
    labels0 = labels[:,0]
    labels1 = labels[:,1]

    def singal_loss(slogits,label0,label1):
        slogits = tf.gather(slogits,label0)
        len = tf.gather(num_classes,label0)
        len = tf.reshape(len,[1])
        slogits = tf.slice(slogits,[0],len)
        label1 = tf.reshape(label1,[1])
        slogits = tf.reshape(slogits,[1,-1])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=slogits,labels=label1)
        return tf.reshape(loss,shape=())

    losses = tf.map_fn(lambda x:singal_loss(*x),(logits,labels0,labels1),dtype=tf.float32)
    return losses
'''
get the label in second level
N: the branch size
Y: maximum classes number in second level 
logits:[batch_size,N,Y]
fl_labels: [batch_size,N]
num_classes:[N]
return:
the label in second level
'''
def hierarchical_prediction(logits,fl_labels,num_classes,scope="hierarchical_prediction"):

    a_op = tf.Assert(tf.less(tf.reduce_max(fl_labels),2),[fl_labels])
    with tf.control_dependencies([a_op]):
        logits = wmlt.select_2thdata_by_index_v2(logits,fl_labels)

    with tf.name_scope(scope):
        def singal_loss(slogits,label0):
            len = tf.gather(num_classes,label0)
            len = tf.reshape(len,[1])
            slogits = tf.slice(slogits,[0],len)
            prob = tf.nn.softmax(slogits)
            prob,label = tf.nn.top_k(prob,k=1)
            return label,prob

        labels,probs = tf.map_fn(lambda x:singal_loss(*x),(logits,fl_labels),dtype=(tf.int32,tf.float32))
        if probs.get_shape().ndims>1:
            probs = tf.squeeze(probs,axis=1)
        if labels.get_shape().ndims > 1:
            labels = tf.squeeze(labels, axis=1)

    return labels,probs

'''
input0:[batch_size,W,H,C0]
input1:[batch_size,1,1,C1]
将input1扩展为[batch_size,W,H,C1]后与input0 concat在一起
'''
def concat_conv(input0,input1):
    shape0 = tf.shape(input0)
    shape1 = tf.shape(input1)
    return tf.concat([input0,tf.ones([shape0[0],shape0[1],shape0[2],shape1[3]])*input1],axis=3)
'''
net: [batch_size,1,1,C]
output: [batch_size,size[0],size[1],C]
'''
def expand_spatial(net,size):
    shape = tf.shape(net)
    return tf.ones([shape[0], size[0], size[1], shape[3]]) * net

def get_variables_of_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,scopes=None,re_pattern=None):
    scopes_list = []
    if scopes is None and re_pattern is None:
        return tf.get_collection(key)
    elif isinstance(scopes,str):
        scopes_list = [scope.strip() for scope in scopes.split(',')]
    elif isinstance(scopes,list):
        scopes_list = scopes

    variables_to_return = []
    if scopes is not None:
        for scope in scopes_list:
            variables = tf.get_collection(key, scope)
            variables_to_return.extend(variables)
    else:
        variables_to_return = tf.get_collection(key)

    if re_pattern is not None:
        pattern = re.compile(re_pattern)
        variables_to_return = list(filter(lambda x: pattern.match(x.name) is not None,variables_to_return))
    return variables_to_return

'''
alpha:每个类别的权重，一般为样本中类别数的逆频率
'''
def sparse_softmax_cross_entropy_with_logits_FL(
    _sentinel=None,  # pylint: disable=invalid-name
    labels=None,
    logits=None,
    gamma=2.,
    alpha=None,
    name=None):
    with tf.variable_scope(name,default_name="sparse_softmax_cross_entropy_with_logits_FL"):
        per_entry_cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits)
        probability = tf.nn.softmax(logits)
        #labels = tf.expand_dims(labels,axis=-1)
        #r_probability = wmlt.batch_gather(probability,labels)
        #r_probability = tf.squeeze(r_probability,axis=-1)
        oh_labels = tf.one_hot(labels,depth=probability.get_shape().as_list()[-1],on_value=True,off_value=False,dtype=tf.bool)
        r_probability = tf.boolean_mask(probability,oh_labels)

        beta = tf.pow((1.-r_probability),gamma)
        if alpha is not None:
            alpha = tf.expand_dims(alpha,axis=0)*tf.ones_like(logits);
            alpha_t = tf.boolean_mask(alpha,oh_labels)
            alpha_t = tf.squeeze(alpha_t,axis=-1)
            beta = alpha_t*beta
        loss = beta*per_entry_cross_ent
        return loss

def sigmoid_cross_entropy_with_logits_FL(  # pylint: disable=invalid-name
    _sentinel=None,
    labels=None,
    logits=None,
    gamma=2.0,
    alpha=0.25,
    name=None):
    with tf.name_scope(name,default_name="sigmoid_cross_entropy_with_logits_FL"):
        per_entry_cross_ent = (tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=logits))
        prediction_probabilities = tf.sigmoid(logits)
        p_t = ((labels* prediction_probabilities) +
               ((1 - labels) * (1 - prediction_probabilities)))
        modulating_factor = 1.0
        if gamma is not None:
            modulating_factor = tf.pow(1.0 - p_t, gamma)
        alpha_weight_factor = 1.0
        if alpha is not None:
          alpha_weight_factor = (labels* alpha +
                                 (1 - labels) * (1 - alpha))
        focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor *
                                    per_entry_cross_ent)
        return focal_cross_entropy_loss

def sparse_softmax_cross_entropy_with_logits_alpha_balanced(
        _sentinel=None,  # pylint: disable=invalid-name
        labels=None,
        logits=None,
        alpha="auto",
        max_alpha_scale=10.0,
        name=None):
    if alpha is None:
        return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits,name=name)
    elif alpha != "auto":
        raise ValueError(f"alphpa shold be None or auto, not {alpha}.")
    with tf.variable_scope(name,default_name="sparse_softmax_cross_entropy_with_logits_alpha_balanced"):
        probability = tf.nn.softmax(logits)
        labels = tf.expand_dims(labels,axis=-1)
        r_probability = wmlt.batch_gather(probability,labels)
        r_probability = tf.squeeze(r_probability,axis=-1)
        r_probability = tf.maximum(1e-10*(1+r_probability),r_probability)
        num_classes = logits.get_shape().as_list()[-1]
        labels = tf.squeeze(labels,axis=-1)
        one_hot_labels = tf.one_hot(indices=labels,depth=num_classes)
        count = tf.reduce_sum(one_hot_labels,axis=list(range(one_hot_labels.get_shape().ndims-1)),keepdims=True)
        count = tf.cast(tf.maximum(count,1),tf.float32)
        ratio = tf.cast(tf.reduce_prod(tf.shape(labels)),tf.float32)/(count*float(num_classes))
        alpha_t = tf.minimum(max_alpha_scale,1.0/ratio)
        alpha_t = alpha_t*tf.ones_like(one_hot_labels,dtype=tf.float32)
        labels = tf.expand_dims(labels,axis=-1)
        alpha_t = wmlt.batch_gather(params=alpha_t,indices=labels)
        alpha_t = tf.squeeze(alpha_t,axis=-1)
        loss = -alpha_t*tf.math.log(r_probability)
        return loss

def save_pbfile(sess,pb_path,output_names):
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_names)
    dirpath = os.path.dirname(pb_path)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    with tf.gfile.FastGFile(pb_path, mode='wb') as f:
        f.write(constant_graph.SerializeToString())
    print(f"Save pb file {pb_path}.")
