#coding=utf-8
import tensorflow as tf
import wml_tfutils as wml
import object_detection.utils as utils
import wnn
import logging

slim = tf.contrib.slim


class ODLoss:
    def __init__(self,num_classes,reg_loss_weight=3.,
            scope="Loss",
            classes_wise=False,
            neg_multiplier=2.0,
            scale=10.0):
        self.num_classes = num_classes
        self.reg_loss_weight = reg_loss_weight
        self.scope = scope
        self.classes_wise = classes_wise
        self.neg_multiplier = neg_multiplier
        self.scale = scale
    '''
    与Faster-RCNN中定义的Smooth L1 loss完全一致
    '''
    @staticmethod
    def smooth_l1(x):
        absx = tf.abs(x)
        minx = tf.minimum(absx, 1)
        r = 0.5 * ((absx - 1) * minx + absx)
        return r
        
    '''
    将用于计算正负样本损失的数据分开
    '''
    def split_data(self,gregs,glabels,classes_logits,bboxes_regs,
            bboxes_remove_indices=None):
        gregs = tf.reshape(gregs, [-1, 4])
        glabels = tf.reshape(glabels, shape=[-1])
        def_ftype = gregs.dtype

        with tf.variable_scope(self.scope):
            classes_logits = tf.reshape(classes_logits, [-1, self.num_classes])
            if self.classes_wise:
                # regs have no background
                bboxes_regs = tf.reshape(bboxes_regs, [-1, self.num_classes - 1, 4])
            else:
                bboxes_regs = tf.reshape(bboxes_regs, [-1, 4])

            if bboxes_remove_indices is not None:
                if bboxes_remove_indices.get_shape().ndims > 1:
                    bboxes_remove_indices = tf.reshape(bboxes_remove_indices, [-1])
                keep_indices = tf.logical_not(bboxes_remove_indices)
                bboxes_regs = tf.boolean_mask(bboxes_regs, keep_indices)
                classes_logits = tf.boolean_mask(classes_logits, keep_indices)
                gregs = tf.boolean_mask(gregs, keep_indices)
                glabels = tf.boolean_mask(glabels, keep_indices)

            pmask = tf.greater(glabels, 0)
            psize = tf.reduce_sum(tf.cast(pmask, def_ftype))
            nmask = tf.logical_not(pmask)
            fnmask = tf.cast(nmask, def_ftype)
            class_prediction = slim.softmax(classes_logits)
            # 负样本的概率为模型预测为负样本的概率，正样本的地方设置为1
            nclass_prediction = tf.where(nmask, class_prediction[:, 0], 1.0 - fnmask)
            nsize = tf.reduce_sum(fnmask)
            '''
            默认负样本数最多为正样本的neg_multiplier倍+1
            '''
            nsize = tf.minimum(nsize, psize * self.neg_multiplier + 1)
            # 取预测的最不好的nsize个负样本
            nmask = tf.logical_and(nmask, wml.bottom_k_mask(nclass_prediction, k=tf.cast(nsize, tf.int32)))

            p_glabels = tf.boolean_mask(glabels, pmask)
            p_gregs = tf.boolean_mask(gregs, pmask)
            p_logits = tf.boolean_mask(classes_logits, pmask)
            p_pred_regs = tf.boolean_mask(bboxes_regs, pmask)
            
            n_glabels = tf.boolean_mask(glabels, nmask)
            n_logits = tf.boolean_mask(classes_logits, nmask)
            if bboxes_remove_indices is not None:
                pmask = wml.merge_mask(keep_indices,pmask)
                nmask = wml.merge_mask(keep_indices,nmask)

            return p_glabels,p_gregs,p_logits,p_pred_regs,psize,pmask,n_glabels,n_logits,nsize,nmask
        
    def sparse_softmax_cross_entropy_with_logits(self,    
                                                 _sentinel=None,  # pylint: disable=invalid-name
                                                 labels=None,
                                                 logits=None,
                                                 name=None):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits,name=name)
    
    def softmax_cross_entropy_with_logits(self,
                                          _sentinel=None,  # pylint: disable=invalid-name
                                          labels=None,
                                          logits=None,
                                          dim=-1,
                                          name=None):
        return tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits,dim=dim,name=name)
    
    def __call__(self, gregs,glabels,classes_logits,bboxes_regs,
                 bboxes_remove_indices=None,
                 scores=None,
                 call_back=None):
        '''
        :param gregs:
        :param glabels:
        :param classes_logits:
        :param bboxes_regs:
        :param bboxes_remove_indices:
        :param scores:
        :param call_back: func(pmask,nmask)
        :return:
        '''
        if scores is None:
            return self.lossv1(gregs,glabels,classes_logits,bboxes_regs,
                 bboxes_remove_indices,call_back=call_back)
        else:
            print("Use loss with scores.")
            return self.lossv2(gregs, glabels, classes_logits, bboxes_regs,
                               bboxes_remove_indices,scores,call_back=call_back)
    
    def lossv1(self,gregs,glabels,classes_logits,bboxes_regs,
                 bboxes_remove_indices=None,call_back=None):
        batch_size = gregs.get_shape().as_list()[0]
        p_glabels, p_gregs, p_logits, p_pred_regs, psize,pmask,n_glabels, n_logits,nsize,nmask = \
            self.split_data(gregs=gregs,glabels=glabels,classes_logits=classes_logits,
                            bboxes_regs=bboxes_regs,bboxes_remove_indices=bboxes_remove_indices)
        with tf.variable_scope("positive_loss"):
            if self.classes_wise:
                p_pred_regs = wml.select_2thdata_by_index_v2(p_pred_regs, p_glabels - 1)
            loss0 = self.sparse_softmax_cross_entropy_with_logits(logits=p_logits, labels=p_glabels)
            loss0 = tf.cond(tf.less(0.5, psize), lambda: tf.reduce_mean(loss0), lambda: 0.)
            loss1 = ODLoss.smooth_l1(p_gregs - p_pred_regs) * self.reg_loss_weight
            loss1 = tf.cond(tf.less(0.5, psize), lambda: tf.reduce_mean(loss1), lambda: 0.)
        with tf.variable_scope("negative_loss"):
            loss2 = self.sparse_softmax_cross_entropy_with_logits(logits=n_logits, labels=n_glabels)
            loss2 = tf.cond(tf.less(0.5, nsize), lambda: tf.reduce_mean(loss2), lambda: 0.)

        with tf.variable_scope("total_loss"):
            loss = (loss0 + loss1 + loss2) * self.scale
            tf.losses.add_loss(loss)

        wml.variable_summaries_v2((nsize + psize) / batch_size, "total_boxes_size_for_loss")
        if call_back is not None:
            call_back(pmask,nmask)
        '''
        loss0:正样本分类损失
        loss1:正样本回归损失
        loss2:负样本分类损失
        psize / (nsize + psize + 1E-8):正样本占的比重
        '''
        return loss0, loss1, loss2, psize / (nsize + psize + 1E-8)
    
    
    '''
    与lossv1相比lossv2的正样本概率不再是1.，而是由scores指定
    '''
    def lossv2(self,gregs,glabels,classes_logits,bboxes_regs,
               bboxes_remove_indices=None,scores=None,call_back=None):
        assert scores is not None, "scores is none."
        batch_size = gregs.get_shape().as_list()[0]
        scores = tf.reshape(scores,shape=[-1])
        p_glabels, p_gregs, p_logits, p_pred_regs, psize,pmask,n_glabels, n_logits,nsize,nmask = \
            self.split_data(gregs=gregs,glabels=glabels,classes_logits=classes_logits,
                            bboxes_regs=bboxes_regs,bboxes_remove_indices=bboxes_remove_indices)
        with tf.variable_scope("positive_loss"):
            if self.classes_wise:
                p_pred_regs = wml.select_2thdata_by_index_v2(p_pred_regs, p_glabels - 1)
            p_scores = tf.boolean_mask(scores,pmask)
            probibality = utils.get_total_probibality_by_object_probibality(probibality=p_scores,
                                                                            labels=p_glabels,
                                                                            num_classes=self.num_classes)
            loss0 = self.softmax_cross_entropy_with_logits(logits=p_logits,labels=probibality)
            loss0 = tf.cond(tf.less(0.5, psize), lambda: tf.reduce_mean(loss0), lambda: 0.)
            loss1 = ODLoss.smooth_l1(p_gregs - p_pred_regs) * self.reg_loss_weight
            loss1 = tf.cond(tf.less(0.5, psize), lambda: tf.reduce_mean(loss1), lambda: 0.)
        with tf.variable_scope("negative_loss"):
            loss2 = self.sparse_softmax_cross_entropy_with_logits(logits=n_logits, labels=n_glabels)
            loss2 = tf.cond(tf.less(0.5, nsize), lambda: tf.reduce_mean(loss2), lambda: 0.)

        with tf.variable_scope("total_loss"):
            loss = (loss0 + loss1 + loss2) * self.scale
            tf.losses.add_loss(loss)

        wml.variable_summaries_v2((nsize + psize) / batch_size, "total_boxes_size_for_loss")
        if call_back is not None:
            call_back(pmask,nmask)
        '''
        loss0:正样本分类损失
        loss1:正样本回归损失
        loss2:负样本分类损失
        psize / (nsize + psize + 1E-8):正样本占的比重
        '''
        return loss0, loss1, loss2, psize / (nsize + psize + 1E-8)
    
class ODLossWithFocalLoss(ODLoss):
    def __init__(self,
                 gamma=2.,
                 alpha="auto",
                 max_alpha_scale=10.0,*args,**kwargs):
        self.gamma = gamma
        self.alpha = alpha
        self.max_alpha_scale = max_alpha_scale
        super().__init__(*args,**kwargs)
        
    def sparse_softmax_cross_entropy_with_logits(self,
                                                 _sentinel=None,  # pylint: disable=invalid-name
                                                 labels=None,
                                                 logits=None,
                                                 name=None):
        logging.info(f"Use focal loss, gamma={self.gamma}, alpha={self.gamma}, max_alpha_scale={self.max_alpha_scale}.")
        return wnn.sparse_softmax_cross_entropy_with_logits_FL(labels=labels,
                                                               logits=logits,name=name,
                                                               gamma=self.gamma,
                                                               alpha=self.alpha,
                                                               max_alpha_scale=self.max_alpha_scale)
'''
gregs:proposal box/anchor box/default box到ground truth box的回归参数
shape 为[batch_size,X,4]/[X,4] batch_size==1

glabels:每个proposal box所对应的目标类型编号，背景为0
shape为[batch_size,X]/[X], batch_size==1

classes_logits:模型预测的每个proposal box的类型logits
shape为[batch_size,X,num_classes]

bboxes_regs:模型预测的每个proposal box到目标边框的回归参数
shape 为[batch_size,X,4](classes_wise=Flase) 或者[batch_size,X,num_classes,4]classes_wise=True

num_class:类型数量，包含背景
reg_loss_weight:对回归参数loss的权重系数
bboxes_remove_indices:[batch_size,Y], 表明那些box既不是正样本，也不是负样本

neg_multiplier:负样本最多为正样本的neg_multiplier倍+1个

'''
def od_loss(gregs,glabels,classes_logits,bboxes_regs,num_classes,reg_loss_weight=3.,
            bboxes_remove_indices=None,scope="Loss",
            classes_wise=False,
            neg_multiplier=2.0,
            scale=10.0,
            use_focal_loss=False):
    if use_focal_loss:
        logging.info("Use focal loss.")
    batch_size = gregs.get_shape().as_list()[0]
    gregs = tf.reshape(gregs,[-1,4])
    glabels = tf.reshape(glabels,shape=[-1])
    def_ftype = gregs.dtype

    with tf.variable_scope(scope):
        classes_logits = tf.reshape(classes_logits, [-1, num_classes])
        if classes_wise:
            #regs have no background
            bboxes_regs = tf.reshape(bboxes_regs,[-1,num_classes-1,4])
        else:
            bboxes_regs = tf.reshape(bboxes_regs,[-1,4])

        if bboxes_remove_indices is not None:
            if bboxes_remove_indices.get_shape().ndims > 1:
                bboxes_remove_indices = tf.reshape(bboxes_remove_indices,[-1])
            keep_indices = tf.logical_not(bboxes_remove_indices)
            bboxes_regs = tf.boolean_mask(bboxes_regs,keep_indices)
            classes_logits = tf.boolean_mask(classes_logits,keep_indices)
            gregs = tf.boolean_mask(gregs,keep_indices)
            glabels = tf.boolean_mask(glabels,keep_indices)

        pmask = tf.greater(glabels,0)
        psize = tf.reduce_sum(tf.cast(pmask,def_ftype))
        nmask = tf.logical_not(pmask)
        fnmask = tf.cast(nmask,def_ftype)
        class_prediction = slim.softmax(classes_logits)
        #负样本的概率为模型预测为负样本的概率，正样本的地方设置为1
        nclass_prediction = tf.where(nmask,class_prediction[:,0],1.0-fnmask)
        nsize = tf.reduce_sum(fnmask)
        '''
        默认负样本数最多为正样本的2倍+2
        '''
        nsize = tf.minimum(nsize,psize*neg_multiplier+1)
        #取预测的最不好的nsize个负样本
        nmask = tf.logical_and(nmask,wml.bottom_k_mask(nclass_prediction,k=tf.cast(nsize,tf.int32)))

        with tf.variable_scope("positive_loss"):
            p_glabels = tf.boolean_mask(glabels,pmask)
            p_gregs = tf.boolean_mask(gregs,pmask)
            p_logits = tf.boolean_mask(classes_logits,pmask)
            p_bboxes_regs = tf.boolean_mask(bboxes_regs,pmask)
            if classes_wise:
                p_bboxes_regs = wml.select_2thdata_by_index_v2(p_bboxes_regs,p_glabels-1)
            if use_focal_loss:
                loss0 = wnn.sparse_softmax_cross_entropy_with_logits_FL(logits=p_logits,labels=p_glabels)
            else:
                loss0 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=p_logits,labels=p_glabels)
            loss0 = tf.cond(tf.less(0.5, psize), lambda: tf.reduce_mean(loss0), lambda: 0.)
            loss1 = ODLoss.smooth_l1(p_gregs-p_bboxes_regs)*reg_loss_weight
            loss1 = tf.cond(tf.less(0.5, psize), lambda: tf.reduce_mean(loss1), lambda: 0.)
        with tf.variable_scope("negative_loss"):
            n_glabels = tf.boolean_mask(glabels,nmask)
            n_logits = tf.boolean_mask(classes_logits,nmask)
            if use_focal_loss:
                loss2 = wnn.sparse_softmax_cross_entropy_with_logits_FL(logits=n_logits,labels=n_glabels)
            else:
                loss2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=n_logits,labels=n_glabels)
            loss2 = tf.cond(tf.less(0.5, nsize), lambda: tf.reduce_mean(loss2), lambda: 0.)

        with tf.variable_scope("total_loss"):
            loss = (loss0 + loss1 + loss2)*scale
            tf.losses.add_loss(loss)

        with tf.device(":/cpu:0"):
            wml.variable_summaries_v2((nsize+psize)/batch_size,"total_boxes_size_for_loss")
        '''
        loss0:正样本分类损失
        loss1:正样本回归损失
        loss2:负样本分类损失
        psize / (nsize + psize + 1E-8):正样本占的比重
        '''
        return loss0,loss1,loss2,psize / (nsize + psize + 1E-8)


'''
gregs:proposal box/anchor box/default box到ground truth box的回归参数
shape 为[batch_size,X,4]/[X,4] batch_size==1

glabels:每个proposal box所对应的目标类型编号，背景为0, 其它类别从1开始递增
shape为[batch_size,X]/[X], batch_size==1

scores:每个proposal box所对应的目标类型的概率，背景为0
shape为[batch_size,X]/[X], batch_size==1

classes_logits:模型预测的每个proposal box的类型logits
shape为[batch_size,X,num_classes]

bboxes_regs:模型预测的每个proposal box到目标边框的回归参数
shape 为[batch_size,X,4](classes_wise=Flase) 或者[batch_size,X,num_classes,4]classes_wise=True

num_class:类型数量，包含背景
reg_loss_weight:对回归参数loss的权重系数
bboxes_remove_indices:[batch_size,Y], 表明那些box既不是正样本，也不是负样本

neg_multiplier:负样本最多为正样本的neg_multiplier倍+1个

与普通od_loss相比od_lossv2的正样本概率不再是1.，而是由scores指定
'''
def od_lossv2(gregs,glabels,classes_logits,bboxes_regs,num_classes,scores,reg_loss_weight=3.,
            bboxes_remove_indices=None,scope="Loss",
            classes_wise=False,
            neg_multiplier=2.0,
            scale=10.0):

    assert scores is not None, "scores is none."

    batch_size = gregs.get_shape().as_list()[0]
    gregs = tf.reshape(gregs,[-1,4])
    glabels = tf.reshape(glabels,shape=[-1])
    scores = tf.reshape(scores,shape=[-1])

    with tf.variable_scope(scope):
        classes_logits = tf.reshape(classes_logits, [-1, num_classes])
        if classes_wise:
            bboxes_regs = tf.reshape(bboxes_regs,[-1,num_classes-1,4])
        else:
            bboxes_regs = tf.reshape(bboxes_regs,[-1,4])

        if bboxes_remove_indices is not None:
            if bboxes_remove_indices.get_shape().ndims > 1:
                bboxes_remove_indices = tf.reshape(bboxes_remove_indices, [-1])
            keep_indices = tf.logical_not(bboxes_remove_indices)
            bboxes_regs = tf.boolean_mask(bboxes_regs,keep_indices)
            classes_logits = tf.boolean_mask(classes_logits,keep_indices)
            gregs = tf.boolean_mask(gregs,keep_indices)
            glabels = tf.boolean_mask(glabels,keep_indices)

        pmask = tf.greater(glabels,0)
        psize = tf.reduce_sum(tf.cast(pmask,tf.float32))
        nmask = tf.logical_not(pmask)
        fnmask = tf.cast(nmask,tf.float32)
        class_prediction = slim.softmax(classes_logits)
        #负样本的概率为模型预测为负样本的概率，正样本的地方设置为1
        nclass_prediction = tf.where(nmask,class_prediction[:,0],1.0-fnmask)
        nsize = tf.reduce_sum(fnmask)
        '''
        默认负样本数最多为正样本的2倍+2
        '''
        nsize = tf.minimum(nsize,psize*neg_multiplier+1)
        #取预测的最不好的nsize个负样本
        nmask = tf.logical_and(nmask,wml.bottom_k_mask(nclass_prediction,k=tf.cast(nsize,tf.int32)))

        with tf.variable_scope("positive_loss"):
            p_glabels = tf.boolean_mask(glabels,pmask)
            p_gregs = tf.boolean_mask(gregs,pmask)
            p_logits = tf.boolean_mask(classes_logits,pmask)
            p_bboxes_regs = tf.boolean_mask(bboxes_regs,pmask)
            p_scores = tf.boolean_mask(scores,pmask)
            if classes_wise:
                p_bboxes_regs = wml.select_2thdata_by_index_v2(p_bboxes_regs,p_glabels-1)

            probibality = utils.get_total_probibality_by_object_probibality(probibality=p_scores,labels=p_glabels,num_classes=num_classes)
            loss0 = tf.nn.softmax_cross_entropy_with_logits(logits=p_logits,labels=probibality)
            loss0 = tf.cond(tf.less(0.5, psize), lambda: tf.reduce_mean(loss0), lambda: 0.)
            loss1 = ODLoss.smooth_l1(p_gregs-p_bboxes_regs)*reg_loss_weight
            loss1 = tf.cond(tf.less(0.5, psize), lambda: tf.reduce_mean(loss1), lambda: 0.)

        with tf.variable_scope("negative_loss"):
            n_glabels = tf.boolean_mask(glabels,nmask)
            n_logits = tf.boolean_mask(classes_logits,nmask)
            loss2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=n_logits,labels=n_glabels)
            loss2 = tf.cond(tf.less(0.5, nsize), lambda: tf.reduce_mean(loss2), lambda: 0.)

        with tf.variable_scope("total_loss"):
            loss = (loss0 + loss1 + loss2)*scale
            tf.losses.add_loss(loss)

        with tf.device(":/cpu:0"):
            wml.variable_summaries_v2((nsize+psize)/batch_size,"total_boxes_size_for_loss")
        '''
        loss0:正样本分类损失
        loss1:正样本回归损失
        loss2:负样本分类损失
        psize / (nsize + psize + 1E-8):正样本占的比重
        '''
        return loss0,loss1,loss2,psize / (nsize + psize + 1E-8)