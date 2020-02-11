#coding=utf-8
from collections import namedtuple
from .standard_names import *
import tensorflow as tf

EncodedData = namedtuple("EncoderData",[ED_GT_OBJECT_LOGITS,SCORES,ED_INDICES,ED_BOXES,ED_GT_BOXES,ED_GT_LABELS,ED_GT_DELTAS])
ProposalsData = namedtuple("ProposalsData",[PD_BOXES,PD_LOGITS])

def unstack_encoded_data_on_batch_dim(data:EncodedData):
    data = [tf.unstack(x,axis=0) for x in data]
    return EncodedData(*data)
