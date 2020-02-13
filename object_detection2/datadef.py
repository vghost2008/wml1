#coding=utf-8
from collections import namedtuple,OrderedDict
from .standard_names import *
import tensorflow as tf

EncodedData = namedtuple("EncoderData",[ED_GT_OBJECT_LOGITS,SCORES,ED_INDICES,ED_BOXES,ED_GT_BOXES,ED_GT_LABELS])
ProposalsData = namedtuple("ProposalsData",[PD_BOXES,PD_LOGITS])
RCNNResultsData = OrderedDict#key=[RD_BOXES,RD_LABELS,RD_PROBABILITY,RD_INDICES,RD_LENGTH,RD_MASKS,RD_KEYPOINT]

def unstack_encoded_data_on_batch_dim(data:EncodedData):
    data = [tf.unstack(x,axis=0) for x in data]
    return EncodedData(*data)
