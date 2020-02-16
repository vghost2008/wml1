#coding=utf-8
from collections import namedtuple,OrderedDict
from .standard_names import *
import tensorflow as tf
from wml_utils import MDict

ProposalsData = OrderedDict#namedtuple("ProposalsData",[PD_BOXES,PD_LOGITS])
RCNNResultsData = OrderedDict#key=[RD_BOXES,RD_LABELS,RD_PROBABILITY,RD_INDICES,RD_LENGTH,RD_MASKS,RD_KEYPOINT]
EncodedData = namedtuple("EncoderData",[ED_GT_OBJECT_LOGITS,SCORES,ED_INDICES,ED_BOXES,ED_GT_BOXES,ED_GT_LABELS])
class EncodedData(MDict):
    def __init__(self,gt_object_logits=None,scores=None,indices=None,boxes=None,gt_boxes=None,gt_labels=None):
        if gt_object_logits is not None:
            self[ED_GT_OBJECT_LOGITS] = gt_object_logits
        if scores is not None:
            self[ED_SCORES] = scores
        if indices is not None:
            self[ED_INDICES] = indices
        if boxes is not None:
            self[ED_BOXES] = boxes
        if gt_boxes is not None:
            self[ED_GT_BOXES] = gt_boxes
        if gt_labels is not None:
            self[ED_GT_LABELS] = gt_labels

def unstack_encoded_data_on_batch_dim(data:EncodedData):
    data = [tf.unstack(x,axis=0) for x in data]
    return EncodedData(*data)

class SummaryLevel:
    DEBUG=0
    INFO=1
    WARNING=2
    CRITICAL=3
