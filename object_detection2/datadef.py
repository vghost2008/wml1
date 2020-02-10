#coding=utf-8
from collections import namedtuple
from .standard_names import *
EncodedData = namedtuple("EncoderData",[ED_GT_OBJECT_LOGITS,SCORES,ED_INDICES,ED_BOXES,ED_GT_BOXES,ED_GT_LABELS,ED_GT_DELTAS])
ProposalsData = namedtuple("ProposalsData",[PD_BOXES,PD_LOGITS])
