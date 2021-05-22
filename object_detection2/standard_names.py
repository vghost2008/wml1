#coding=utf-8
SCORES = "scores"
INDICES = "indices"
BOXES = "boxes"
LABELS = "labels"
PROBABILITY = "probability"
RAW_PROBABILITY = "raw_probability"
IMAGE = 'image'
HEIGHT = "height"
WIDTH = "width"
IS_CROWD = 'is_crowd'
MASK_AREA = "mask_area"
ORG_HEIGHT = "org_height"
ORG_WIDTH = "org_width"
LOGITS = "logits"
BOXES_REGS = "box_regs"
SEMANTIC = "semantic"
COEFFICIENT = "coefficient"

GT_BOXES = 'gt_boxes'
GT_LENGTH = 'gt_length'
GT_LABELS = 'gt_labels'
GT_MASKS = 'gt_masks'
GT_KEYPOINTS = 'gt_keypoints'
GT_SEMANTIC_LABELS = 'gt_semantic_mask_labels'
FILEINDEX = "fileindex"
FILENAME = "filename"
GT_OBJECT_LOGITS = "gt_object_logits"

#Encoded data
ED_GT_OBJECT_LOGITS = "gt_object_logits"
ED_SCORES = "scores"
ED_INDICES = "indices"
ED_BOXES = "boxes"
ED_GT_BOXES = GT_BOXES
ED_GT_LABELS = GT_LABELS
ED_GT_DELTAS = "deltas"
ED_GT_DELTAS = "deltas"

#Results Data
RD_BOXES = BOXES
RD_BOXES_ABSOLUTE = BOXES+"_absolute"
RD_LABELS = LABELS
RD_PROBABILITY = PROBABILITY
RD_RAW_PROBABILITY = RAW_PROBABILITY
RD_INDICES = INDICES
RD_LENGTH = "length"
RD_MASKS = "masks"  #标准格式为[batch_size,N,h,w]
RD_SEMANTIC = "semantic"  #标准格式为[batch_size,N,h,w]
RD_SPARSE_SEMANTIC = "sparse_semantic"  #标准格式为[batch_size,N,h,w]
RD_FULL_SIZE_MASKS = "full_size_masks"  #标准格式为[batch_size,N,H,W]
RD_RESULT_IMAGE = "result_image"
RD_KEYPOINT = "keypoint"
RD_MASK_AREA = MASK_AREA
RD_ID = "rd_id"

#Proposal network't result
PD_BOXES = "boxes"
PD_PROBABILITY = PROBABILITY


GRADIENT_DEBUG_COLLECTION = "gradient_debug_collection"

