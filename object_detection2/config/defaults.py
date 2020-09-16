from .config import CfgNode as CN

_C = CN()
_C.VERSION = 2

_C.MODEL = CN()
_C.MODEL.MASK_ON = False
_C.MODEL.KEYPOINT_ON = False
_C.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
_C.MODEL.MIN_BOXES_AREA_TEST = 0.
_C.MODEL.PREPROCESS = "ton1p1" #"subimagenetmean","standardization","NONE"
_C.MODEL.WEIGHTS = ""
_C.MODEL.ONLY_SCOPE = ""
_C.MODEL.EXCLUDE_SCOPE = ""

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the smallest side of the image during training
_C.INPUT.MIN_SIZE_TRAIN = (224,256,288)
# Maximum size of the side of the image during training
_C.INPUT.MAX_SIZE_TRAIN = 1333
# Size of the smallest side of the image during testing. Set to zero to disable resize in testing.
_C.INPUT.MIN_SIZE_TEST = 224
# Maximum size of the side of the image during testing
_C.INPUT.MAX_SIZE_TEST = 1333
_C.INPUT.SIZE_ALIGN = 1
_C.INPUT.SIZE_ALIGN_FOR_TEST = 1

_C.INPUT.CROP = CN()
_C.INPUT.CROP.SIZE = [0.1, 1.0]
_C.INPUT.CROP.ASPECT_RATIO = [0.5, 2.0]
_C.INPUT.CROP.MIN_OBJECT_COVERED = 0.5
_C.INPUT.CROP.PROBABILITY = 0.5
_C.INPUT.CROP.FILTER_THRESHOLD = 0.3
_C.INPUT.DATAPROCESS = "coco"
_C.INPUT.STITCH = 0.0
_C.INPUT.ROTATE_ANY_ANGLE = CN()
_C.INPUT.ROTATE_ANY_ANGLE.ENABLE = False
_C.INPUT.ROTATE_ANY_ANGLE.MAX_ANGLE = 6
_C.INPUT.ROTATE_ANY_ANGLE.PROBABILITY = 0.5
_C.INPUT.SHUFFLE_BUFFER_SIZE = 1024  #tensorflow models use 2048 as default value
_C.INPUT.FILTER_EMPTY = True
_C.INPUT.EXTRA_FILTER = ""

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training. Must be registered in DatasetCatalog
_C.DATASETS.TRAIN = ""
# List of the dataset names for testing. Must be registered in DatasetCatalog
_C.DATASETS.TEST = ""
_C.DATASETS.NUM_CLASSES= 90
_C.DATASETS.SKIP_CROWD_DURING_TRAINING = True

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
# if True, the dataloader will filter out images that have no associated
# annotations at train time.
_C.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()

_C.MODEL.BACKBONE.NAME = "build_resnet_backbone"

# ---------------------------------------------------------------------------- #
# FPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.FPN = CN()
# Names of the input feature maps to be used by FPN
# They must have contiguous power of 2 strides
# e.g., ["res2", "res3", "res4", "res5"]
_C.MODEL.FPN.IN_FEATURES = []
_C.MODEL.FPN.OUT_CHANNELS = 256
_C.MODEL.FPN.BACKBONE = ""

# Options: "" (no norm), "GN", "BN"
_C.MODEL.FPN.NORM = ""
_C.MODEL.FPN.ACTIVATION_FN = "relu"
_C.MODEL.FPN.LAST_LEVEL_NUM_CONV = 2

# Types for fusing the FPN top-down and lateral features. Can be either "sum" or "avg"
_C.MODEL.FPN.FUSE_TYPE = "sum"
_C.MODEL.FPN.BACKBONE_HOOK = ("","")
_C.MODEL.FPN.ENABLE_DROPBLOCK = False
_C.MODEL.FPN.DROPBLOCK_SIZE = 7
_C.MODEL.FPN.KEEP_PROB = 0.9

_C.MODEL.TWOWAYFPN = CN()
# Names of the input feature maps to be used by TWOWAYFPN
# They must have contiguous power of 2 strides
# e.g., ["res2", "res3", "res4", "res5"]
_C.MODEL.TWOWAYFPN.IN_FEATURES = []
_C.MODEL.TWOWAYFPN.OUT_CHANNELS = 256
_C.MODEL.TWOWAYFPN.BACKBONE = ""

# Options: "" (no norm), "GN", "BN"
_C.MODEL.TWOWAYFPN.NORM = ""
_C.MODEL.TWOWAYFPN.ACTIVATION_FN = "relu"
_C.MODEL.TWOWAYFPN.LAST_LEVEL_NUM_CONV = 2

# Types for fusing the TWOWAYFPN top-down and lateral features. Can be either "sum" or "avg"
_C.MODEL.TWOWAYFPN.FUSE_TYPE = "sum"
_C.MODEL.TWOWAYFPN.BACKBONE_HOOK = ("","")

_C.MODEL.BIFPN = CN()
# Names of the input feature maps to be used by FPN
# They must have contiguous power of 2 strides
# e.g., ["res2", "res3", "res4", "res5"]
_C.MODEL.BIFPN.IN_FEATURES = []
_C.MODEL.BIFPN.OUT_CHANNELS = 256

# Options: "" (no norm), "GN", "BN"
_C.MODEL.BIFPN.BACKBONE = ""
_C.MODEL.BIFPN.NORM = ""
_C.MODEL.BIFPN.ACTIVATION_FN = "relu"
_C.MODEL.BIFPN.REPEAT= 1

# Types for fusing the FPN top-down and lateral features. Can be either "sum" or "avg"
_C.MODEL.BIFPN.BACKBONE_HOOK = ("","")

# ---------------------------------------------------------------------------- #
# Proposal generator options
# ---------------------------------------------------------------------------- #
_C.MODEL.PROPOSAL_GENERATOR = CN()
# Current proposal generators include "RPN", "RRPN" and "PrecomputedProposals"
_C.MODEL.PROPOSAL_GENERATOR.NAME = "RPN"
# Proposal height and width both need to be greater than MIN_SIZE
# (a the scale used during training or inference)
_C.MODEL.PROPOSAL_GENERATOR.MIN_SIZE = 0

_C.MODEL.PROPOSAL_GENERATOR.SCORE_THRESH_TEST = 0.0


# ---------------------------------------------------------------------------- #
# Anchor generator options
# ---------------------------------------------------------------------------- #
_C.MODEL.ANCHOR_GENERATOR = CN()
# The generator can be any name in the ANCHOR_GENERATOR registry
_C.MODEL.ANCHOR_GENERATOR.NAME = "DefaultAnchorGenerator"
# anchor sizes given in absolute pixels w.r.t. the scaled network input.
# Format: list of lists of sizes. SIZES[i] specifies the list of sizes
# to use for IN_FEATURES[i]; len(SIZES) == len(IN_FEATURES) must be true,
# or len(SIZES) == 1 is true and size list SIZES[0] is used for all
# IN_FEATURES.
_C.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256, 512]]
# Anchor aspect ratios.
# Format is list of lists of sizes. ASPECT_RATIOS[i] specifies the list of aspect ratios
# to use for IN_FEATURES[i]; len(ASPECT_RATIOS) == len(IN_FEATURES) must be true,
# or len(ASPECT_RATIOS) == 1 is true and aspect ratio list ASPECT_RATIOS[0] is used
# for all IN_FEATURES.
_C.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]

# ---------------------------------------------------------------------------- #
# RPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.RPN = CN()
_C.MODEL.RPN.HEAD_NAME = "StandardRPNHead"  # used by RPN_HEAD_REGISTRY
_C.MODEL.RPN.OUTPUTS = "RPNGIOUOutputs"

# Names of the input feature maps to be used by RPN
# e.g., ["p2", "p3", "p4", "p5", "p6"] for FPN
_C.MODEL.RPN.IN_FEATURES = ["res4"]
# Remove RPN anchors that go outside the image by BOUNDARY_THRESH pixels
# Set to -1 or a large value, e.g. 100000, to disable pruning anchors
_C.MODEL.RPN.BOUNDARY_THRESH = -1
# IOU overlap ratios [BG_IOU_THRESHOLD, FG_IOU_THRESHOLD]
# Minimum overlap required between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a positive example (IoU >= FG_IOU_THRESHOLD
# ==> positive RPN example: 1)
# Maximum overlap allowed between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a negative examples (IoU < BG_IOU_THRESHOLD
# ==> negative RPN example: 0)
# Anchors with overlap in between (BG_IOU_THRESHOLD <= IoU < FG_IOU_THRESHOLD)
# are ignored (-1)
_C.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.7]
_C.MODEL.RPN.IOU_LABELS = [0, -1, 1]
# Total number of RPN examples per image
_C.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
# Target fraction of foreground (positive) examples per RPN minibatch
_C.MODEL.RPN.POSITIVE_FRACTION = 0.5
# Weights on (dx, dy, dw, dh) for normalizing RPN anchor regression targets
_C.MODEL.RPN.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# The transition point from L1 to L2 loss. Set to 0.0 to make the loss simply L1.
_C.MODEL.RPN.SMOOTH_L1_BETA = 0.0
_C.MODEL.RPN.LOSS_WEIGHT = 1.0
# Number of top scoring RPN proposals to keep before applying NMS
# When FPN is used, this is *per FPN level* (not total)
_C.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 12000
_C.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000
# Number of top scoring RPN proposals to keep after applying NMS
# When FPN is used, this limit is applied per level and then again to the union
# of proposals from all levels
# NOTE: When FPN is used, the meaning of this config is different from Detectron1.
# It means per-batch topk in Detectron1, but per-image topk here.
# See "modeling/rpn/rpn_outputs.py" for details.
_C.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
_C.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
# NMS threshold used on RPN proposals
_C.MODEL.RPN.NMS_THRESH = 0.7
_C.MODEL.RPN.SORT_RESULTS = False
_C.MODEL.RPN.MATCHER = "Matcher"
_C.MODEL.RPN.NORM = ""
_C.MODEL.RPN.ACTIVATION_FN = "relu"

# ---------------------------------------------------------------------------- #
# RetinaNet proposal generator options
# ---------------------------------------------------------------------------- #
_C.MODEL.RETINANET_PG = CN()
_C.MODEL.RETINANET_PG.HEAD_NAME = "StandardRETINANET_PGHead"  # used by RETINANET_PG_HEAD_REGISTRY
_C.MODEL.RETINANET_PG.OUTPUTS = "PGRetinaNetGIOUOutputs"

_C.MODEL.RETINANET_PG.NUM_CLASSES = 1
# Names of the input feature maps to be used by RETINANET_PG
# e.g., ["p2", "p3", "p4", "p5", "p6"] for FPN
_C.MODEL.RETINANET_PG.IN_FEATURES = ["P3", "P4", "P5", "P6"]
# Remove RETINANET_PG anchors that go outside the image by BOUNDARY_THRESH pixels
# Set to -1 or a large value, e.g. 100000, to disable pruning anchors
_C.MODEL.RETINANET_PG.BOUNDARY_THRESH = -1
# IOU overlap ratios [BG_IOU_THRESHOLD, FG_IOU_THRESHOLD]
# Minimum overlap required between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a positive example (IoU >= FG_IOU_THRESHOLD
# ==> positive RETINANET_PG example: 1)
# Maximum overlap allowed between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a negative examples (IoU < BG_IOU_THRESHOLD
# ==> negative RETINANET_PG example: 0)
# Anchors with overlap in between (BG_IOU_THRESHOLD <= IoU < FG_IOU_THRESHOLD)
# are ignored (-1)
_C.MODEL.RETINANET_PG.IOU_THRESHOLDS = [0.4, 0.5]
_C.MODEL.RETINANET_PG.IOU_LABELS = [0, -1, 1]
# Weights on (dx, dy, dw, dh) for normalizing RETINANET_PG anchor regression targets
_C.MODEL.RETINANET_PG.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
_C.MODEL.RETINANET_PG.LOSS_WEIGHT = 1.0
# Number of top scoring RETINANET_PG proposals to keep before applying NMS
# When FPN is used, this is *per FPN level* (not total)
_C.MODEL.RETINANET_PG.PRE_NMS_TOPK_TRAIN = 12000
_C.MODEL.RETINANET_PG.PRE_NMS_TOPK_TEST = 6000
# Number of top scoring RETINANET_PG proposals to keep after applying NMS
# When FPN is used, this limit is applied per level and then again to the union
# of proposals from all levels
# NOTE: When FPN is used, the meaning of this config is different from Detectron1.
# It means per-batch topk in Detectron1, but per-image topk here.
# See "modeling/rpn/rpn_outputs.py" for details.
_C.MODEL.RETINANET_PG.POST_NMS_TOPK_TRAIN = 2000
_C.MODEL.RETINANET_PG.POST_NMS_TOPK_TEST = 1000
# NMS threshold used on RETINANET_PG proposals
_C.MODEL.RETINANET_PG.NMS_THRESH = 0.7
_C.MODEL.RETINANET_PG.FOCAL_LOSS_GAMMA = 2.0
_C.MODEL.RETINANET_PG.FOCAL_LOSS_ALPHA = 0.25
_C.MODEL.RETINANET_PG.SMOOTH_L1_LOSS_BETA = 0.1
# Convolutions to use in the cls and bbox tower
# NOTE: this doesn't include the last conv for logits
_C.MODEL.RETINANET_PG.NUM_CONVS = 4
# Prior prob for rare case (i.e. foreground) at the beginning of training.
# This is used to set the bias for the logits layer of the classifier subnet.
# This improves training stability in the case of heavy class imbalance.
_C.MODEL.RETINANET_PG.PRIOR_PROB = 0.01

_C.MODEL.RETINANET_PG.SCORE_THRESH_TEST = 0.05
_C.MODEL.RETINANET_PG.TOPK_CANDIDATES_TEST = 1000
_C.MODEL.RETINANET_PG.NMS_THRESH_TEST = 0.5
_C.MODEL.RETINANET_PG.FAST_MODE = True
_C.MODEL.RETINANET_PG.MATCHER = "Matcher"
_C.MODEL.RETINANET_PG.NORM = "BN"
_C.MODEL.RETINANET_PG.ACTIVATION_FN = "relu"

# ---------------------------------------------------------------------------- #
# ROI HEADS options
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_HEADS = CN()
_C.MODEL.ROI_HEADS.NAME = "Res5ROIHeads"
_C.MODEL.ROI_HEADS.HOOK = ""
_C.MODEL.ROI_HEADS.BACKBONE = ""
# Number of foreground classes
_C.MODEL.ROI_HEADS.NUM_CLASSES = 80
# Names of the input feature maps to be used by ROI heads
# Currently all heads (box, mask, ...) use the same input feature map list
# e.g., ["p2", "p3", "p4", "p5"] is commonly used for FPN
_C.MODEL.ROI_HEADS.IN_FEATURES = ["res4"]
# IOU overlap ratios [IOU_THRESHOLD]
# Overlap threshold for an RoI to be considered background (if < IOU_THRESHOLD)
# Overlap threshold for an RoI to be considered foreground (if >= IOU_THRESHOLD)
_C.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5]
_C.MODEL.ROI_HEADS.IOU_LABELS = [0, 1]
_C.MODEL.ROI_HEADS.POS_LABELS_THRESHOLD = -1.0

# RoI minibatch size *per image* (number of regions of interest [ROIs])
# Total number of RoIs per training minibatch =
#   ROI_HEADS.BATCH_SIZE_PER_IMAGE * SOLVER.IMS_PER_BATCH
# E.g., a common configuration is: 512 * 16 = 8192
_C.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
# Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)
_C.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25
_C.MODEL.ROI_HEADS.BALANCED_POS_SAMPLE = True
_C.MODEL.ROI_HEADS.BALANCED_NEG_SAMPLE = True
_C.MODEL.ROI_HEADS.BALANCED_NEG_SAMPLE_LOW_VALUE = -0.2

# Only used on test mode

# Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
# balance obtaining high recall with not having too many low precision
# detections that will slow down inference post processing steps (like NMS)
# A default threshold of 0.0 increases AP by ~0.2-0.3 but significantly slows down
# inference.
_C.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
_C.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
# If True, augment proposals with ground-truth boxes before sampling proposals to
# train ROI heads.
_C.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT = True
_C.MODEL.ROI_HEADS.PROPOSAL_APPEND_HUGE_NUM_GT = False

_C.MODEL.ROI_HEADS.PRED_IOU = False
_C.MODEL.ROI_HEADS.PRED_IOU_VERSION = 0
#预测时box结果的最小相对面积

_C.MODEL.ROI_HEADS.OUTPUTS = "FastRCNNOutputs"
_C.MODEL.ROI_HEADS.BOX_REG_LOSS_SCALE = 1.0
_C.MODEL.ROI_HEADS.BOX_CLS_LOSS_SCALE = 1.0

# ---------------------------------------------------------------------------- #
# Box Head
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_BOX_HEAD = CN()
# C4 don't use head name option
# Options for non-C4 models: FastRCNNConvFCHead,
_C.MODEL.ROI_BOX_HEAD.NAME = ""
_C.MODEL.ROI_BOX_HEAD.OUTPUTS_LAYER = "FastRCNNOutputLayers"
# Default weights on (dx, dy, dw, dh) for normalizing bbox regression targets
# These are empirically chosen to approximately lead to unit variance targets
_C.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS = (10.0, 10.0, 5.0, 5.0)
# The transition point from L1 to L2 loss. Set to 0.0 to make the loss simply L1.
_C.MODEL.ROI_BOX_HEAD.bin_size = (2,2)
_C.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 0
# Type of pooling operation applied to the incoming feature map for each RoI
_C.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlign"
_C.MODEL.ROI_BOX_HEAD.canonical_box_size = 224.0
_C.MODEL.ROI_BOX_HEAD.canonical_level = 1

_C.MODEL.ROI_BOX_HEAD.NUM_FC = 0
# Hidden layer dimension for FC layers in the RoI box head
_C.MODEL.ROI_BOX_HEAD.FC_DIM = 1024
_C.MODEL.ROI_BOX_HEAD.NUM_CONV = 0
# Channel dimension for Conv layers in the RoI box head
_C.MODEL.ROI_BOX_HEAD.CONV_DIM = 256
# Normalization method for the convolution layers.
# Options: "" (no norm), "GN", "SyncBN".
_C.MODEL.ROI_BOX_HEAD.NORM = ""
_C.MODEL.ROI_BOX_HEAD.ACTIVATION_FN = "relu"
# Whether to use class agnostic for bbox regression
_C.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = False
# If true, RoI heads use bounding boxes predicted by the box head rather than proposal boxes.
_C.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES = False

# ---------------------------------------------------------------------------- #
# Cascaded Box Head
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_BOX_CASCADE_HEAD = CN()
# The number of cascade stages is implicitly defined by the length of the following two configs.
_C.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS = (
    (10.0, 10.0, 5.0, 5.0),
    (20.0, 20.0, 10.0, 10.0),
    (30.0, 30.0, 15.0, 15.0),
)
_C.MODEL.ROI_BOX_CASCADE_HEAD.IOUS = (0.5, 0.6, 0.7)


# ---------------------------------------------------------------------------- #
# Mask Head
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_MASK_HEAD = CN()
_C.MODEL.ROI_MASK_HEAD.NAME = "MaskRCNNConvUpsampleHead"
_C.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_MASK_HEAD.NUM_CONV = 0  # The number of convs in the mask head
_C.MODEL.ROI_MASK_HEAD.CONV_DIM = 256
_C.MODEL.ROI_MASK_HEAD.bin_size = (2,2)
# Normalization method for the convolution layers.
# Options: "" (no norm), "GN", "SyncBN".
_C.MODEL.ROI_MASK_HEAD.NORM = "BN"
_C.MODEL.ROI_MASK_HEAD.ACTIVATION_FN = "relu"
# Whether to use class agnostic for mask prediction
_C.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK = False
# Type of pooling operation applied to the incoming feature map for each RoI
_C.MODEL.ROI_MASK_HEAD.POOLER_TYPE = "ROIAlignV2"
_C.MODEL.ROI_MASK_HEAD.canonical_box_size = 224.0
_C.MODEL.ROI_MASK_HEAD.canonical_level = 1


# ---------------------------------------------------------------------------- #
# Keypoint Head
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_KEYPOINT_HEAD = CN()
_C.MODEL.ROI_KEYPOINT_HEAD.NAME = "KRCNNConvDeconvUpsampleHead"
_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_KEYPOINT_HEAD.CONV_DIMS = tuple(512 for _ in range(8))
_C.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 17  # 17 is the number of keypoints in COCO.
_C.MODEL.ROI_KEYPOINT_HEAD.bin_size = (2,2)

# Images with too few (or no) keypoints are excluded from training.
_C.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE = 1
# Normalize by the total number of visible keypoints in the minibatch if True.
# Otherwise, normalize by the total number of keypoints that could ever exist
# in the minibatch.
# The keypoint softmax loss is only calculated on visible keypoints.
# Since the number of visible keypoints can vary significantly between
# minibatches, this has the effect of up-weighting the importance of
# minibatches with few visible keypoints. (Imagine the extreme case of
# only one visible keypoint versus N: in the case of N, each one
# contributes 1/N to the gradient compared to the single keypoint
# determining the gradient direction). Instead, we can normalize the
# loss by the total number of keypoints, if it were the case that all
# keypoints were visible in a full minibatch. (Returning to the example,
# this means that the one visible keypoint contributes as much as each
# of the N keypoints.)
_C.MODEL.ROI_KEYPOINT_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS = True
# Multi-task loss weight to use for keypoints
# Recommended values:
#   - use 1.0 if NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS is True
#   - use 4.0 if NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS is False
_C.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT = 1.0
# Type of pooling operation applied to the incoming feature map for each RoI
_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE = "ROIAlignV2"
_C.MODEL.ROI_KEYPOINT_HEAD.canonical_box_size = 224.0
_C.MODEL.ROI_KEYPOINT_HEAD.canonical_level = 1

# ---------------------------------------------------------------------------- #
# SSD Head
# ---------------------------------------------------------------------------- #
_C.MODEL.SSD = CN()
# This is the number of foreground classes.
_C.MODEL.SSD.NUM_CLASSES = 90
_C.MODEL.SSD.BATCH_SIZE_PER_IMAGE = 128  #RetinaNet的实验中说SSD 128时最好，256最反而变差了？？？
_C.MODEL.SSD.POSITIVE_FRACTION = 0.25
_C.MODEL.SSD.NUM_CONVS = 4
_C.MODEL.SSD.PRIOR_PROB = 0.01
_C.MODEL.SSD.IOU_THRESHOLDS = [0.4, 0.5]
_C.MODEL.SSD.IOU_LABELS = [0, -1, 1]
_C.MODEL.SSD.IN_FEATURES = ["P3", "P4", "P5", "P6"]
_C.MODEL.SSD.SCORE_THRESH_TEST = 0.05
_C.MODEL.SSD.TOPK_CANDIDATES_TEST = 1000
_C.MODEL.SSD.NMS_THRESH_TEST = 0.5

# ---------------------------------------------------------------------------- #
# RetinaNet Head
# ---------------------------------------------------------------------------- #
_C.MODEL.RETINANET = CN()

_C.MODEL.RETINANET.HEAD_NAME = "RetinaNetHead"
# This is the number of foreground classes.
_C.MODEL.RETINANET.NUM_CLASSES = 80

_C.MODEL.RETINANET.IN_FEATURES = ["P3", "P4", "P5", "P6"]

# Convolutions to use in the cls and bbox tower
# NOTE: this doesn't include the last conv for logits
_C.MODEL.RETINANET.NUM_CONVS = 4

# IoU overlap ratio [bg, fg] for labeling anchors.
# Anchors with < bg are labeled negative (0)
# Anchors  with >= bg and < fg are ignored (-1)
# Anchors with >= fg are labeled positive (1)
_C.MODEL.RETINANET.IOU_THRESHOLDS = [0.4, 0.5]
_C.MODEL.RETINANET.IOU_LABELS = [0, -1, 1]

# Prior prob for rare case (i.e. foreground) at the beginning of training.
# This is used to set the bias for the logits layer of the classifier subnet.
# This improves training stability in the case of heavy class imbalance.
_C.MODEL.RETINANET.PRIOR_PROB = 0.01

# Inference cls score threshold, only anchors with score > INFERENCE_TH are
# considered for inference (to improve speed)
_C.MODEL.RETINANET.SCORE_THRESH_TEST = 0.05
_C.MODEL.RETINANET.TOPK_CANDIDATES_TEST = 1000
_C.MODEL.RETINANET.NMS_THRESH_TEST = 0.5

# Weights on (dx, dy, dw, dh) for normalizing Retinanet anchor regression targets
_C.MODEL.RETINANET.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

# Loss parameters
_C.MODEL.RETINANET.FOCAL_LOSS_GAMMA = 2.0
_C.MODEL.RETINANET.FOCAL_LOSS_ALPHA = 0.25
_C.MODEL.RETINANET.SMOOTH_L1_LOSS_BETA = 0.1
_C.MODEL.RETINANET.OUTPUTS = "RetinaNetOutputs"
_C.MODEL.RETINANET.NORM = "BN"
_C.MODEL.RETINANET.ACTIVATION_FN = "relu"
_C.MODEL.RETINANET.MATCHER = "Matcher"
_C.MODEL.RETINANET.BOX_REG_LOSS_SCALE = 1.0
_C.MODEL.RETINANET.BOX_CLS_LOSS_SCALE = 1.0
_C.MODEL.RETINANET.CLASSES_WISE_NMS = True

# ---------------------------------------------------------------------------- #
# CornerNet Head
# ---------------------------------------------------------------------------- #
_C.MODEL.CENTERNET = CN()

# This is the number of foreground classes.
_C.MODEL.CENTERNET.NUM_CLASSES = 80

_C.MODEL.CENTERNET.IN_FEATURES = ["P3"]

_C.MODEL.CENTERNET.SCORE_THRESH_TEST = 0.05
_C.MODEL.CENTERNET.TOPK_CANDIDATES_TEST = 1000
_C.MODEL.CENTERNET.NMS_THRESH_TEST = 0.5
_C.MODEL.CENTERNET.DIS_THRESHOLD = 1.0

# Loss parameters
_C.MODEL.CENTERNET.OUTPUTS = "CenterNetOutputs"
_C.MODEL.CENTERNET.NORM = "BN"
_C.MODEL.CENTERNET.ACTIVATION_FN = "relu"
_C.MODEL.CENTERNET.K = 100
_C.MODEL.CENTERNET.SIZE_THRESHOLD = 130
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
# FCOS Head
# ---------------------------------------------------------------------------- #
_C.MODEL.FCOS = CN()

# This is the number of foreground classes.
_C.MODEL.FCOS.NUM_CLASSES = 80

_C.MODEL.FCOS.IN_FEATURES = ["P3"]

_C.MODEL.FCOS.SCORE_THRESH_TEST = 0.05
_C.MODEL.FCOS.TOPK_CANDIDATES_TEST = 1000
_C.MODEL.FCOS.NMS_THRESH_TEST = 0.5
_C.MODEL.FCOS.NUM_CONVS = 4
_C.MODEL.FCOS.PRIOR_PROB = 0.01


# Loss parameters
_C.MODEL.FCOS.OUTPUTS = "FCOSGIOUOutputs"
_C.MODEL.FCOS.NORM = "BN"
_C.MODEL.FCOS.ACTIVATION_FN = "relu"
_C.MODEL.FCOS.SIZE_THRESHOLD = [64,128,256,512]
_C.MODEL.FCOS.FOCAL_LOSS_GAMMA = 2.0
_C.MODEL.FCOS.FOCAL_LOSS_ALPHA = 0.25

# ---------------------------------------------------------------------------- #
# FCOSPG Head
# ---------------------------------------------------------------------------- #
_C.MODEL.FCOSPG = CN()

# This is the number of foreground classes.
_C.MODEL.FCOSPG.NUM_CLASSES = 1
_C.MODEL.FCOSPG.OUTPUTS = "PGFCOSOutputs"

_C.MODEL.FCOSPG.IN_FEATURES = ["P3"]

_C.MODEL.FCOSPG.SCORE_THRESH_TEST = 0.0
_C.MODEL.FCOSPG.TOPK_CANDIDATES_TEST = 1000
_C.MODEL.FCOSPG.NMS_THRESH_TEST = 0.7
_C.MODEL.FCOSPG.NUM_CONVS = 4
_C.MODEL.FCOSPG.PRIOR_PROB = 0.01


# Loss parameters
_C.MODEL.FCOSPG.OUTPUTS = "FCOSPGGIOUOutputs"
_C.MODEL.FCOSPG.NORM = "BN"
_C.MODEL.FCOSPG.ACTIVATION_FN = "relu"
_C.MODEL.FCOSPG.SIZE_THRESHOLD = [64,128,256,512]
_C.MODEL.FCOSPG.FOCAL_LOSS_GAMMA = 2.0
_C.MODEL.FCOSPG.FOCAL_LOSS_ALPHA = 0.25
_C.MODEL.FCOSPG.PRE_NMS_TOPK_TRAIN = 12000
_C.MODEL.FCOSPG.PRE_NMS_TOPK_TEST = 6000
# Number of top scoring RPN proposals to keep after applying NMS
# When FPN is used, this limit is applied per level and then again to the union
# of proposals from all levels
# NOTE: When FPN is used, the meaning of this config is different from Detectron1.
# It means per-batch topk in Detectron1, but per-image topk here.
# See "modeling/rpn/rpn_outputs.py" for details.
_C.MODEL.FCOSPG.POST_NMS_TOPK_TRAIN = 2000
_C.MODEL.FCOSPG.POST_NMS_TOPK_TEST = 1000
_C.MODEL.FCOSPG.LOSS_SCALE = 0.25

# FUSIONPG Head
# ---------------------------------------------------------------------------- #
_C.MODEL.FUSIONPG = CN()

_C.MODEL.FUSIONPG.NAMES = [""]
_C.MODEL.FUSIONPG.NMS_THRESH_TEST = 0.7

_C.MODEL.FUSIONPG.PRE_NMS_TOPK_TRAIN = 12000
_C.MODEL.FUSIONPG.PRE_NMS_TOPK_TEST = 6000
# Number of top scoring RPN proposals to keep after applying NMS
# When FPN is used, this limit is applied per level and then again to the union
# of proposals from all levels
# NOTE: When FPN is used, the meaning of this config is different from Detectron1.
# It means per-batch topk in Detectron1, but per-image topk here.
# See "modeling/rpn/rpn_outputs.py" for details.
_C.MODEL.FUSIONPG.POST_NMS_TOPK_TRAIN = 2000
_C.MODEL.FUSIONPG.POST_NMS_TOPK_TEST = 1000
# RetinaNet Head
# ---------------------------------------------------------------------------- #
_C.MODEL.YOLACT = CN()

# This is the number of foreground classes.
_C.MODEL.YOLACT.NUM_CLASSES = 80

_C.MODEL.YOLACT.IN_FEATURES = ["P3", "P4", "P5", "P6"]

# Convolutions to use in the cls and bbox tower
# NOTE: this doesn't include the last conv for logits
_C.MODEL.YOLACT.NUM_CONVS = 4

# IoU overlap ratio [bg, fg] for labeling anchors.
# Anchors with < bg are labeled negative (0)
# Anchors  with >= bg and < fg are ignored (-1)
# Anchors with >= fg are labeled positive (1)
_C.MODEL.YOLACT.IOU_THRESHOLDS = [0.4, 0.5]
_C.MODEL.YOLACT.IOU_LABELS = [0, -1, 1]

# Prior prob for rare case (i.e. foreground) at the beginning of training.
# This is used to set the bias for the logits layer of the classifier subnet.
# This improves training stability in the case of heavy class imbalance.
_C.MODEL.YOLACT.PRIOR_PROB = 0.01

# Inference cls score threshold, only anchors with score > INFERENCE_TH are
# considered for inference (to improve speed)
_C.MODEL.YOLACT.SCORE_THRESH_TEST = 0.05
_C.MODEL.YOLACT.TOPK_CANDIDATES_TEST = 1000
_C.MODEL.YOLACT.NMS_THRESH_TEST = 0.5

# Weights on (dx, dy, dw, dh) for normalizing Retinanet anchor regression targets
_C.MODEL.YOLACT.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

# Loss parameters
_C.MODEL.YOLACT.FOCAL_LOSS_GAMMA = 2.0
_C.MODEL.YOLACT.FOCAL_LOSS_ALPHA = 0.25
_C.MODEL.YOLACT.SMOOTH_L1_LOSS_BETA = 0.1
_C.MODEL.YOLACT.PROTONET_NR = 32
_C.MODEL.YOLACT.MASK_SIZE = 63
_C.MODEL.YOLACT.MASK_THRESHOLD = 0.1
_C.MODEL.YOLACT.OUTPUTS = "YOLACTOutputs"
# ---------------------------------------------------------------------------- #
# ResNe[X]t options (ResNets = {ResNet, ResNeXt}
# Note that parts of a resnet may be used for both the backbone and the head
# These options apply to both
# ---------------------------------------------------------------------------- #
_C.MODEL.RESNETS = CN()

_C.MODEL.RESNETS.DEPTH = 50
_C.MODEL.RESNETS.OUT_FEATURES = ["res4"]  # res4 for C4 backbone, res2..5 for FPN backbone

# Number of groups to use; 1 ==> ResNet; > 1 ==> ResNeXt
_C.MODEL.RESNETS.NUM_GROUPS = 1

# Baseline width of each group.
# Scaling this parameters will scale the width of all bottleneck layers.
_C.MODEL.RESNETS.WIDTH_PER_GROUP = 64

# Place the stride 2 conv on the 1x1 filter
# Use True only for the original MSRA ResNet; use False for C2 and Torch models
_C.MODEL.RESNETS.STRIDE_IN_1X1 = True

# Apply dilation in stage "res5"
_C.MODEL.RESNETS.RES5_DILATION = 1

# Output width of res2. Scaling this parameters will scale the width of all 1x1 convs in ResNet
_C.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
_C.MODEL.RESNETS.STEM_OUT_CHANNELS = 64

# Apply Deformable Convolution in stages
# Specify if apply deform_conv on Res2, Res3, Res4, Res5
_C.MODEL.RESNETS.DEFORM_ON_PER_STAGE = [False, False, False, False]
# Use True to use modulated deform_conv (DeformableV2, https://arxiv.org/abs/1811.11168);
# Use False for DeformableV1.
_C.MODEL.RESNETS.DEFORM_MODULATED = False
# Number of groups in deformable conv.
_C.MODEL.RESNETS.DEFORM_NUM_GROUPS = 1
_C.MODEL.RESNETS.batch_norm_decay = 0.999
_C.MODEL.RESNETS.FROZEN_BN = False
_C.MODEL.RESNETS.MAKE_C6C7 = ""
#for C6C7
_C.MODEL.RESNETS.NORM = "evo_norm_s0"
_C.MODEL.RESNETS.ACTIVATION_FN = "NA"
_C.MODEL.RESNETS.OUT_CHANNELS = 256

_C.MODEL.SHUFFLENETS = CN()
_C.MODEL.SHUFFLENETS.use_max_pool = False
_C.MODEL.SHUFFLENETS.later_max_pool = False 
_C.MODEL.SHUFFLENETS.MAKE_C6C7 = ""
#for C6C7
_C.MODEL.SHUFFLENETS.NORM = "evo_norm_s0"
_C.MODEL.SHUFFLENETS.ACTIVATION_FN = "NA"
_C.MODEL.SHUFFLENETS.OUT_CHANNELS = 512


_C.MODEL.MOBILENETS = CN()

# Options: FrozenBN, GN, "SyncBN", "BN"
_C.MODEL.MOBILENETS.VERSION = 3

_C.MODEL.MOBILENETS.DEPTH_MULTIPLIER = 1.0
_C.MODEL.MOBILENETS.batch_norm_decay = 0.999
_C.MODEL.MOBILENETS.FROZEN_BN = False

_C.MODEL.EFFICIENTNETS = CN()

# Options: FrozenBN, GN, "SyncBN", "BN"
_C.MODEL.EFFICIENTNETS.TYPE = 0
_C.MODEL.EFFICIENTNETS.FROZEN_BN = False
_C.MODEL.EFFICIENTNETS.MAKE_C6C7 = "C6"
_C.MODEL.EFFICIENTNETS.NORM = "BN"
_C.MODEL.EFFICIENTNETS.ACTIVATION_FN = "relu"
#
_C.MODEL.BBDNET = CN()
_C.MODEL.BBDNET.USE_GLOBAL_ATTR = True
_C.MODEL.BBDNET.USE_GLOBAL_LOSS = True
_C.MODEL.BBDNET.USE_EDGE_LOSS = True
_C.MODEL.BBDNET.MAP_DATA = "P3"
_C.MODEL.BBDNET.END2END_TRAIN = False
_C.MODEL.BBDNET.SCORE_THRESH_TEST = 0.02
_C.MODEL.BBDNET.ABSOLUTE_BBOXES = False
_C.MODEL.BBDNET.USE_SENT_EDGES_FOR_NODE = True
_C.MODEL.BBDNET.NUM_PREPROCESSING_STEPS = 2
_C.MODEL.BBDNET.NUM_PROCESSING_STEPS = 3
_C.MODEL.BBDNET.NAME = "BBDNET3"

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()

_C.SOLVER.MAX_ITER = 40000

_C.SOLVER.BASE_LR = 0.001

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0001
# The weight decay that's applied to parameters of normalization layers
# (typically the affine transformation)
_C.SOLVER.WEIGHT_DECAY_NORM = 0.0

_C.SOLVER.GAMMA = 0.1
# The iteration number to decrease learning rate by GAMMA.
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.WARMUP_ITERS = 1000

# Save a checkpoint after every this number of iterations
_C.SOLVER.CHECKPOINT_PERIOD = 5000

_C.SOLVER.CLIP_NORM = 16

# Number of images per batch across all machines.
# If we have 16 GPUs and IMS_PER_BATCH = 32,
# each GPU will see 2 images per batch.
_C.SOLVER.IMS_PER_BATCH = 16

# Detectron v1 (and previous detection code) used a 2x higher LR and 0 WD for
# biases. This is not useful (at least for recent models). You should avoid
# changing these and they exist only to reproduce Detectron v1 training if
# desired.
_C.SOLVER.BIAS_LR_FACTOR = 1.0
_C.SOLVER.WEIGHT_DECAY_BIAS = _C.SOLVER.WEIGHT_DECAY
_C.SOLVER.LR_DECAY_TYPE = "piecewise"
_C.SOLVER.LR_DECAY_FACTOR = 0.1
_C.SOLVER.TRAIN_SCOPES = ""
_C.SOLVER.TRAIN_REPATTERN = ""
_C.SOLVER.FILTER_NAN_AND_INF_GRADS = False

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
# The sigmas used to calculate keypoint OKS.
# When empty it will use the defaults in COCO.
# Otherwise it should have the same length as ROI_KEYPOINT_HEAD.NUM_KEYPOINTS.
_C.TEST.KEYPOINT_OKS_SIGMAS = []
# Maximum number of detections to return per image during inference (100 is
# based on the limit established for the COCO dataset).
_C.TEST.DETECTIONS_PER_IMAGE = 100

# global config is for quick hack purposes.
# You can set them in command line or config files,
# and access it with:
#
# from config import global_cfg
# print(global_cfg.HACK)
#
# Do not commit any configs into it.
_C.GLOBAL = CN()
_C.GLOBAL.DEBUG = True
_C.GLOBAL.PROJ_NAME = "Demon"
_C.GLOBAL.SUMMARY_LEVEL = 0
_C.GLOBAL.RESEARCH = [""] #result_classes, result_bboxes
_C.GLOBAL.LOG_STEP = 200
_C.GLOBAL.SAVE_STEP = 500
_C.GLOBAL.GPU_MEM_FRACTION=0.0
_C.log_dir = ""
_C.ckpt_dir = ""
#BalanceBackboneHookV2
