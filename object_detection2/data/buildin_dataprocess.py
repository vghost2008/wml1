import iotoolkit.transform as trans
from .build_dataprocess import DATAPROCESS_REGISTRY


@DATAPROCESS_REGISTRY.register()
def simple(cfg, is_training):
    if is_training:
        trans_on_single_img = [trans.MaskNHW2HWN(),
                                    trans.ResizeToFixedSize(),
                                    trans.MaskHWN2NHW(),
                                    trans.WRemoveCrowdInstance(cfg.DATASETS.SKIP_CROWD_DURING_TRAINING),
                                    trans.AddBoxLens(),
                                    trans.UpdateHeightWidth(),
                                    ]
        trans_on_batch_img = [trans.FixDataInfo()]
    else:
        trans_on_single_img = [trans.MaskNHW2HWN(),
                               trans.ResizeToFixedSize(),
                               trans.MaskHWN2NHW(),
                               trans.BBoxesRelativeToAbsolute(),
                               trans.AddBoxLens(),
                               ]
        trans_on_batch_img = [trans.BBoxesAbsoluteToRelative(),
                              trans.FixDataInfo()]

    return (trans_on_single_img, trans_on_batch_img)

@DATAPROCESS_REGISTRY.register()
def AA(cfg,is_training):
    if is_training:
        trans_on_single_img = [trans.AutoAugment(),
                               trans.ResizeShortestEdge(short_edge_length=cfg.INPUT.MIN_SIZE_TRAIN,
                                                        max_size=cfg.INPUT.MAX_SIZE_TRAIN,
                                                        align=cfg.INPUT.SIZE_ALIGN),
                               trans.RandomFlipLeftRight(),
                               trans.BBoxesRelativeToAbsolute(),
                               trans.WRemoveCrowdInstance(cfg.DATASETS.SKIP_CROWD_DURING_TRAINING),
                               trans.AddBoxLens(),
                               trans.UpdateHeightWidth(),
                               ]
        if cfg.INPUT.SIZE_ALIGN > 1:
            trans_on_batch_img = [trans.PadtoAlign(align=cfg.INPUT.SIZE_ALIGN),
                                  trans.BBoxesAbsoluteToRelative(),
                                  trans.FixDataInfo()]
        else:
            trans_on_batch_img = [trans.BBoxesAbsoluteToRelative(),
                                  trans.FixDataInfo()]
    else:
        trans_on_single_img = [trans.MaskNHW2HWN(),
                               trans.ResizeShortestEdge(short_edge_length=cfg.INPUT.MIN_SIZE_TEST,
                                                        max_size=cfg.INPUT.MAX_SIZE_TEST,
                                                        align=cfg.INPUT.SIZE_ALIGN_FOR_TEST),
                               trans.MaskHWN2NHW(),
                               trans.BBoxesRelativeToAbsolute(),
                               trans.AddBoxLens(),
                               ]
        trans_on_batch_img = [trans.BBoxesAbsoluteToRelative(),
                              trans.FixDataInfo()]

    return (trans_on_single_img,trans_on_batch_img)

@DATAPROCESS_REGISTRY.register()
def WAA(cfg,is_training):
    if is_training:
        trans0 = [trans.WRandomTranslateX(),trans.WRandomEqualize()]
        trans1 = [trans.WRandomCutout()]
        trans2 = [trans.RandomRotateAnyAngle(max_angle=30, #max angle=30
                                             rotate_probability=0.6, #prob = 0.6
                                             enable=True),trans.WColor()]
        trans3 = [trans.NoTransform()]
        aa = trans.RandomSelectSubTransform([trans0,trans1,trans2,trans3])

        trans_on_single_img = [trans.MaskNHW2HWN(),trans.ResizeShortestEdge(short_edge_length=cfg.INPUT.MIN_SIZE_TRAIN,
                                                        max_size=cfg.INPUT.MAX_SIZE_TRAIN,
                                                        align=cfg.INPUT.SIZE_ALIGN),
                               trans.RandomFlipLeftRight(),
                               trans.MaskHWN2NHW(),
                               trans.BBoxesRelativeToAbsolute(),
                               aa,
                               trans.WRemoveCrowdInstance(cfg.DATASETS.SKIP_CROWD_DURING_TRAINING),
                               trans.AddBoxLens(),
                               trans.UpdateHeightWidth(),
                               ]
        if cfg.INPUT.SIZE_ALIGN > 1:
            trans_on_batch_img = [trans.PadtoAlign(align=cfg.INPUT.SIZE_ALIGN),
                                  trans.BBoxesAbsoluteToRelative(),
                                  trans.FixDataInfo()]
        else:
            trans_on_batch_img = [trans.BBoxesAbsoluteToRelative(),
                                  trans.FixDataInfo()]
    else:
        trans_on_single_img = [trans.MaskNHW2HWN(),
                               trans.ResizeShortestEdge(short_edge_length=cfg.INPUT.MIN_SIZE_TEST,
                                                        max_size=cfg.INPUT.MAX_SIZE_TEST,
                                                        align=cfg.INPUT.SIZE_ALIGN_FOR_TEST),
                               trans.MaskHWN2NHW(),
                               trans.BBoxesRelativeToAbsolute(),
                               trans.AddBoxLens(),
                               ]
        trans_on_batch_img = [trans.BBoxesAbsoluteToRelative(),
                              trans.FixDataInfo()]

    return (trans_on_single_img,trans_on_batch_img)

@DATAPROCESS_REGISTRY.register()
def coco(cfg,is_training):
    if is_training:
        trans_on_single_img = [trans.MaskNHW2HWN(),
                               trans.RandomFlipLeftRight(),
                                    trans.ResizeShortestEdge(short_edge_length=cfg.INPUT.MIN_SIZE_TRAIN,
                                                             max_size=cfg.INPUT.MAX_SIZE_TRAIN,
                                                             align=cfg.INPUT.SIZE_ALIGN),
                                    trans.MaskHWN2NHW(),
                                    trans.BBoxesRelativeToAbsolute(),
                                    trans.WRemoveCrowdInstance(cfg.DATASETS.SKIP_CROWD_DURING_TRAINING),
                                    trans.RandomRotateAnyAngle(max_angle=cfg.INPUT.ROTATE_ANY_ANGLE.MAX_ANGLE,
                                                               rotate_probability=cfg.INPUT.ROTATE_ANY_ANGLE.PROBABILITY,
                                                               enable=cfg.INPUT.ROTATE_ANY_ANGLE.ENABLE),
                                    trans.AddBoxLens(),
                                    trans.UpdateHeightWidth(),
                                    ]
        if cfg.INPUT.SIZE_ALIGN > 1:
            trans_on_batch_img = [trans.PadtoAlign(align=cfg.INPUT.SIZE_ALIGN),
                                       trans.BBoxesAbsoluteToRelative(),
                                       trans.FixDataInfo()]
        else:
            trans_on_batch_img = [trans.BBoxesAbsoluteToRelative(),
                                       trans.FixDataInfo()]
    else:
        trans_on_single_img = [trans.MaskNHW2HWN(),
                                    trans.ResizeShortestEdge(short_edge_length=cfg.INPUT.MIN_SIZE_TEST,
                                                             max_size=cfg.INPUT.MAX_SIZE_TEST,
                                                             align=cfg.INPUT.SIZE_ALIGN_FOR_TEST),
                                    trans.MaskHWN2NHW(),
                                    trans.BBoxesRelativeToAbsolute(),
                                    trans.AddBoxLens(),
                                    ]
        trans_on_batch_img = [trans.BBoxesAbsoluteToRelative(),
                                   trans.FixDataInfo()]

    return (trans_on_single_img,trans_on_batch_img)


@DATAPROCESS_REGISTRY.register()
def coco_nodirection(cfg, is_training):
    if is_training:
        trans_on_single_img = [trans.MaskNHW2HWN(),
                               trans.RandomFlipLeftRight(),
                                trans.RandomFlipUpDown(),
                                trans.RandomRotate(clockwise=True),
                               # trans.RandomRotate(clockwise=False), 因为已经有了上下及左右翻转的辅助，已经可以覆盖每一个角度
                               trans.ResizeShortestEdge(short_edge_length=cfg.INPUT.MIN_SIZE_TRAIN,
                                                        max_size=cfg.INPUT.MAX_SIZE_TRAIN,
                                                        align=cfg.INPUT.SIZE_ALIGN),
                               trans.MaskHWN2NHW(),
                               trans.BBoxesRelativeToAbsolute(),
                               trans.WRemoveCrowdInstance(cfg.DATASETS.SKIP_CROWD_DURING_TRAINING),
                               trans.AddBoxLens(),
                               trans.UpdateHeightWidth(),
                               ]
        if cfg.MODEL.INPUT_ALIGN > 1:
            trans_on_batch_img = [trans.PadtoAlign(align=cfg.MODEL.INPUT_ALIGN),
                                  trans.BBoxesAbsoluteToRelative(),
                                  trans.FixDataInfo()]
        else:
            trans_on_batch_img = [trans.BBoxesAbsoluteToRelative(),
                                  trans.FixDataInfo()]
    else:
        trans_on_single_img = [trans.MaskNHW2HWN(),
                               trans.ResizeShortestEdge(short_edge_length=cfg.INPUT.MIN_SIZE_TEST,
                                                        max_size=cfg.INPUT.MAX_SIZE_TEST,
                                                        align=cfg.INPUT.SIZE_ALIGN_FOR_TEST),
                               trans.MaskHWN2NHW(),
                               trans.BBoxesRelativeToAbsolute(),
                               trans.AddBoxLens(),
                               ]
        trans_on_batch_img = [trans.BBoxesAbsoluteToRelative(),
                              trans.FixDataInfo()]

    return (trans_on_single_img, trans_on_batch_img)


@DATAPROCESS_REGISTRY.register()
def SSD(cfg, is_training):
    if is_training:
        trans_on_single_img = [trans.MaskNHW2HWN(),
                               trans.RandomFlipLeftRight(),
                               trans.RandomSampleDistortedBoundingBox(min_object_covered=cfg.INPUT.CROP.MIN_OBJECT_COVERED,
                                                                      aspect_ratio_range=cfg.INPUT.CROP.ASPECT_RATIO,
                                                                      area_range=cfg.INPUT.CROP.SIZE,
                                                                      filter_threshold=cfg.INPUT.CROP.FILTER_THRESHOLD,
                                                                      probability=cfg.INPUT.CROP.PROBABILITY),
                               trans.MaskHWN2NHW(),
                               trans.BBoxesRelativeToAbsolute(),
                               trans.WRemoveCrowdInstance(cfg.DATASETS.SKIP_CROWD_DURING_TRAINING),
                               trans.AddBoxLens(),
                               trans.UpdateHeightWidth(),
                               ]
        if cfg.MODEL.INPUT_ALIGN > 1:
            trans_on_batch_img = [trans.PadtoAlign(align=cfg.MODEL.INPUT_ALIGN),
                                  trans.BBoxesAbsoluteToRelative(),
                                  trans.FixDataInfo()]
        else:
            trans_on_batch_img = [trans.BBoxesAbsoluteToRelative(),
                                  trans.FixDataInfo()]
    else:
        trans_on_single_img = [trans.MaskNHW2HWN(),
                               trans.ResizeShortestEdge(short_edge_length=cfg.INPUT.MIN_SIZE_TEST,
                                                        max_size=cfg.INPUT.MAX_SIZE_TEST,
                                                        align=cfg.INPUT.SIZE_ALIGN_FOR_TEST),
                               trans.MaskHWN2NHW(),
                               trans.BBoxesRelativeToAbsolute(),
                               trans.AddBoxLens(),
                               ]
        trans_on_batch_img = [trans.BBoxesAbsoluteToRelative(),
                              trans.FixDataInfo()]

    return (trans_on_single_img, trans_on_batch_img)

@DATAPROCESS_REGISTRY.register()
def SSD_Fix_Size(cfg, is_training):
    if is_training:
        size = cfg.INPUT.MIN_SIZE_TRAIN[0]
        trans_on_single_img = [trans.MaskNHW2HWN(),
                               trans.RandomFlipLeftRight(),
                               trans.RandomSampleDistortedBoundingBox(min_object_covered=cfg.INPUT.CROP.MIN_OBJECT_COVERED,
                                                                      aspect_ratio_range=cfg.INPUT.CROP.ASPECT_RATIO,
                                                                      area_range=cfg.INPUT.CROP.SIZE,
                                                                      filter_threshold=cfg.INPUT.CROP.FILTER_THRESHOLD,
                                                                      probability=cfg.INPUT.CROP.PROBABILITY),
                               trans.ResizeToFixedSize(size=[size,size]),
                               trans.MaskHWN2NHW(),
                               trans.BBoxesRelativeToAbsolute(),
                               trans.WRemoveCrowdInstance(cfg.DATASETS.SKIP_CROWD_DURING_TRAINING),
                               trans.AddBoxLens(),
                               trans.UpdateHeightWidth(),
                               ]
        trans_on_batch_img = [trans.BBoxesAbsoluteToRelative(),
                               trans.FixDataInfo()]
    else:
        size = cfg.INPUT.MIN_SIZE_TEST
        trans_on_single_img = [trans.MaskNHW2HWN(),
                               trans.ResizeToFixedSize(size=[size,size]),
                               trans.MaskHWN2NHW(),
                               trans.BBoxesRelativeToAbsolute(),
                               trans.AddBoxLens(),
                               ]
        trans_on_batch_img = [trans.BBoxesAbsoluteToRelative(),
                              trans.FixDataInfo()]

    return (trans_on_single_img, trans_on_batch_img)