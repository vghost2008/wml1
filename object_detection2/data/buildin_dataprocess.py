import object_detection2.data.transforms.transform as trans
from .build_dataprocess import DATAPROCESS_REGISTRY


@DATAPROCESS_REGISTRY.register()
def NONE(cfg, is_training):
    trans_on_single_img = [trans.AddBoxLens(),
                           trans.BBoxesRelativeToAbsolute(),
                           trans.UpdateHeightWidth(),
                           ]
    trans_on_batch_img = [trans.FixDataInfo(),
                          trans.BBoxesAbsoluteToRelative()]
    return (trans_on_single_img, trans_on_batch_img)

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
def simple_semantic(cfg, is_training):
    if is_training:
        trans_on_single_img = [trans.MaskNHW2HWN(),
                                    trans.ResizeToFixedSize(),
                                    trans.MaskHWN2NHW(),
                                    trans.WRemoveCrowdInstance(cfg.DATASETS.SKIP_CROWD_DURING_TRAINING),
                                    trans.GetSemanticMaskFromCOCO(num_classes=cfg.MODEL.NUM_CLASSES,no_background=False),
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
        trans0 = [trans.WRandomTranslate(translate_horizontal=True),trans.WRandomEqualize()]
        trans1 = [trans.WRandomTranslate(pixels=40,translate_horizontal=False),trans.WRandomCutout()]
        trans2 = [trans.WShear(shear_horizontal=False),trans.WRandomTranslate(pixels=40,translate_horizontal=False)]
        trans3 = [trans.RandomRotateAnyAngle(max_angle=30, #max angle=30
                                             rotate_probability=0.6, #prob = 0.6
                                             enable=True),trans.WColor()]
        trans4 = [trans.NoTransform()]
        aa = trans.RandomSelectSubTransform([trans0,trans1,trans2,trans3,trans4])

        trans_on_single_img = [
                                trans.BBoxesRelativeToAbsolute(),
                                aa,
                                trans.BBoxesAbsoluteToRelative(),
                                trans.MaskNHW2HWN(),
                                trans.ResizeShortestEdge(short_edge_length=cfg.INPUT.MIN_SIZE_TRAIN,
                                                                              max_size=cfg.INPUT.MAX_SIZE_TRAIN,
                                                                              align=cfg.INPUT.SIZE_ALIGN),
                                trans.RandomFlipLeftRight(),
                                trans.BBoxesRelativeToAbsolute(),
                                trans.MaskHWN2NHW(),
                                trans.WRemoveCrowdInstance(cfg.DATASETS.SKIP_CROWD_DURING_TRAINING),
                                trans.RemoveZeroAreaBBox(2),
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
                                       trans.CheckBBoxes(),
                                       trans.FixDataInfo()]
        else:
            trans_on_batch_img = [trans.BBoxesAbsoluteToRelative(),
                                  trans.CheckBBoxes(),
                                  trans.FixDataInfo()]
    else:
        trans_on_single_img = [trans.AddSize(),
                               trans.MaskNHW2HWN(),
                               trans.ResizeShortestEdge(short_edge_length=cfg.INPUT.MIN_SIZE_TEST,
                                                             max_size=cfg.INPUT.MAX_SIZE_TEST,
                                                             align=cfg.INPUT.SIZE_ALIGN_FOR_TEST),
                               trans.MaskHWN2NHW(),
                               trans.BBoxesRelativeToAbsolute(),
                               trans.AddBoxLens(),
                               ]
        if cfg.INPUT.SIZE_ALIGN_FOR_TEST > 1:
            trans_on_batch_img = [trans.PadtoAlign(align=cfg.INPUT.SIZE_ALIGN_FOR_TEST),
                                  trans.BBoxesAbsoluteToRelative(),
                                  trans.FixDataInfo()]
        else:
            trans_on_batch_img = [trans.BBoxesAbsoluteToRelative(),
                                  trans.FixDataInfo()]

    return (trans_on_single_img,trans_on_batch_img)

@DATAPROCESS_REGISTRY.register()
def coco_fixed_size(cfg,is_training):
    if is_training:
        trans_on_single_img = [trans.MaskNHW2HWN(),
                               trans.RandomFlipLeftRight(),
                               trans.WTransImgToFloat(),
                               trans.RandomCrop(crop_size=cfg.INPUT.FIXED_SIZE_TRAIN),
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
                                  trans.CheckBBoxes(),
                                  trans.FixDataInfo()]
        else:
            trans_on_batch_img = [trans.BBoxesAbsoluteToRelative(),
                                  trans.CheckBBoxes(),
                                  trans.FixDataInfo()]
    else:
        trans_on_single_img = [trans.AddSize(),
                               trans.MaskNHW2HWN(),
                               trans.ResizeShortestEdge(short_edge_length=cfg.INPUT.MIN_SIZE_TEST,
                                                        max_size=cfg.INPUT.MAX_SIZE_TEST,
                                                        align=cfg.INPUT.SIZE_ALIGN_FOR_TEST),
                               trans.MaskHWN2NHW(),
                               trans.BBoxesRelativeToAbsolute(),
                               trans.AddBoxLens(),
                               ]
        if cfg.INPUT.SIZE_ALIGN_FOR_TEST > 1:
            trans_on_batch_img = [trans.PadtoAlign(align=cfg.INPUT.SIZE_ALIGN_FOR_TEST),
                                  trans.BBoxesAbsoluteToRelative(),
                                  trans.FixDataInfo()]
        else:
            trans_on_batch_img = [trans.BBoxesAbsoluteToRelative(),
                                  trans.FixDataInfo()]

    return (trans_on_single_img,trans_on_batch_img)
@DATAPROCESS_REGISTRY.register()
def coco_semantic(cfg,is_training):
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
                               trans.GetSemanticMaskFromCOCO(num_classes=cfg.MODEL.NUM_CLASSES,no_background=False),
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
        trans_on_single_img = [trans.AddSize(),
                               trans.MaskNHW2HWN(),
                               trans.ResizeShortestEdge(short_edge_length=cfg.INPUT.MIN_SIZE_TEST,
                                                        max_size=cfg.INPUT.MAX_SIZE_TEST,
                                                        align=cfg.INPUT.SIZE_ALIGN_FOR_TEST),
                               trans.MaskHWN2NHW(),
                               trans.BBoxesRelativeToAbsolute(),
                               trans.GetSemanticMaskFromCOCO(num_classes=cfg.MODEL.NUM_CLASSES, no_background=False),
                               trans.AddBoxLens(),
                               ]
        if cfg.INPUT.SIZE_ALIGN_FOR_TEST > 1:
            trans_on_batch_img = [trans.PadtoAlign(align=cfg.INPUT.SIZE_ALIGN_FOR_TEST),
                                  trans.BBoxesAbsoluteToRelative(),
                                  trans.FixDataInfo()]
        else:
            trans_on_batch_img = [trans.BBoxesAbsoluteToRelative(),
                                  trans.FixDataInfo()]

    return (trans_on_single_img,trans_on_batch_img)



@DATAPROCESS_REGISTRY.register()
def coco_nodirection(cfg, is_training):
    if is_training:
        trans_on_single_img = [trans.MaskNHW2HWN(),
                               trans.RandomFlipLeftRight(),
                               trans.RandomFlipUpDown(),
                               trans.RandomRotate(clockwise=True), #上述三个组合就可以构成各个方位0.125概率的覆盖
                               trans.ResizeShortestEdge(short_edge_length=cfg.INPUT.MIN_SIZE_TRAIN,
                                                        max_size=cfg.INPUT.MAX_SIZE_TRAIN,
                                                        align=cfg.INPUT.SIZE_ALIGN),
                               trans.MaskHWN2NHW(),
                               trans.BBoxesRelativeToAbsolute(),
                               trans.WRemoveCrowdInstance(cfg.DATASETS.SKIP_CROWD_DURING_TRAINING),
                               trans.RemoveZeroAreaBBox(2),
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
        if cfg.INPUT.SIZE_ALIGN> 1:
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

    return (trans_on_single_img, trans_on_batch_img)

@DATAPROCESS_REGISTRY.register()
def SSD_Fix_Size(cfg, is_training):
    if is_training:
        size = cfg.INPUT.MIN_SIZE_TRAIN[0]
        trans_on_single_img = [
                               trans.WRemoveCrowdInstance(cfg.DATASETS.SKIP_CROWD_DURING_TRAINING),
                               trans.MaskNHW2HWN(),
                               trans.RandomFlipLeftRight(),
                               trans.WTransImgToFloat(),
                               trans.RandomSampleDistortedBoundingBox(min_object_covered=cfg.INPUT.CROP.MIN_OBJECT_COVERED,
                                                                      aspect_ratio_range=cfg.INPUT.CROP.ASPECT_RATIO,
                                                                      area_range=cfg.INPUT.CROP.SIZE,
                                                                      filter_threshold=cfg.INPUT.CROP.FILTER_THRESHOLD,
                                                                      probability=cfg.INPUT.CROP.PROBABILITY),
                               trans.ResizeToFixedSize(size=[size,size]),
                               trans.MaskHWN2NHW(),
                               trans.BBoxesRelativeToAbsolute(),
                               trans.AddBoxLens(),
                               trans.UpdateHeightWidth(),
                               ]
        trans_on_batch_img = [trans.BBoxesAbsoluteToRelative(),
                               trans.FixDataInfo()]
    else:
        size = cfg.INPUT.MIN_SIZE_TEST
        trans_on_single_img = [trans.AddSize(),
                               trans.MaskNHW2HWN(),
                               trans.ResizeToFixedSize(size=[size,size]),
                               trans.MaskHWN2NHW(),
                               trans.BBoxesRelativeToAbsolute(),
                               trans.AddBoxLens(),
                               ]
        trans_on_batch_img = [trans.BBoxesAbsoluteToRelative(),
                              trans.FixDataInfo()]

    return (trans_on_single_img, trans_on_batch_img)

@DATAPROCESS_REGISTRY.register()
def SSD_Fix_Size_semantic(cfg, is_training):
    if is_training:
        size = cfg.INPUT.MIN_SIZE_TRAIN[0]
        trans_on_single_img = [
            trans.WRemoveCrowdInstance(cfg.DATASETS.SKIP_CROWD_DURING_TRAINING),
            trans.MaskNHW2HWN(),
            trans.RandomFlipLeftRight(),
            trans.WTransImgToFloat(),
            trans.RandomSampleDistortedBoundingBox(min_object_covered=cfg.INPUT.CROP.MIN_OBJECT_COVERED,
                                                   aspect_ratio_range=cfg.INPUT.CROP.ASPECT_RATIO,
                                                   area_range=cfg.INPUT.CROP.SIZE,
                                                   filter_threshold=cfg.INPUT.CROP.FILTER_THRESHOLD,
                                                   probability=cfg.INPUT.CROP.PROBABILITY),
            trans.ResizeToFixedSize(size=[size,size]),
            trans.MaskHWN2NHW(),
            trans.BBoxesRelativeToAbsolute(),
            trans.GetSemanticMaskFromCOCO(num_classes=cfg.MODEL.NUM_CLASSES,no_background=False),
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
                               trans.GetSemanticMaskFromCOCO(num_classes=cfg.MODEL.NUM_CLASSES, no_background=False),
                               trans.AddBoxLens(),
                               ]
        trans_on_batch_img = [trans.BBoxesAbsoluteToRelative(),
                              trans.FixDataInfo()]

    return (trans_on_single_img, trans_on_batch_img)

@DATAPROCESS_REGISTRY.register()
def NULL(cfg, is_training):
    if is_training:
        size = cfg.INPUT.MIN_SIZE_TRAIN[0]
        trans_on_single_img = [trans.NoTransform(),
                               trans.AddBoxLens(),
                               trans.UpdateHeightWidth(),
                               ]
        trans_on_batch_img = [trans.FixDataInfo()]
    else:
        size = cfg.INPUT.MIN_SIZE_TEST
        trans_on_single_img = [trans.NoTransform(),
                               trans.AddBoxLens(),
                               ]
        trans_on_batch_img = [trans.FixDataInfo()]

    return (trans_on_single_img, trans_on_batch_img)

@DATAPROCESS_REGISTRY.register()
def TRANS1(cfg,is_training):
    if is_training:
        trans0 = [trans.WRandomTranslate(translate_horizontal=True)]
        trans1 = [trans.WRandomTranslate(pixels=20,translate_horizontal=False),trans.WRandomCutout()]
        trans2 = [trans.WRandomTranslate(pixels=20,translate_horizontal=False)]
        trans3 = [trans.NoTransform()]
        aa = trans.RandomSelectSubTransform([trans0,trans1,trans2,trans3])
        trans4 = [trans.ResizeShortestEdge(short_edge_length=cfg.INPUT.MIN_SIZE_TRAIN,
                         max_size=2048,
                         align=1)]
        trans5 = [trans.NoTransform()]
        aa1 = trans.RandomSelectSubTransform([trans4,trans5])

        trans_on_single_img = [
            trans.RemoveMask(),
            trans.WTransImgToFloat(),
            trans.WRemoveOverlap(threshold=0.6),
            trans.MaskNHW2HWN(),
            aa1,
            trans.MaskHWN2NHW(),
            trans.BBoxesRelativeToAbsolute(),
            trans.RandomRotateAnyAngle(max_angle=15,
                                       use_mask=False,
                                       rotate_probability=0.3,
                                       rotate_bboxes_type=0),
            trans.BBoxesAbsoluteToRelative(),
            trans.MaskNHW2HWN(),
            trans.SampleDistortedBoundingBox(area_range=cfg.INPUT.CROP.SIZE,
                                             aspect_ratio_range=cfg.INPUT.CROP.ASPECT_RATIO,
                                             filter_threshold=0.7,
                                             use_image_if_no_bounding_boxes=True),
            trans.MaskHWN2NHW(),
            trans.RandomNoise(0.2,10),
            trans.BBoxesRelativeToAbsolute(),
            aa,
            trans.BBoxesAbsoluteToRelative(),
            trans.MaskNHW2HWN(),
            trans.RandomFlipLeftRight(),
            trans.RandomFlipUpDown(),
            trans.RandomRotate(clockwise=True),
            trans.BBoxesRelativeToAbsolute(),
            trans.MaskHWN2NHW(),
            trans.RemoveZeroAreaBBox(2),
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
        trans_on_batch_img.append(trans.RemoveFakeInstance())
    else:
        trans_on_single_img = [
            trans.AddSize(),
            trans.WTransImgToFloat(),
            trans.BBoxesRelativeToAbsolute(),
            trans.AddBoxLens(),
            ]
        trans_on_batch_img = [
                              trans.BBoxesAbsoluteToRelative(),
                              trans.FixDataInfo()]

    return (trans_on_single_img,trans_on_batch_img)

@DATAPROCESS_REGISTRY.register()
def OPENPOSE(cfg, is_training):
    if is_training:
        size = cfg.INPUT.MIN_SIZE_TRAIN[0]
        trans_on_single_img = [
            trans.MaskNHW2HWN(),
            trans.RandomFlipLeftRight(cfg=cfg),
            trans.RemoveMask(),
            trans.WTransImgToFloat(),
            #trans.ShowInfo("INFO0"),
            trans.ResizeToFixedSize(size=[size,size]),
            trans.MaskHWN2NHW(),
            trans.BBoxesRelativeToAbsolute(),
            trans.RandomRotateAnyAngle(max_angle=cfg.INPUT.ROTATE_ANY_ANGLE.MAX_ANGLE,
                                       rotate_probability=cfg.INPUT.ROTATE_ANY_ANGLE.PROBABILITY,
                                       enable=cfg.INPUT.ROTATE_ANY_ANGLE.ENABLE),
            trans.AddBoxLens(),
            trans.UpdateHeightWidth(),
            trans.WRemoveCrowdInstance(cfg.DATASETS.SKIP_CROWD_DURING_TRAINING),
            #trans.ShowInfo("INFO1"),
        ]
        trans_on_batch_img = [trans.BBoxesAbsoluteToRelative(),
                              trans.CheckBBoxes(),
                              trans.FixDataInfo()]
    else:
        size = cfg.INPUT.MIN_SIZE_TEST
        trans_on_single_img = [trans.AddSize(),
                               trans.MaskNHW2HWN(),
                               trans.ResizeToFixedSize(size=[size,size]),
                               trans.MaskHWN2NHW(),
                               trans.BBoxesRelativeToAbsolute(),
                               trans.AddBoxLens(),
                               ]
        trans_on_batch_img = [trans.BBoxesAbsoluteToRelative(),
                              trans.FixDataInfo()]

    return (trans_on_single_img, trans_on_batch_img)

@DATAPROCESS_REGISTRY.register()
def SIMPLE_KP(cfg, is_training):
    if is_training:
        size = cfg.INPUT.MIN_SIZE_TRAIN[0]
        trans_on_single_img = [
            trans.MaskNHW2HWN(),
            trans.WTransImgToFloat(),
            #trans.ShowInfo("INFO0"),
            trans.ResizeToFixedSize(size=[size,size]),
            trans.MaskHWN2NHW(),
            trans.BBoxesRelativeToAbsolute(),
            trans.AddBoxLens(),
            trans.UpdateHeightWidth(),
            trans.WRemoveCrowdInstance(cfg.DATASETS.SKIP_CROWD_DURING_TRAINING),
            #trans.ShowInfo("INFO1"),
        ]
        trans_on_batch_img = [trans.BBoxesAbsoluteToRelative(),
                              trans.CheckBBoxes(),
                              trans.FixDataInfo()]
    else:
        size = cfg.INPUT.MIN_SIZE_TEST
        trans_on_single_img = [trans.AddSize(),
                               trans.MaskNHW2HWN(),
                               trans.ResizeToFixedSize(size=[size,size]),
                               trans.MaskHWN2NHW(),
                               trans.BBoxesRelativeToAbsolute(),
                               trans.AddBoxLens(),
                               ]
        trans_on_batch_img = [trans.BBoxesAbsoluteToRelative(),
                              trans.FixDataInfo()]

    return (trans_on_single_img, trans_on_batch_img)

