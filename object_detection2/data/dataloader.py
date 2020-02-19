#coding=utf-8
import wmodule
import iotoolkit.transform as trans
import wsummary
import tensorflow as tf

class DataLoader(wmodule.WModule):
    def __init__(self,cfg,*args,**kwargs):
        super().__init__(cfg,*args,**kwargs)
        if self.is_training:
            self.trans_on_single_img = [trans.MaskNHW2HWN(),
                                        #trans.ResizeToFixedSize(),
                                        trans.RandomFlipLeftRight(),
                                        trans.RandomFlipUpDown(),
                                        trans.RandomRotate(clockwise=True),
                                        #trans.RandomRotate(clockwise=False), 因为已经有了上下及左右翻转的辅助，已经可以覆盖每一个角度
                                        trans.ResizeShortestEdge(short_edge_length=self.cfg.INPUT.MIN_SIZE_TRAIN,
                                                                 max_size=self.cfg.INPUT.MAX_SIZE_TRAIN,
                                                                 align=self.cfg.INPUT.SIZE_ALIGN),
                                        trans.MaskHWN2NHW(),
                                        trans.BBoxesRelativeToAbsolute(),
                                        trans.AddBoxLens(),
                                        trans.UpdateHeightWidth(),
                                        ]
            self.trans_on_batch_img = [trans.BBoxesAbsoluteToRelative(),
                                       trans.FixDataInfo()]
            '''self.trans_on_single_img = [trans.MaskNHW2HWN(),
                                        trans.ResizeToFixedSize(),
                                        trans.UpdateHeightWidth(),
                                        ]
            self.trans_on_batch_img = [trans.FixDataInfo()]'''
        else:
            self.trans_on_single_img = [trans.MaskNHW2HWN(),
                                        #trans.ResizeToFixedSize(),
                                        trans.ResizeShortestEdge(short_edge_length=self.cfg.INPUT.MIN_SIZE_TEST,
                                                                 max_size=self.cfg.INPUT.MAX_SIZE_TEST,
                                                                 align=self.cfg.INPUT.SIZE_ALIGN),
                                        trans.MaskHWN2NHW(),
                                        trans.BBoxesRelativeToAbsolute(),
                                        trans.AddBoxLens(),
                                        ]
            self.trans_on_batch_img = [trans.BBoxesAbsoluteToRelative(),
                                       trans.FixDataInfo()]

    @staticmethod
    def get_pad_shapes(dataset):
        shapes = dataset.output_shapes
        res = {}
        for k,v in shapes.items():
            shape = v.as_list()
            res[k] = shape
        return res

    def load_data(self,path,func,num_classes,batch_size=None,is_training=True):
        data = func(path,transforms=self.trans_on_single_img)
        if is_training:
            data = data.repeat()
            batch_size = self.cfg.SOLVER.IMS_PER_BATCH
            data = data.shuffle(batch_size*4)
        else:
            batch_size = 1 if batch_size is None else batch_size
        data = data.padded_batch(batch_size,self.get_pad_shapes(data),drop_remainder=True)
        if len(self.trans_on_batch_img) == 1:
            data = data.map(self.trans_on_batch_img[0])
        elif len(self.trans_on_batch_img) > 1:
            data = data.map(trans.WTransformList(self.trans_on_batch_img))
        if batch_size>0:
            data = data.prefetch(2)
        return data.make_one_shot_iterator(),num_classes

    @staticmethod
    def detection_image_summary(inputs,
                           category_index=None,
                           max_boxes_to_draw=20,
                           min_score_thresh=0.2,name="detection_image_summary",max_outputs=3):
        image = inputs.get('image',None)

        if 'gt_boxes' not in inputs:
            if image is not None:
                wsummary.image_summaries(image,
                                     name=name+"_onlyimg")
            return
        
        boxes = inputs.get('gt_boxes',None)
        classes = inputs.get('gt_labels',None)
        instance_masks = inputs.get('gt_masks',None)
        lengths = inputs.get('gt_length',None)
        if instance_masks is not None:
            wsummary.image_summaries(image,
                                     name=name+"_onlyimg")
            wsummary.detection_image_summary(tf.ones_like(image,dtype=tf.float32)*0.5,
                                             boxes,classes,instance_masks=instance_masks,
                                             lengths=lengths,category_index=category_index,
                                             max_boxes_to_draw=max_boxes_to_draw,
                                             min_score_thresh=min_score_thresh,
                                             name=name+"_onlymask")
        else:
            wsummary.detection_image_summary(image,boxes,classes,instance_masks=instance_masks,
                                             lengths=lengths,category_index=category_index,
                                             max_boxes_to_draw=max_boxes_to_draw,
                                             min_score_thresh=min_score_thresh,
                                             name=name)


