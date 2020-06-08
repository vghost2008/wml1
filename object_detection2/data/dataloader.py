#coding=utf-8
import wmodule
import iotoolkit.transform as trans
import wsummary
import tensorflow as tf
from .build_dataprocess import DATAPROCESS_REGISTRY
import socket
from collections import Iterable

class DataLoader(wmodule.WModule):
    category_index = None
    def __init__(self,cfg,*args,**kwargs):
        super().__init__(cfg=cfg,*args,**kwargs)
        self.trans_on_single_img,self.trans_on_batch_img = DATAPROCESS_REGISTRY.get(self.cfg.INPUT.DATAPROCESS)(cfg,self.is_training)
        if not isinstance(self.trans_on_batch_img,Iterable):
            self.trans_on_batch_img = [self.trans_on_batch_img]
        if self.cfg.INPUT.STITCH > 0.001:
            self.trans_on_batch_img = [trans.Stitch(self.cfg.INPUT.STITCH)]+self.trans_on_batch_img

    @staticmethod
    def get_pad_shapes(dataset):
        shapes = dataset.output_shapes
        res = {}
        for k,v in shapes.items():
            shape = v.as_list()
            res[k] = shape
        return res

    def load_data(self,path,func,num_classes,category_index,batch_size=None,is_training=True):
        if "vghost" in socket.gethostname():
            num_parallel = 8
        else:
            num_parallel = 12
        print(f"Num parallel {num_parallel}")

        print("Trans on single img:",self.trans_on_single_img)
        data = func(path,transforms=self.trans_on_single_img,num_parallel=num_parallel)
        if is_training:
            data = data.repeat()
            batch_size = self.cfg.SOLVER.IMS_PER_BATCH
            data = data.shuffle(batch_size*4)
        else:
            batch_size = 1 if batch_size is None else batch_size
        DataLoader.category_index = category_index
        data = data.padded_batch(batch_size,self.get_pad_shapes(data),drop_remainder=True)
        print("Trans on batch img:",self.trans_on_batch_img)
        if len(self.trans_on_batch_img) == 1:
            data = data.map(self.trans_on_batch_img[0],num_parallel_calls=num_parallel)
        elif len(self.trans_on_batch_img) > 1:
            data = data.map(trans.WTransformList(self.trans_on_batch_img),num_parallel_calls=num_parallel)
        if batch_size>0:
            data = data.prefetch(8)
        return data.make_one_shot_iterator(),num_classes

    @staticmethod
    def detection_image_summary(inputs,
                           max_boxes_to_draw=20,
                           min_score_thresh=0.2,name="detection_image_summary",max_outputs=3,show_mask=True):
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
        if instance_masks is not None and show_mask:
            wsummary.detection_image_summary(image,
                                             boxes,classes,instance_masks=instance_masks,
                                             lengths=lengths,category_index=DataLoader.category_index,
                                             max_boxes_to_draw=max_boxes_to_draw,
                                             min_score_thresh=min_score_thresh,
                                             max_outputs=max_outputs,
                                             name=name)
        else:
            wsummary.detection_image_summary(image,boxes,classes,
                                             lengths=lengths,category_index=DataLoader.category_index,
                                             max_boxes_to_draw=max_boxes_to_draw,
                                             min_score_thresh=min_score_thresh,
                                             max_outputs=max_outputs,
                                             name=name)


