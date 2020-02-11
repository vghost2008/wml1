#coding=utf-8
import wmodule
import iotoolkit.transform as trans
import wsummary

class DataLoader(wmodule.WModule):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    @staticmethod
    def get_pad_shapes(dataset):
        shapes = dataset.output_shapes
        res = {}
        for k,v in shapes.items():
            shape = v.as_list()
            res[k] = shape
        return res

    def load_data(self,path,func,num_classes):
        data = func(path,transforms=[trans.MaskNHW2HWN(),trans.ResizeToFixedSize(),trans.MaskHWN2NHW(),trans.AddBoxLens()])
        data = data.repeat()
        data = data.shuffle(32)
        #data = data.batch(self.cfg.SOLVER.IMS_PER_BATCH,drop_remainder=True)
        data = data.padded_batch(self.cfg.SOLVER.IMS_PER_BATCH,self.get_pad_shapes(data),drop_remainder=True)
        return data.make_one_shot_iterator(),num_classes

    @staticmethod
    def detection_image_summary(inputs,
                           category_index=None,
                           max_boxes_to_draw=20,
                           min_score_thresh=0.2,name="detection_image_summary",max_outputs=3):
        image = inputs.get('image',None)
        boxes = inputs.get('gt_boxes',None)
        classes = inputs.get('gt_labels',None)
        instance_masks = inputs.get('gt_masks',None)
        lengths = inputs.get('gt_length',None)
        wsummary.detection_image_summary(image,boxes,classes,instance_masks=instance_masks,
                                         lengths=lengths,category_index=category_index,
                                         max_boxes_to_draw=max_boxes_to_draw,
                                         min_score_thresh=min_score_thresh,
                                         name=name)

