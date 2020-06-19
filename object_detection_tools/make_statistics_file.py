#coding=utf-8
from object_detection_tools.statistics_tools import *
import sys
import pickle
from functools import partial

if len(sys.argv)==1:
    save_path = wmlu.home_dir("ai/mldata2/0day/coco_statics.dat")
    save_path = wmlu.home_dir("ai/mldata2/0day/ocr_statics.dat")
    save_path = wmlu.home_dir("ai/mldata2/0day/cocoval_statics.dat")
else:
    save_path = sys.argv[-1]
if __name__ == "__main__":
    print(f"Save path {save_path}")
    sys.stdout.flush()

    #statics = statistics_boxes_with_datas(pascal_voc_dataset(),
    #statics=statistics_boxes_with_datas(labelme_dataset(),
    #statics=statistics_boxes_with_datas(coco_dataset(),
    statics=statistics_boxes_with_datas(coco2014_val_dataset(),
                                          label_encoder=default_encode_label,
                                          labels_to_remove=None,
                                          max_aspect=None, absolute_size=True,
                                          #trans_img_size=None,
                                          #trans_img_size = partial(trans_img_long_size_to,long_size=512))
                                          trans_img_size = partial(trans_img_short_size_to,short_size=576))
    with open(save_path,"wb") as file:
        pickle.dump(statics,file)
    print("Finish.")
    sys.stdout.flush()
