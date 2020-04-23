#coding=utf-8
from object_detection_tools.statistics_tools import *
import sys
import pickle

save_path = sys.argv[-1]
if __name__ == "__main__":
    print(f"Save path {save_path}")
    sys.stdout.flush()

    statics = statistics_boxes_with_datas(coco_dataset(),
    #statics=statistics_boxes_with_datas(pascal_voc_dataset(),
                                          label_encoder=default_encode_label,
                                          labels_to_remove=None,
                                          max_aspect=None, absolute_size=True,
                                          trans_img_size=None)
    with open(save_path,"wb") as file:
        pickle.dump(statics,file)
    print("Finish.")
    sys.stdout.flush()
