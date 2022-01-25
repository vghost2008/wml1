from iotoolkit.mat_data import MatData
import object_detection2.keypoints as odk
import numpy as np
import object_detection2.bboxes as odb
import glob
import os.path as osp
'''
1.  head       
2.  left_shoulder  3.  right_shoulder
4.  left_elbow     5.  right_elbow
6.  left_wrist     7.  right_wrist     
8.  left_hip       9.  right_hip 
10. left_knee      11. right_knee 
12. left_ankle     13. right_ankle
'''
def __read_one_file(file_path):
    data = MatData(file_path).data
    x = data['x'].astype(np.float32)
    y = data['y'].astype(np.float32)
    visibility = data['visibility'].astype(np.float32)
    kps = np.stack([x,y,visibility],axis=-1)
    bbox = data['bbox'].astype(np.float32)

    return kps,bbox

def read_penn_action_data(labels_path):
    all_files = glob.glob(osp.join(labels_path,"*.mat"))

    res = []
    for file in all_files:
        kps,bbox = __read_one_file(file)
        res.append([file,kps,bbox])

    return res

class Trans2COCO:
    def __init__(self) -> None:
        self.dst_idxs = [5,6,7,8,9,10,11,12,13,14,15,16]
        self.src_idxs = np.array([2,3,4,5,6,7,8,9,10,11,12,13],dtype=np.int32)-1
        self.coco_idxs = [0,1,2,3,4]

    def __call__(self,mpii_kps,coco_kps):
        if len(mpii_kps.shape)==2:
            return self.trans_one(mpii_kps,coco_kps)
        res = []
        for mp,coco in zip(mpii_kps,coco_kps):
            res.append(self.trans_one(mp,coco))
        return np.array(res)

    def trans_one(self,mpii_kps,coco_kps):
        '''
        img: [RGB]
        '''
        res = np.zeros([17,3],dtype=np.float32)
        res[self.dst_idxs] = mpii_kps[self.src_idxs]
        res[self.coco_idxs] = coco_kps[self.coco_idxs]
        return res
