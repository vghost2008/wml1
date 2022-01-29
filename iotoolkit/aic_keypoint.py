import json
import numpy as np
import os.path as osp

'''
       0: 'Right Shoulder',
        1: 'Right Elbow',
        2: 'Right Wrist',
        3: 'Left Shoulder',
        4: 'Left Elbow',
        5: 'Left Wrist',
        6: 'Right Hip',
        7: 'Right Knee',
        8: 'Right Ankle',
        9: 'Left Hip',
        10: 'Left Knee',
        11: 'Left Ankle',
        12: 'Head top',
        13: 'Neck'
'''
def read_aic_keypoint(file_path,sub_dir_name=None):
    with open(file_path,"r") as f:
        datas = json.load(f)

    res = []
    for data in datas:
        img_name = data['image_id']
        kps = []
        bboxes = []
        human_data = data['human_annotations']
        for k,v in data['keypoint_annotations'].items():
            v = np.array(v,dtype=np.float32)
            v = np.reshape(v,[-1,3])
            mask = v[...,-1]==3
            v[mask] = 0
            bbox = np.array(human_data[k],dtype=np.float32)
            kps.append(v)
            bboxes.append(bbox)
        kps = np.array(kps)
        bboxes = np.array(bboxes)
        if sub_dir_name is not None:
            img_name = osp.join(sub_dir_name,img_name)
        res.append([img_name+".jpg",bboxes,kps])

    return res

class Trans2COCO:
    def __init__(self) -> None:
        self.dst_idxs = [5,6,7,8,9,10,11,12,13,14,15,16]
        self.src_idxs = [3,0,4,1,5,2,9,6,10,7,11,8]
        self.coco_idxs = [0,1,2,3,4]

    def __call__(self,mpii_kps,coco_kps=None):
        if len(mpii_kps.shape)==2:
            return self.trans_one(mpii_kps,coco_kps)
        res = []
        if coco_kps is not None:
            for mp,coco in zip(mpii_kps,coco_kps):
                res.append(self.trans_one(mp,coco))
        else:
            for mp in mpii_kps:
                res.append(self.trans_one(mp))
        return np.array(res)

    def trans_one(self,mpii_kps,coco_kps=None):
        '''
        img: [RGB]
        '''
        res = np.zeros([17,3],dtype=np.float32)
        res[self.dst_idxs] = mpii_kps[self.src_idxs]
        if coco_kps is not None:
            res[self.coco_idxs] = coco_kps[self.coco_idxs]
        return res