from iotoolkit.mat_data import MatData
import object_detection2.keypoints as odk
import numpy as np
import object_detection2.bboxes as odb
'''
Right ankle
				Right knee
				Right hip
				Left hip
				Left knee
				Left ankle
				Right wrist 6
				Right elbow 7
				Right shoulder 8
 				Left shoulder 9
				Left elbow 10
				Left wrist 11
				Neck
				Head top
'''
def read_lspet_data(path):
    data = MatData(path).data['joints']
    kps = np.transpose(data, [2, 0, 1])
    bboxes = odk.npbatchget_bboxes(kps)
    return kps,bboxes

class Trans2COCO:
    def __init__(self) -> None:
        self.dst_idxs = [5,6,7,8,9,10,11,12,13,14,15,16]
        self.src_idxs = [9,8,10,7,11,6,3,2,4,1,5,0]
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