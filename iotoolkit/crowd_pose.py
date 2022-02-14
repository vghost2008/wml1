import json
import numpy as np
import object_detection2.visualization as odv
import object_detection2.bboxes as odb
import os.path as osp
import wml_utils as wmlu
import img_utils as wmli
import object_detection2.keypoints as odk
import math

'''
'left_shoulder', 'right_shoulder', 0,1
'left_elbow', 'right_elbow', 2,3
 'left_wrist', 'right_wrist', 4,5
'left_hip', 'right_hip', 6,7
 'left_knee', 'right_knee', 8,9
 'left_ankle', 'right_ankle', 10,11
 'head', 'neck' 12,13
'''
def read_crowd_pose(file_path):
    with open(file_path,"r") as f:
        datas = json.load(f)
    id2img = {}
    for data in datas['images']:
        id = data['id']
        file_name = data['file_name']
        id2img[id] = file_name

    res = {}
    for data in datas['annotations']:
        kps = data['keypoints']
        kps = np.array(kps,dtype=np.float32)
        kps = np.reshape(kps,[-1,3])
        if data['iscrowd']:
            continue
        mask = kps[...,2]
        if np.sum(mask)<2:
            continue
        if np.count_nonzero(mask>0.1)<2:
            continue
        bbox = data['bbox']
        bbox = [bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]]
        kps_bbox = odk.npget_bbox(kps)
        bbox = odb.bbox_of_boxes([bbox,kps_bbox])
        id = data['image_id']
        image = id2img[id]
        if id not in res:
            res[id] = [image,[kps],[bbox]]
        else:
            res[id][1].append(kps)
            res[id][2].append(bbox)
    
    r_res = []
    for k,v in res.items():
        r_res.append([v[0],np.array(v[1]),np.array(v[2])])

    return r_res

class Trans2COCO:
    def __init__(self) -> None:
        self.dst_idxs = [5,6,7,8,9,10,11,12,13,14,15,16]
        self.src_idxs = np.array([0,1,2,3,4,5,6,7,8,9,10,11],dtype=np.int32)
        self.coco_idxs = [0,1,2,3,4]
    
    @staticmethod
    def get_head_pos(kp):
        p0 = kp[12]
        p1 = kp[13]
        if p0[2]>0 and p1[2]>0:
            dx = p0[0] - p1[0]
            dy = p0[1] - p1[1]
            dis = math.sqrt(dx*dx+dy*dy)*0.75
            cx = (p0[0] + p1[0])/2
            cy = (p0[1] + p1[1])/2
            x0 = cx-dis
            x1 = cx+dis
            y0 = cy-dis
            y1 = cy+dis
            return [x0,y0,x1,y1]
        else:
            return None
    
    @staticmethod 
    def kps_in_bbox(kps,bbox):
        nr = kps.shape[0]
        for i in range(nr):
            if kps[i,2]<= 0:
                return False
            x = kps[i,0]
            y = kps[i,1]
            if x<bbox[0] or x>bbox[2] or y<bbox[1] or y>bbox[3]:
                return False
        return True


    def __call__(self,mpii_kps,coco_kps):
        if len(mpii_kps.shape)==2:
            return self.trans_one(mpii_kps,coco_kps)
        res = []
        for mp,coco in zip(mpii_kps,coco_kps):
            res.append(self.trans_one(mp,coco))
        return np.array(res)

    '''def trans_one(self,mpii_kps,coco_kps):
        res = np.zeros([17,3],dtype=np.float32)
        res[self.dst_idxs] = mpii_kps[self.src_idxs]
        res[self.coco_idxs] = coco_kps[self.coco_idxs]
        return res'''
    
    def trans_one(self,mpii_kps,coco_kps=None):
        '''
        img: [RGB]
        '''
        res = np.zeros([17,3],dtype=np.float32)
        res[self.dst_idxs] = mpii_kps[self.src_idxs]
        if coco_kps is not None:
            left_right_pairs = [[5,6],[11,12]]
            is_good = True
            for pair in left_right_pairs:
                l,r = pair
                if res[l,0]<res[r,0] or res[l,2]<0.1 or res[r,2]<0.1:
                    is_good = False
                    break

            head_bbox = self.get_head_pos(mpii_kps)
            if is_good and head_bbox is not None and self.kps_in_bbox(coco_kps[self.coco_idxs],head_bbox):
                res[self.coco_idxs] = coco_kps[self.coco_idxs]
        return res

if __name__ == "__main__":
    file_path = '/home/wj/ai/mldata1/crowd_pose/CrowdPose/crowdpose_train.json'
    images_dir = '/home/wj/ai/mldata1/crowd_pose/images'
    save_dir = '/home/wj/ai/mldata1/crowd_pose/tmp/vis'
    wmlu.create_empty_dir(save_dir,remove_if_exists=False)
    datas = read_crowd_pose(file_path)
    do_vis = True
    for data in datas:
        image_name,kps,bbox = data
        image = osp.join(images_dir,image_name)
        img = wmli.imread(image)
        img = odv.draw_keypoints(img, kps, no_line=True)
        t_bboxes = np.array([bbox])
        t_bboxes = odb.npchangexyorder(t_bboxes)
        img = odv.draw_bboxes(img, bboxes=t_bboxes, is_relative_coordinate=False)
        save_path = osp.join(save_dir,image_name)
        wmli.imwrite(save_path, img)