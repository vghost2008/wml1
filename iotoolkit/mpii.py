from iotoolkit.mat_data import MatData
import object_detection2.keypoints as odk
import numpy as np
import object_detection2.bboxes as odb
'''
id - joint id (0 - r ankle, 1 - r knee, 2 - r hip, 
3 - l hip, 4 - l knee, 5 - l ankle, 6 - pelvis, 
7 - thorax, 8 - upper neck, 9 - head top,
10 - r wrist, 11 - r elbow, 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist)
'''
def read_mpii_data(path):
    mpii_data = MatData(path).RELEASE
    annolist = mpii_data.annolist
    img_train = mpii_data.img_train

    res = []
    for i in range(0,len(annolist)):
        annorect_item = annolist.annorect.get_array_item(i)
        if annorect_item is None:
            continue
        image = str(annolist.image[i].name)
        image_datas = []
        try:
            for j in range(len(annorect_item)):
                person_anno = annorect_item[j]
                x1 = person_anno.x1
                y1 = person_anno.y1
                x2 = person_anno.x2
                y2 = person_anno.y2
                if 'annopoints' not in person_anno.data_keys():
                    continue
                points = person_anno.annopoints
                if points is None:
                    continue
                kps = person_anno.annopoints.point
                res_kps = np.zeros([16,3],dtype=np.float32)
                for k in range(len(kps)):
                    kp = kps[k]
                    id = kp.id
                    res_kps[id,0] = kp.x
                    res_kps[id,1] = kp.y
                    is_visible = kp.is_visible
                    res_kps[id,2] = float(is_visible)+1 if is_visible is not None else 0
                person_data = [np.array([x1,y1,x2,y2]),res_kps]
                image_datas.append(person_data)
        except Exception as e:
            if img_train.data[i]==1:
                print(f"ERROR",i,e,img_train.data[i])
        if len(image_datas)>0:
            image_datas = list(zip(*image_datas))
            bboxes = np.array(image_datas[0])
            kps = np.array(image_datas[1])
            _bboxes = odk.npbatchget_bboxes(kps)
            r_bboxes = []
            for bbox0,bbox1 in zip(bboxes,_bboxes):
                bbox = odb.bbox_of_boxes([bbox0,bbox1])
                r_bboxes.append(bbox)
            bboxes = np.array(r_bboxes,dtype=np.float32)
            res.append([image,bboxes,kps])

    return res

class Trans2COCO:
    def __init__(self) -> None:
        self.dst_idxs = [5,6,7,8,9,10,11,12,13,14,15,16]
        self.src_idxs = [13,12,14,11,15,10,3,2,4,1,5,0]
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
