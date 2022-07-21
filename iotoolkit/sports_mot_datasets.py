import wml_utils as wmlu
import os
import numpy as np
import os.path as osp

class SportsMOTDatasets(object):
    def __init__(self,dirs,absolute_coord=False,use_det=False):
        if not isinstance(dirs,list):
            dirs = [dirs]
        self.dirs = dirs
        self.tid_curr = 0
        self.dir_curr = 0
        self.absolute_coord = absolute_coord
        self.use_det = use_det

    def get_data_items(self):
        dir_idx = -1
        for seq_root in self.dirs:
            seqs = wmlu.get_subdir_in_dir(seq_root)
            for seq in seqs:
                dir_idx += 1
                seq_info = open(os.path.join(seq_root, seq, 'seqinfo.ini')).read()
                seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
                seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

                if self.use_det:
                    gt_txt = osp.join(seq_root,seq,"det","det.txt")
                else:
                    gt_txt = os.path.join(seq_root, seq, 'gt', 'gt.txt')
                if not os.path.exists(gt_txt):
                    print(f"{gt_txt} not exists.")
                    continue
                gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
                idx = np.lexsort(gt.T[:2, :])
                gt = gt[idx, :]

                fid_datas = {}
                for v in gt:
                    fid, tid, x, y, w, h, *_ = v[:9]
                    if fid in fid_datas:
                        fid_datas[fid].append(v[:9])
                    else:
                        fid_datas[fid] = [v[:9]]
                tid_to_cls = {}
                for fid,datas in fid_datas.items():
                    labels = []
                    bboxes = []
                    for _, tid, x, y, w, h, *_ in datas:
                        fid = int(fid)
                        tid = int(tid)
                        if not tid in tid_to_cls:
                            self.tid_curr += 1
                            tid_to_cls[tid] = self.tid_curr
                            cls = self.tid_curr
                        else:
                            cls = tid_to_cls[tid]
                        if not self.absolute_coord:
                            xmin = x/seq_width
                            ymin = y/seq_height
                            xmax = (x+w)/seq_width
                            ymax = (y+h)/seq_height
                        else:
                            xmin = x
                            ymin = y
                            xmax = x + w
                            ymax = y + h
                        labels.append(cls)
                        bboxes.append([ymin,xmin,ymax,xmax])

                    if len(labels)>0:
                        img_name = '{:06d}.jpg'.format(fid)
                        img_path = os.path.join(seq_root,seq,"img1",img_name)
                        yield img_path, [seq_height,seq_width], labels, None, bboxes, None, None, None,dir_idx

        print(f"Last tid value {self.tid_curr}")