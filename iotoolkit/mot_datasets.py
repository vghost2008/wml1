import wml_utils as wmlu
import os
import numpy as np

class MOTDatasets(object):
    def __init__(self,dirs):
        self.dirs = dirs
        self.tid_curr = 0
        self.dir_curr = 0

    def get_data_items(self):
        for seq_root in self.dirs:
            seqs = wmlu.get_subdir_in_dir(seq_root)
            for seq in seqs:
                seq_info = open(os.path.join(seq_root, seq, 'seqinfo.ini')).read()
                seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
                seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

                gt_txt = os.path.join(seq_root, seq, 'gt', 'gt.txt')
                if not os.path.exists(gt_txt):
                    print(f"{gt_txt} not exists!")
                    continue
                gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
                idx = np.lexsort(gt.T[:2, :])
                gt = gt[idx, :]

                fid_datas = {}
                for v in gt:
                    fid, tid, x, y, w, h, mark, label, _ = v[:9]
                    if mark == 0:
                        continue
                    if 'MOT15' not in seq_root and label !=1:
                        continue
                    if fid in fid_datas:
                        fid_datas[fid].append(v[:9])
                    else:
                        fid_datas[fid] = [v[:9]]
                tid_to_cls = {}
                for fid,datas in fid_datas.items():
                    tmp_data = []
                    for _, tid, x, y, w, h, mark, label, _ in datas:
                        fid = int(fid)
                        tid = int(tid)
                        if not tid in tid_to_cls:
                            self.tid_curr += 1
                            tid_to_cls[tid] = self.tid_curr
                            cls = self.tid_curr
                        else:
                            cls = tid_to_cls[tid]
                        xmin = x/seq_width
                        ymin = y/seq_height
                        xmax = (x+w)/seq_width
                        ymax = (y+h)/seq_height
                        tmp_data.append([cls,[ymin,xmin,ymax,xmax]])

                    if len(tmp_data)>0:
                        img_name = '{:06d}.jpg'.format(fid)
                        img_path = os.path.join(seq_root,seq,"img1",img_name)
                        img_data = {'img_width':seq_width,'img_height':seq_height,'img_path':img_path}
                        yield img_data,tmp_data

        print(f"Last tid value {self.tid_curr}")