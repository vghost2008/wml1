import numpy as np
import pickle
import wml_utils as wmlu
import random
import img_utils as wmli
import os.path as osp

def dump_files(file,save_dir):
    print(f"Process {file}")
    wmlu.create_empty_dir(save_dir,remove_if_exists=False)
    with open(file,"rb") as f:
        data = pickle.load(f)
        nr = len(data)
        idxs = list(range(nr))
        for idx in idxs:
            img = wmli.decode_img(data[idx])
            save_path_name = osp.join(save_dir,f"img{idx}.jpg")
            #print(save_path_name)
            wmli.imwrite(save_path_name,img)
            if img is None or img.shape[0] < 2 or img.shape[1] < 2 or img.shape[2] != 3:
                print(f"Read {file}:{idx} faild.")
    return (file,nr)

if __name__ == "__main__":
    src_dir = "/home/wj/ai/mldata1/driver_actions/train_data1"
    out_dir = "/home/wj/ai/tmp/npframes"
    files = wmlu.recurse_get_filepath_in_dir(src_dir,suffix=".np")
    random.shuffle(files)
    dump_files(files[0],out_dir)