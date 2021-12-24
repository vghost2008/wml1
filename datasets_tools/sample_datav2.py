import os.path as osp
import os
import random
import wml_utils as wmlu
import shutil

def sample_in_one_dir(dir_path,nr):
    print(f"Sample in {dir_path}")
    files = wmlu.recurse_get_filepath_in_dir(dir_path)
    random.shuffle(files)
    return files[:nr]

def append_to_dict(dict,key,data):
    if key in dict:
        dict[key].extend(data)
    else:
        dict[key] = data

def sample_in_dir(dir_path,nr,split_nr=None):
    '''
    sample data in dir_path's sub dirs
    sample nr images in each sub dir, if split_nr is not None, sampled nr images will be split 
    to split_nr part and saved in different dir
    '''
    res = {}
    dirs = wmlu.get_subdir_in_dir(dir_path,absolute_path=True)
    print(f"Find dirs in {dir_path}")
    wmlu.show_list(dirs)

    for dir in dirs:
        data = sample_in_one_dir(dir,nr)
        if split_nr is None:
            append_to_dict(res,0,data)
        else:
            data = wmlu.list_to_2dlistv2(data,split_nr)
            for i,d in enumerate(data):
                append_to_dict(res,i,d)

    return res

def save_data(data,save_dir):
    for k,v in data.items():
        tsd = osp.join(save_dir,str(k))
        wmlu.create_empty_dir(tsd,False)
        for f in v:
            dir_name = wmlu.base_name(osp.dirname(f))
            if dir_name == "":
                print(f"Get dir name faild {f}.")
            name = dir_name+"_"+osp.join(osp.basename(f))
            os.link(f,osp.join(tsd,name))
            #shutil.copy(f,osp.join(tsd,name))

if __name__ == "__main__":
    data_dir = "/home/wj/ai/mldata/basketball_datasets/multisports/rawframes"
    save_dir = "/home/wj/ai/mldata/basketball_objs/data2label"
    data_dir = "/home/wj/ai/mldata/0day/basketball"
    save_dir = "/home/wj/ai/mldata/basketball_objs/data2label/ba1"
    wmlu.create_empty_dir(save_dir,False)
    data = sample_in_dir(data_dir,150,1)
    print(f"Save_path {save_dir}")
    save_data(data,save_dir)