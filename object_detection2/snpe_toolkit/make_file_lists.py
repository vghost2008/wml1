import os
import wml_utils as wmlu
from object_detection2.snpe_toolkit.toolkit import *

save_dir = "/home/wj/0day/data"
src_dir = "/home/wj/ai/mldata/MOT/MOT15/test/ETH-Crossing/img1"
dir_path_in_target_file = None
max_file_nr = 20
input_size = (256,480)

if dir_path_in_target_file is None:
    dir_path_in_target_file = save_dir
wmlu.create_empty_dir(save_dir)
images = wmlu.recurse_get_filepath_in_dir(src_dir,suffix=".jpg")
if max_file_nr is not None:
    images = images[:max_file_nr]

#output_layers = []
output_layers=["shared_head/l2_normalize/Square", "shared_head/hw_regr/Conv_1/Conv2D",
                "shared_head/ct_regr/Conv_1/Conv2D","shared_head/heat_ct/Conv_1/Conv2D"],
input_file_list = os.path.join(save_dir,"input.txt")

with open(input_file_list, "w") as f:
    if output_layers is not None and len(output_layers)>0:
        v = f"#{output_layers[0]}"
        for x in output_layers[1:]:
            v += f" {x}"
        v += "\n"
        f.write(v)
    for file in images:
        raw_name = wmlu.base_name(file)+".raw"
        raw_path = os.path.join(save_dir,raw_name)
        save_raw_path = os.path.join(dir_path_in_target_file,raw_name)
        file_to_snpe_raw(file,raw_path,input_size=input_size)
        f.write(save_raw_path+"\n")
