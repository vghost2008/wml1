import wml_utils as wmlu
import os
import sys

def add_foler_for_images(dir_path,save_dir):
    files = wmlu.recurse_get_filepath_in_dir(dir_path,suffix=".jpg;;..png;;.jpeg")
    for file in files:
        base_name = wmlu.base_name(file)
        cur_save_dir = os.path.join(save_dir,base_name)
        if not os.path.exists(cur_save_dir):
            os.makedirs(cur_save_dir)
        os.link(file,os.path.join(cur_save_dir,os.path.basename(file)))

if __name__ == "__main__":
    add_foler_for_images(sys.argv[1],sys.argv[2])