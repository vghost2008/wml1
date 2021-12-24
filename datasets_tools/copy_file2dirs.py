import cv2 as cv
import imageio
import glob
import os
import sys
import os.path as osp
import img_utils as wmli
import wml_utils as wmlu
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="build gif")
    parser.add_argument("src_dir",type=str,help="src dir")
    parser.add_argument("out_dir",type=str,help="out dir")
    parser.add_argument("--ext",type=str,default=".gif",help="img ext")
    parser.add_argument("--files_nr",type=int,default=500,help="files nr")
    parser.add_argument("--dir_name",type=str,default="2124",help="save dir name")
    args = parser.parse_args()
    return args

def save2dirs(files,save_dir,files_nr,dir_name):
    files = wmlu.list_to_2dlist(files,files_nr)
    for i,lfiles in enumerate(files):
        cur_save_dir = osp.join(save_dir,dir_name+f"_{i}")
        wmlu.create_empty_dir(cur_save_dir,remove_if_exists=False)
        for f in lfiles:
            wmlu.try_link(f,cur_save_dir)

if __name__ == "__main__":
    args = parse_args()
    files = wmlu.recurse_get_filepath_in_dir(args.src_dir,suffix=args.ext)
    save2dirs(files,
              args.out_dir,
              args.files_nr,
              args.dir_name)

