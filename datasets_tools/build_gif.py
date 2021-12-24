import numpy as np
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
    parser.add_argument("--video_ext",type=str,default=None,help="video ext")
    parser.add_argument("--img_ext",type=str,default=".jpg",help="img ext")
    parser.add_argument("--fps",type=int,default=10,help="output fps")
    parser.add_argument("--beg_idx",type=int,default=0,help="output fps")
    parser.add_argument("--total_nr",type=int,default=-1,help="output fps")
    args = parser.parse_args()
    return args


def trans_one(src_data,out_dir,fps,beg_idx=0,total_nr=-1):
    reader = wmli.VideoReader(src_data)
    if total_nr>1:
        frames = []
        for i in range(beg_idx,beg_idx+total_nr):
            frames.append(reader[i])
    else:
        frames = [x for x in reader]
    save_name = wmlu.base_name(src_data)+".gif"
    save_path = osp.join(out_dir,save_name)
    if osp.exists(save_path):
        print(f"Error {save_path} exists.")
    imageio.mimsave(save_path,frames,fps=fps)


if __name__ == "__main__":
    args = parse_args()
    wmlu.create_empty_dir(args.src_dir,remove_if_exists=False)
    if args.video_ext is not None:
        files = wmlu.recurse_get_filepath_in_dir(args.src_dir,suffix=args.video_ext)
        if args.total_nr>1:
            print(f"ERROR: can't specify total nr for video.")
        for file in files:
            trans_one(file,args.out_dir,args.fps,args.beg_idx,args.total_nr)
    else:
        _sub_dirs = wmlu.recurse_get_subdir_in_dir(args.src_dir,append_self=True)
        sub_dirs = []
        for sd in _sub_dirs:
            rd = osp.join(args.src_dir,sd)
            files = glob.glob(osp.join(rd,args.img_ext))
            if len(files)>3:
                sub_dirs.append(rd)
        for sd in sub_dirs:
            trans_one(sd,args.out_dir,args.fps,args.beg_idx,args.total_nr)
