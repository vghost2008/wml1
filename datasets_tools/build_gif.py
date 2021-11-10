import numpy as np
import cv2 as cv
import imageio
import glob
import os
import sys
import os.path as osp
import img_utils as wmli

data_dir = sys.argv[1]
output_path = sys.argv[2]
beg_idx = 1
total_nr = -1
int_val = 4
if len(sys.argv)>3:
    beg_idx = int(sys.argv[3])
if len(sys.argv)>4:
    total_nr = int(sys.argv[4])
fps = 30.0
gif_images = []
if osp.isdir(data_dir):
    files = glob.glob(os.path.join(data_dir,"*.jpg"))
    total_frames_nr = len(files)
    if total_nr>0:
        total_frames_nr = min(total_nr,total_frames_nr)
    files = []
    for i in range(0,total_frames_nr,int_val):
        files.append(os.path.join(data_dir,f'img_{i+beg_idx:05d}.jpg'))
    for ifn in files:
        frame = imageio.imread(ifn)
        gif_images.append(frame)
else:
    reader = wmli.VideoReader(data_dir)
    if total_nr<1:
        total_nr = 1e9
    for i,frame in enumerate(reader):
        if i+1<beg_idx or i+1-beg_idx>total_nr:
            continue
        frame = wmli.resize_width(frame,640)
        gif_images.append(frame)
imageio.mimsave(output_path,gif_images,fps=fps)
