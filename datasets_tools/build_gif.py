import numpy as np
import cv2 as cv
import imageio
import glob
import os
import sys

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
files = glob.glob(os.path.join(data_dir,"*.jpg"))
total_frames_nr = len(files)
if total_nr>0:
    total_frames_nr = min(total_nr,total_frames_nr)
files = []
for i in range(0,total_frames_nr,int_val):
    files.append(os.path.join(data_dir,f'img_{i+beg_idx:05d}.jpg'))
gif_images = []
for ifn in files:
    frame = imageio.imread(ifn)
    gif_images.append(frame)
imageio.mimsave(output_path,gif_images,fps=5)
