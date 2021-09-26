import numpy as np
import cv2 as cv
import glob
import os
import sys

data_dir = sys.argv[1]
output_path = sys.argv[2]
format_str = "img_{:05d}.jpg"
beg_idx = 1
if len(sys.argv)>3:
    beg_idx = int(sys.argv[3])
if len(sys.argv)>=5:
    format_str = sys.argv[4]
fps = 30.0
files = glob.glob(os.path.join(data_dir,"*.jpg"))
total_frames_nr = len(files)
files = []
for i in range(total_frames_nr):
    files.append(os.path.join(data_dir,format_str.format(i+beg_idx)))
test_img = cv.imread(files[0])
size = (test_img.shape[1],test_img.shape[0])
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter(output_path,fourcc, fps, size)
for ifn in files:
    frame = cv.imread(ifn)
    out.write(frame)
out.release()
