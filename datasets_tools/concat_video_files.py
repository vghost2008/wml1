import sys
import cv2
import os

import numpy as np

import img_utils as wmli

def get_video_info(video_path):
    video_reader = cv2.VideoCapture(video_path)
    frame_cnt = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_reader.get(cv2.CAP_PROP_FPS)
    video_reader.release()
    return (width,height,fps,frame_cnt)

def write_video(writer,size,video_path,repeat=1):
    print(f"Use video file {video_path}")
    video_reader = cv2.VideoCapture(video_path)
    frame_cnt = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(frame_cnt):
        res,frame = video_reader.read()
        if not res:
            break
        frame = wmli.resize_and_pad(frame,size)
        frame = frame.astype(np.uint8)
        writer.write(frame)
        if repeat>1:
            for _ in range(repeat-1):
                writer.write(frame)
    video_reader.release()

def init_writer(save_path,video_info):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(save_path,fourcc,video_info[2],(video_info[0],video_info[1]))
    return video_writer

if __name__ == "__main__":
    if len(sys.argv)<=3:
        print(f"Usage: python concat_video_files.py save_path video_path0 video_path1 ...")
        exit(0)
    video_info = get_video_info(sys.argv[2])
    writer = init_writer(sys.argv[1],video_info)
    for i,video_path in enumerate(sys.argv[2:]):
        write_video(writer,(video_info[0],video_info[1]),video_path)
    writer.release()

