import cv2 as cv
import os.path as osp
import numpy as np
import img_utils as wmli
import wml_utils as wmlu
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="build gif")
    #parser.add_argument("--out_dir",type=str,default='/home/wj/ai/mldata1/0day',help="out dir")
    parser.add_argument("--out_dir",type=str,default='/home/wj/videos',help="out dir")
    parser.add_argument("--cameras",nargs='+',type=int,default=[6,2,4],help="cameras")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    wmlu.create_empty_dir(args.out_dir,remove_if_exists=False)
    out_dir = args.out_dir
    cameras = args.cameras
    print(f"Cameras {cameras}")
    video_captures = [cv.VideoCapture(x) for x in cameras]
    video_writer = [wmli.VideoWriter(osp.join(out_dir,f"video{i}.mp4")) for i in range(len(video_captures))]
    while True:
        imgs = [x.read()[1] for x in video_captures]
        for vw,img in zip(video_writer,imgs):
            try:
                vw.write(img[...,::-1])
            except Exception as e:
                print(e)
        if len(imgs)==0:
            continue
        try:
            img = np.concatenate(imgs,axis=1)
            cv.imshow('img',img)
        except Exception as e:
            print(e)
            for x in imgs:
                if x is not None:
                    print(x.shape)
                else:
                    print('None')
        if (cv.waitKey(30)&255)==27:
            break
    [x.release() for x in video_writer]
