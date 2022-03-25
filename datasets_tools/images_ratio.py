import wml_utils as wmlu
import cv2
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="build gif")
    parser.add_argument("src_dir",type=str,help="src dir")
    parser.add_argument("--img_ext",type=str,default=".jpg",help="img ext")
    args = parser.parse_args()
    return args

def get_ratios(src_dir,ext):
    files = wmlu.recurse_get_filepath_in_dir(src_dir,suffix=ext)
    ratios = []
    total_skip = 0
    for file in files:
        img = cv2.imread(file)
        shape = img.shape
        if shape[1]>0 and shape[0]>0:
            ratios.append(shape[1]/shape[0])
            total_skip += 1
    print(f"Total find {len(files)} files, total skip {total_skip} files.")
    ratios = np.array(ratios)
    mean = np.mean(ratios)
    std = np.std(ratios)
    return mean,std

if __name__ == "__main__":
    args = parse_args()
    mean,std = get_ratios(args.src_dir,args.img_ext)
    print(f"W/H mean: {mean:.3f}, std= {std:.3f}")