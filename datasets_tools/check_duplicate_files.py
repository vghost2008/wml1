import sys
import wml_utils as wmlu
import argparse
import os.path as osp

def parse_args():
    parser = argparse.ArgumentParser(description="build gif")
    parser.add_argument("src_dir",type=str,help="src dir")
    parser.add_argument("--ext",type=str,default=".gif",help="img ext")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    all_files = wmlu.recurse_get_filepath_in_dir(args.src_dir,suffix=args.ext)
    name = [osp.basename(x) for x in all_files]
    nr0 = len(name)
    name = set(name)
    nr1 = len(name)
    print(f"{nr0} {nr1}")