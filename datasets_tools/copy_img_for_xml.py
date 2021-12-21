import argparse
import glob
import os.path as osp
import wml_utils as wmlu

def parse_args():
    parser = argparse.ArgumentParser(
        description='arguments')
    parser.add_argument('xml_dir', default=None,type=str,help='xml_dir')
    parser.add_argument('img_dir', default=None,type=str,help='img_dir')
    args = parser.parse_args()
    return args

def copy_imgfiles(xml_dir,img_dir,img_suffix=".jpg"):
    xml_files = glob.glob(osp.join(xml_dir,"*.xml"))
    for xf in xml_files:
        base_name = wmlu.base_name(xf)
        img_name = base_name+img_suffix
        img_path = osp.join(img_dir,img_name)
        dst_img_path = osp.join(xml_dir,img_name)
        if osp.exists(img_path):
            wmlu.try_link(img_path,dst_img_path)

if __name__ == "__main__":
    args = parse_args()
    copy_imgfiles(args.xml_dir,args.img_dir)