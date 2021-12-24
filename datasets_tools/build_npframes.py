import argparse
import glob
import os
import os.path as osp
import sys
import warnings
from multiprocessing import Pool
import img_utils as wmli
import mmcv
import numpy as np
import pickle
import wml_utils as wmlu

#img_process_fn = None
def img_process_fn(img):
    H,W,_ = img.shape
    return img[:,W//2:]

def extract_frame(vid_item):
    """Generate optical flow using dense flow.

    Args:
        vid_item (list): Video item containing video full path,
            video (short) path, video id.

    Returns:
        bool: Whether generate optical flow successfully.
    """
    full_path, subdir, save_path = vid_item

    out_full_path = save_path

    vr = wmli.VideoReader(full_path)
    print(f"{full_path} fps {vr.fps}.")
    # for i in range(len(vr)):
    all_frames = []
    try:
        for i, vr_frame in enumerate(vr):
            if vr_frame is not None:
                if img_process_fn is not None:
                    vr_frame = img_process_fn(vr_frame)
                w, h, _ = np.shape(vr_frame)
                if args.new_short == 0:
                    if args.new_width == 0 or args.new_height == 0:
                        # Keep original shape
                        out_img = vr_frame
                    else:
                        out_img = mmcv.imresize(vr_frame,
                                                (args.new_width,
                                                 args.new_height))
                else:
                    if min(h, w) == h:
                        new_h = args.new_short
                        new_w = int((new_h / h) * w)
                    else:
                        new_w = args.new_short
                        new_h = int((new_w / w) * h)
                    out_img = mmcv.imresize(vr_frame, (new_h, new_w))
                all_frames.append(wmli.encode_img(out_img))
            else:
                warnings.warn(
                    'Length inconsistent!'
                    f'Early stop with {i + 1} out of {len(vr)} frames.')
                break

        with open(out_full_path,"wb") as f:
            pickle.dump(all_frames,f)
    except Exception as e:
        print(f"Process {full_path} faild, {e}")

    print(f'{full_path} -> {out_full_path} done')
    sys.stdout.flush()
    return True


def parse_args():
    parser = argparse.ArgumentParser(description='extract optical flows')
    parser.add_argument('src_dir', type=str, help='source video directory')
    parser.add_argument('out_dir', type=str, help='output rawframe directory')
    parser.add_argument('--num_worker', default=8,type=int, help='num worker')
    parser.add_argument(
        '--ext',
        type=str,
        default='mp4',
        #choices=['avi', 'mp4', 'webm','MOV'],
        help='video file extensions')
    parser.add_argument(
        '--new-width', type=int, default=0, help='resize image width')
    parser.add_argument(
        '--new-height', type=int, default=0, help='resize image height')
    parser.add_argument(
        '--new-short',
        type=int,
        default=256,
        help='resize image short side length keeping ratio')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not osp.isdir(args.out_dir):
        print(f'Creating folder: {args.out_dir}')
        os.makedirs(args.out_dir)

    _sub_dirs = wmlu.recurse_get_subdir_in_dir(args.src_dir,append_self=True)
    datas = []
    for sd in _sub_dirs:
        rsd = osp.join(args.src_dir,sd)
        files = glob.glob(osp.join(rsd,"*."+args.ext))

        if len(files)>0:
            for f in files:
                v_save_dir = osp.join(args.out_dir,sd)
                v_save_path = osp.join(v_save_dir,wmlu.base_name(f)+".np")
                wmlu.create_empty_dir(v_save_dir,remove_if_exists=False)
                datas.append([f,sd,v_save_path])
    print(f"Total find {len(datas)} files.")
    sys.stdout.flush()
    pool = Pool(args.num_worker)
    pool.map(
        extract_frame,
        datas)