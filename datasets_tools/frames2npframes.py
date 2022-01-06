import argparse
import wml_utils as wmlu
import os.path as osp
import glob
import pickle
import img_utils as wmli

data_type = "call"
def parse_args():
    parser = argparse.ArgumentParser(description='extract optical flows')
    parser.add_argument('--src_dir', default="/home/wj/ai/mldata1/driver_actions/train_data/call",type=str, help='source video directory')
    parser.add_argument('--out_dir', default="/home/wj/ai/mldata1/driver_actions/train_data/call_np",type=str, help='output rawframe directory')
    parser.add_argument(
        '--new_short',
        type=int,
        default=256,
        help='resize image short side length keeping ratio')
    args = parser.parse_args()

    return args

def trans_dir(src_dir,out_dir):
    sub_dirs = wmlu.recurse_get_subdir_in_dir(src_dir)
    pattern = "img_{:05d}.jpg"
    wmlu.create_empty_dir(out_dir,remove_if_exists=False)
    for sd in sub_dirs:
        if sd[-1] == "/":
            sd = sd[:-1]
        rsd = osp.join(src_dir,sd)

        files = glob.glob(osp.join(rsd,"*.jpg"))
        if len(files)==0:
            continue

        save_name = osp.join(out_dir,sd+f"_{data_type}_{len(files)}.np")
        save_dir_name = osp.dirname(save_name)
        wmlu.create_empty_dir(save_dir_name,remove_if_exists=False)

        all_frames = []
        for i in range(len(files)):
            file_name = pattern.format(i+1)
            file_path = osp.join(rsd,file_name)
            if not osp.exists(file_path):
                print(f"File {file_path} not exists, len={len(files)}")
                continue
            with open(file_path,"rb") as f:
                data = f.read()
            if args.new_short>2:
                img = wmli.decode_img(data)
                img = wmli.resize_short_size(img,args.new_short)
                data = wmli.encode_img(img)
            all_frames.append(data)

        print(f"Save {save_name}")
        with open(save_name,"wb") as f:
            pickle.dump(all_frames,f)


if __name__ == "__main__":
    args = parse_args()
    trans_dir(args.src_dir,args.out_dir)