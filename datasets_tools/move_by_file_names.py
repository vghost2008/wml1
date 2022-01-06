import argparse
import wml_utils as wmlu
import os.path as osp
import os
import glob
import pickle

data_type = "normal"
def parse_args():
    parser = argparse.ArgumentParser(description='extract optical flows')
    parser.add_argument('--src_dir', default="/home/wj/ai/mldata/drive_and_act/phone/training/images/2",type=str, help='source video directory')
    parser.add_argument('--src_file', default="/home/wj/ai/mldata/drive_and_act/phone/training/images/2",type=str, help='source video directory')
    parser.add_argument('--out_dir', default="/home/wj/ai/mldata/drive_and_act/phone/training/images/2_np",type=str, help='output rawframe directory')
    parser.add_argument('--ext', default=".gif",type=str, help='file ext')
    parser.add_argument(
        "--reverse", default=False, action="store_true", help="resume training"
    )
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    files = wmlu.recurse_get_filepath_in_dir(args.src_file,suffix=args.ext)

    base_names = [wmlu.base_name(x) for x in files]

    subdirs = wmlu.get_subdir_in_dir(args.src_dir,append_self=False,absolute_path=False)
    dirs_need_to_move = []
    move_data = []
    dirs_dont_move = []
    for sd in subdirs:
        rdir = osp.join(args.src_dir,sd)
        ddir = args.out_dir
        tdir = wmlu.base_name(sd,process_suffix=False)
        move = False
        if args.reverse:
            if tdir not in base_names:
                move = True
        else:
            if tdir in base_names:
                move = True

        if move:
            dirs_need_to_move.append(sd)
            move_data.append((rdir,ddir))
        else:
            dirs_dont_move.append(sd)

    print(f"dirs don't need move:")
    wmlu.show_list(dirs_dont_move)
    print(f"Total don't move dir {len(dirs_dont_move)}")
    print(f"dirs need move:")
    wmlu.show_list(dirs_need_to_move)
    print(f"Total dirs need to move {len(dirs_need_to_move)}")

    ans = input("Move dirs?[y/n]")
    if ans == 'y':
        wmlu.create_empty_dir(args.out_dir,remove_if_exists=False)
        for rdir,ddir in move_data:
            cmd = f"mv  \"{rdir}\" \"{ddir}\""
            print(cmd)
            os.system(cmd)

