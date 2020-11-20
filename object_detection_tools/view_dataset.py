import object_detection_tools.cmp_datasets as cmpd
import os
from iotoolkit.pascal_voc_toolkit import PascalVOCData
import wml_utils as wmlu
import object_detection2.visualization as odv
import img_utils as wmli
import argparse


def default_argument_parser():
    """
    Create a parser with some common arguments used by detectron2 users.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="ObjectDetection2 Training")
    parser.add_argument("--test_dir", default="/2_data/wj/mldata/coco/val2017",type=str,help="path to test data dir")
    parser.add_argument("--save_dir", default="/2_data/wj/mldata/coco/coco_results",type=str,help="path to save data dir")
    return parser

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    save_path = args.save_dir
    dataset = PascalVOCData()
    # data_path0 = get_data_dir2("annotationed_data/verify_p03_1020_1_proc_m")
    data_path = args.test_dir

    if not dataset.read_data(data_path):
        exit(0)
    wmlu.create_empty_dir(save_path, remove_if_exists=True, yes_to_all=True)
    for data in dataset.get_items():
        full_path, shape, category_ids, category_names, boxes, binary_masks, area, is_crowd, num_annotations_skipped = data
        print(f"Process {full_path}")
        img = wmli.imread(full_path)
        base_name = wmlu.base_name(full_path) + "_a.jpg"
        save_file_path = os.path.join(save_path, base_name)
        img = odv.draw_bboxes(img, classes=category_names, bboxes=boxes, show_text=True, text_color=(255., 0., 0.))
        wmli.imsave(save_file_path, img)
