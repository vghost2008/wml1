import os
from object_detection2.bboxes import change_bboxes_nr
from iotoolkit.pascal_voc_toolkit import PascalVOCData
import sys


def counting_changed_bboxes_nr(lh_ds, rh_ds):
    '''
    :param lh_ds:
    :param rh_ds: as gt datasets
    :param num_classes:
    :param mask_on:
    :return:
    '''
    rh_ds_dict = {}

    for data in rh_ds:
        full_path, shape, category_ids, category_names, boxes, binary_masks, area, is_crowd, num_annotations_skipped = data
        rh_ds_dict[os.path.basename(full_path)] = data

    total_diff_nr = 0
    for i, data in enumerate(lh_ds):
        full_path, shape, category_ids, category_names, boxes, binary_masks, area, is_crowd, num_annotations_skipped = data

        base_name = os.path.basename(full_path)
        if base_name not in rh_ds_dict:
            print(f"Error find {base_name} in rh_ds faild.")
            continue
        rh_data = rh_ds_dict[base_name]
        diff_nr = change_bboxes_nr(boxes,category_names,rh_data[4],rh_data[3],threshold=0.8)
        total_diff_nr += diff_nr
        print(f"{base_name}: {diff_nr}, total diff nr {total_diff_nr}")

if __name__ == "__main__":
    data0 = PascalVOCData(label_text2id=None)
    data1 = PascalVOCData(label_text2id=None)

    if len(sys.argv) >= 3:
        data0.read_data(sys.argv[1])
        data1.read_data(sys.argv[2])
    else:
        data0.read_data("/home/wj/ai/mldata3/cell/data2annotation_width_fake_instance/check_2")
        data1.read_data("/2_data/wj/ds07_anno")
        #data0.read_data("/home/vghost/0day/a")
        #data1.read_data("/home/vghost/0day/b")

    counting_changed_bboxes_nr(data0.get_items(),data1.get_items())

