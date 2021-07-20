import shutil
import wml_utils as wmlu
import os
from iotoolkit.pascal_voc_toolkit import PascalVOCData

def copy_to_dir(files,save_dir):
    wmlu.create_empty_dir(save_dir,remove_if_exists=False)
    for f in files:
        wmlu.safe_copy(f,save_dir)

def split_file_by_type(datas,save_dir,get_buddy_files=None):
    for data in datas:
        img_file, shape, labels, labels_names, bboxes, _, _, difficult, _ = data
        names = set(labels_names)
        files = [img_file]
        if get_buddy_files is not None:
            files = files + get_buddy_files(img_file)

        for name in names:
            t_save_dir = os.path.join(save_dir,name)
            copy_to_dir(files,t_save_dir)


if __name__ == "__main__":
    dataset = PascalVOCData()
    dataset.read_data("/3_data/wj/mldata/cell/deep_data/")
    def get_buddy_file(img_file):
        return [wmlu.change_suffix(img_file,"xml")]
    split_file_by_type(dataset.get_items(),"/3_data/wj/mldata/cell/deep_data_split2",get_buddy_files=get_buddy_file)
