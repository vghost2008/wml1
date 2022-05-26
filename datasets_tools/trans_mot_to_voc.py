from iotoolkit.sports_mot_datasets import SportsMOTDatasets
from iotoolkit.pascal_voc_toolkit import *

def trans_data(data_dir):
    dataset = SportsMOTDatasets(data_dir, absolute_coord=True)
    for img_path, shape, labels, _, bboxes,*_ in dataset.get_data_items():
        labels = [0]*len(labels)
        writeVOCXml(img_path,bboxes,labels,img_shape=shape,is_relative_coordinate=False)


if __name__ == "__main__":
    trans_data("/home/wj/ai/mldata1/SportsMOT-2022-4-24/data/sportsmot_publish/dataset/train_copy")
