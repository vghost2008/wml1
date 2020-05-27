#coding=utf-8
from object_detection_tools.statistics_tools import *
import sys
import pickle

data_path = sys.argv[-1]
data_path = wmlu.home_dir("ai/mldata2/0day/test.dat")
print(f"Data path {data_path}")

if __name__ == "__main__":
    with open(data_path,"rb") as file:
        statics = pickle.load(file)
    nr = 100
    statistics_boxes(statics[0], nr=nr)
    statistics_boxes_by_different_area(statics[0], nr=nr, bin_size=5)
    statistics_boxes_by_different_ratio(statics[0], nr=nr, bin_size=5)
    # show_boxes_statistics(statics)
    show_classwise_boxes_statistics(statics[1], nr=nr)