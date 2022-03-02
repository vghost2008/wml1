from PIL import Image
import os
from semantic.visualization_utils import RGB_STANDARD_COLORS
import wml_utils as wmlu
import numpy as np

class SemanticData(object):
    def __init__(self,img_suffix=".jpg",label_suffix=".png",img_sub_dir=None,label_sub_dir=None,
                 color_map=None):
        if color_map is None:
            color_map = RGB_STANDARD_COLORS
        self.color_map = []
        for x in color_map:
            self.color_map += list(x)
        self.img_suffix = img_suffix
        self.label_suffix = label_suffix
        self.img_sub_dir = img_sub_dir
        self.label_sub_dir = label_sub_dir
        self.files = []

    def read_data(self,dir_path):
        label_dir = dir_path if self.label_sub_dir is None else os.path.join(dir_path,self.label_sub_dir)
        img_dir = dir_path if self.img_sub_dir is None else os.path.join(dir_path,self.img_sub_dir)
        image_files = wmlu.recurse_get_filepath_in_dir(img_dir,suffix=self.img_suffix)
        self.files = []
        for ifn in image_files:
            base_name = wmlu.base_name(ifn)
            label_path = os.path.join(label_dir,base_name+self.label_suffix)
            if not os.path.exists(label_path):
                print(f"ERROR: Find label file {label_path} for image {ifn} faild.")
                continue
            else:
                self.files.append([ifn,label_path])

    def __len__(self):
        return len(self.files)

    def get_items(self):
        for i in range(len(self.files)):
            yield self.__getitem__(i)

    def __getitem__(self, item):
        ifn,lfn = self.files[item]
        img = Image.open(ifn).convert('RGB')
        '''
        mask:[H,W], value is label
        '''
        mask = Image.open(lfn)
        return [ifn,lfn],np.array(img),np.array(mask)