import numpy as np
import img_utils as wmli
import cv2

def array_to_snpe_raw(data, raw_filepath,input_size=None):
    img_array = np.array(data)  # read it
    if input_size is not None:
        print(f"Resize data to {input_size}")
        img_array = cv2.resize(img_array, (input_size[1],input_size[0]))
    # save
    with open(raw_filepath, 'wb') as fid:
        img_array.tofile(fid)

def file_to_snpe_raw(img_path, raw_filepath,input_size=None):
    img = wmli.imread(img_path)
    img = img.astype(np.float32)
    array_to_snpe_raw(img,raw_filepath,input_size)
