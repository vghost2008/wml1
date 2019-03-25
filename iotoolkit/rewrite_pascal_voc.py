#coding=utf-8
import os
import object_detection.utils as odu
import wml_utils as wmlu
import shutil

def rewrite(input_dir,output_dir,image_sub_dir="JPEGImages",xml_sub_dir="Annotations",img_suffix=".jpg",begin_index=1):
    if output_dir is None:
        output_dir = os.path.join(input_dir,"output")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    files = odu.getVOCFiles(input_dir,image_sub_dir=image_sub_dir,xml_sub_dir=xml_sub_dir,img_suffix=img_suffix)
    print(f"Nr {len(files)}")
    index = begin_index
    for img_file,xml_file in files:
        suffix = wmlu.suffix(img_file)
        img_out_path = os.path.join(output_dir,f"IMG_{index:0>4}."+suffix)
        xml_out_path = os.path.join(output_dir,f"IMG_{index:0>4}."+"xml")
        if os.path.exists(img_out_path):
            print(f"Error file {img_out_path} exists.")
            break
        print(img_file,"->",img_out_path)
        print(xml_file,"->",xml_out_path)
        index += 1

        shutil.copy(img_file,img_out_path)
        shape, bboxes, labels_text, difficult, truncated = odu.read_voc_xml(xml_file)
        odu.write_voc_xml(xml_out_path, img_out_path, shape, bboxes, labels_text, difficult,truncated)
