import tensorflow as tf
import object_detection_tools.predictmodel as predm
import wml_utils as wmlu
import glob
import img_utils as wmli
import numpy as np
import os
import shutil
import iotoolkit.pascal_voc_toolkit as pvt
import iotoolkit.labelme_toolkit as lmt
import iotoolkit.coco_toolkit as cocot
from object_detection2.engine.defaults import default_argument_parser
from object_detection2.standard_names import *


def main(_):
    args = default_argument_parser().parse_args()
    test_dir = args.test_data_dir
    save_dir = args.save_data_dir
    wmlu.create_empty_dir(save_dir,remove_if_exists=True)
    files =  glob.glob(os.path.join(test_dir,"*.jpg"))
    m = predm.PredictModel()
    m.restoreVariables()
    m.remove_batch()

    def id_to_text(id):
        return m.trainer.category_index[id]

    for file in files:
        img = wmli.imread(file)
        img = np.expand_dims(img,axis=0)
        m.predictImages(img)
        save_path = os.path.join(save_dir,os.path.basename(file))
        xml_path = wmlu.change_suffix(save_path,"xml")
        shutil.copy(file,save_path)
        labels = [id_to_text(id) for id in m.res_data[RD_LABELS]]
        pvt.writeVOCXml(xml_path,m.res_data[RD_BOXES],labels)
        if RD_FULL_SIZE_MASKS in m.res_data:
            annotations = lmt.trans_odresult_to_annotations_list(m.res_data)
            json_path = wmlu.change_suffix(save_path,"json")
            lmt.save_labelme_datav1(json_path,file,img,annotations,label_to_text=id_to_text)
        img_save_path = wmlu.change_suffix(xml_path,"jpg")
        wmli.imwrite(img_save_path,m.res_data[RD_RESULT_IMAGE])

if __name__ == "__main__":
    tf.app.run()

