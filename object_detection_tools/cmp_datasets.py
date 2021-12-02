from object_detection2.metrics.toolkit import *
import os

def cmp_datasets(lh_ds,rh_ds,num_classes=90,mask_on=False,model=COCOEvaluation,classes_begin_value=1,**kwargs):
    '''
    :param lh_ds:
    :param rh_ds: as gt datasets
    :param num_classes:
    :param mask_on:
    :return:
    '''
    rh_ds_dict = {}
    rh_total_box_nr = 0
    lh_total_box_nr = 0
    
    for data in rh_ds:
        full_path, shape, category_ids, category_names, boxes, binary_masks, area, is_crowd, num_annotations_skipped = data
        rh_ds_dict[os.path.basename(full_path)] = data
        rh_total_box_nr += len(category_ids)
    
    eval = model(num_classes=num_classes,mask_on=mask_on,**kwargs)
    #eval2 = ClassesWiseModelPerformace(num_classes=num_classes,model_type=model)
    eval2 = ClassesWiseModelPerformace(num_classes=num_classes,model_type=COCOEvaluation,
                                       classes_begin_value=classes_begin_value)
    for i,data in enumerate(lh_ds):
        full_path, shape, category_ids, category_names, boxes, binary_masks, area, is_crowd, num_annotations_skipped = data
        lh_total_box_nr += len(category_ids)

        base_name = os.path.basename(full_path)
        if base_name not in rh_ds_dict:
            print(f"Error find {base_name} in rh_ds faild.")
            continue
        rh_data = rh_ds_dict[base_name]
            
        kwargs = {}
        kwargs['gtboxes'] = rh_data[4]
        kwargs['gtlabels'] = rh_data[2]
        kwargs['boxes'] = boxes
        kwargs['labels'] = category_ids
        kwargs['probability'] = np.ones_like(category_ids,np.float32)
        kwargs['img_size'] = shape
        eval(**kwargs)
        eval2(**kwargs)

        if i % 100 == 0:
            eval.show()
    
    eval.show()
    eval2.show()
    print(f"bboxes nr {lh_total_box_nr} vs {rh_total_box_nr}")

if __name__ == "__main__":
    from iotoolkit.pascal_voc_toolkit import PascalVOCData
    DC_CLASSES_TO_ID = {"car": 0, "bus": 1, "truck": 2, "van": 2, "dangerous_sign": 3, "tank_truck": 4}
    def text2id(x):
        return DC_CLASSES_TO_ID[x]
    data_path0 = "/home/wj/ai/smldata/boedcvehicle/data2label/data"
    data_path1 = "/home/wj/桌面/data_dc_20211117"
    data0 = PascalVOCData(label_text2id=text2id)
    data1 = PascalVOCData(label_text2id=text2id)
    data0.read_data(data_path0, silent=True)
    data1.read_data(data_path1, silent=True)
    cmp_datasets(data0.get_items(),data1.get_items(),num_classes=5,mask_on=False,model=COCOEvaluation,classes_begin_value=0)
