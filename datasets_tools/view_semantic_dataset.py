import json
from iotoolkit.semantic_data import SemanticData
import wml_utils as wmlu
from object_detection2.visualization import *
import img_utils as wmli
import os

ID_TO_NAME = {
0:"construction--flat--road",
1:"construction--flat--sidewalk",
2:"object--street-light",
3:"construction--structure--bridge",
4:"construction--structure--building",
5:"human",
6:"object--support--pole",
7:"marking--continuous--dashed",
8:"marking--continuous--solid",
9:"marking--discrete--crosswalk-zebra",
10:"nature--sand",
11:"nature--sky",
12:"nature--snow",
13:"nature--terrain",
14:"nature--vegetation",
15:"nature--water",
16:"object--vehicle--bicycle",
17:"object--vehicle--boat",
18:"object--vehicle--bus",
19:"object--vehicle--car",
20:"object--vehicle--caravan",
21:"object--vehicle--motorcycle",
22:"object--vehicle--on-rails",
23:"object--vehicle--truck",
24:"construction--flat--pedestrian-area",
25:"construction--structure--tunnel",
26:"nature--wasteland",
}
ID_TO_READABLE_NAME = {
    0:"road",
    1:"sidewalk",
    2:"light",
    3:"bridge",
    4:"building",
    5:"human",
    6:"pole",
    7:"dashed",
    8:"solid",
    9:"crosswalk-zebra",
    10:"sand",
    11:"sky",
    12:"snow",
    13:"terrain",
    14:"vegetation",
    15:"water",
    16:"bicycle",
    17:"boat",
    18:"bus",
    19:"car",
    20:"caravan",
    21:"motorcycle",
    22:"on-rails",
    23:"truck",
    24:"pedestrian-area",
    25:"tunnel",
    26:"wasteland",
}

NAME_TO_ID = {}
for k,v in ID_TO_NAME.items():
    NAME_TO_ID[v] = k

NAME_TO_MAPILLARY_NAME= {
"construction--flat--road":"construction--flat--road",
"construction--flat--sidewalk":"construction--flat--sidewalk",
"object--street-light":"object--street-light",
"construction--structure--bridge":"construction--structure--bridge",
"construction--structure--building":"construction--structure--building",
"human":"human--person--individual",
"object--support--pole":"object--support--pole",
"marking--continuous--dashed":"marking--continuous--dashed",
"marking--continuous--solid":"marking--continuous--solid",
"marking--discrete--crosswalk-zebra":"marking--discrete--crosswalk-zebra",
"nature--sand":"nature--sand",
"nature--sky":"nature--sky",
"nature--snow":"nature--snow",
"nature--terrain":"nature--terrain",
"nature--vegetation":"nature--vegetation",
"nature--water":"nature--water",
"object--vehicle--bicycle":"object--vehicle--bicycle",
"object--vehicle--boat":"object--vehicle--boat",
"object--vehicle--bus":"object--vehicle--bus",
"object--vehicle--car":"object--vehicle--car",
"object--vehicle--caravan":"object--vehicle--caravan",
"object--vehicle--motorcycle":"object--vehicle--motorcycle",
"object--vehicle--on-rails":"object--vehicle--on-rails",
"object--vehicle--truck":"object--vehicle--truck",
"construction--flat--pedestrian-area":"construction--flat--pedestrian-area",
"construction--structure--tunnel":"construction--structure--tunnel",
"nature--wasteland":"void--ground",
}

def fill_colormap_and_names(config_fn):
    """
    Mapillary code for color map and class names

    Outputs
    -------
    trainid_to_name
    color_mapping
    """
    with open(config_fn) as config_file:
        config = json.load(config_file)
    config_labels = config['labels']

    mapillary_name2color = {}

    for i in range(0, len(config_labels)):
        label = config_labels[i]['name']
        color = config_labels[i]['color']
        mapillary_name2color[label] = color

    # calculate label color mapping
    colormap = []
    for i in range(0, len(ID_TO_NAME)):
        name = ID_TO_NAME[i]
        m_name = NAME_TO_MAPILLARY_NAME[name]
        color = mapillary_name2color[m_name]
        colormap = colormap + color
    colormap += [0, 0, 0] * (256 - len(ID_TO_NAME))
    return colormap

if __name__ == "__main__":
    '''dataset = SemanticData(img_suffix=".jpg",label_suffix=".png",img_sub_dir="boe_labels",label_sub_dir="boe_labels")
    dataset.read_data("/home/wj/ai/mldata/boesemantic")
    save_dir = wmlu.home_dir("ai/tmp/boe_images2")
    wmlu.create_empty_dir(save_dir,remove_if_exists=False)
    color_map = fill_colormap_and_names("/home/wj/ai/mldata/mapillary_vistas/config_v2.0.json")
    def text_fn(l):
        if l in ID_TO_READABLE_NAME:
            return ID_TO_READABLE_NAME[l]
        else:
            return "NA"
    def color_fn(l):
        return color_map[l*3:l*3+3]

    legend_img = draw_legend(list(ID_TO_NAME.keys()),text_fn,img_size=(2448,300),color_fn=color_fn)
    for ifn,img,mask in dataset.get_items():
        base_name = wmlu.base_name(ifn)
        wmlu.safe_copy(ifn,save_dir)
        rgb_mask = convert_semantic_to_rgb(mask,color_map,True)
        if rgb_mask.shape[0] != legend_img.shape[0]:
            legend_img = wmli.resize_img(legend_img,(legend_img.shape[1],rgb_mask.shape[0]))
        rgb_mask = np.concatenate([rgb_mask,legend_img],axis=1)
        mask_path = os.path.join(save_dir,base_name+".png")
        wmli.imwrite(mask_path,rgb_mask)

        mask_image = draw_semantic_on_image(img,mask,color_map,ignored_label=255)
        mask_image = np.concatenate([mask_image,legend_img],axis=1)
        mask_image_path = os.path.join(save_dir,base_name+"1.png")
        wmli.imwrite(mask_image_path,mask_image)'''

    dataset = SemanticData(img_suffix=".jpg",label_suffix=".png",img_sub_dir=None,label_sub_dir=None)
    #dataset.read_data("/home/wj/ai/mldata1/safety_belt/boe_labels_train")
    #dataset.read_data("/home/wj/ai/mldata1/safety_belt/trans_train_1/")
    dataset.read_data("/home/wj/ai/mldata1/safety_belt/training/safetybelt_seg_imgs")
    save_dir = wmlu.get_unused_path("/home/wj/ai/mldata1/safety_belt/tmp/view")
    wmlu.create_empty_dir(save_dir,remove_if_exists=False)
    color_map = fill_colormap_and_names("/home/wj/ai/mldata/mapillary_vistas/config_v2.0.json")
    ID_TO_READABLE_NAME = {0:"person",1:"tie",2:'seat_belt'}
    def text_fn(l):
        if l in ID_TO_READABLE_NAME:
            return ID_TO_READABLE_NAME[l]
        else:
            return "NA"
    def color_fn(l):
        return color_map[l*3:l*3+3]

    legend_img = draw_legend(list(ID_TO_NAME.keys()),text_fn,img_size=(2448,300),color_fn=color_fn)
    for ifn,img,mask in dataset.get_items():
        base_name = wmlu.base_name(ifn)
        wmlu.safe_copy(ifn,save_dir)
        rgb_mask = convert_semantic_to_rgb(mask,color_map,True)
        if rgb_mask.shape[0] != legend_img.shape[0]:
            legend_img = wmli.resize_img(legend_img,(legend_img.shape[1],rgb_mask.shape[0]))
        rgb_mask = np.concatenate([rgb_mask,legend_img],axis=1)
        mask_path = os.path.join(save_dir,base_name+".png")
        wmli.imwrite(mask_path,rgb_mask)

        mask_image = draw_semantic_on_image(img,mask,color_map,ignored_label=255)
        mask_image = np.concatenate([mask_image,legend_img],axis=1)
        mask_image_path = os.path.join(save_dir,base_name+"1.png")
        wmli.imwrite(mask_image_path,mask_image)

