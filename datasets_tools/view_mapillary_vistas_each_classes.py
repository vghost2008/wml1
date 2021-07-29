from iotoolkit.mapillary_vistas_toolkit import *
import img_utils as wmli
import object_detection_tools.visualization as odv
import matplotlib.pyplot as plt
import wml_utils as wmlu

lid = 0


def view_data(name,save_dir,nr=20):
    print(f"View {name}")
    raw_name = name
    names = name.split("--")
    if names[0]=="void" or "ambiguous" in raw_name:
        return
    if "road" not in raw_name:
        return
    for x in names:
        save_dir = os.path.join(save_dir,x)

    wmlu.create_empty_dir(save_dir,remove_if_exists=False)

    allowed_names = [raw_name]
    NAME2ID = {}
    ID2NAME = {}


    def name_to_id(x):
        global lid
        if x in NAME2ID:
            return NAME2ID[x]
        else:
            NAME2ID[x] = lid
            ID2NAME[lid] = x
            lid += 1
            return NAME2ID[x]


    data = MapillaryVistasData(label_text2id=name_to_id, shuffle=False, ignored_labels=None,
                               label_map=None,
                               allowed_labels_fn=allowed_names)
    data.read_data(wmlu.home_dir("ai/mldata/mapillary_vistas"))

    i = 0
    for x in data.get_items():
        full_path, img_info, category_ids, category_names, boxes, binary_mask, area, is_crowd, num_annotations_skipped = x
        img = wmli.imread(full_path)

        def text_fn(classes, scores):
            return f"{ID2NAME[classes]}"

        if len(category_ids) == 0:
            continue

        wmlu.show_dict(NAME2ID)
        odv.draw_bboxes_and_maskv2(
            img=img, classes=category_ids, scores=None, bboxes=boxes, masks=binary_mask, color_fn=None,
            text_fn=text_fn, thickness=4,
            show_text=True,
            fontScale=0.8)
        base_name = os.path.basename(full_path)
        save_path = os.path.join(save_dir,base_name)
        wmli.imwrite(save_path,img)
        i += 1
        if i>= nr:
            break


if __name__ == "__main__":
    with open("/home/wj/ai/mldata/mapillary_vistas/config_v2.0.json") as f:
        data = json.load(f)
    save_dir = wmlu.home_dir("ai/tmp/mv2")
    for d in data['labels']:
        view_data(d['name'], save_dir)