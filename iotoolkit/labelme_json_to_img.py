#!/opt/anaconda3/bin/python
import argparse
import json
import os
import sys
import os.path as osp
 
import PIL.Image
import yaml
 
from labelme import utils
 
parser = argparse.ArgumentParser()
parser.add_argument('-json_file',type=str,default="",help="json file path")
flags,unparsed = parser.parse_known_args(sys.argv[1:])
 
def main():
 
    json_file = flags.json_file
 
    out_dir = osp.basename(json_file).replace('.', '_')
    out_dir = osp.join(osp.dirname(json_file), out_dir)
    os.mkdir(out_dir)
 
    data = json.load(open(json_file))
 
    img = utils.img_b64_to_arr(data['imageData'])
    cls = utils.shapes_to_label(img.shape, data['shapes'],{"1":255,"2":128,"3":128})
 
    PIL.Image.fromarray(cls).save(osp.join(out_dir, 'img.png'))
    '''
    lbl, lbl_names = utils.shapes_to_label(img.shape, data['shapes'],{"1":1,"2":2,"3":3})
 
    lbl_viz = utils.draw_label(lbl, img, lbl_names)
 
    PIL.Image.fromarray(img).save(osp.join(out_dir, 'img.png'))
    PIL.Image.fromarray(lbl).save(osp.join(out_dir, 'label.png'))
    PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, 'label_viz.png'))
 
    info = dict(label_names=lbl_names)
 
    with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
        yaml.safe_dump(info, f, default_flow_style=False)
 
    print('wrote data to %s' % out_dir)
    '''
 
if __name__ == '__main__':
    main()
