#coding=utf-8
ID_TO_TEXT= {
  1:'aeroplane',
  2:'bicycle',
  3:'bird',
  4:'boat',
  5:'bottle',
  6:'bus',
  7:'car',
  8:'cat',
  9:'chair',
  10:'cow',
  11:'diningtable',
  12:'dog',
  13:'horse',
  14:'motorbike',
  15:'person',
  16:'pottedplant',
  17:'sheep',
  18:'sofa',
  19:'train',
  20:'tvmonitor'
}
TEXT_TO_ID = {
    'aeroplane':  1,
    'bicycle':  2,
    'bird':  3,
    'boat':  4,
    'bottle':  5,
    'bus':  6,
    'car':  7,
    'cat':  8,
    'chair':  9,
    'cow':  10,
    'diningtable':  11,
    'dog':  12,
    'horse':  13,
    'motorbike':  14,
    'person':  15,
    'pottedplant':  16,
    'sheep':  17,
    'sofa':  18,
    'train':  19,
    'tvmonitor':20
}
def get_id_to_textv2():
    res = {}
    for k,v in ID_TO_TEXT.items():
        res[k] = {"name":v}
    return res

