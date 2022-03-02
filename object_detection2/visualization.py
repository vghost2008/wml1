#coding=utf-8
import cv2
import random
import numpy as np
import semantic.visualization_utils as smv
from PIL import Image
from iotoolkit.coco_toolkit import JOINTS_PAIR as COCO_JOINTS_PAIR
from .basic_datadef import colors_tableau as _colors_tableau
from .basic_datadef import DEFAULT_COLOR_MAP as _DEFAULT_COLOR_MAP

colors_tableau = _colors_tableau
DEFAULT_COLOR_MAP = _DEFAULT_COLOR_MAP

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def draw_rectangle(img, p1, p2, color=[255, 0, 0], thickness=2):
    cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)

def draw_bbox(img, bbox, shape=None, label=None, color=[255, 0, 0], thickness=2,is_relative_bbox=False,xy_order=True):
    if is_relative_bbox:
        p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
        p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
    else:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[2]), int(bbox[3]))
    if xy_order:
        p1 = p1[::-1]
        p2 = p2[::-1]
    cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
    p1 = (p1[0]+15, p1[1])
    if label is not None:
        cv2.putText(img, str(label), p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
    return img

def get_text_pos_fn(pmin,pmax,bbox,label):
    if bbox[0]<0.1:
        p1 = (pmax[0],pmin[1])
    else:
        p1 = pmin
    return (p1[0]-5,p1[1])

def random_color_fn(label):
    del label
    nr = len(colors_tableau)
    return colors_tableau[random.randint(0,nr-1)]

def default_text_fn(label,score):
    return str(label)

'''
color_fn: tuple(3) (*f)(label)
text_fn: str (*f)(label,score)
get_text_pos_fn: tuple(2) (*f)(lt_corner,br_corner,bboxes,label)
'''
def draw_bboxes(img, classes=None, scores=None, bboxes=None,
                        color_fn=random_color_fn,
                        text_fn=default_text_fn,
                        get_text_pos_fn=get_text_pos_fn,
                        thickness=4,show_text=True,font_scale=1.2,text_color=(0.,255.,0.),
                        is_relative_coordinate=True,
                        is_show_text=None,
                        fill_bboxes=False):
    bboxes = np.array(bboxes)
    if classes is None:
        classes = np.zeros([bboxes.shape[0]],dtype=np.int32)
    bboxes_thickness = thickness if not fill_bboxes else -1
    if is_relative_coordinate:
        shape = img.shape
    else:
        shape = [1.0,1.0]
    if len(img.shape)<2:
        print(f"Error img size {img.shape}.")
        return img
    img = np.array(img)
    if scores is None:
        scores = np.ones_like(classes,dtype=np.float32)
    if not isinstance(bboxes,np.ndarray):
        bboxes = np.array(bboxes)
    for i in range(bboxes.shape[0]):
        try:
            bbox = bboxes[i]
            if color_fn is not None:
                color = color_fn(classes[i])
            else:
                color = (int(random.random()*255), int(random.random()*255), int(random.random()*255))
            p10 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
            p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
            cv2.rectangle(img, p10[::-1], p2[::-1], color, bboxes_thickness)
            if show_text and text_fn is not None:
                f_show_text = True
                if is_show_text is not None:
                    f_show_text = is_show_text(p10,p2)

                if f_show_text:
                    s = text_fn(classes[i], scores[i])
                    p = get_text_pos_fn(p10,p2,bbox,classes[i])
                    cv2.putText(img, s, p[::-1], cv2.FONT_HERSHEY_DUPLEX,
                                fontScale=font_scale,
                                color=text_color,
                                thickness=1)
        except Exception as e:
            bbox = bboxes[i]
            p10 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
            p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
            if color_fn is not None:
                color = color_fn(classes[i])
            else:
                color = (random.random()*255, random.random()*255, random.random()*255)
            print("Error:",img.shape,shape,bboxes[i],classes[i],p10,p2,color,thickness,e)
            

    return img

def draw_legend(labels,text_fn,img_size,color_fn,thickness=4,font_scale=1.2,text_color=(0.,255.,0.),fill_bboxes=True):
    '''
    Generate a legend image
    Args:
        labels: list[int] labels
        text_fn: str fn(label) trans label to text
        img_size: (H,W) the legend image size, the legend is drawed in veritical direction
        color_fn: tuple(3) fn(label): trans label to RGB color
        thickness: text thickness
        font_scale: font size
        text_color: text color
    Returns:

    '''
    boxes_width = max(img_size[1]//3,20)
    boxes_height = img_size[0]/(2*len(labels))
    def lget_text_pos_fn(pmin, pmax, bbox, label):
        p1 = (pmax[0]+5, pmax[1]+5)
        return p1

    bboxes = []
    for i,l in enumerate(labels):
        xmin = 5
        xmax = xmin+boxes_width
        ymin = int((2*i+0.5)*boxes_height)
        ymax = ymin + boxes_height
        bboxes.append([ymin,xmin,ymax,xmax])
    img = np.ones([img_size[0],img_size[1],3],dtype=np.uint8)
    def _text_fn(x,_):
        return text_fn(x)
    return draw_bboxes(img,labels,bboxes=bboxes,color_fn=color_fn,text_fn=_text_fn,
                get_text_pos_fn=lget_text_pos_fn,
                thickness=thickness,
                show_text=True,
                font_scale=font_scale,
                text_color=text_color,
                is_relative_coordinate=False,
                fill_bboxes=fill_bboxes)



'''
mask only include the area within bbox
'''
def draw_bboxes_and_mask(img,classes,scores,bboxes,masks,color_fn=None,text_fn=None,thickness=4,show_text=False,fontScale=0.8):
    masks = masks.astype(np.uint8)
    for i,bbox in enumerate(bboxes):
        if color_fn is not None:
            color = list(color_fn(classes[i]))
        else:
            color = [random.random()*255, random.random()*255, random.random()*255]
        x = int(bbox[1]*img.shape[1])
        y = int(bbox[0]*img.shape[0])
        w = int((bbox[3]-bbox[1])*img.shape[1])
        h = int((bbox[2]-bbox[0])*img.shape[0])
        if w<=0 or h<=0:
            continue
        mask = masks[i]
        mask = cv2.resize(mask,(w,h))
        mask = np.expand_dims(mask,axis=-1)
        img[y:y+h,x:x+w,:] = (img[y:y+h,x:x+w,:]*(np.array([[[1]]],dtype=np.float32)-mask*0.4)).astype(np.uint8)+(mask*color*0.4).astype(np.uint8)

    img = draw_bboxes(img,classes,scores,bboxes,
                               color_fn=color_fn,
                               text_fn=text_fn,
                               thickness=thickness,
                               show_text=show_text,
                               fontScale=fontScale)
    return img

'''
mask include the area of whole image
'''
def draw_bboxes_and_maskv2(img,classes,scores,bboxes,masks,color_fn=None,text_fn=None,thickness=4,
                           show_text=False,
                           fontScale=0.8):
    if not isinstance(masks,np.ndarray):
        masks = np.array(masks)
    masks = masks.astype(np.uint8)
    for i,bbox in enumerate(bboxes):
        if color_fn is not None:
            color = list(color_fn(classes[i]))
        else:
            color = [random.random()*255, random.random()*255, random.random()*255]
        x = int(bbox[1]*img.shape[1])
        y = int(bbox[0]*img.shape[0])
        w = int((bbox[3]-bbox[1])*img.shape[1])
        h = int((bbox[2]-bbox[0])*img.shape[0])
        if w<=0 or h<=0:
            continue
        mask = masks[i]
        img = smv.draw_mask_on_image_array(img,mask,color=color,alpha=0.4)

    img = draw_bboxes(img,classes,scores,bboxes,
                               color_fn=color_fn,
                               text_fn=text_fn,
                               thickness=thickness,
                               show_text=show_text,
                               fontScale=fontScale)
    return img

def convert_semantic_to_rgb(semantic,color_map,return_nparray=False):
    '''
    convert semantic label map to rgb PIL image or a np.ndarray
    Args:
        semantic: [H,W] label value
        color_map: list[int], [r0,g0,b0,r1,g1,b1,....]
    Returns:
        image: [H,W,3]
    '''
    new_mask = Image.fromarray(semantic.astype(np.uint8)).convert('P')
    new_mask.putpalette(color_map)
    if return_nparray:
        return np.array(new_mask.convert('RGB'))
    return new_mask

def draw_semantic_on_image(image,semantic,color_map,alpha=0.4,ignored_label=0):
    '''
    draw semantic on image
    Args:
        image:
        semantic: [H,W] label value
        color_map: list[int], [r0,g0,b0,r1,g1,b1,....]
        alpha:
        ignored_label:
    Returns:
        return image*(1-alpha)+semantic+alpha
    '''
    mask = convert_semantic_to_rgb(semantic,color_map=color_map,return_nparray=True)
    new_img = image.astype(np.float32)*(1-alpha)+mask.astype(np.float32)*alpha
    new_img = np.clip(new_img,0,255).astype(np.uint8)
    pred = np.expand_dims(semantic!=ignored_label,axis=-1)
    new_img = np.where(pred,new_img,image)
    return new_img

def add_jointsv1(image, joints, color, r=5,no_line=False,joints_pair=None,left_node=None):

    def link(a, b, color):
        jointa = joints[a]
        jointb = joints[b]
        cv2.line(
                image,
                (int(jointa[0]), int(jointa[1])),
                (int(jointb[0]), int(jointb[1])),
                color, 2 )

    # add link
    if not no_line and joints_pair is not None:
        for pair in joints_pair:
            link(pair[0], pair[1], color)

    # add joints
    node_color = None
    for i, joint in enumerate(joints):
        if left_node is None:
            node_color = colors_tableau[i]
        elif i in left_node:
            node_color = (0,255,0)
        else:
            node_color = (0,0,255)
        cv2.circle(image, (int(joint[0]), int(joint[1])), r, node_color, -1)

    return image

def add_jointsv2(image, joints, color, r=5,no_line=False,joints_pair=None,left_node=None):

    def link(a, b, color):
        jointa = joints[a]
        jointb = joints[b]
        if jointa[2] > 0.01 and jointb[2] > 0.01:
            cv2.line(
                image,
                (int(jointa[0]), int(jointa[1])),
                (int(jointb[0]), int(jointb[1])),
                color, 2 )

    # add link
    if not no_line and joints_pair is not None:
        for pair in joints_pair:
            link(pair[0], pair[1], color)

    # add joints
    for i, joint in enumerate(joints):
        if joint[2] > 0.05 and joint[0] > 1 and joint[1] > 1:
            if left_node is None:
                node_color = colors_tableau[i]
            elif i in left_node:
                node_color = (0,255,0)
            else:
                node_color = (0,0,255)
            cv2.circle(image, (int(joint[0]), int(joint[1])), r, node_color, -1)

    return image

def draw_keypoints(image, joints, color=[0,255,0],no_line=False,joints_pair=COCO_JOINTS_PAIR,left_node=list(range(1,17,2))):
    '''

    Args:
        image: [H,W,3]
        joints: [N,kps_nr,2] or [kps_nr,2]
        color:
        no_line:
        joints_pair: [[first idx,second idx],...]
    Returns:

    '''
    image = np.ascontiguousarray(image)
    joints = np.array(joints)
    if color is None:
        use_random_color=True
    else:
        use_random_color = False
    if len(joints.shape)==2:
        joints = [joints]
    else:
        assert len(joints.shape)==3,"keypoints need to be 3-dimensional."

    for person in joints:
        if use_random_color:
            color = np.random.randint(0, 255, size=3)
            color = [int(i) for i in color]

        if person.shape[-1] == 3:
            add_jointsv2(image, person, color=color,no_line=no_line,joints_pair=joints_pair,left_node=left_node)
        else:
            add_jointsv1(image, person, color=color,no_line=no_line,joints_pair=joints_pair,left_node=left_node)

    return image


def draw_keypoints_diff(image, joints0, joints1,color=[0,255,0]):
    image = np.ascontiguousarray(image)
    joints0 = np.array(joints0)
    joints1 = np.array(joints1)
    if color is None:
        use_random_color=True
    else:
        use_random_color = False
    if len(joints0.shape)==2:
        points_nr = joints0.shape[0]
        joints0 = [joints0]
        joints1 = [joints1]
    else:
        points_nr = joints0.shape[1]
        assert len(joints0.shape)==3,"keypoints need to be 3-dimensional."

    for person0,person1 in zip(joints0,joints1):
        if use_random_color:
            color = np.random.randint(0, 255, size=3)
            color = [int(i) for i in color]
        for i in range(points_nr):
            jointa = person0[i]
            jointb = person1[i]
            if person0.shape[-1] == 3:
                if person0[i][-1]>0.015 and person1[i][-1]>0.015:
                    cv2.line(
                    image,
                    (int(jointa[0]), int(jointa[1])),
                    (int(jointb[0]), int(jointb[1])),
                     color, 2 )
            else:
                cv2.line(
                    image,
                    (int(jointa[0]), int(jointa[1])),
                    (int(jointb[0]), int(jointb[1])),
                     color, 2 )

    return image
