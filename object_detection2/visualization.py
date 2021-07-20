#coding=utf-8
import cv2
import random
import numpy as np
import semantic.visualization_utils as smv

colors_tableau = [(255, 255, 255), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                  (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                  (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                  (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                  (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def draw_rectangle(img, p1, p2, color=[255, 0, 0], thickness=2):
    cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)

def draw_bbox(img, bbox, shape, label, color=[255, 0, 0], thickness=2):
    p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
    p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
    cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
    p1 = (p1[0]+15, p1[1])
    cv2.putText(img, str(label), p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)

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
def draw_bboxes(img, classes, scores=None, bboxes=None,
                        color_fn=random_color_fn,
                        text_fn=default_text_fn,
                        get_text_pos_fn=get_text_pos_fn,
                        thickness=4,show_text=True,font_scale=1.2,text_color=(0.,255.,0.),
                        is_relative_coordinate=True,
                        is_show_text=None):
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
                color = (random.random()*255, random.random()*255, random.random()*255)
            p10 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
            p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
            cv2.rectangle(img, p10[::-1], p2[::-1], color, thickness)
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
        except:
            bbox = bboxes[i]
            p10 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
            p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
            if color_fn is not None:
                color = color_fn(classes[i])
            else:
                color = (random.random()*255, random.random()*255, random.random()*255)
            print("Error:",img.shape,shape,bboxes[i],classes[i],p10,p2,color,thickness)
            

    return img

def draw_legend(labels,text_fn,img_size,color_fn,thickness=4,font_scale=1.2,text_color=(0.,255.,0.)):
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
    return draw_bboxes(img,labels,bboxes=bboxes,color_fn=color_fn,text_fn=text_fn,
                get_text_pos_fn=lget_text_pos_fn,
                thickness=thickness,
                show_text=True,
                font_scale=font_scale,
                text_color=text_color,
                is_relative_coordinate=False)



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
