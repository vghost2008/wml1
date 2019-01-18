#coding=utf-8
import cv2
import matplotlib
import platform
#if not platform.platform().startswith("Darwin"):
    #matplotlib.use('agg')
import matplotlib.image as mpimg
import matplotlib.cm as mpcm
import matplotlib.pyplot as plt
import random
import numpy as np
import wml_utils as wmlu


def colors_subselect(colors, num_classes=21):
    dt = len(colors) // num_classes
    sub_colors = []
    for i in range(num_classes):
        color = colors[i*dt]
        if isinstance(color[0], float):
            sub_colors.append([int(c * 255) for c in color])
        else:
            sub_colors.append([c for c in color])
    return sub_colors

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


def bboxes_draw_on_img(img, classes, scores, bboxes,
                       colors={0:(0.,0.,0.),1:(0.,0.,255.),2:(255.,0.,0.),3:(255.,255.,0.)},
                       thickness=4,show_text=False,fontScale=1.2):
    shape = img.shape
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        if classes[i] in colors:
            color = colors[classes[i]]
        else:
            colors[classes[i]] = (random.random()*255, random.random()*255, random.random()*255)
            color = colors[classes[i]]
        p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
        p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
        cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
        if show_text:
            s = '%d %.1f' % (classes[i], scores[i])
            p = (p1[0]-5, p1[1])
            cv2.putText(img, s, p[::-1], cv2.FONT_HERSHEY_DUPLEX, fontScale=fontScale, color=(255.,255.,255.), thickness=1)


'''
sigsize:(w,h)图像大小
'''
def plt_bboxes(img, classes, scores, bboxes, figsize=(10,10), linewidth=1.5,cmap=None,show_text=True,title=None,save_path=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(img,cmap=cmap)
    height = img.shape[0]
    width = img.shape[1]
    colors = {1:(0.,0.,1.),2:(1.,0.,0.)}
    if isinstance(classes,list):
        classes = np.array(classes)
    if isinstance(bboxes,list):
        bboxes = np.array(bboxes)
    for i in range(classes.shape[0]):
        cls_id = int(classes[i])
        if cls_id >= 0:
            score = scores[i]
            if cls_id not in colors:
                colors[cls_id] = (random.random(), random.random(), random.random())
            ymin = int(bboxes[i, 0] * height)
            xmin = int(bboxes[i, 1] * width)
            ymax = int(bboxes[i, 2] * height)
            xmax = int(bboxes[i, 3] * width)
            rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                 ymax - ymin, fill=False,
                                 edgecolor=colors[cls_id],
                                 linewidth=linewidth)
            plt.gca().add_patch(rect)
            class_name = str(cls_id)
            if show_text:
                plt.gca().text(xmin, ymin - 2,
                               '{:s} | {:.3f}'.format(class_name, score),
                               bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                               fontsize=12, color='white')
    if title is not None:
        plt.gca().text(width/2, 16 ,
                       title,
                       bbox=None,
                       fontsize=12, color='red')
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

def plt_bboxesv2(img, classes, scores, bboxes, linewidth=1.5,cmap=None,show_text=True,title=None,hold=False):
    plt.imshow(img,cmap=cmap)
    height = img.shape[0]
    width = img.shape[1]
    colors = {1:(0.,0.,1.),2:(1.,0.,0.)}
    if isinstance(classes,list):
        classes = np.array(classes)
    if isinstance(bboxes,list):
        bboxes = np.array(bboxes)
    for i in range(classes.shape[0]):
        cls_id = int(classes[i])
        if cls_id >= 0:
            score = scores[i]
            if cls_id not in colors:
                colors[cls_id] = (random.random(), random.random(), random.random())
            ymin = int(bboxes[i, 0] * height)
            xmin = int(bboxes[i, 1] * width)
            ymax = int(bboxes[i, 2] * height)
            xmax = int(bboxes[i, 3] * width)
            rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                 ymax - ymin, fill=False,
                                 edgecolor=colors[cls_id],
                                 linewidth=linewidth)
            plt.gca().add_patch(rect)
            class_name = str(cls_id)
            if show_text:
                plt.gca().text(xmin, ymin - 2,
                               '{:s} | {:.3f}'.format(class_name, score),
                               bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                               fontsize=12, color='white')
    if title is not None:
        plt.gca().text(width/2, 16 ,
                       title,
                       bbox=None,
                       fontsize=12, color='red')
    if not hold:
        plt.show()
