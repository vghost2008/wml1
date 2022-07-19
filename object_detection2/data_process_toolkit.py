import img_utils as wmli
from bboxes import *
import cv2 as cv

'''
cnt is a contour in a image, cut the area of rect
cnt:[[x,y],[x,y],...]
rect:[ymin,xmin,ymax,xmax]
output:
the new contours in sub image
'''
def cut_contour(cnt,rect):
    bbox = bbox_of_contour(cnt)
    width = max(bbox[3],rect[3])
    height = max(bbox[2],rect[2])
    img = np.zeros(shape=(height,width),dtype=np.uint8)
    segmentation = cv.drawContours(img,[cnt],-1,color=(1),thickness=cv.FILLED)
    cuted_img = wmli.sub_image(segmentation,rect)
    contours,hierarchy = cv.findContours(cuted_img,cv.CV_RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
    return contours

'''
find the contours in rect of segmentation
segmentation:[H,W]
rect:[ymin,xmin,ymax,xmax]
output:
the new contours in sub image and correspond bbox
'''
def cut_contourv2(segmentation,rect):
    org_contours,org_hierarchy = cv.findContours(segmentation,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
    max_area = 1e-8
    for cnt in org_contours:
        area = cv.contourArea(cnt)
        max_area = max(max_area,area)
    cuted_img = wmli.sub_image(segmentation,rect)
    contours,hierarchy = cv.findContours(cuted_img,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
    boxes = []
    ratio = []
    for cnt in contours:
        boxes.append(bbox_of_contour(cnt))
        ratio.append(cv.contourArea(cnt)/max_area)
    return contours,boxes,ratio

def remove_class_in_image(bboxes,labels,labels_to_remove,image,default_value=127,scale=1.1):
    bboxes = bboxes.astype(np.int32)
    mask = np.ones_like(labels,dtype=np.bool)
    for l in labels_to_remove:
        tm = labels==l
        mask = np.logical_and(tm,mask)
    keep_mask = np.logical_not(mask)
    keep_bboxes = bboxes[keep_mask]
    remove_bboxes = bboxes[mask]
    img_mask = np.ones(image.shape[:2],dtype=np.bool)

    wmli.remove_boxes_of_img(img_mask,remove_bboxes,False)
    if scale>1.0:
        t_keep_bboxes = npscale_bboxes(keep_bboxes,scale).astype(np.int32)
    else:
        t_keep_bboxes = keep_bboxes
    wmli.remove_boxes_of_img(img_mask,t_keep_bboxes,True)

    img_mask = np.expand_dims(img_mask,axis=-1)
    img_mask = np.tile(img_mask,[1,1,3])
    remove_image = np.ones_like(image)*default_value
    image = np.where(img_mask,image,remove_image)
    return image,keep_bboxes,labels[keep_mask]