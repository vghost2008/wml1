#coding=utf-8
import tensorflow as tf
from iotoolkit.coco_tf_decode import get_data as _get_data
import wml_tfutils as wmlt
import object_detection.od_toolkit as od
import wml_tfutils as wml

MIN_OBJECT_COVERED = 0.8
CROP_RATIO_RANGE = (0.75, 1.33)
MIN_AREA_RANGE=0.25
FILTER_THRESHOLD=0.3
NUM_CLASSES=91
MAX_BOXES_NR=100

def r_preprocess_for_train(image, labels, bboxes,
                           out_shape,
                           area_range=(0.25, 1.0),
                           scope='preprocessing_train'):

    with tf.name_scope(scope, 'ssd_preprocessing_train', [image, labels, bboxes]):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')
        dst_image, labels, bboxes, bbox_begin,bbox_size,_= \
            od.distorted_bounding_box_crop(image, labels, bboxes,
                                           area_range=area_range,
                                           min_object_covered=MIN_OBJECT_COVERED,
                                           aspect_ratio_range=CROP_RATIO_RANGE,
                                           filter_threshold=FILTER_THRESHOLD)
        dst_image = tf.image.resize_images(dst_image, out_shape)

        return dst_image, labels, bboxes

def preprocess_for_train(image,labels, bboxes,
                         out_shape,
                         area_range=(0.25, 1.0),
                         scope='preprocess'):
    with tf.name_scope("preprocess1"):
        bboxes = tf.clip_by_value(bboxes,0.,1.0);

        #image = wml.probability_case([(0.5,lambda:wml.distort_color(image,color_ordering=6)),
        #(0.5,lambda:image)])
        #def add_noise():
        #    return image+tf.truncated_normal(shape=tf.shape(image),stddev=0.003)
        #image = wml.probability_case([(0.5,lambda:image),(0.5,add_noise)])
        image = tf.cast(image,tf.float32)
        image = image/255.0
        image = (image-0.5)*2.

        r_image,r_labels,r_bboxes= r_preprocess_for_train(image,labels, bboxes, out_shape,area_range=area_range,scope=scope)
        raw_image = tf.image.resize_images(image,out_shape)
        p_image,p_labels,p_bboxes= tf.cond(tf.greater(tf.random_uniform(shape=[]),0.5),
                                           lambda: (raw_image,labels,bboxes),
                                           lambda: (r_image, r_labels, r_bboxes))
        p_image,p_bboxes = wml.probability_case(
            [(0.5,lambda:(p_image,p_bboxes)),
             (0.5,lambda:od.flip_left_right(p_image,p_bboxes))
             ])
        #p_image = wmli.random_blur(p_image,prob=0.12)

        '''p_image = tf.expand_dims(p_image,axis=0)
        p_image = wnnl.dropblock(p_image,0.985,block_size=11,is_training=True,all_channel=True)
        p_image = tf.squeeze(p_image,axis=0)'''

        return p_image,p_labels, p_bboxes
    
def get_coco_data(data_dir,batch_size,output_shape=[512,512],preprocess=preprocess_for_train,num_samples=100):
    ids = range(NUM_CLASSES)
    id_to_label = dict(zip(ids,ids))
    image, labels, bboxes,mask = _get_data(data_dir=data_dir,batch_size=batch_size,
                                           num_samples=num_samples,num_classes=NUM_CLASSES,id_to_label=id_to_label,
                                           has_file_index=True)
    image = tf.cast(image,tf.float32)
    image = wmlt.assert_equal(image,[tf.shape(labels),tf.shape(mask)[0]],name="assert_equal0")
    [image,glabels, bboxes] = preprocess(image,labels, bboxes,out_shape=output_shape)
    #image = wmlt.assert_equal(image,[tf.shape(glabels),tf.shape(mask)[2]],name="assert_equal0")
    log_image = image
    len = tf.shape(glabels)[0]
    glabels = tf.pad(glabels,paddings=[[0,MAX_BOXES_NR-len]])
    glabels = tf.reshape(glabels,[MAX_BOXES_NR])
    bboxes = tf.pad(bboxes,paddings=[[0,MAX_BOXES_NR-len],[0,0]])
    bboxes = tf.reshape(bboxes,[MAX_BOXES_NR,4])
    return image,glabels,bboxes,len

