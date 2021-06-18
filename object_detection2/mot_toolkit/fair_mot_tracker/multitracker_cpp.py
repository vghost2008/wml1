import tensorflow as tf
from object_detection2.standard_names import *
from object_detection2.mot_toolkit.tracking_utils.utils import *
import tfop
import object_detection2.bboxes as odb
from object_detection2.visualization import colors_tableau
import object_detection2.visualization as odv

class CPPTracker(object):
    def __init__(self, model, det_thresh=0.1, frame_rate=30, track_buffer=30):
        self.model = model
        self.det_thresh = det_thresh
        self.frame_rate = frame_rate
        self.track_buffer = track_buffer
        self.frame_id = 0
        self.post_process_net()

    def post_process_net(self):
        self.is_first_frame = tf.placeholder(tf.bool, shape=())
        self.bboxes = tf.placeholder(tf.float32, shape=[None, 4])
        self.probs = tf.placeholder(tf.float32, shape=[None])
        self.embedding = tf.placeholder(tf.float32, shape=[None, None])
        self.tracked_id,self.tracked_bboxes = tfop.fair_mot(self.bboxes,self.probs,self.embedding,
                                                           self.is_first_frame,
                                                           det_thredh=self.det_thresh,
                                                           frame_rate=self.frame_rate,
                                                           track_buffer=self.track_buffer)

    def draw_tracks(self, img, tracked_objs):
        labels,bboxes = tracked_objs

        def color_fn(l):
            index = l % len(colors_tableau)
            return colors_tableau[index]

        def text_fn(l, scires):
            return f"{l}"

        print(f"bboxes:", bboxes)
        img = odv.draw_bboxes(img, labels, bboxes=bboxes,
                              color_fn=color_fn, text_fn=text_fn,
                              show_text=True, is_relative_coordinate=False)
        return img

    def update(self, img0):
        self.frame_id += 1
        width = img0.shape[1]
        height = img0.shape[0]

        output = self.model(np.expand_dims(img0, axis=0))
        if len(output[RD_BOXES].shape) == 3:
            l = output[RD_LENGTH][0]
            bboxes = output[RD_BOXES][0][:l]
            bboxes = odb.relative_boxes_to_absolutely_boxes(bboxes, width=width, height=height)
            probs = output[RD_PROBABILITY][0][:l]
            ids = output[RD_ID][0][:l]
            pass
        else:
            bboxes = output[RD_BOXES]
            bboxes = odb.relative_boxes_to_absolutely_boxes(bboxes, width=width, height=height)
            probs = output[RD_PROBABILITY]
            ids = output[RD_ID]
            pass
        feed_dict = {self.is_first_frame:self.frame_id==1,
                     self.bboxes:bboxes,
                     self.probs:probs,
                     self.embedding:ids}
        tracked_ids,tracked_bboxes = self.model.sess.run([self.tracked_id,self.tracked_bboxes],feed_dict=feed_dict)
        return tracked_ids,tracked_bboxes
