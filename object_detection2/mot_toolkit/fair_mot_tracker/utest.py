import tensorflow as tf
import numpy as np
from object_detection2.mot_toolkit.tracking_utils.kalman_filter import KalmanFilter

def get_bboxes(i):
    return np.array([100,100,1,20],dtype=np.float32)+np.array([i*1,i*-1,i*0.001,-i*0.01])
def measure_bboxes(bboxes):
    #return bboxes+np.random.randn(4)*np.array([0.1,0.1,0.001,0.001],dtype=np.float32)
    return bboxes+np.random.randn(4)*np.array([1,1,0.1,1.],dtype=np.float32)

kf = KalmanFilter()
bboxes = get_bboxes(0)
bboxes = measure_bboxes(bboxes)
mean,cov = kf.initiate(bboxes)
for i in range(1,100):
    p_mean,p_cov = kf.predict(mean,cov)
    bboxes = get_bboxes(i)
    bboxes = measure_bboxes(bboxes)
    mean,cov = kf.update(p_mean,p_cov,bboxes)

