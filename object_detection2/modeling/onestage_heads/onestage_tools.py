#coding=utf-8
import wml_tfutils as wmlt
import tensorflow as tf

def reshape_to_N_HWA_K(tensor, K):
    """
    Transpose/reshape a tensor from (N, (A x K), H, W) to (N, (HxWxA), K)
    """
    assert len(tensor.get_shape()) == 4, tensor.shape
    N, H, W, _ = wmlt.combined_static_and_dynamic_shape(tensor)
    tensor = tf.reshape(tensor,[N,-1,K])
    return tensor

def permute_all_cls_and_box_to_N_HWA_K_and_concat(box_cls, box_delta, num_classes=80):
    """
    box_cls:list of [batch_size,H,W,A*K]
    box_delta:list of [batch_size,H,W,A*4]
    return:
    box_cls [batch_size,HWA(concat),K]
    box_delta [batch_size,HWA(concat),4]
    """
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_delta
    box_cls_flattened = [reshape_to_N_HWA_K(x, num_classes) for x in box_cls]
    box_delta_flattened = [reshape_to_N_HWA_K(x, 4) for x in box_delta]
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = tf.concat(box_cls_flattened, axis=1)
    box_delta = tf.concat(box_delta_flattened, axis=1)
    return box_cls, box_delta