import logging
import copy
import re
import torch
import numpy as np

def convert_basic_c2_names(original_keys):
    """
    Apply some basic name conversion to names in C2 weights.
    It only deals with typical backbone models.

    Args:
        original_keys (list[str]):
    Returns:
        list[str]: The same number of strings matching those in original_keys.
    """
    layer_keys = copy.deepcopy(original_keys)
    layer_keys = [
        {"pred_b": "linear_b", "pred_w": "linear_w"}.get(k, k) for k in layer_keys
    ]  # some hard-coded mappings

    layer_keys = [k.replace("_", ".") for k in layer_keys]
    layer_keys = [re.sub("\\.b$", ".bias", k) for k in layer_keys]
    layer_keys = [re.sub("\\.w$", ".weight", k) for k in layer_keys]
    # Uniform both bn and gn names to "norm"
    layer_keys = [re.sub("bn\\.s$", "norm.weight", k) for k in layer_keys]
    layer_keys = [re.sub("bn\\.bias$", "norm.bias", k) for k in layer_keys]
    layer_keys = [re.sub("bn\\.rm", "norm.running_mean", k) for k in layer_keys]
    layer_keys = [re.sub("bn\\.running.mean$", "norm.running_mean", k) for k in layer_keys]
    layer_keys = [re.sub("bn\\.riv$", "norm.running_var", k) for k in layer_keys]
    layer_keys = [re.sub("bn\\.running.var$", "norm.running_var", k) for k in layer_keys]
    layer_keys = [re.sub("bn\\.gamma$", "norm.weight", k) for k in layer_keys]
    layer_keys = [re.sub("bn\\.beta$", "norm.bias", k) for k in layer_keys]
    layer_keys = [re.sub("gn\\.s$", "norm.weight", k) for k in layer_keys]
    layer_keys = [re.sub("gn\\.bias$", "norm.bias", k) for k in layer_keys]

    # stem
    layer_keys = [re.sub("^res\\.conv1\\.norm\\.", "conv1.norm.", k) for k in layer_keys]
    # to avoid mis-matching with "conv1" in other components (e.g. detection head)
    layer_keys = [re.sub("^conv1\\.", "stem.conv1.", k) for k in layer_keys]

    # layer1-4 is used by torchvision, however we follow the C2 naming strategy (res2-5)
    # layer_keys = [re.sub("^res2.", "layer1.", k) for k in layer_keys]
    # layer_keys = [re.sub("^res3.", "layer2.", k) for k in layer_keys]
    # layer_keys = [re.sub("^res4.", "layer3.", k) for k in layer_keys]
    # layer_keys = [re.sub("^res5.", "layer4.", k) for k in layer_keys]

    # blocks
    layer_keys = [k.replace(".branch1.", ".shortcut.") for k in layer_keys]
    layer_keys = [k.replace(".branch2a.", ".conv1.") for k in layer_keys]
    layer_keys = [k.replace(".branch2b.", ".conv2.") for k in layer_keys]
    layer_keys = [k.replace(".branch2c.", ".conv3.") for k in layer_keys]

    # DensePose substitutions
    layer_keys = [re.sub("^body.conv.fcn", "body_conv_fcn", k) for k in layer_keys]
    layer_keys = [k.replace("AnnIndex.lowres", "ann_index_lowres") for k in layer_keys]
    layer_keys = [k.replace("Index.UV.lowres", "index_uv_lowres") for k in layer_keys]
    layer_keys = [k.replace("U.lowres", "u_lowres") for k in layer_keys]
    layer_keys = [k.replace("V.lowres", "v_lowres") for k in layer_keys]
    return layer_keys

def convert_c2_detectron_names(weights):
    """
    Map Caffe2 Detectron weight names to Detectron2 names.

    Args:
        weights (dict): name -> tensor

    Returns:
        dict: detectron2 names -> tensor
        dict: detectron2 names -> C2 names
    """
    logger = logging.getLogger(__name__)
    logger.info("Renaming Caffe2 weights ......")
    original_keys = sorted(weights.keys())
    layer_keys = copy.deepcopy(original_keys)

    layer_keys = convert_basic_c2_names(layer_keys)

    # --------------------------------------------------------------------------
    # RPN hidden representation conv
    # --------------------------------------------------------------------------
    # FPN case
    # In the C2 model, the RPN hidden layer conv is defined for FPN level 2 and then
    # shared for all other levels, hence the appearance of "fpn2"
    layer_keys = [
        k.replace("conv.rpn.fpn2", "proposal_generator.rpn_head.conv") for k in layer_keys
    ]
    # Non-FPN case
    layer_keys = [k.replace("conv.rpn", "proposal_generator.rpn_head.conv") for k in layer_keys]

    # --------------------------------------------------------------------------
    # RPN box transformation conv
    # --------------------------------------------------------------------------
    # FPN case (see note above about "fpn2")
    layer_keys = [
        k.replace("rpn.bbox.pred.fpn2", "proposal_generator.rpn_head.anchor_deltas")
        for k in layer_keys
    ]
    layer_keys = [
        k.replace("rpn.cls.logits.fpn2", "proposal_generator.rpn_head.objectness_logits")
        for k in layer_keys
    ]
    # Non-FPN case
    layer_keys = [
        k.replace("rpn.bbox.pred", "proposal_generator.rpn_head.anchor_deltas") for k in layer_keys
    ]
    layer_keys = [
        k.replace("rpn.cls.logits", "proposal_generator.rpn_head.objectness_logits")
        for k in layer_keys
    ]

    # --------------------------------------------------------------------------
    # Fast R-CNN box head
    # --------------------------------------------------------------------------
    layer_keys = [re.sub("^bbox\\.pred", "bbox_pred", k) for k in layer_keys]
    layer_keys = [re.sub("^cls\\.score", "cls_score", k) for k in layer_keys]
    layer_keys = [re.sub("^fc6\\.", "box_head.fc1.", k) for k in layer_keys]
    layer_keys = [re.sub("^fc7\\.", "box_head.fc2.", k) for k in layer_keys]
    # 4conv1fc head tensor names: head_conv1_w, head_conv1_gn_s
    layer_keys = [re.sub("^head\\.conv", "box_head.conv", k) for k in layer_keys]

    # --------------------------------------------------------------------------
    # FPN lateral and output convolutions
    # --------------------------------------------------------------------------
    def fpn_map(name):
        """
        Look for keys with the following patterns:
        1) Starts with "fpn.inner."
           Example: "fpn.inner.res2.2.sum.lateral.weight"
           Meaning: These are lateral pathway convolutions
        2) Starts with "fpn.res"
           Example: "fpn.res2.2.sum.weight"
           Meaning: These are FPN output convolutions
        """
        splits = name.split(".")
        norm = ".norm" if "norm" in splits else ""
        if name.startswith("fpn.inner."):
            # splits example: ['fpn', 'inner', 'res2', '2', 'sum', 'lateral', 'weight']
            stage = int(splits[2][len("res") :])
            return "fpn_lateral{}{}.{}".format(stage, norm, splits[-1])
        elif name.startswith("fpn.res"):
            # splits example: ['fpn', 'res2', '2', 'sum', 'weight']
            stage = int(splits[1][len("res") :])
            return "fpn_output{}{}.{}".format(stage, norm, splits[-1])
        return name

    layer_keys = [fpn_map(k) for k in layer_keys]

    # --------------------------------------------------------------------------
    # Mask R-CNN mask head
    # --------------------------------------------------------------------------
    # roi_heads.StandardROIHeads case
    layer_keys = [k.replace(".[mask].fcn", "mask_head.mask_fcn") for k in layer_keys]
    layer_keys = [re.sub("^\\.mask\\.fcn", "mask_head.mask_fcn", k) for k in layer_keys]
    layer_keys = [k.replace("mask.fcn.logits", "mask_head.predictor") for k in layer_keys]
    # roi_heads.Res5ROIHeads case
    layer_keys = [k.replace("conv5.mask", "mask_head.deconv") for k in layer_keys]

    # --------------------------------------------------------------------------
    # Keypoint R-CNN head
    # --------------------------------------------------------------------------
    # interestingly, the keypoint head convs have blob names that are simply "conv_fcnX"
    layer_keys = [k.replace("conv.fcn", "roi_heads.keypoint_head.conv_fcn") for k in layer_keys]
    layer_keys = [
        k.replace("kps.score.lowres", "roi_heads.keypoint_head.score_lowres") for k in layer_keys
    ]
    layer_keys = [k.replace("kps.score.", "roi_heads.keypoint_head.score.") for k in layer_keys]

    # --------------------------------------------------------------------------
    # Done with replacements
    # --------------------------------------------------------------------------
    assert len(set(layer_keys)) == len(layer_keys)
    assert len(original_keys) == len(layer_keys)

    new_weights = {}
    new_keys_to_original_keys = {}
    for orig, renamed in zip(original_keys, layer_keys):
        new_keys_to_original_keys[renamed] = orig
        if renamed.startswith("bbox_pred.") or renamed.startswith("mask_head.predictor."):
            # remove the meaningless prediction weight for background class
            new_start_idx = 4 if renamed.startswith("bbox_pred.") else 1
            new_weights[renamed] = weights[orig][new_start_idx:]
            logger.info(
                "Remove prediction weight for background class in {}. The shape changes from "
                "{} to {}.".format(
                    renamed, tuple(weights[orig].shape), tuple(new_weights[renamed].shape)
                )
            )
        elif renamed.startswith("cls_score."):
            # move weights of bg class from original index 0 to last index
            logger.info(
                "Move classification weights for background class in {} from index 0 to "
                "index {}.".format(renamed, weights[orig].shape[0] - 1)
            )
            new_weights[renamed] = torch.cat([weights[orig][1:], weights[orig][:1]])
        else:
            new_weights[renamed] = weights[orig]

    return new_weights, new_keys_to_original_keys

def convert_ndarray_to_tensor(state_dict) -> None:
    """
    In-place convert all numpy arrays in the state_dict to torch tensor.
    Args:
        state_dict (dict): a state-dict to be loaded to the model.
            Will be modified.
    """
    # model could be an OrderedDict with _metadata attribute
    # (as returned by Pytorch's state_dict()). We should preserve these
    # properties.
    for k in list(state_dict.keys()):
        v = state_dict[k]
        if not isinstance(v, np.ndarray) and not isinstance(v, torch.Tensor):
            raise ValueError(
                "Unsupported type found in checkpoint! {}: {}".format(k, type(v))
            )
        if not isinstance(v, torch.Tensor):
            state_dict[k] = torch.from_numpy(v)