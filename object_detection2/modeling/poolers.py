# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math
import sys
import wmodule
import object_detection2.bboxes as odbox
from object_detection2.config.config import global_cfg
from object_detection2.wlayers import *
import wsummary

__all__ = ["ROIPooler"]


def assign_boxes_to_levels(bboxes, min_level, max_level, canonical_box_size, canonical_level):
    """
    Map each box in `box_lists` to a feature map level index and return the assignment
    vector.

    Args:
        box_lists (list[Boxes] | list[RotatedBoxes]): A list of N Boxes or N RotatedBoxes,
            where N is the number of images in the batch.
        min_level (int): Smallest feature map level index. The input is considered index 0,
            the output of stage 1 is index 1, and so.
        max_level (int): Largest feature map level index.
        canonical_box_size (int): A canonical box size in pixels (sqrt(box area)).
        canonical_level (int): The feature map level index on which a canonically-sized box
            should be placed.

    Returns:
        [batch_size,box_nr]
        A tensor of length M, where M is the total number of boxes aggregated over all
            N batch images. The memory layout corresponds to the concatenation of boxes
            from all images. Each element is the feature map index, as an offset from
            `self.min_level`, for the corresponding box (so value i means the box is at
            `self.min_level + i`).
    """
    with tf.name_scope("assign_boxes_to_levels"):
        eps = 1e-6
        box_sizes = tf.sqrt(odbox.box_area(bboxes))
        # Eqn.(1) in FPN paper
        level_assignments = tf.floor(
            canonical_level + tf.log(box_sizes / canonical_box_size + eps)/math.log(2)
        )
        # clamp level to (min, max), in case the box size is too large or too small
        # for the available feature maps
        level_assignments = tf.cast(level_assignments,tf.int32)
        level_assignments = tf.clip_by_value(level_assignments, min_level, max_level)
        return level_assignments - min_level




class ROIPooler(wmodule.WChildModule):
    """
    Region of interest feature map pooler that supports pooling from one or more
    feature maps.
    """

    def __init__(
        self,
        cfg,
        parent,
        output_size=[7,7],
        bin_size=[2,2],
        pooler_type="ROIAlign",
        **kwargs,
    ):
        """
        Args:
            output_size (int, tuple[int] or list[int]): output size of the pooled region,
                e.g., 14 x 14. If tuple or list is given, the length must be 2.
            pooler_type (string): Name of the type of pooling operation that should be applied.
                For instance, "ROIPool" or "ROIAlignV2".
            canonical_box_size (int): A canonical box size in pixels (sqrt(box area)). The default
                is heuristically defined as 224 pixels in the FPN paper (based on ImageNet
                pre-training).
            canonical_level (int): The feature map level index from which a canonically-sized box
                should be placed. The default is defined as level 4 (stride=16) in the FPN paper,
                i.e., a box of size 224x224 will be placed on the feature with stride=16.
                The box placement for all boxes will be determined from their sizes w.r.t
                canonical_box_size. For example, a box whose area is 4x that of a canonical box
                should be used to pool features from feature level ``canonical_level+1``.

                Note that the actual input feature maps given to this module may not have
                sufficiently many levels for the input boxes. If the boxes are too large or too
                small for the input feature maps, the closest level will be used.
        """
        super().__init__(cfg=cfg,parent=parent,**kwargs)

        canonical_box_size=cfg.canonical_box_size,
        canonical_level=cfg.canonical_level,

        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        assert len(output_size) == 2
        assert isinstance(output_size[0], int) and isinstance(output_size[1], int)
        self.output_size = output_size
        self.canonical_box_size = canonical_box_size
        self.canonical_level = canonical_level

        if pooler_type == "ROIAlign" or pooler_type == "ROIAlignV2":
            self.level_pooler = WROIAlign(bin_size=bin_size,output_size=output_size)
        elif pooler_type == "ROIPool":
            self.level_pooler = WROIPool(bin_size=bin_size,output_size=output_size)
        else:
            raise ValueError("Unknown pooler type: {}".format(pooler_type))

    def forward(self, x, bboxes):
        """
        Args:
            x (list[Tensor]): tensorshape is [batch_size,H,W,C] resolution from high to low
            bboxes:[batch_size,box_nr,4]

        Returns:
            Tensor:
                A tensor of shape (M, output_size, output_size,C) where M is the total number of
                boxes aggregated over all N batch images and C is the number of channels in `x`.
        """
        assert isinstance(x, list),"Arguments to pooler must be lists"
        level_num = len(x)

        with tf.name_scope("ROIPoolers"):
            if level_num == 1:
                return self.level_pooler(x[0], bboxes)

            level_assignments = assign_boxes_to_levels(
                bboxes, 0, level_num-1, self.canonical_box_size, self.canonical_level
            )
            features = []
            for net in x:
                features.append(self.level_pooler(net,bboxes))

            features = tf.stack(features,axis=1)
            level_assignments = tf.reshape(level_assignments,[-1])

            if global_cfg.GLOBAL.DEBUG:
                wsummary.histogram_or_scalar(level_assignments,"level_assignments")

            output = wmlt.batch_gather(features,level_assignments)


            return output
