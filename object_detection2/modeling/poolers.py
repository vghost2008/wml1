# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math
import sys
import wmodule

from object_detection2.wlayers import *

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
        A tensor of length M, where M is the total number of boxes aggregated over all
            N batch images. The memory layout corresponds to the concatenation of boxes
            from all images. Each element is the feature map index, as an offset from
            `self.min_level`, for the corresponding box (so value i means the box is at
            `self.min_level + i`).
    """
    '''
    eps = 1e-6
    box_sizes = torch.sqrt(cat([boxes.area() for boxes in box_lists]))
    # Eqn.(1) in FPN paper
    level_assignments = torch.floor(
        canonical_level + torch.log2(box_sizes / canonical_box_size + eps)
    )
    # clamp level to (min, max), in case the box size is too large or too small
    # for the available feature maps
    level_assignments = torch.clamp(level_assignments, min=min_level, max=max_level)
    return level_assignments.to(torch.int64) - min_level
    '''




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
        canonical_box_size=224,
        canonical_level=4,
        min_level=0,
        max_level=0,
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

        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        assert len(output_size) == 2
        assert isinstance(output_size[0], int) and isinstance(output_size[1], int)
        self.output_size = output_size
        self.min_level = min_level
        self.max_level = max_level
        self.level_num = max_level-min_level+1
        self.canonical_box_size = canonical_box_size
        self.canonical_level = canonical_level

        if pooler_type == "ROIAlign":
            self.level_pooler = WROIAlign(bin_size=bin_size,output_size=output_size)
        elif pooler_type == "ROIPool":
            self.level_pooler = WROIPool(bin_size=bin_size,output_size=output_size)
        else:
            raise ValueError("Unknown pooler type: {}".format(pooler_type))

    def forward(self, x, bboxes):
        """
        Args:
            x (list[Tensor]): tensorshape is [batch_size,H,W,C]
            bboxes:[batch_size,box_nr,4]

        Returns:
            Tensor:
                A tensor of shape (M, output_size, output_size,C) where M is the total number of
                boxes aggregated over all N batch images and C is the number of channels in `x`.
        """
        assert isinstance(x, list),"Arguments to pooler must be lists"
        assert self.level_num == len(x), "Error input feature map size"

        if self.level_num == 1:
            return self.level_pooler(x[0], bboxes)

        '''level_assignments = assign_boxes_to_levels(
            bboxes, self.min_level, self.max_level, self.canonical_box_size, self.canonical_level
        )

        return output'''
