# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
from thirdparty.config import CfgNode as _CfgNode


class CfgNode(_CfgNode):

    def dump(self, *args, **kwargs):
        """
        Returns:
            str: a yaml string representation of the config
        """
        # to make it show up in docs
        return super().dump(*args, **kwargs)


global_cfg = CfgNode()


def get_cfg() -> CfgNode:
    """
    Get a copy of the default config.

    Returns:
        a detectron2 CfgNode instance.
    """
    from .defaults import _C

    return _C.clone()


def set_global_cfg(cfg: CfgNode) -> None:
    """
    Let the global config point to the given cfg.

    Assume that the given "cfg" has the key "KEY", after calling
    `set_global_cfg(cfg)`, the key can be accessed by:

    .. code-block:: python

        from detectron2.config import global_cfg
        print(global_cfg.KEY)

    By using a hacky global config, you can access these configs anywhere,
    without having to pass the config object or the values deep into the code.
    This is a hacky feature introduced for quick prototyping / research exploration.
    """
    global global_cfg
    global_cfg.clear()
    global_cfg.update(cfg)
