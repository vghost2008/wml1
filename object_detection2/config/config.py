# -*- coding: utf-8 -*-
import logging
from thirdparty.config import CfgNode as _CfgNode
import os


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


def get_config_file(name:str):
    CONFIG_DIR = "/home/vghost/ai/work/wml/object_detection2/default_configs/"
    COCOCONFIG_DIR = "/home/vghost/ai/work/wml/object_detection2/default_configs/coco/"
    MODCONFIG_DIR = "/home/vghost/ai/work/wml/object_detection2/default_configs/mnistod/"
    search_dirs = [COCOCONFIG_DIR,MODCONFIG_DIR,CONFIG_DIR]
    if os.path.exists(name):
        return name
    if not name.endswith(".yaml"):
        name = name+".yaml"

    for dir in search_dirs:
        path = os.path.join(dir,name)
        if os.path.exists(path):
            return path

    return name


