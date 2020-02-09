from abc import ABCMeta, abstractmethod
import wmodule

__all__ = ["Backbone"]


class Backbone(wmodule.WModule):
    """
    Abstract base class for network backbones.
    """

    def __init__(self,cfg,*args,**kwargs):
        """
        The `__init__` method of any subclass can specify its own set of arguments.
        """
        wmodule.WModule.__init__(self,cfg,*args,**kwargs)

    def forward(self):
        """
        Subclasses must override this method, but adhere to the same return type.

        Returns:
            dict[str->Tensor]: mapping from feature name (e.g., "res2") to tensor
        """
        pass

    @property
    def size_divisibility(self):
        """
        Some backbones require the input height and width to be divisible by a
        specific integer. This is typically true for encoder / decoder type networks
        with lateral connection (e.g., FPN) for which feature maps need to match
        dimension in the "bottom up" and "top down" paths. Set to 0 if no specific
        input size divisibility is required.
        """
        return 0
