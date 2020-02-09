#coding=utf-8

class WModule(object):
    def __init__(self,cfg,parent=None,is_training=False):
        self.cfg = cfg
        self.parent = parent
        self.childs = []
        if self.parent is not None:
            self.parent.add_child(self)
        self.maped_attr = {"is_training":is_training}

    def add_child(self,obj):
        self.childs.append(obj)

    def __getattr__(self,item):
        if item in self.maped_attr:
            if self.parent is None:
                return self.maped_attr[item]
            else:
                return self.parent.maped_attr[item]
        else:
            return self.__dict__[item]

    def __setattr__(self, key, value):
        if key in self.maped_attr:
            if self.parent is None:
                self.maped_attr[key] = value
            else:
                self.parent.maped_attr[key] = value
        else:
            self.__dict__[key] = value

class WChildModule(WModule):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        assert self.parent is not None
