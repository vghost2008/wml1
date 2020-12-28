#coding=utf-8

class WModule(object):
    def __init__(self,cfg,parent=None,is_training=False,*args,**kwargs):
        self.maped_attr = {"is_training":is_training}
        self.local_attr = {}
        self.cfg = cfg
        self.parent = parent
        self.childs = []
        if self.parent is not None:
            self.parent.add_child(self)

    def add_child(self,obj):
        self.childs.append(obj)

    def __getattr__(self,item):
        if item == "maped_attr" or item == 'local_attr':
            return self.__dict__[item]
        elif item in self.local_attr:
            return self.local_attr[item]
        elif item in self.maped_attr:
            if self.parent is None:
                return self.maped_attr[item]
            elif item in self.parent.local_attr:
                return self.parent.local_attr[item]
            elif item in self.parent.maped_attr:
                return self.parent.maped_attr[item]
        else:
            if item in self.__dict__:
                return self.__dict__[item]
            else:
                raise AttributeError(r"object has no attribute '%s'" % item)

    def __setattr__(self, key, value):
        if key == "maped_attr":
            self.__dict__[key] = value
        elif key in self.maped_attr:
            if self.parent is None:
                self.maped_attr[key] = value
            else:
                self.parent.maped_attr[key] = value
        else:
            self.__dict__[key] = value

    def __call__(self, *args, **kwargs):
        return self.forward(*args,**kwargs)

    def disable_training(self):
        self.local_attr['is_training'] = False

class WChildModule(WModule):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        assert self.parent is not None

class WModelList(WModule):
    def __init__(self,models,*args,**kwargs):
        super().__init__(*args,**kwargs)
        assert self.parent is not None
        self.models = models
        assert len(self.models)>0

    def farward(self,*args,**kwargs):
        res = self.models[0](*args,**kwargs)
        for m in self.models[1:]:
            res = m(res)
        return res

