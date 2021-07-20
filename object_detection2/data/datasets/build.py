#coding=utf-8
class DatasetRegistry(object):
    def __init__(self,name):
        self.name = name
        self.dict = {}

    def register(self,name,v):
        self.dict[name] = v

    def __getitem__(self,name):
        return self.dict[name]

DATASETS_REGISTRY = DatasetRegistry("DATASETS")


