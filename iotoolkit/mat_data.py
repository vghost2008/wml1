import scipy.io as scio
import numpy as np

class MatData:
    def __init__(self,mat_path=None,data=None):
        if mat_path is not None:
            assert data is None,"Error arguments"
            data = scio.loadmat(mat_path)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self,item):
        return self.get_item(item)

    def __getattr__(self, item):
        return self.get_item(item)

    def get_item(self,item): 
        if item == 'shape' and 'shape' in self.data.__dict__:
            return self.data.shape

        if item in self.__dict__:
            return self.__dict__[item]
        else:
            data = self.data[item]
            shape = data.shape
            if len(shape)==2:
                if shape[0] == 1:
                    if shape[1] == 1:
                        data=data[0,0]
                        if data.shape==() and self.isscalar(data):
                            return data
                        return MatData(data=data)
                    elif shape[1]>1:
                        return MatData(data=data[0])
                    else:
                        return MatData(data=data)
                elif shape[0] == 0 or shape[1]==0:
                    return None
                else:
                    return MatData(data=data)
            elif len(shape)==1:
                if shape[0] == 1:
                    data=data[0]
                    if data.shape==() and self.isscalar(data):
                        return data
                    else:
                        return MatData(data=data)
                elif shape[0] == 0:
                    return None
                else:
                    return MatData(data=data)
            else:
                return MatData(data=data)
    def get_array_item(self,item): 
        if item == 'shape' and 'shape' in self.data.__dict__:
            return self.data.shape

        if item in self.__dict__:
            return self.__dict__[item]
        else:
            data = self.data[item]
            shape = data.shape
            if len(shape)==2:
                if shape[0] == 1:
                    return MatData(data=data[0])
                elif shape[0] == 0 or shape[1]==0:
                    return None
                else:
                    return MatData(data=data)
            elif len(shape)==1:
                if shape[0] == 0:
                    return None
                else:
                    return MatData(data=data)
            else:
                return MatData(data=data)

    @staticmethod
    def isscalar(data):
        type_str = type(data)
        return type_str in [np.uint8,np.int8,np.uint16,np.int16,np.int32,np.uint32,np.int64,np.uint64,np.float,np.float64,np.str_,np.str]

    def item(self):
        return self.data

    def data_keys(self):
        return self.data.dtype.names
