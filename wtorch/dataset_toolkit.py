import wml_utils as wmlu
import random

class DataUnit:
    def __init__(self,data):
        if not isinstance(data,(list,tuple)):
            raise RuntimeError("Error data type")
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def sample(self):
        return random.choice(self.data)

def make_data_unit(datas,total_nr=None,nr_per_unit=None):
    assert total_nr is None or nr_per_unit is None, "Error arguments"
    if total_nr is not None:
        if total_nr>=len(datas):
            return datas
        datas = wmlu.list_to_2dlistv2(datas,total_nr)
    else:
        if nr_per_unit<=1:
            return datas
        datas = wmlu.list_to_2dlist(data,nr_per_unit)

    datas = [DataUnit(x) for x in datas]
    return datas
