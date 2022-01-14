import os
class IterdatasetWrapper:
    def __init__(self,dataset,max_iter=100000):
        self.dataset = dataset
        self.max_iter = max_iter
        self.iter = iter(self.dataset)

    def __len__(self):
        return self.max_iter

    def __getitem__(self, item):
        try:
            data = next(self.iter)
        except StopIteration as e:
            self.iter = iter(self.dataset)
            try:
                data = next(self.iter)
            except Exception as e:
                print('ERROR: IterdatasetWrapper:',e)
                raise e
        except Exception as e:
            print('ERROR: IterdatasetWrapper:',e)
            raise e
        return data


