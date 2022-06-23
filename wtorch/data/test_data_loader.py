from wtorch.data.dataloader import DataLoader
import numpy as np
from wtorch.data import SequentialSampler

class ExampleDataset(object):
    def __init__(self):
        self.data = np.array(list(range(100)))
        self.data = np.reshape(self.data,[50,2])

    def __len__(self):
        return len(self.data)

    def __getitem__(self,item):
        return self.data[item]


dataset = ExampleDataset()
dataloader = DataLoader(ExampleDataset(),4,shuffle=True,num_workers=4,pin_memory=True,batch_split_nr=2)
                        #sampler=SequentialSampler(dataset))

idx = 0
for i in range(3):
    for x in iter(dataloader):
        print(idx,x)
        idx += 1

