import torch
from torch.utils.data import Dataset, DataLoader

class DataLoaderClass:
    """ """
    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.data_transforms = None
        self.dataset = None
        self.dataloader = None
        
    def set_transform(self, data_transforms):
        """

        :param data_transforms: 

        """
        self.data_transforms = data_transforms
        
    def create_dataset(self):
        """ """
        self.dataset = MyDataset(self.data, self.target)
        self.dataset = self.dataset.transform(self.data_transforms)
        
    def create_dataloader(self,batch_size, shuffle=True):
        """

        :param batch_size: 
        :param shuffle:  (Default value = True)

        """
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)
        
    def return_dataloader(self):
        """ """
        return self.dataloader

class MyDataset(Dataset):
    """ """
    def __init__(self, data, target):
        self.data = data
        self.target = target
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]