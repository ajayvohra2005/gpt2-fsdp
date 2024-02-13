
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import math

class GptDataset(Dataset):

    def __init__(self, data_path:str,  block_size:int=1024, max_len:int = math.inf):
        super().__init__()
  
        self.__data_path = data_path
        data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.__block_size = block_size
        self.__length = min(len(data) // block_size, max_len)
        del data

    def __getitem__(self, index):
        i = index * self.__block_size
        j = i+self.__block_size
     
        data = np.memmap(self.__data_path, dtype=np.uint16, mode='r')
        x = torch.from_numpy(data[i:j].copy().astype(np.int64))
        y = torch.from_numpy(data[i+1:j+1].copy().astype(np.int64))
        
        del data

        return x, y
    
    def __len__(self):
        return self.__length
    
def test():
    from tqdm import tqdm as tq
    from torch.utils.data import RandomSampler, DataLoader
    data_path = "data/openwebtext/train.bin"
    ds = GptDataset(data_path=data_path)

    sampler = RandomSampler(ds)
        
    kwargs = {'batch_size': 12, 'sampler': sampler}
    cuda_kwargs = {'num_workers': 1, 'prefetch_factor': 2, 'shuffle': False}
    kwargs.update(cuda_kwargs)

    data_loader = DataLoader(ds,**kwargs)
    inner_pbar = tq(range(len(data_loader)), colour="blue", desc="Data sample")
    for X,Y in data_loader:
        X, Y = X.to(device="cuda:0"), Y.to(device="cuda:0")
        inner_pbar.update(1)

if __name__ == "__main__":
    test()