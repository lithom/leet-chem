import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class NPYDataset(Dataset):
    def __init__(self, filenames):
        self.filenames = filenames

        # Load the .npy files
        self.x_data = np.load(filenames['x'])
        self.target_data = np.load(filenames['target'])
        self.bondsType_data = np.load(filenames['bondsType'])
        self.structureInfo_data = np.load(filenames['structureInfo'])

    def __len__(self):
        # Return the number of batches
        return len(self.x_data)

    def __getitem__(self, idx):
        # Get a batch for each type of data and convert them to tensors
        x = torch.tensor(self.x_data[idx], dtype=torch.float32)
        target = torch.tensor(self.target_data[idx], dtype=torch.float32)
        bondsType = torch.tensor(self.bondsType_data[idx], dtype=torch.bool)
        structureInfo = torch.tensor(self.structureInfo_data[idx], dtype=torch.bool)

        return x, target, structureInfo, bondsType


class NPYDataset2(Dataset):
    def __init__(self, filenames):
        self.filenames = filenames

        self.entrymap = dict()

        idx_tot = 0
        for idx, fi in enumerate( self.filenames ):
            data_i = np.load(fi+'_structureInfo.npy')
            #idx_file = 0
            for xi in range(0,len(data_i)):
                self.entrymap[idx_tot] = (idx,xi)
                idx_tot+=1

        self.length = idx_tot
        self.loaded_file = -1

        self.x_data = None
        self.target_data = None
        self.bondsType_data = None
        self.structureInfo_data = None

    def load_i(self,i):
        # Load the .npy files
        if(self.loaded_file==i):
            return
        self.loaded_file = i
        self.x_data = np.load(self.filenames[i]+'_x.npy')
        self.target_data = np.load(self.filenames[i]+'_target.npy')
        self.bondsType_data = np.load(self.filenames[i]+'_bondsType.npy')
        self.structureInfo_data = np.load(self.filenames[i]+'_structureInfo.npy')

    def __len__(self):
        # Return the number of batches
        return self.length

    def __getitem__(self, idx):
        if self.entrymap[idx][0] != self.loaded_file:
            self.load_i(self.entrymap[idx][0])

        idx2 = self.entrymap[idx][1]

        # Get a batch for each type of data and convert them to tensors
        x = torch.tensor(self.x_data[idx2], dtype=torch.float32)
        target = torch.tensor(self.target_data[idx2], dtype=torch.float32)
        bondsType = torch.tensor(self.bondsType_data[idx2], dtype=torch.bool)
        structureInfo = torch.tensor(self.structureInfo_data[idx2], dtype=torch.bool)

        # add some noise?
        if True:
            noise_level = 0.001
            # Generate random noise with the same shape as 'data'
            noise_x = torch.randn_like(x)
            noise_t = torch.randn_like(target)
            # Scale the noise according to the noise_level
            scaled_noise_x = torch.abs(noise_x * noise_level)
            scaled_noise_t = torch.abs(noise_t * noise_level)
            x = x + scaled_noise_x
            target = target + scaled_noise_t

        return x, target, structureInfo, bondsType
