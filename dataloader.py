import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
import torch

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class MovementDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the csv data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels = os.listdir(root_dir)
        self.all_files = []
        for i in self.labels:
            x = os.listdir(os.path.join(root_dir, i))
            self.all_files += [(os.path.join(root_dir, i, j), i) for j in x]
        random.shuffle(self.all_files)
        self.label_size = len(self.labels)

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        with open(self.all_files[idx][0]) as file:
            data = pd.read_csv(file)
            file.close()
        data = torch.tensor(data.to_numpy())
        data = data.type(torch.FloatTensor)
        return data, self.labels.index(self.all_files[idx][1])