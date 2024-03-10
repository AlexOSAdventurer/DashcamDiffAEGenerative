import torch
from torch.utils.data import Dataset
import numpy as np

class OriginalImageDataset(Dataset):
    def __init__(self, images_path):
        self.images_data = np.load(images_path, mmap_mode='r')
        self.total_sequences = self.images_data.shape[0]
        self.dataset_len = self.total_sequences
        self.sequence_length = self.images_data.shape[1]
        self.depth = self.images_data.shape[2]
        self.size = self.images_data.shape[3]
        print(self.total_sequences, self.dataset_len, self.depth, self.size)

    def __getitem__(self, index):
        return torch.clamp(((torch.from_numpy(self.images_data[index].copy()).type(torch.FloatTensor) / 255.0) - 0.5) * 2.0, -1.0, 1.0)

    def __len__(self):
        return self.total_sequences


class ImageDataset(Dataset):
    def __init__(self, images_path):
        self.images_data = np.load(images_path, mmap_mode='r')
        self.total_sequences = self.images_data.shape[0]
        self.dataset_len = self.total_sequences
        self.sequence_length = self.images_data.shape[1]
        self.depth = self.images_data.shape[2]
        self.size = self.images_data.shape[3]
        print(self.total_sequences, self.dataset_len, self.depth, self.size)

    def __getitem__(self, index):
        return torch.from_numpy(self.images_data[index].copy()).type(torch.FloatTensor)
        #return torch.clamp(((torch.from_numpy(self.images_data[index].copy()).type(torch.FloatTensor) / 255.0) - 0.5) * 2.0, -1.0, 1.0)

    def __len__(self):
        return self.total_sequences

class DiffusionLatentImageDataset(Dataset):
    def __init__(self, images_path, latents_path):
        self.images_dataset = ImageDataset(images_path)
        self.latents_data = np.load(latents_path, mmap_mode='r')
        self.total_sequences = self.latents_data.shape[0]
        self.sequence_length = self.latents_data.shape[1]
        self.size = self.latents_data.shape[2]
        print(self.total_sequences, self.sequence_length, self.size)

    def __getitem__(self, index):
        return torch.from_numpy(self.latents_data[index].copy()).type(torch.FloatTensor), self.images_dataset[index]

    def __len__(self):
        return self.total_sequences
