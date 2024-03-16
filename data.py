import torch
from torch.utils.data import Dataset
import numpy as np

class OriginalImageDataset(Dataset):
    def __init__(self, images_path, train_dataset_length=None, latent_path=None):
        self.images_path = images_path
        self.images_data = np.load(images_path, mmap_mode='r')
        self.total_sequences = self.images_data.shape[0]
        self.dataset_len = self.total_sequences
        self.depth = self.images_data.shape[1]
        self.size = self.images_data.shape[2]
        self.train_dataset_length = train_dataset_length
        if (self.train_dataset_length is None):
            self.train_dataset_length = self.total_sequences
        print(self.total_sequences, self.dataset_len, self.depth, self.size, self.train_dataset_length)

    def __getitem__(self, index):
        #del self.images_data
        #self.images_data = np.load(self.images_path, mmap_mode='r')
        return torch.clamp(((torch.from_numpy(self.images_data[(index % self.train_dataset_length)].copy()).type(torch.FloatTensor) / 255.0) - 0.5) * 2.0, -1.0, 1.0)

    def __len__(self):
        return self.total_sequences


class ImageDataset(Dataset):
    def __init__(self, images_path, train_dataset_length=None, latent_path=None):
        self.images_path = images_path
        self.images_data = np.load(images_path, mmap_mode='r')
        self.total_sequences = self.images_data.shape[0]
        self.dataset_len = self.total_sequences
        self.depth = self.images_data.shape[1]
        self.size = self.images_data.shape[2]
        self.train_dataset_length = train_dataset_length
        if (self.train_dataset_length is None):
            self.train_dataset_length = self.total_sequences
        print(self.total_sequences, self.dataset_len, self.depth, self.size, self.train_dataset_length)

    def __getitem__(self, index):
        #del self.images_data
        #self.images_data = np.load(self.images_path, mmap_mode='r')
        return torch.from_numpy(self.images_data[(index % self.train_dataset_length)].copy()).type(torch.FloatTensor)

    def __len__(self):
        return self.total_sequences

class ImageLabelDataset(Dataset):
    def __init__(self, images_path, label_path):
        self.images_data = np.load(images_path, mmap_mode='r')
        self.label_data = np.load(label_path, mmap_mode='r')
        self.total_sequences = self.images_data.shape[0]
        self.dataset_len = self.total_sequences
        self.depth = self.images_data.shape[1]
        self.size = self.images_data.shape[2]
        print(self.total_sequences, self.dataset_len, self.depth, self.size)

    def __getitem__(self, index):
        return torch.from_numpy(self.images_data[index].copy()).type(torch.FloatTensor), torch.from_numpy(self.label_data[index].copy()).type(torch.FloatTensor)

    def __len__(self):
        return self.total_sequences

class ImageSemanticDataset(Dataset):
    def __init__(self, images_path, semantic_path):
        self.images_data = np.load(images_path, mmap_mode='r')
        self.semantic_data = np.load(semantic_path, mmap_mode='r')
        self.total_sequences = self.images_data.shape[0]
        self.dataset_len = self.total_sequences
        self.depth = self.images_data.shape[1]
        self.size = self.images_data.shape[2]
        print(self.total_sequences, self.dataset_len, self.depth, self.size)

    def __getitem__(self, index):
        return torch.clamp(((torch.from_numpy(self.images_data[index].copy()).type(torch.FloatTensor) / 255.0)), 0.0, 1.0), torch.from_numpy(self.semantic_data[index].copy()).type(torch.FloatTensor)

    def __len__(self):
        return self.total_sequences
