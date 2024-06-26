import torch
from torch.utils.data import Dataset
import numpy as np

def _process_index_through_max_length(index, max_length):
    if (isinstance(index, slice)):
        new_index_start = index.start
        new_index_stop = index.stop
        new_index_step = index.step
        if (new_index_start is not None):
            new_index_start = (new_index_start % max_length)
        if (new_index_stop is not None):
            new_index_stop = (new_index_stop % max_length)
        index = slice(new_index_start, new_index_stop, index.step)
    else:
        index = index % max_length
    return index

class OriginalImageDataset(Dataset):
    def __init__(self, images_path, train_dataset_length=None, latent_path=None, reset_mmap=False):
        self.images_path = images_path
        self.images_data = np.load(images_path, mmap_mode='r')
        self.total_sequences = self.images_data.shape[0]
        self.dataset_len = self.total_sequences
        self.depth = self.images_data.shape[1]
        self.size = self.images_data.shape[2]
        self.reset_mmap = reset_mmap
        self.train_dataset_length = train_dataset_length
        if (self.train_dataset_length is None):
            self.train_dataset_length = self.total_sequences
        print(self.total_sequences, self.dataset_len, self.depth, self.size, self.train_dataset_length)

    def __getitem__(self, index):
        if self.reset_mmap:
            del self.images_data
            self.images_data = np.load(self.images_path, mmap_mode='r')
        index = _process_index_through_max_length(index, self.train_dataset_length)
        return torch.clamp(((torch.from_numpy(self.images_data[index].copy()).type(torch.FloatTensor) / 255.0) - 0.5) * 2.0, -1.0, 1.0)

    def __len__(self):
        return self.total_sequences


class ImageDataset(Dataset):
    def __init__(self, images_path, train_dataset_length=None, latent_path=None, reset_mmap=False):
        self.images_path = images_path
        self.images_data = np.load(images_path, mmap_mode='r')
        self.total_sequences = self.images_data.shape[0]
        self.dataset_len = self.total_sequences
        self.depth = self.images_data.shape[1]
        self.size = self.images_data.shape[2]
        self.reset_mmap = reset_mmap
        self.train_dataset_length = train_dataset_length
        if (self.train_dataset_length is None):
            self.train_dataset_length = self.total_sequences
        print(self.total_sequences, self.dataset_len, self.depth, self.size, self.train_dataset_length)

    def __getitem__(self, index):
        if self.reset_mmap:
            del self.images_data
            self.images_data = np.load(self.images_path, mmap_mode='r')
        index = _process_index_through_max_length(index, self.train_dataset_length)
        return torch.from_numpy(self.images_data[index].copy()).type(torch.FloatTensor)

    def __len__(self):
        return self.total_sequences

class ImageLabelDataset(Dataset):
    def __init__(self, semantic_path, label_path, train_dataset_length=None):
        self.semantic_data = np.load(semantic_path, mmap_mode='r')
        self.label_data = np.load(label_path, mmap_mode='r')
        self.total_sequences = self.semantic_data.shape[0]
        self.dataset_len = self.total_sequences
        self.latent_space = self.semantic_data.shape[1]
        self.train_dataset_length = train_dataset_length
        if (self.train_dataset_length is None):
            self.train_dataset_length = self.total_sequences
        print(self.total_sequences, self.dataset_len, self.latent_space, self.train_dataset_length)

    def __getitem__(self, index):
        index = _process_index_through_max_length(index, self.train_dataset_length)
        return torch.from_numpy(self.semantic_data[index].copy()).type(torch.FloatTensor), torch.from_numpy(self.label_data[index].copy()).type(torch.FloatTensor)

    def get_conds_mean(self):
        return self.semantic_data[:self.train_dataset_length].mean(axis=0)

    def get_conds_std(self):
        return self.semantic_data[:self.train_dataset_length].std(axis=0)

    def __len__(self):
        return self.total_sequences

class ImageSemanticDataset(Dataset):
    def __init__(self, semantic_path, train_dataset_length=None):
        self.semantic_data = np.load(semantic_path, mmap_mode='r')
        self.total_sequences = self.semantic_data.shape[0]
        self.dataset_len = self.total_sequences
        self.latent_space = self.semantic_data.shape[1]
        self.train_dataset_length = train_dataset_length
        if (self.train_dataset_length is None):
            self.train_dataset_length = self.total_sequences
        print(self.total_sequences, self.dataset_len, self.latent_space)

    def __getitem__(self, index):
        index = _process_index_through_max_length(index, self.train_dataset_length)
        return torch.from_numpy(self.semantic_data[index].copy()).type(torch.FloatTensor)

    def __len__(self):
        return self.total_sequences
