import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import glob
from pytorch_lightning.cli import LightningCLI
import yaml
import numpy
import os
import sys

if __name__ == '__main__':

    def add_root_to_path():
        sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    add_root_to_path()
    from data import OriginalImageDataset as ImageDataset
    import first_stage_autoencoder

    base_dir = os.path.dirname(__file__)
    dataset_path_train = os.path.join(base_dir, "train_images.npy")
    dataset_path_val = os.path.join(base_dir, "val_images.npy")
    new_dataset_path_train = os.path.join(base_dir, "train_images_64x64_latent.npy")
    new_dataset_path_val = os.path.join(base_dir, "val_images_64x64_latent.npy")
    batch_size = 32
    device = 'cuda'
    # Create datasets and data loaders
    train_dataset = ImageDataset(dataset_path_train)
    val_dataset = ImageDataset(dataset_path_val)

    autoencoder_model = first_stage_autoencoder.generate_pretrained_model()
    autoencoder_model = autoencoder_model.eval().to(device)

    def convertData(dataset, new_path):
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)
        output_memmap = numpy.lib.format.open_memmap(new_path, dtype=float, shape=(len(dataset), 6, 64, 64), mode='w+')
        with torch.no_grad():
            for i, data in enumerate(loader, 0):
                data = data.to(device)
                result = autoencoder_model.encode_raw(data).to('cpu')
                output_memmap[(i * batch_size):((i * batch_size) + data.shape[0])] = result.numpy()
                print(i)

    print("Train")
    convertData(train_dataset, new_dataset_path_train)
    print("Val")
    convertData(val_dataset, new_dataset_path_val)