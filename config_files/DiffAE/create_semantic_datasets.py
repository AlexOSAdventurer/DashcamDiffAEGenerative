import sys
import os
def add_root_to_path():
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
add_root_to_path()
import torch
import numpy
from lightning_training_model import DiffusionModel
from data import ImageDataset
import custom_diffusion as diffusion
from torch.utils.data import DataLoader
import yaml
import subprocess
from multiprocessing import Pool

def convertData(model, dataset, new_path, latent_space, device):
    batch_size = 128
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    output_memmap = numpy.lib.format.open_memmap(new_path, dtype=float, shape=(len(dataset), latent_space), mode='w+')
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            data = data.to(device)
            encoding = model.fetch_encoding(data, False)
            result = diffusion.encode_semantic(model.unet_autoencoder.encoder, encoding).cpu()
            output_memmap[(i * batch_size):((i * batch_size) + data.shape[0])] = result.numpy()
            print(i)

def launch_conversion(args):
    folder, i = args
    print(folder, i)
    current_folder = os.path.dirname(__file__)
    root_folder = os.path.join(current_folder, "..", "..")
    folder = os.path.join(current_folder, "grid_search", folder)
    diffusion_config_path = os.path.join(folder, "diffusion_model.yaml")
    diffusion_checkpoint_path = os.path.join(folder, "diffusion_model.ckpt")
    config_data = yaml.safe_load(open(diffusion_config_path))
    dataset_path_train = os.path.join(root_folder, config_data['data']['train'])
    dataset_path_val = os.path.join(root_folder, config_data['data']['val'])
    dataset_path_train_new = os.path.join(folder, "train_semantic.npy")
    dataset_path_val_new = os.path.join(folder, "val_semantic.npy")

    cuda_device = f"cuda:{i % 4}"
    model = DiffusionModel.load_from_checkpoint(diffusion_checkpoint_path, config=config_data).eval().to(cuda_device)
    latent_space = int(yaml.safe_load(open(config_data["model"]["unet_config"]))["encoder_model"]["latent_space"])
    dataset_type = ImageDataset if config_data["model"]["first_stage_needed"] else OriginalImageDataset

    train_dataset = dataset_type(dataset_path_train, reset_mmap=True)
    val_dataset = dataset_type(dataset_path_val, reset_mmap=True)
    convertData(model, train_dataset, dataset_path_train_new, latent_space, cuda_device)
    convertData(model, val_dataset, dataset_path_val_new, latent_space, cuda_device)

if __name__ == '__main__':
    parallel_training_units = 12
    current_folder = os.path.dirname(__file__)
    torch.set_float32_matmul_precision('medium')
    grid_search_folder = os.path.join(current_folder, "grid_search")
    configuration_folders = os.listdir(grid_search_folder)
    configuration_folders.sort(reverse=True)
    with Pool(parallel_training_units) as p:
        p.map(launch_conversion, zip(configuration_folders, range(len(configuration_folders))))