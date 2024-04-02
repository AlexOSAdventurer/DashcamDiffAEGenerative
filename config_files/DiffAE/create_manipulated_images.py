import sys
import os
def add_root_to_path():
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
add_root_to_path()
import torch
import numpy
import math
from pytorch_lightning import seed_everything
from lightning_training_classifier_model import ClsModel
from data import OriginalImageDataset
import custom_diffusion as diffusion
from torch.utils.data import DataLoader
import yaml
import subprocess
from multiprocessing import Pool
import torchvision.utils
import torchvision.transforms.functional as F

resolution = 256

def do_manipulations(model, images, latent_space, cls_index, result_path):
    with torch.no_grad():
        diffae_model = model.diffae_model
        images_64x64 = diffae_model.fetch_encoding(diffae_model.create_distribution(images), False)

        print("Create original encoding!")
        original_encoding = diffusion.encode_semantic(diffae_model.unet_autoencoder.encoder, images_64x64)
        print("Create x_t!")

        x_t = diffusion.stochastic_encode_process_multiple_images(
        diffae_model.unet_autoencoder, images_64x64,
        original_encoding, diffae_model.t_range,
        diffae_model.beta_small, diffae_model.beta_large)

        print("Create new encoding!")
        original_encoding_norm = model.normalize(original_encoding)
        new_encoding_norm_positive = original_encoding_norm + (0.5 * math.sqrt(latent_space) * torch.nn.functional.normalize(model.classifier.weight[cls_index][None, :], dim=1))
        new_encoding_norm_negative = original_encoding_norm - (0.5 * math.sqrt(latent_space) * torch.nn.functional.normalize(model.classifier.weight[cls_index][None, :], dim=1))
        new_encoding_positive = model.denormalize(new_encoding_norm_positive).float()
        new_encoding_negative = model.denormalize(new_encoding_norm_negative).float()

        print("Create reversed x_0!")
        result_modified_training_positive = diffusion.denoise_process_multiple_images(diffae_model.unet_autoencoder, x_t, new_encoding_positive,
        diffae_model.t_range, diffae_model.beta_small, diffae_model.beta_large)
        result_modified_training_negative = diffusion.denoise_process_multiple_images(diffae_model.unet_autoencoder, x_t, new_encoding_negative,
        diffae_model.t_range, diffae_model.beta_small, diffae_model.beta_large)

        print("Processing output!")
        final_result_encoding = torch.cat((images_64x64, result_modified_training_positive, result_modified_training_negative))
        final_result_256_x_256 = model.diffae_model.decode_encoding(final_result_encoding)

        final_result = torchvision.utils.make_grid(final_result_256_x_256, nrow=images.shape[0])
        final_result_img = F.to_pil_image(final_result)
        final_result_img.save(result_path)

def launch_creation(args):
    folder, i = args
    print(folder, i)
    batch_size = 48
    start_index = 200
    current_folder = os.path.dirname(__file__)
    root_folder = os.path.join(current_folder, "..", "..")
    folder = os.path.join(current_folder, "grid_search", folder)
    cls_config_path = os.path.join(folder, "cls_model.yaml")
    cls_checkpoint_path = os.path.join(folder, "cls_model.ckpt")
    config_data = yaml.safe_load(open(cls_config_path))
    validation_dataset_path = os.path.join(root_folder, "image_dataset_synthesis", "val_images.npy")
    val_dataset_images = OriginalImageDataset(validation_dataset_path, reset_mmap=True)
    with torch.no_grad():
        cuda_device = f"cuda:{i % 4}"
        model = ClsModel.load_from_checkpoint(cls_checkpoint_path, config=config_data).eval().to(cuda_device)
        latent_space = int(config_data["model"]["latent_space"])

        val_images = val_dataset_images[start_index:(start_index + batch_size)].to(cuda_device)
        destination_file_daytime = os.path.join(folder, "daytime.png")
        destination_file_highway = os.path.join(folder, "highway.png")
        destination_file_clouds = os.path.join(folder, "clouds.png")

        do_manipulations(model, val_images, latent_space, 0, destination_file_daytime)
        do_manipulations(model, val_images, latent_space, 1, destination_file_highway)
        do_manipulations(model, val_images, latent_space, 2, destination_file_clouds)
        del model
        torch.cuda.empty_cache()

if __name__ == '__main__':
    parallel_training_units = 2
    current_folder = os.path.dirname(__file__)
    torch.set_float32_matmul_precision('medium')
    torch.manual_seed(1234567)
    seed_everything(1234567, workers=True)
    grid_search_folder = os.path.join(current_folder, "grid_search")
    configuration_folders = os.listdir(grid_search_folder)
    configuration_folders.sort(reverse=True)
    with Pool(parallel_training_units) as p:
        p.map(launch_creation, zip(configuration_folders, range(len(configuration_folders))))