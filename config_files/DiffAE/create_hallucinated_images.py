import sys
import os
def add_root_to_path():
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
add_root_to_path()
import torch
import numpy
from pytorch_lightning import seed_everything
from lightning_training_latent_model import LatentModel
from data import OriginalImageDataset, ImageSemanticDataset
import custom_diffusion as diffusion
from torch.utils.data import DataLoader
import yaml
import subprocess
from multiprocessing import Pool
import torchvision.utils
import torchvision.transforms.functional as F

resolution = 256

def find_min(model, hallucinated, training_dataset, latent_space, semantic_or_pixel="pixel"):
    current_mse = None
    current_index = None
    hallucinated_value = None
    if (semantic_or_pixel == "pixel"):
        hallucinated_value = hallucinated.reshape(1,3,resolution,resolution)
    elif (semantic_or_pixel == "semantic"):
        hallucinated_64x64 = model.fetch_encoding(model.diffae_model.create_distribution(hallucinated.reshape(1, 3, resolution, resolution).to(model.device)), sample=False)
        hallucinated_value = diffusion.encode_semantic(model.diffae_model.unet_autoencoder.encoder, hallucinated_64x64).reshape(1,latent_space).cpu()
        del hallucinated_64x64
    for i in range(len(training_dataset)):
        print(i)
        pixel_or_semantic = training_dataset[i]
        training_value = None
        if (semantic_or_pixel == "pixel"):
            training_value = (pixel_or_semantic.reshape(1,3,resolution, resolution).cpu().clamp(-1, 1) + 1) / 2.0
        elif (semantic_or_pixel == "semantic"):
            training_value = pixel_or_semantic.reshape(1,latent_space).cpu()
        new_mse = float(torch.nn.functional.mse_loss(hallucinated_value, training_value))
        if ((current_mse is None) or (new_mse < current_mse)):
            current_mse = new_mse
            current_index = i
    return current_index

def do_comparisons(model, train_dataset_images, train_dataset_semantic, unconditional_samples, latent_space, destination_file):
    number_of_images = unconditional_samples.shape[0]
    result_mse_pixel = torch.empty((number_of_images, 3, resolution, resolution))
    result_mse_semantic = torch.empty((number_of_images, 3, resolution, resolution))
    for i in range(number_of_images):
        hallucinated = unconditional_samples[i]
        pixel_image = (train_dataset_images[find_min(model, hallucinated, train_dataset_images, latent_space, "pixel")].reshape(1, 3, resolution, resolution).clamp(-1, 1) + 1) / 2.0
        semantic_image = (train_dataset_images[find_min(model, hallucinated, train_dataset_semantic, latent_space, "semantic")].reshape(1, 3, resolution, resolution).clamp(-1, 1) + 1) / 2.0
        result_mse_semantic[i] = semantic_image.reshape(3, resolution, resolution)
        result_mse_pixel[i] = pixel_image.reshape(3, resolution, resolution)

    final_result = torchvision.utils.make_grid(torch.cat((unconditional_samples, result_mse_pixel, result_mse_semantic)), nrow=number_of_images)
    final_result_img = F.to_pil_image(final_result)
    final_result_img.save(destination_file)

def launch_creation(args):
    folder, i = args
    print(folder, i)
    batch_size = 8
    current_folder = os.path.dirname(__file__)
    root_folder = os.path.join(current_folder, "..", "..")
    folder = os.path.join(current_folder, "grid_search", folder)
    diffusion_config_path = os.path.join(folder, "latent_model.yaml")
    diffusion_checkpoint_path = os.path.join(folder, "latent_model.ckpt")
    destination_file = os.path.join(folder, "hallucinated.png")
    config_data = yaml.safe_load(open(diffusion_config_path))
    train_dataset_length = int(config_data['data']['train_dataset_length'])
    dataset_path_train_images = os.path.join(root_folder, "image_dataset_synthesis", "train_images.npy")
    dataset_path_train_semantic = os.path.join(root_folder, config_data['data']['train'])
    train_dataset_images = OriginalImageDataset(dataset_path_train_images, train_dataset_length=train_dataset_length, reset_mmap=True)
    train_dataset_semantic = ImageSemanticDataset(dataset_path_train_semantic, train_dataset_length=train_dataset_length)

    with torch.no_grad():
        cuda_device = f"cuda:{i % 4}"
        model = LatentModel.load_from_checkpoint(diffusion_checkpoint_path, config=config_data).eval().to(cuda_device)
        latent_space = int(config_data["model"]["latent_space"])

        unconditional_samples = model.create_unconditional_samples(batch_size).cpu()
        do_comparisons(model, train_dataset_images, train_dataset_semantic, unconditional_samples, latent_space, destination_file)
        del model
        torch.cuda.empty_cache()

if __name__ == '__main__':
    parallel_training_units = 4
    current_folder = os.path.dirname(__file__)
    torch.set_float32_matmul_precision('medium')
    torch.manual_seed(1234567)
    seed_everything(1234567, workers=True)
    grid_search_folder = os.path.join(current_folder, "grid_search")
    configuration_folders = os.listdir(grid_search_folder)
    configuration_folders.sort(reverse=True)
    with Pool(parallel_training_units) as p:
        p.map(launch_creation, zip(configuration_folders, range(len(configuration_folders))))