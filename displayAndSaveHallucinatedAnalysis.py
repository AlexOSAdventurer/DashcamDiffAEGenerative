import torch
from data import ImageDataset, ImageSemanticDataset
import pytorch_lightning as pl
from lightning_training_model import DiffusionModel
import custom_diffusion as diffusion
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.cli import LightningCLI
import yaml
import numpy
import sys
import torchvision.utils
import torchvision.transforms.functional as F

# Training hyperparameters
#base_dir = "/work/cseos2g/papapalpi/data/images_labels"
diffae_model_name = sys.argv[1]
checkpoint_path = f"/work/cseos2g/papapalpi/code/DashcamDiffusionAutoencoder64x64New/saved_diffusion_models/{diffae_model_name}/model.ckpt"
config_data = yaml.safe_load(open(f"diffusion_model_64x64x3_{diffae_model_name}.yaml"))
training_dataset_image_path = sys.argv[2]
training_dataset_semantic_path = sys.argv[3]
hallucinated_dataset_path = sys.argv[4]
destination_file = sys.argv[5]
hallucinated_dataset_indices = [793,420,99,75]
device = 'cuda'
# Create datasets and data loaders
training_dataset = ImageSemanticDataset(training_dataset_image_path, training_dataset_semantic_path)
hallucinated_dataset = ImageDataset(hallucinated_dataset_path)

model = DiffusionModel.load_from_checkpoint(checkpoint_path, config=config_data).eval().to(device)

def find_min(hallucinated, training_dataset, semantic_or_pixel="pixel"):
    current_mse = None
    current_image = None
    hallucinated_value = None
    if (semantic_or_pixel == "pixel"):
        hallucinated_value = hallucinated.reshape(1,3,256,256)
    elif (semantic_or_pixel == "semantic"):
        hallucinated_64x64 = model.fetch_encoding(model.create_distribution(hallucinated.reshape(1, 3, 256, 256).to(device)), sample=False)
        hallucinated_value = diffusion.encode_semantic(model.unet_autoencoder.encoder, hallucinated_64x64).reshape(1,512).cpu()
    for i in range(len(training_dataset)):
        print(i)
        pixel, semantic = training_dataset[i]
        training_value = None
        if (semantic_or_pixel == "pixel"):
            training_value = pixel.reshape(1,3,256,256)
        elif (semantic_or_pixel == "semantic"):
            training_value = semantic.reshape(1,512)
        current_mse_loss = float(torch.nn.functional.mse_loss(hallucinated_value, training_value))
        if ((current_mse is None) or (current_mse_loss < current_mse)):
            current_mse = current_mse_loss
            current_image = pixel
    return current_image


def doComparisons(training_dataset, hallucinated_dataset, indices):
    result_original_hallucinated = torch.empty((len(indices), 3, 256, 256))
    result_mse_pixel = torch.empty((len(indices), 3, 256, 256))
    result_mse_semantic = torch.empty((len(indices), 3, 256, 256))
    for i in range(len(indices)):
        hallucinated = hallucinated_dataset[indices[i]]
        result_original_hallucinated[i] = hallucinated
        result_mse_semantic[i] = find_min(hallucinated, training_dataset, "semantic")
        result_mse_pixel[i] = find_min(hallucinated, training_dataset, "pixel")
        print(i)
    final_result = torchvision.utils.make_grid(torch.cat((result_original_hallucinated, result_mse_pixel, result_mse_semantic)), nrow=len(indices))
    final_result_img = F.to_pil_image(final_result)
    final_result_img.save(destination_file)

doComparisons(training_dataset, hallucinated_dataset, hallucinated_dataset_indices)
