import torch
from data import ImageDataset, ImageSemanticDataset
import pytorch_lightning as pl
from lightning_training_classifier_model import ClsModel
import custom_diffusion as diffusion
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.cli import LightningCLI
import yaml
import numpy
import sys
import math
import torchvision.utils
import torchvision.transforms.functional as tF

# Training hyperparameters
#base_dir = "/work/cseos2g/papapalpi/data/images_labels"
diffae_model_name = sys.argv[1]
checkpoint_path = f"/work/cseos2g/papapalpi/code/DashcamDiffusionAutoencoder64x64New/saved_classifier_models/{diffae_model_name}/model.ckpt"
config_data = yaml.safe_load(open(f"cls_diffusion_model_64x64x3_{diffae_model_name}.yaml"))
validation_dataset_image_path = sys.argv[2]
destination_file = sys.argv[3]
validation_dataset_indices = [18,19,20]
class_attribute_index = 2
device = 'cuda'
# Create datasets and data loaders
validation_dataset = ImageDataset(validation_dataset_image_path)

model = ClsModel.load_from_checkpoint(checkpoint_path, config=config_data).eval().to(device)

def doManipulations(validation_dataset, indices):
    with torch.no_grad():
        result_original_training_encoding = torch.empty((len(indices), 6, 64, 64))
        for i in range(len(indices)):
            original = validation_dataset[indices[i]]
            result_original_training_encoding[i] = original
        result_original_training = model.fetch_encoding(result_original_training_encoding.to(device), sample=False)
        print("Create original encoding!")
        original_encoding = diffusion.encode_semantic(model.diffae_model.unet_autoencoder.encoder, result_original_training)
        print("Create x_t!")

        x_t = diffusion.stochastic_encode_process_multiple_images(
        model.diffae_model.unet_autoencoder, result_original_training,
        original_encoding, model.diffae_model.t_range,
        model.diffae_model.beta_small, model.diffae_model.beta_large)

        print("Create new encoding!")
        original_encoding_norm = model.normalize(original_encoding)

        new_encoding_norm = original_encoding_norm + (0.5 * math.sqrt(512) * torch.nn.functional.normalize(model.classifier.weight[class_attribute_index][None, :], dim=1))

        new_encoding = model.denormalize(new_encoding_norm)

        print("Create reversed x_0!")
        result_modified_training = diffusion.denoise_process_multiple_images(model.diffae_model.unet_autoencoder, x_t, new_encoding,
        model.diffae_model.t_range, model.diffae_model.beta_small, model.diffae_model.beta_large)

        print("Processing output!")
        final_result_encoding = torch.cat((result_original_training, result_modified_training))
        final_result_256_x_256 = model.diffae_model.decode_encoding(final_result_encoding)

        final_result = torchvision.utils.make_grid(final_result_256_x_256, nrow=len(indices))
        final_result_img = tF.to_pil_image(final_result)
        final_result_img.save(destination_file)

doManipulations(validation_dataset, validation_dataset_indices)
