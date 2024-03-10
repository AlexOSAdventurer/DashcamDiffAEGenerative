import torch
from data import ImageDataset
import pytorch_lightning as pl
from lightning_training_model import DiffusionModel
import custom_diffusion as diffusion
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.cli import LightningCLI
import yaml
import numpy
import sys

# Training hyperparameters
#base_dir = "/work/cseos2g/papapalpi/data/images_labels"
diffae_model_name = sys.argv[1]
checkpoint_path = f"/work/cseos2g/papapalpi/code/DashcamDiffusionAutoencoder64x64New/saved_diffusion_models/{diffae_model_name}/model.ckpt"
config_data = yaml.safe_load(open(f"diffusion_model_64x64x3_{diffae_model_name}.yaml"))
dataset_path = sys.argv[2]
target_path = sys.argv[3]
batch_size = 32
device = 'cuda'
# Create datasets and data loaders
dataset = ImageDataset(dataset_path)

model = DiffusionModel.load_from_checkpoint(checkpoint_path, config=config_data).eval().to(device)

def convertData(dataset, new_path):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    output_memmap = numpy.lib.format.open_memmap(new_path, dtype=numpy.float, shape=(len(dataset), 512), mode='w+')
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            data = data.to(device)
            encoding = model.fetch_encoding(data, False)
            result = diffusion.encode_semantic(model.unet_autoencoder.encoder, encoding).cpu()
            output_memmap[(i * batch_size):((i * batch_size) + data.shape[0])] = result.numpy()
            print(i)

print(diffae_model_name)
convertData(dataset, target_path)
