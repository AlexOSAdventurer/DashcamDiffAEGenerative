import torch
from data import ImageDataset, OriginalImageDataset
import pytorch_lightning as pl
from lightning_training_latent_model import LatentModel
from torch.utils.data import DataLoader
import glob
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.strategies import *
import yaml
import os

#filename = "config_files/latent_diffusion_model_64x64x3_our_largest_model.yaml"
filename = os.environ['LATENT_DDIM_CONFIG']
# Training hyperparameters
#dataset_choice = "BergerAblation_1to4_Channels_64_8_Attn_DDI"
dataset_choice = f"{filename}"
base_dir = "/work/cseos2g/papapalpi/"
config_data = yaml.safe_load(open(filename))
base_dir = config_data['data']['base_dir']
dataset_path_train = base_dir + config_data['data']['train']
dataset_path_val = base_dir + config_data['data']['val']

# Loading parameters
load_model = False
load_version_num = 17

# Code for optionally loading model
last_checkpoint = None

if load_model:
    last_checkpoint = glob.glob(
        f"./lightning_logs/{dataset_choice}/version_{load_version_num}/checkpoints/*.ckpt"
    )[-1]

dataset_type = ImageDataset

# Create datasets and data loaders
train_dataset = dataset_type(dataset_path_train)
val_dataset = dataset_type(dataset_path_val)

train_loader = DataLoader(train_dataset, batch_size=config_data["model"]["batch_size"], num_workers=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config_data["model"]["batch_size"], num_workers=16, shuffle=False)

if load_model:
    model = LatentModel.load_from_checkpoint(last_checkpoint, config=config_data)
else:
    model = LatentModel(config_data)

# Load Trainer model
tb_logger = pl.loggers.TensorBoardLogger(
    "lightning_logs/",
    name=dataset_choice,
    version=None,
)

def getModel() -> pl.LightningModule:
    return model

class ImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 4,
        workers: int = 8,
        **kwargs,
    ):
        super().__init__()

    def train_dataloader(self):
        return train_loader

    def val_dataloader(self):
        return val_loader

    def test_dataloader(self):
        return val_loader

cli = LightningCLI(
        model_class=getModel,
        datamodule_class=ImageDataModule,
        seed_everything_default=123,
        trainer_defaults=dict(
            accelerator="gpu",
            max_steps=int(config_data['model']['total_samples']) // (int(config_data['model']['batch_size']) * 8),
            precision=16,
            strategy=DDPStrategy(find_unused_parameters=True),
            logger=tb_logger
        ),
)

cli.trainer.fit(cli.model, datamodule=cli.datamodule)
