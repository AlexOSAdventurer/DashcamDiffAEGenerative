import torch
from data import ImageDataset, OriginalImageDataset, DiffusionLatentImageDataset
import pytorch_lightning as pl
from lightning_training_model import VideoPredictionModel
from torch.utils.data import DataLoader
import glob
from pytorch_lightning.utilities.cli import LightningCLI
import yaml

# Training hyperparameters
dataset_choice = "AIinCPS"
base_dir = "/work/cseos2g/papapalpi/"
config_data = yaml.safe_load(open("video_model_64x64x3.yaml"))
base_dir = config_data['data']['base_dir']
dataset_path_train_images = base_dir + config_data['data']['train_images']
dataset_path_val_images = base_dir + config_data['data']['val_images']
dataset_path_train_latents = base_dir + config_data['data']['train_latents']
dataset_path_val_latents = base_dir + config_data['data']['val_latents']

# Loading parameters
load_model = False
load_version_num = 62

# Code for optionally loading model
last_checkpoint = None

if load_model:
    last_checkpoint = glob.glob(
        f"./lightning_logs/{dataset_choice}/version_{load_version_num}/checkpoints/*.ckpt"
    )[-1]

dataset_type = DiffusionLatentImageDataset
# Create datasets and data loaders
train_dataset = dataset_type(dataset_path_train_images, dataset_path_train_latents)
val_dataset = dataset_type(dataset_path_val_images, dataset_path_val_latents)

train_loader = DataLoader(train_dataset, batch_size=config_data["model"]["batch_size"], num_workers=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, num_workers=16, shuffle=False)

if load_model:
    model = VideoPredictionModel.load_from_checkpoint(last_checkpoint, config=config_data)
else:
    model = VideoPredictionModel(config_data)

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

def runCLI():
    cli = LightningCLI(
            description="PyTorch Video Next-Frame Prediction Model Using the Semantic Latents of a Diffusion Autoencoder as the basis",
            model_class=getModel,
            datamodule_class=ImageDataModule,
            seed_everything_default=123,
            save_config_overwrite=True,
            trainer_defaults=dict(
                devices=2,
                num_nodes=1,
                accelerator="gpu",
                max_epochs=4000,
                precision=16,
                strategy="ddp",
                logger=tb_logger
            ),
    )

    cli.trainer.fit(cli.model, datamodule=cli.datamodule)

def runNOCLI():
    model = getModel()

    trainer = pl.Trainer(
        gpus=2,
        num_nodes=6,
        strategy="ddp",
        max_epochs=-1,
        logger=tb_logger
    )

    trainer.fit(model, train_loader, val_loader)

runNOCLI()
