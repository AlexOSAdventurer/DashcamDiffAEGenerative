import torch
from data import ImageLabelDataset
import pytorch_lightning as pl
from lightning_training_classifier_model import ClsModel
from torch.utils.data import DataLoader
import glob
from pytorch_lightning.utilities.cli import LightningCLI
import yaml

filename = "config_files/cls_diffusion_model_64x64x3_our_largest_model.yaml"
dataset_choice = f"RealCLS_{filename}"
base_dir = "/work/cseos2g/papapalpi/"
config_data = yaml.safe_load(open(filename))
base_dir = config_data['data']['base_dir']
dataset_path_train = base_dir + config_data['data']['train']
dataset_path_val = base_dir + config_data['data']['val']
dataset_path_train_labels = base_dir + config_data['data']['train_labels']
dataset_path_val_labels = base_dir + config_data['data']['val_labels']

# Loading parameters
load_model = False
load_version_num = 17

# Code for optionally loading model
last_checkpoint = None

if load_model:
    last_checkpoint = glob.glob(
        f"./lightning_logs/{dataset_choice}/version_{load_version_num}/checkpoints/*.ckpt"
    )[-1]

dataset_type = ImageLabelDataset

# Create datasets and data loaders
train_dataset = dataset_type(dataset_path_train, dataset_path_train_labels)
val_dataset = dataset_type(dataset_path_val, dataset_path_val_labels)

train_loader = DataLoader(train_dataset, batch_size=config_data["model"]["batch_size"], num_workers=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config_data["model"]["batch_size"], num_workers=16, shuffle=False)

if load_model:
    model = ClsModel.load_from_checkpoint(last_checkpoint, config=config_data)
else:
    model = ClsModel(config_data)

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
        description="PyTorch Latent Classifier",
        model_class=getModel,
        datamodule_class=ImageDataModule,
        seed_everything_default=123,
        save_config_overwrite=True,
        trainer_defaults=dict(
            accelerator="gpu",
            max_epochs=200,
            precision=16,
            strategy="ddp",
            logger=tb_logger
        ),
)

cli.trainer.fit(cli.model, datamodule=cli.datamodule)
