import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = os.environ['USER_SET_MASTER_PORT']
os.environ['PL_TORCH_DISTRIBUTED_BACKEND'] = 'gloo'
import torch
from data import ImageDataset, OriginalImageDataset, ImageSemanticDataset
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer, seed_everything
from lightning_training_latent_model import LatentModel
from torch.utils.data import DataLoader
import glob
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
import yaml

config_file = os.environ['LATENT_DDIM_CONFIG']
config_data = yaml.safe_load(open(config_file))
config_choice = os.path.dirname(config_file)
base_dir = os.path.dirname(__file__)
dataset_path_train = os.path.join(base_dir, config_data['data']['train'])
dataset_path_val = os.path.join(base_dir, config_data['data']['val'])
dataset_train_dataset_length = int(config_data['data']['train_dataset_length'])
diffusion_config = yaml.safe_load(open(config_data["diffae"]["diffae_config"]))
dataset_type = ImageSemanticDataset

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    # Create datasets and data loaders
    train_dataset = dataset_type(dataset_path_train, dataset_train_dataset_length)
    val_dataset = dataset_type(dataset_path_val)

    train_loader = DataLoader(train_dataset, batch_size=config_data["model"]["batch_size"], num_workers=1, shuffle=True, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=8, num_workers=1, shuffle=False, pin_memory=True, persistent_workers=True)
    seed_everything(1234567, workers=True)
    ddp = DDPStrategy(process_group_backend="gloo", find_unused_parameters=True)
    model = LatentModel(config_data)

    # Load Trainer model
    tb_logger = pl.loggers.TensorBoardLogger(
        "lightning_logs/",
        name=config_choice,
        version=None,
    )

    class ImageDataModule(pl.LightningDataModule):
        def __init__(
            self,
            batch_size: int = 4,
            workers: int = 1,
            **kwargs,
        ):
            super().__init__()

        def train_dataloader(self):
            return train_loader

        def val_dataloader(self):
            return val_loader

        def test_dataloader(self):
            return val_loader

    checkpoint_callback = ModelCheckpoint(every_n_epochs=1, save_on_train_epoch_end=True)
    trainer = Trainer(
        devices=int(config_data['model']['gpus']),
        accelerator="gpu",
        num_nodes = int(config_data['model']['nodes']),
        check_val_every_n_epoch=125,
        max_steps=int(config_data['model']['total_samples']) // (int(config_data['model']['batch_size']) * int(config_data['model']['gpus']) * int(config_data['model']['nodes'])),
        precision="bf16-mixed",
        strategy=ddp,
        logger=tb_logger,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, ImageDataModule())