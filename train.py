import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = os.environ['USER_SET_MASTER_PORT']
os.environ['PL_TORCH_DISTRIBUTED_BACKEND'] = 'gloo'
import torch
from data import ImageDataset, OriginalImageDataset
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer, seed_everything
from lightning_training_model import DiffusionModel
from torch.utils.data import DataLoader
import glob
from pytorch_lightning.cli import LightningCLI
import yaml
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint

config_file = os.environ['AUTOENCODER_CONFIG']
config_data = yaml.safe_load(open(config_file))
config_choice = os.path.dirname(config_file)
base_dir = os.path.dirname(__file__)
dataset_path_train = os.path.join(base_dir, config_data['data']['train'])
dataset_path_val = os.path.join(base_dir, config_data['data']['val'])
dataset_train_dataset_length = int(config_data['data']['train_dataset_length'])

# Loading parameters
load_model = config_data['model']['recycle_previous_version']
load_version_num = config_data['model']['previous_version']

# Code for optionally loading model
last_checkpoint = None

if load_model:
    last_checkpoint = glob.glob(
        f"./lightning_logs/{config_choice}/version_{load_version_num}/checkpoints/*.ckpt"
    )[-1]

dataset_type = ImageDataset if config_data['model']['first_stage_needed'] else OriginalImageDataset

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    # Create datasets and data loaders
    train_dataset = dataset_type(dataset_path_train, dataset_train_dataset_length)
    val_dataset = dataset_type(dataset_path_val)

    train_loader = DataLoader(train_dataset, batch_size=config_data["model"]["batch_size"], num_workers=1, shuffle=True, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=8, num_workers=1, shuffle=False, pin_memory=True, persistent_workers=True)
    seed_everything(1234567, workers=True)
    ddp = DDPStrategy(process_group_backend="gloo", find_unused_parameters=True)

    if load_model:
        model = DiffusionModel.load_from_checkpoint(last_checkpoint, config=config_data)
        model.generate_first_stage() #Overwrite the old first stage autoencoder if we've changed first stages for some reason
    else:
        model = DiffusionModel(config_data)

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
        check_val_every_n_epoch=25,
        max_steps=int(config_data['model']['total_samples']) // (int(config_data['model']['batch_size']) * int(config_data['model']['gpus']) * int(config_data['model']['nodes'])),
        precision="bf16-mixed",
        strategy=ddp,
        logger=tb_logger,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, ImageDataModule())
    