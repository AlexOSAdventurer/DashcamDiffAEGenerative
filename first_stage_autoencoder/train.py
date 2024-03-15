import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ['PL_TORCH_DISTRIBUTED_BACKEND'] = 'gloo'
import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader
import glob
from pytorch_lightning.cli import LightningCLI
import yaml
import model
import sys
from pytorch_lightning.strategies import DDPStrategy

def add_root_to_path():
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

add_root_to_path()
from data import ImageDataset, OriginalImageDataset


'''
cli = LightningCLI(
        model_class=getModel,
        datamodule_class=ImageDataModule,
        seed_everything_default=123,
        trainer_defaults=dict(
            accelerator="gpu",
            max_steps=int(config_data['model']['total_samples']) // (int(config_data['model']['batch_size']) * int(config_data['model']['gpus']) * int(config_data['model']['nodes'])),
            precision=16,
            strategy="ddp",
            logger=tb_logger
        ),
)

#cli.trainer.fit(cli.model, datamodule=cli.datamodule)
'''
if __name__ == '__main__':
    # Training hyperparameters
    torch.set_float32_matmul_precision('medium')
    base_dir = os.path.dirname(__file__)
    config_file = os.path.join(base_dir, "autoencoder_kl_64x64x3.yaml")
    config_data = yaml.safe_load(open(config_file))
    dataset_path_train = os.path.join(base_dir, "..", "image_dataset_synthesis", "train_images.npy")
    dataset_path_val = os.path.join(base_dir, "..", "image_dataset_synthesis", "val_images.npy")

    # Create datasets and data loaders
    train_dataset = OriginalImageDataset(dataset_path_train)
    val_dataset = OriginalImageDataset(dataset_path_val)

    train_loader = DataLoader(train_dataset, batch_size=config_data["model"]["batch_size"], num_workers=2, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, num_workers=2, shuffle=False, pin_memory=True)
    seed_everything(123, workers=True)
    ddp = DDPStrategy(process_group_backend="gloo", find_unused_parameters=True)
    model = model.AutoencoderKL(ddconfig=config_data['model']['params']['ddconfig'],
                 lossconfig=config_data['model']['params']['lossconfig'],
                 embed_dim=config_data['model']['params']['embed_dim'],
                 base_learning_rate=config_data['model']['base_learning_rate'])

    # Load Trainer model
    tb_logger = pl.loggers.TensorBoardLogger(
        "lightning_logs/",
        name="pretrained_autoencoder",
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
    trainer = Trainer(
        accelerator="gpu",
        max_steps=int(config_data['model']['total_samples']) // (int(config_data['model']['batch_size']) * int(config_data['model']['gpus']) * int(config_data['model']['nodes'])),
        precision="16-mixed",
        num_nodes = int(config_data['model']['nodes']),
        strategy=ddp,
        logger=tb_logger
    )
    trainer.fit(model, ImageDataModule())