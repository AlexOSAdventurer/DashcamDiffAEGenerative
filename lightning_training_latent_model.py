import torch
import torch.nn as nn
import pytorch_lightning as pl
import math
import sys
import numpy as np
import torch.nn.functional as F
import yaml
import first_stage_autoencoder
from first_stage_autoencoder.distribution import DiagonalGaussianDistribution
import unet_autoencoder
import custom_diffusion as diffusion
from lightning_training_model import DiffusionModel

class LatentModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.diffae_model_config = yaml.safe_load(open(config["diffae"]["diffae_config"]))
        self.beta_small = self.diffae_model_config["model"]["beta_small"]
        self.beta_large = self.diffae_model_config["model"]["beta_large"]
        self.t_range = self.diffae_model_config["model"]["t_range"]
        self.diffae_checkpoint_file = config["diffae"]["checkpoint_file"]
        self.batch_size = config["model"]["batch_size"]
        self.latent_space = config["model"]["latent_space"]
        self.save_hyperparameters()
        self.diffae_model = DiffusionModel.load_from_checkpoint(self.diffae_checkpoint_file, config=self.diffae_model_config)
        self.diffae_model.generate_first_stage() #Overwrite the old first stage autoencoder
        self.diffae_model = self.diffae_model.eval()
        self.latent_model = unet_autoencoder.generate_ddim_model(self.latent_space)

    def fetch_encoding(self, batch, sample):
        return self.diffae_model.fetch_encoding(batch, sample)

    def decode_encoding(self, encoding):
        return self.diffae_model.decode_encoding(encoding)

    def forward(self, x, t):
        return diffusion.estimate_noise(self.latent_model, x, t)

    def get_loss(self, x, batch_idx):
        number_of_images = x.shape[0]
        time_steps = diffusion.create_random_time_steps(number_of_images, self.t_range, self.device)
        noised_zsem, source_noise = diffusion.diffuse_images(x, time_steps, self.t_range, self.beta_small, self.beta_large)
        estimated_noise = self.forward(noised_zsem, time_steps)
        return F.mse_loss(estimated_noise, source_noise)

    def training_step(self, batch, batch_idx):
        '''
            encoding = self.fetch_encoding(batch, False)
            z_sem = diffusion.encode_semantic(self.diffae_model.unet_autoencoder.encoder, encoding)
        '''
        z_sem = batch
        loss = self.get_loss(z_sem, batch_idx)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        '''
            encoding = self.fetch_encoding(batch, False)
            z_sem = diffusion.encode_semantic(self.diffae_model.unet_autoencoder.encoder, encoding)
        '''
        z_sem = batch
        loss = self.get_loss(z_sem, batch_idx)
        self.log("val/loss", loss)
        return loss

    def on_validation_epoch_end(self):
        if (self.global_rank != 0):
            return
        # Get tensorboard logger
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, pl.loggers.TensorBoardLogger):
                tb_logger = logger.experiment
                break

        if tb_logger is None:
                raise ValueError('TensorBoard Logger not found')
        batch_dim = (10, self.latent_space)
        z_sem_noised = torch.randn(batch_dim).to(self.device)
        z_sem = diffusion.denoise_process_multiple_images(self.latent_model, z_sem_noised, None, self.t_range, self.beta_small, self.beta_large)
        print("Z_sem created!")
        x_t = torch.randn((batch_dim[0], 3, 64, 64)).to(self.device)
        reconstructed_x_0 = diffusion.denoise_process_multiple_images(self.diffae_model.unet_autoencoder, x_t, z_sem, self.t_range, self.beta_small, self.beta_large)
        print("Denoising done!")
        reconstructed_images = self.decode_encoding(reconstructed_x_0)
        print("Decoding done!")
        tb_logger.add_images(f"val/hallucinated_images", reconstructed_images, self.current_epoch)

        print("Images added!")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.latent_model.parameters()), lr=self.diffae_model_config["model"]["base_learning_rate"])
        return optimizer
