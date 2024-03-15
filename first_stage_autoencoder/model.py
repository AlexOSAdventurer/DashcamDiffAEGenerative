import sys
import torch
import pytorch_lightning as pl

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.diffusionmodules.openaimodel import UNetModel

from ldm.util import instantiate_from_config

from distribution import DiagonalGaussianDistribution

class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 base_learning_rate
                 ):
        super().__init__()
        self.automatic_optimization = False
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.learning_rate = base_learning_rate
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        
    def encode_raw(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        return moments

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def training_step_prim(self, batch, batch_idx, optimizer_idx):
        inputs = batch
        reconstructions, posterior = self(inputs)
        opt = self.optimizers()[optimizer_idx]
        loss = None
        with opt.toggle_model():
            if optimizer_idx == 0:
                # train encoder+decoder+logvar
                aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
                self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
                loss = aeloss

            if optimizer_idx == 1:
                # train the discriminator
                discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                    last_layer=self.get_last_layer(), split="train")

                self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
                loss = discloss
            self.manual_backward(loss)
            opt.step()
            opt.zero_grad()
    
    def training_step(self, batch, batch_idx):
        self.training_step_prim(batch, batch_idx, 0)
        self.training_step_prim(batch, batch_idx, 1)

    def validation_step(self, batch, batch_idx):
        inputs = batch
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        if (batch_idx == 0):
            self.val_batch = batch
        return self.log_dict
        
    def on_validation_epoch_end(self):
        # Get tensorboard logger
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, pl.loggers.TensorBoardLogger):
                tb_logger = logger.experiment
                break

        if tb_logger is None:
                raise ValueError('TensorBoard Logger not found')
        sample_batch_size = self.val_batch.shape[0]
        decoding, _ = self(self.val_batch)
        x = torch.cat([self.val_batch, decoding], dim=0)
        x = (x.clamp(-1, 1) + 1) / 2.0
        
        tb_logger.add_images(f"val/output_images", x, self.current_epoch)

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight
