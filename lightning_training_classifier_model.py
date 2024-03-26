import torch
import torch.nn as nn
import pytorch_lightning as pl
import math
import sys
import numpy as np
import torch.nn.functional as F
import yaml
import first_stage_autoencoder
from data import ImageLabelDataset
from first_stage_autoencoder.distribution import DiagonalGaussianDistribution
import custom_diffusion as diffusion
from lightning_training_model import DiffusionModel
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score

class ClsModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.diffae_model_config = yaml.safe_load(open(config["diffae"]["diffae_config"]))
        self.beta_small = self.diffae_model_config["model"]["beta_small"]
        self.beta_large = self.diffae_model_config["model"]["beta_large"]
        self.t_range = self.diffae_model_config["model"]["t_range"]
        self.diffae_checkpoint_file = config["diffae"]["checkpoint_file"]
        self.train_input_file_path = config["data"]["train"]
        self.train_label_file_path = config["data"]["train_labels"]
        self.train_length = int(config["data"]["train_dataset_length"])
        self.latent_space = int(config["model"]["latent_space"])
        self.get_conds_stats()
        self.save_hyperparameters()
        self.diffae_model = DiffusionModel.load_from_checkpoint(self.diffae_checkpoint_file, config=self.diffae_model_config).eval()
        self.diffae_model.generate_first_stage() #Overwrite the old first stage autoencoder
        self.classifier = nn.Linear(self.latent_space, 3)
        self.accuracy = BinaryAccuracy()
        self.f1_score = BinaryF1Score()

    def get_conds_stats(self):
        train_dataset = ImageLabelDataset(self.train_input_file_path, self.train_label_file_path, self.train_length)
        self.conds_mean = train_dataset.get_conds_mean()
        self.conds_std = train_dataset.get_conds_std()

    def fetch_encoding(self, batch, sample):
        return self.diffae_model.fetch_encoding(batch, sample)

    def decode_encoding(self, encoding):
        return self.diffae_model.decode_encoding(encoding)

    def forward(self, images_or_semantic, images=True):
        x = images_or_semantic
        if images:
            x = diffusion.encode_semantic(self.diffae_model.unet_autoencoder.encoder, x)
        output_predicted_labels = self.classifier(x)
        return output_predicted_labels

    def normalize(self, cond):
        cond = (cond - self.conds_mean.to(self.device)) / self.conds_std.to(
            self.device)
        return cond

    def denormalize(self, cond):
        cond = (cond * self.conds_std.to(self.device)) + self.conds_mean.to(
            self.device)
        return cond

    def get_loss(self, x, labels):
        output_predicted_labels = self(x, images=False)
        return F.binary_cross_entropy_with_logits(output_predicted_labels, labels), output_predicted_labels

    def compute_accuracies(self, predicted_labels, target_labels):
        pred_daytime_label = predicted_labels[:, 0]
        pred_highway_label = predicted_labels[:, 1]
        pred_clouds_label = predicted_labels[:, 2]
        target_daytime_label = target_labels[:, 0]
        target_highway_label = target_labels[:, 1]
        target_clouds_label = target_labels[:, 2]

        daytime_accuracy = self.accuracy(pred_daytime_label, target_daytime_label)
        highway_accuracy = self.accuracy(pred_highway_label, target_highway_label)
        clouds_accuracy = self.accuracy(pred_clouds_label, target_clouds_label)
        return {'daytime': daytime_accuracy, 'highway': highway_accuracy, 'clouds': clouds_accuracy}

    def compute_f1s(self, predicted_labels, target_labels):
        pred_daytime_label = predicted_labels[:, 0]
        pred_highway_label = predicted_labels[:, 1]
        pred_clouds_label = predicted_labels[:, 2]
        target_daytime_label = target_labels[:, 0]
        target_highway_label = target_labels[:, 1]
        target_clouds_label = target_labels[:, 2]

        daytime_accuracy = self.f1_score(pred_daytime_label, target_daytime_label)
        highway_accuracy = self.f1_score(pred_highway_label, target_highway_label)
        clouds_accuracy = self.f1_score(pred_clouds_label, target_clouds_label)
        return {'daytime': daytime_accuracy, 'highway': highway_accuracy, 'clouds': clouds_accuracy}

    def log_accuracies(self, accuracies, root):
        for k in accuracies:
            self.log(f"{root}/{k}/accuracy", accuracies[k])

    def log_f1s(self, f1s, root):
        for k in f1s:
            self.log(f"{root}/{k}/f1", f1s[k])

    def training_step(self, batch, batch_idx):
        x, labels = batch
        loss, pred = self.get_loss(x, labels)

        self.log("train/loss", loss)
        self.log_accuracies(self.compute_accuracies(pred, labels), "train")
        self.log_f1s(self.compute_f1s(pred, labels), "train")
        return loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        loss, pred = self.get_loss(x, labels)

        self.log("val/loss", loss)
        self.log_accuracies(self.compute_accuracies(pred, labels), "val")
        self.log_f1s(self.compute_f1s(pred, labels), "val")
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.classifier.parameters()), lr=self.diffae_model_config["model"]["base_learning_rate"])
        return optimizer
