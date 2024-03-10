import os
import skvideo
import skvideo.io
import cv2
import numpy
import json
import joblib
import sys
import yaml
import torch
import json
import math
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from lightning_training_model import DiffusionModel

t_length = 200
latent_size = 512
total_nodes = 3
node_id = int(os.environ['SLURM_PROCID'])

class Image256Dataset(Dataset):
    def __init__(self, npy_file, node_index, total_nodes):
        self.data = numpy.load(npy_file, mmap_mode='r')
        self.node_index = node_index
        self.total_nodes = total_nodes
        self.total_length = self.data.shape[0]
        self.block_length = int(self.total_length / self.total_nodes)
        self.remainder = (self.total_length % self.total_nodes)
        self.our_length = self.block_length
        if (((self.remainder) != 0) and (node_id == (total_nodes - 1))):
            self.our_length = self.our_length + self.remainder
        self.start = self.block_length * self.node_index
        print(node_id, self.total_length, self.block_length, self.start, self.our_length)
        sys.stdout.flush()
    def __len__(self):
        return self.our_length
    def __getitem__(self, idx):
        return self.data[self.start + idx].copy()

def computeBatch(model, batch):
    batch = batch.to(model.device)
    batch = ((batch / 255.0) - 0.5) * 2.0
    batch = torch.clamp(batch, -1.0, 1.0)
    batch = batch.reshape(-1, 3, 256, 256)
    #torch.permute(torch.clamp(((torch.cat(current_queue_frames) / 255.0) - 0.5) * 2.0, -1.0, 1.0), (2, 0, 1)).reshape(-1, 3, 256, 256)
    return model.encode_semantic(batch)

if __name__ == "__main__":
    input_file_frames = "./video_train_3x256x256.npy"
    output_file_latent = "./video_train_512.npy"

    dataset = Image256Dataset(input_file_frames, node_id, total_nodes)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)
    output_memmap_latent = numpy.lib.format.open_memmap(output_file_latent, dtype=float, shape=(len(dataset), t_length, latent_size), mode='r+')

    def runItAll():
        torch.backends.cudnn.benchmark = True
        with torch.no_grad():
            with torch.autocast("cuda"):
                path = "/work/cseos2g/papapalpi/data/models/DashcamDiffusionAutoencoder64x64New/saved_diffusion_models/our_largest_model/model.ckpt"
                config_path = "./config_files/diffusion_model_64x64x3_largest_model.yaml"
                config = yaml.safe_load(open(config_path))
                device1 = f"cuda:0"
                device2 = f"cuda:1"
                model1 = DiffusionModel.load_from_checkpoint(path, config=config).eval().to(device1)
                model2 = DiffusionModel.load_from_checkpoint(path, config=config).eval().to(device2)
                for i, sample in enumerate(dataloader):
                    sample = sample.reshape(200,3,256,256)
                    data_batch_1 = sample[:50]
                    data_batch_2 = sample[50:100]
                    data_batch_3 = sample[100:150]
                    data_batch_4 = sample[150:]
                    latent_batch_1 = computeBatch(model1, data_batch_1)
                    latent_batch_2 = computeBatch(model2, data_batch_2)
                    output_memmap_latent[dataset.start + i][:50] = latent_batch_1.cpu().detach()
                    output_memmap_latent[dataset.start + i][50:100] = latent_batch_2.cpu().detach()
                    latent_batch_3 = computeBatch(model1, data_batch_3)
                    latent_batch_4 = computeBatch(model2, data_batch_4)
                    output_memmap_latent[dataset.start + i][100:150] = latent_batch_3.cpu().detach()
                    output_memmap_latent[dataset.start + i][150:] = latent_batch_4.cpu().detach()
                    print(i)
                    if ((i % 100) == 0):
                        sys.stdout.flush()

    runItAll()
