import torch
import torch.multiprocessing as mp
from data import OriginalImageDataset as ImageDataset
import pytorch_lightning as pl
import first_stage_autoencoder
from torch.utils.data import DataLoader
import imageio
import glob
from pytorch_lightning.utilities.cli import LightningCLI
import yaml
import numpy
import sys

# Training hyperparameters
dataset_choice = "Testing"
base_dir = "./"
dataset_path_train = base_dir + "video_train_256x256.npy"
dataset_path_val = base_dir + "video_val_256x256.npy"
new_dataset_path_train =  base_dir + "video_train_256x256_latent_2.npy"
new_dataset_path_val =  base_dir + "video_val_256x256_latent_2.npy"
max_epoch = 10
batch_size = 1
gpus = 2
device = 'cuda'
max_queue = 100
# Create datasets and data loaders
train_dataset = ImageDataset(dataset_path_train)
val_dataset = ImageDataset(dataset_path_val)

def infer_thread(rank, input_queue, output_queue):
    with torch.no_grad():
        dev = f"cuda:{rank}"
        autoencoder_model = first_stage_autoencoder.generate_pretrained_model()
        autoencoder_model = autoencoder_model.eval().to(dev)
        while True:
            i, data = input_queue.get()
            if (i is None):
                break
            number_of_sequences = data.shape[0]
            sequence_length = data.shape[1]
            data = torch.reshape(data, (-1, 3, 256, 256))
            data = data.to(dev)
            #data_tuple = torch.split(data, 1)
            #data_result = []
            #for data_entry in data_tuple:
            #    data_entry = data_entry.to(dev)
            #    data_result.append(autoencoder_model.encode_raw(data_entry).to('cpu'))
            #result = torch.reshape(torch.cat(data_result), (number_of_sequences, sequence_length, 6, 64, 64))
            result = torch.reshape(autoencoder_model.encode_raw(data).to('cpu'), (number_of_sequences, sequence_length, 6, 64, 64))
            output_queue.put((i, result))
            print(f"Inferred {i}")
            sys.stdout.flush()
            del i, data, number_of_sequences, sequence_length

def mmap_thread(input_queue, total_expected, mmap_out):
    i = 0
    print("LAUNCHING MMAP THREAD!")
    while (i < total_expected):
        i, result = input_queue.get()
        mmap_out[(i * batch_size):((i * batch_size) + result.shape[0])] = result.numpy()
        print(f"Writing {i}")
        sys.stdout.flush()
        i = i + 1

def convertData(dataset, new_path):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    output_memmap = numpy.lib.format.open_memmap(new_path, dtype=numpy.float, shape=(len(dataset), dataset.sequence_length, 6, 64, 64), mode='w+')
    input_queue = mp.Queue(max_queue)
    mmap_queue = mp.Queue(max_queue)
    total_expected = output_memmap.shape[0]
    processes = []
    for rank in range(gpus):
        p = mp.Process(target=infer_thread, args=(rank, input_queue, mmap_queue))
        p.start()
        processes.append(p)
    mmap_process = mp.Process(target=mmap_thread, args=(mmap_queue, total_expected, output_memmap))
    mmap_process.start()
    processes.append(mmap_process)
    for i, data in enumerate(loader, 0):
        input_queue.put((i, data))
        print(f"Injecting {i}")
        sys.stdout.flush()
    for _ in range(gpus):
        input_queue.put((None, None))
    for p in processes:
        p.join()

print("Train")
convertData(train_dataset, new_dataset_path_train)
print("Val")
convertData(val_dataset, new_dataset_path_val)
