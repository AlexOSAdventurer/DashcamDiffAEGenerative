import torch
import pytorch_lightning as pl
from lightning_training_latent_model import LatentModel
import custom_diffusion as diffusion
import yaml
import numpy

base_dir = "/work/cseos2g/papapalpi/data/"
model_name = "largest_model"
config_path = f"latent_diffusion_model_64x64x3_{model_name}.yaml"
checkpoint_path = f"saved_latent_diffusion_models/{model_name}/model.ckpt"
config_data = yaml.safe_load(open(config_path))
new_dataset_path =  base_dir + f"images_hallucinated/{config_path}.npy"
batch_size = 32
batches = 64
device = 'cuda:1'

model = LatentModel.load_from_checkpoint(checkpoint_path, config=config_data).eval().to(device)

def createSamples(batch_dim):
    z_sem_noised = torch.randn(batch_dim).to(device)
    z_sem = diffusion.denoise_process_multiple_images(model.latent_model, z_sem_noised, None, model.t_range, model.beta_small, model.beta_large)
    print("Z_sem created!")
    x_t = torch.randn((batch_dim[0], 3, 64, 64)).to(device)
    reconstructed_x_0 = diffusion.denoise_process_multiple_images(model.diffae_model.unet_autoencoder, x_t, z_sem, model.t_range, model.beta_small, model.beta_large)
    print("Denoising done!")
    reconstructed_images = model.decode_encoding(reconstructed_x_0)
    print("Decoding done!")
    return reconstructed_images

def convertData(new_path):
    output_memmap = numpy.lib.format.open_memmap(new_path, dtype=numpy.float, shape=(batch_size*batches, 3, 256, 256), mode='w+')
    with torch.no_grad():
        for i in range(batches):
            result = createSamples((batch_size, 512)).to('cpu')
            output_memmap[(i * batch_size):((i * batch_size) + result.shape[0])] = result.numpy()
            print(i)

print("Creating data!")
convertData(new_dataset_path)
