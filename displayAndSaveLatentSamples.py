import sys
import torch
import numpy
import matplotlib.pyplot as plt
import first_stage_autoencoder

original_data = numpy.load("../data/val_float_256x256.npy", mmap_mode='r')
latent_data = numpy.load("../data/val_float_256x256_latent_2.npy", mmap_mode='r')

model = first_stage_autoencoder.generate_pretrained_model()

def displayAndSave(d, n):
    plt.subplot(121)
    plt.imshow(numpy.transpose(numpy.clip((d[0] + 1) / 2.0, 0.0, 1.0), [1,2,0]))
    plt.savefig(n)
    
    
original_img = numpy.clip(((original_data[:1] / 255.0) * 2) - 1, -1.0, 1.0).copy()
displayAndSave(original_img, "original.png")
original_img_tensor = torch.from_numpy(original_img).type(torch.FloatTensor)
original_img_encoding = model.encode(original_img_tensor)
original_img_encoding_mode = original_img_encoding.mode()
original_img_decoding = model.decode(original_img_encoding_mode)
displayAndSave(original_img_decoding.detach().numpy(), "original_decoded.png")
latent_moment = torch.from_numpy(latent_data[:1].copy()).type(torch.FloatTensor)
latent_distribution = first_stage_autoencoder.distribution.DiagonalGaussianDistribution(latent_moment)
latent_distribution_mode = latent_distribution.mode()
latent_decoding = model.decode(latent_distribution_mode)
displayAndSave(latent_decoding.detach().numpy(), "latent_decoded.png")