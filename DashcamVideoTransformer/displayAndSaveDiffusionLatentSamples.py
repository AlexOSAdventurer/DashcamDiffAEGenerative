import sys
import torch
import numpy
import matplotlib.pyplot as plt
import first_stage_autoencoder
import diffusion_model as df

dev = 'cuda:0'
original_data = numpy.load("./video_train_256x256.npy", mmap_mode='r')
latent_data = numpy.load("./video_train_256x256_latent_diffusion.npy", mmap_mode='r')

first_stage_model = first_stage_autoencoder.generate_pretrained_model().eval().to('cuda:0')
diffusion_model = df.DiffusionModel_load_pretrained().eval().to(dev)

def displayAndSave(d, n):
    plt.subplot(121)
    plt.imshow(numpy.transpose(numpy.clip((d[0] + 1) / 2.0, 0.0, 1.0), [1,2,0]))
    plt.savefig(n)

#Pedestrian ones: 7890
#Bad ones: 37
#Black endings (possibly incomplete?): 217
i = 10684 #Interesting ones: 938, 1048, 8708, 4182, 1112, 3999, 10009

x_t = torch.randn((1,3,64,64)).to(dev)
print(original_data[i][79:].shape, "ASDFASDFASDF")

with torch.no_grad():
    original_img = numpy.clip(((original_data[i][79:] / 255.0) * 2) - 1, -1.0, 1.0).copy()
    displayAndSave(original_img, f"original_{i}.png")
    original_img_tensor = torch.from_numpy(original_img).type(torch.FloatTensor).to(dev)
    original_img_encoding = first_stage_model.encode(original_img_tensor)
    original_img_encoding_mode = original_img_encoding.mode()
    print(original_img_encoding_mode.shape, "ASDFASDFDS")
    original_img_encoding_mode_zsem = diffusion_model.create_semantic(original_img_encoding.parameters.to(dev), False)
    print(f"zsem shape {original_img_encoding_mode_zsem.shape}")
    original_img_zsem_decoding = diffusion_model.denoise_zsem(original_img_encoding_mode_zsem, x_t)
    del original_img_encoding_mode_zsem
    print(f"decoding {original_img_zsem_decoding.shape}")
    original_img_decoding = first_stage_model.decode(original_img_zsem_decoding)
    displayAndSave(original_img_decoding.to('cpu').detach().numpy(), f"original_decoded_{i}.png")
    latent = torch.from_numpy(latent_data[i][79:].copy()).type(torch.FloatTensor).to(dev)
    latent_zsem_decoding = diffusion_model.denoise_zsem(latent, x_t)
    latent_decoding = first_stage_model.decode(latent_zsem_decoding)
    displayAndSave(latent_decoding.to('cpu').detach().numpy(), f"latent_decoded_{i}.png")
