import os

base_channels = [16, 32, 48, 64]
train_dataset_lengths = [10000, 40000, 70000]
total_samples = [1000000]
current_folder = os.path.dirname(__file__)

def readUNetFile():
    return open("template/unet_template.yaml", "r").read()

def readLatentModelFile():
    return open("template/latent_model_template.yaml", "r").read()

def readDiffusionModelFile():
    return open("template/diffusion_model_template.yaml", "r").read()

def readCLSModelFile():
    return open("template/cls_model_template.yaml", "r").read()

def createUNetFile(folder_name, unet_template, c, t):
    new_path = os.path.join(folder_name, "unet_model.yaml")
    f = open(new_path, "w+")
    f.write(unet_template.format(base_channels = str(c), latent_space = str(c * 8)))
    f.close()

def createLatentModelFile(folder_name, latent_template, c, t, s):
    new_path = os.path.join(folder_name, "latent_model.yaml")
    f = open(new_path, "w+")
    diffae_config = os.path.join(folder_name, "diffusion_model.yaml")
    diffae_checkpoint = os.path.join(folder_name, "diffusion_model.ckpt")
    f.write(latent_template.format(train_dataset_length=t, total_samples=s, diffae_config=diffae_config, diffae_checkpoint=diffae_checkpoint))
    f.close()

def createDiffusionModelFile(folder_name, diffusion_template, c, t, s):
    new_path = os.path.join(folder_name, "diffusion_model.yaml")
    f = open(new_path, "w+")
    unet_config = os.path.join(folder_name, "unet_model.yaml")
    f.write(diffusion_template.format(train_dataset_length=t, total_samples=s, unet_config=unet_config))
    f.close()

def createCLSModelFile(folder_name, cls_template, c, t, s):
    new_path = os.path.join(folder_name, "cls_model.yaml")
    f = open(new_path, "w+")
    diffae_config = os.path.join(folder_name, "diffusion_model.yaml")
    diffae_checkpoint = os.path.join(folder_name, "diffusion_model.ckpt")
    f.write(cls_template.format(train_dataset_length=t, total_samples=s, diffae_config=diffae_config, diffae_checkpoint=diffae_checkpoint))
    f.close()

for c in base_channels:
    for t in train_dataset_lengths:
        for s in total_samples:
            folder_name = os.path.join("grid_search", f"channels_{c}_dataset_length_{t}_total_samples_{s}")
            folder_path = os.path.join(current_folder, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            createUNetFile(folder_path, readUNetFile(), c, t)
            createLatentModelFile(folder_path, readLatentModelFile(), c, t, s)
            createDiffusionModelFile(folder_path, readDiffusionModelFile(), c, t, s)
            createCLSModelFile(folder_path, readCLSModelFile(), c, t, s)