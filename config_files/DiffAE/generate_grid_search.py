import os

base_channels = [16, 32, 48, 64]
train_dataset_lengths = [10000, 40000, 70000]
total_samples = [100000000]
current_folder = os.path.dirname(__file__)

def read_unet_file():
    return open(os.path.join(current_folder, "template/unet_template.yaml"), "r").read()

def read_latent_model_file():
    return open(os.path.join(current_folder, "template/latent_model_template.yaml"), "r").read()

def read_diffusion_model_file():
    return open(os.path.join(current_folder, "template/diffusion_model_template.yaml"), "r").read()

def read_cls_model_file():
    return open(os.path.join(current_folder, "template/cls_model_template.yaml"), "r").read()

def create_unet_file(folder_name, unet_template, c, t, l):
    new_path = os.path.join(folder_name, "unet_model.yaml")
    f = open(new_path, "w+")
    f.write(unet_template.format(base_channels = str(c), latent_space = str(l)))
    f.close()

def create_latent_model_file(folder_name, latent_template, c, t, s, l):
    new_path = os.path.join(folder_name, "latent_model.yaml")
    f = open(new_path, "w+")
    diffae_config = os.path.join(folder_name, "diffusion_model.yaml")
    diffae_checkpoint = os.path.join(folder_name, "diffusion_model.ckpt")
    f.write(latent_template.format(train_dataset_length=t, total_samples=s, latent_space=l, diffae_config=diffae_config, diffae_checkpoint=diffae_checkpoint))
    f.close()

def create_diffusion_model_file(folder_name, diffusion_template, c, t, s):
    new_path = os.path.join(folder_name, "diffusion_model.yaml")
    f = open(new_path, "w+")
    unet_config = os.path.join(folder_name, "unet_model.yaml")
    f.write(diffusion_template.format(train_dataset_length=t, total_samples=s, unet_config=unet_config))
    f.close()

def create_cls_model_file(folder_name, cls_template, c, t, s, l):
    new_path = os.path.join(folder_name, "cls_model.yaml")
    f = open(new_path, "w+")
    diffae_config = os.path.join(folder_name, "diffusion_model.yaml")
    diffae_checkpoint = os.path.join(folder_name, "diffusion_model.ckpt")
    f.write(cls_template.format(train_dataset_length=t, total_samples=s, latent_space=l, diffae_config=diffae_config, diffae_checkpoint=diffae_checkpoint))
    f.close()

for c in base_channels:
    for t in train_dataset_lengths:
        for s in total_samples:
            l = c * 8
            folder_name = os.path.join("grid_search", f"channels_{c}_dataset_length_{t}_total_samples_{s}")
            folder_path = os.path.join(current_folder, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            create_unet_file(folder_path, read_unet_file(), c, t, l)
            create_latent_model_file(folder_path, read_latent_model_file(), c, t, s, l)
            create_diffusion_model_file(folder_path, read_diffusion_model_file(), c, t, s)
            create_cls_model_file(folder_path, read_cls_model_file(), c, t, s, l)