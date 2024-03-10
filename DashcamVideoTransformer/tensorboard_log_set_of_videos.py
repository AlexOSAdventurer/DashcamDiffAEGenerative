import torch
import torchvision
from data import ImageDataset, OriginalImageDataset, DiffusionLatentImageDataset
import pytorch_lightning as pl
from lightning_training_model import VideoPredictionModel
from torch.utils.data import DataLoader
import glob
from pytorch_lightning.utilities.cli import LightningCLI
import yaml

# Training hyperparameters
model_choice = "AIinCPS"
base_dir = "/work/cseos2g/papapalpi/"
config_data = yaml.safe_load(open("video_model_64x64x3.yaml"))
base_dir = config_data['data']['base_dir']
dataset_path_val_images = base_dir + config_data['data']['val_images']
dataset_path_val_latents = base_dir + config_data['data']['val_latents']
load_version_num = 66
device = "cuda:0"

last_checkpoint = glob.glob(
    f"./lightning_logs/{model_choice}/version_{load_version_num}/checkpoints/*.ckpt"
)[-1]

dataset_type = DiffusionLatentImageDataset
# Create datasets and data loaders
val_dataset = dataset_type(dataset_path_val_images, dataset_path_val_latents)

model = VideoPredictionModel.load_from_checkpoint(last_checkpoint, config=config_data).to(device)

#x_t_bigger = torch.randn((80,3,64,64)).to(device)
x_t_smaller = torch.randn((79,3,64,64)).to(device)


# Load Trainer model
tb_logger = pl.loggers.TensorBoardLogger(
    "lightning_logs/",
    name="paper_analysis",
    version=None,
)

#latents, originals = val_dataset[[0, 109, 938, 708]]
latents, originals = val_dataset[[47, 209, 538, 991]]

def get_predicted_target(latent_input):
    predicted_transitions_for_target = model(latent_input, True).reshape(-1, 512)
    latent_input = latent_input.reshape(-1, 512)
    return model.decode_64x64(model.diffusion_model.denoise_zsem(latent_input + predicted_transitions_for_target, x_t_smaller)).reshape(79, 3, 256, 256)

def get_hallucinated_target(latent_input):
    latent_input = latent_input.reshape(79, 512)
    seed_input = latent_input[0:1]
    for i in range(latent_input.shape[0]):
        print(seed_input.shape)
        current_output = model(seed_input.reshape(1, -1, 512), True).reshape(-1, 512)
        last_frame = seed_input[i:(i + 1)]
        current_output = current_output[i:(i + 1)]
        print(last_frame.shape, current_output.shape)
        seed_input = torch.cat([seed_input, last_frame + current_output], dim=0)
    seed_input = seed_input[1:]
    print(seed_input.shape)
    return model.decode_64x64(model.diffusion_model.denoise_zsem(seed_input)).reshape(79, 3, 256, 256)

def get_reconstructed_target(latent_target):
    latent_target = latent_target.reshape(-1, 512)
    return model.decode_64x64(model.diffusion_model.denoise_zsem(latent_target)).reshape(79, 3, 256, 256)

def get_original_target(original_target):
    return model.decode_64x64(model.diffusion_model.fetch_encoding(original_target.reshape(-1, 6, 64, 64), False)).reshape(79, 3, 256, 256)

def create_grid(tensor, row_length):
    return torchvision.utils.make_grid(tensor, nrow=row_length)

def log_large_sequence(name, original, reconstructed, predicted, hallucinated, step=10):
    original = original[::step]
    reconstructed = reconstructed[::step]
    predicted = predicted[::step]
    hallucinated = hallucinated[::step]
    combined = torch.cat([original, reconstructed, predicted, hallucinated], dim=0)
    grid = create_grid(combined, original.shape[0])

    tb_logger.experiment.add_image(name, grid, 0)

def log_input_versus_target_sequence(name, reconstructed_input, predicted, step=10):
    reconstructed_input = reconstructed_input[::step]
    predicted = predicted[::step]
    combined = torch.cat([reconstructed_input, predicted], dim=0)

    grid = create_grid(combined, predicted.shape[0])
    tb_logger.experiment.add_image(name, grid, 0)

with torch.no_grad():
    for i in range(latents.shape[0]):
        current_latents = latents[i:(i + 1)].to(device)
        current_originals = originals[i:(i + 1)].to(device)
        latent_input, latent_target = model.get_input_and_target(current_latents)
        original_input, original_target = model.get_input_and_target(current_originals)
        original_target = get_original_target(original_target)
        hallucinated_target = get_hallucinated_target(latent_input)
        predicted_target = get_predicted_target(latent_input)
        reconstructed_target = get_reconstructed_target(latent_target)
        reconstructed_input = get_reconstructed_target(latent_input)
        log_large_sequence(f"output_large_{i}", original_target, reconstructed_target, predicted_target, hallucinated_target)
        log_input_versus_target_sequence(f"output_versus_{i}", reconstructed_input, predicted_target)
