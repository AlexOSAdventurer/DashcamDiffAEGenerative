import sys
import os
import time
import glob
from multiprocessing import Pool

def launch_training(folder):
    root_folder = os.path.join(os.dirname(__file__), "..")
    train_script = os.path.join(root_folder, "train.py")
    diffusion_config_path = os.path.join(folder, "diffusion_model.yaml")
    subprocess.run(f"python {train_script}", shell=True, env={"AUTOENCODER_CONFIG": diffusion_config_path})
    time.sleep(5) # wait 5 seconds for IO to finish
    source_glob = os.path.join(root_folder, "lightning_logs", "config_files", "DiffAE", "grid_search", os.path.basename(folder), "**", "checkpoints", "*.ckpt")
    source_ckpt = glob.glob(source_glob)
    source_ckpt.sort()
    source_ckpt = source_ckpt[-1]
    target_ckpt = os.path.join(folder, "diffusion_model.ckpt")
    subprocess.run(["cp", source_ckpt, target_ckpt], shell=True)

if __name__ == '__main__':
    parallel_training_units = 3
    current_folder = os.dirname(__file__)
    grid_search_folder = os.path.join(current_folder, "grid_search")
    configuration_folders = os.listdir(grid_search_folder)
    with Pool(parallel_training_units) as p:
        p.map(launch_training, configuration_folders)