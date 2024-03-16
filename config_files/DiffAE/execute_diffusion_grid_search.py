import sys
import os
import time
import glob
import subprocess
from multiprocessing import Pool
powershell_path = "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe"

def copy_environment():
    result = {}
    for k in os.environ:
        result[k] = os.environ[k]
    return result

def launch_training(args):
    folder, i = args
    time.sleep(i * 30)
    print(folder, i)
    current_folder = os.path.dirname(__file__)
    root_folder = os.path.join(current_folder, "..", "..")
    train_script = os.path.join(root_folder, "train.py")
    diffusion_config_path = os.path.join(current_folder, "grid_search", folder, "diffusion_model.yaml")
    env = copy_environment()
    env["AUTOENCODER_CONFIG"] = diffusion_config_path
    env["USER_SET_MASTER_PORT"] = str(12345 + i)
    env["CUDA_VISIBLE_DEVICES"] = str(i % 4)
    train_result = subprocess.run(f"conda activate dashcam_diffae_generative; python {train_script}", shell=True, executable=powershell_path, env=env)
    if (train_result.returncode != 0):
        Pool.terminate()
        return
    time.sleep(5) # wait 5 seconds for IO to finish
    source_glob = os.path.join(root_folder, "lightning_logs", "config_files", "DiffAE", "grid_search", os.path.basename(folder), "**", "checkpoints", "*.ckpt")
    source_ckpt = glob.glob(source_glob)
    source_ckpt.sort()
    source_ckpt = source_ckpt[-1]
    target_ckpt = os.path.join(folder, "diffusion_model.ckpt")
    subprocess.run(["cp", source_ckpt, target_ckpt], shell=True)

if __name__ == '__main__':
    parallel_training_units = 8
    current_folder = os.path.dirname(__file__)
    grid_search_folder = os.path.join(current_folder, "grid_search")
    configuration_folders = os.listdir(grid_search_folder)
    configuration_folders.sort(reverse=True)
    with Pool(parallel_training_units) as p:
        p.map(launch_training, zip(configuration_folders, range(len(configuration_folders))))