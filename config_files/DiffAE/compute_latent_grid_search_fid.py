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

def launch_test(args):
    folder, i = args
    print(folder, i)
    current_folder = os.path.dirname(__file__)
    root_folder = os.path.join(current_folder, "..", "..")
    test_script = os.path.join(root_folder, "test_latent.py")
    folder = os.path.join(current_folder, "grid_search", folder)
    latent_config_path = os.path.join(folder, "latent_model.yaml")
    env = copy_environment()
    env["LATENT_DDIM_CONFIG"] = latent_config_path
    env["USER_SET_MASTER_PORT"] = str(12343 + i)
    env["CUDA_VISIBLE_DEVICES"] = str(i % 4)
    train_result = subprocess.run(f"conda activate dashcam_diffae_generative; python {test_script}", shell=True, executable=powershell_path, env=env)
    if (train_result.returncode != 0):
        Pool.terminate()
        return

if __name__ == '__main__':
    parallel_training_units = 4
    current_folder = os.path.dirname(__file__)
    grid_search_folder = os.path.join(current_folder, "grid_search")
    configuration_folders = os.listdir(grid_search_folder)
    configuration_folders.sort(reverse=True)
    with Pool(parallel_training_units) as p:
        p.map(launch_test, zip(configuration_folders, range(len(configuration_folders))))