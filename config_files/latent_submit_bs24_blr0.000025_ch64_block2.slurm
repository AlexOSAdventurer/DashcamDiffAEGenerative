#!/bin/bash -l
#SBATCH --job-name=latent_grid_search_bs24_blr0.000025_ch64_block2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=20
#SBATCH --nodes=4
#SBATCH --mem=128gb
#SBATCH --time=168:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --output=/common/cseos2g/papapalpi/logs/%x-%j.out
#SBATCH --constraint=gpu_32gb

export PYTHONPATH="/home/cseos2g/papapalpi/.local/lib/python3.10/site-packages/:$PYTHONPATH"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
srun singularity exec --env "PYTHONPATH=/home/cseos2g/papapalpi/.local/lib/python3.10/site-packages/" --env "LATENT_DDIM_CONFIG=config_files/latent_model_bs24_blr0.000025_ch64_block2.yaml" --nv $WORK/evelyn_container.sif python3 train_latent.py fit --trainer.num_nodes 4  --trainer.devices 2
