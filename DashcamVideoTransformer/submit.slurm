#!/bin/bash -l
#SBATCH --job-name=VideoPrediction_1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --nodes=1
#SBATCH --mem=128gb
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --output=/work/cseos2g/papapalpi/logs/%x-%j.out

export PYTHONPATH="/home/cseos2g/papapalpi/.local/lib/python3.8/site-packages/:$PYTHONPATH"
srun singularity exec --nv $WORK/torchgan_container_old_new_latest.sif python3 train.py fit --trainer.num_nodes 1 --trainer.gpus 2
