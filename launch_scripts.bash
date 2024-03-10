#!/bin/bash

for p in config_files/latent_*.slurm
do
	echo $p
	sbatch $p
done
