#Previous Total samples: 50000000

autoencoder_config_template = """model:
  batch_size: {batch_size}
  first_stage_needed: True
  base_learning_rate: {base_learning_rate}
  total_samples: 50000000
  in_size: 4096
  in_size_sqrt: 64
  t_range: 1000
  img_depth: 3
  beta_small: 0.0001
  beta_large: 0.02
  unet_config: {unet_config}
  recycle_previous_version: True
  previous_version: 1
data:
  base_dir: /common/cseos2g/papapalpi/
  train: data/train_float_256x256_64x64_latent_better.npy
  val: data/val_float_256x256_64x64_latent_better.npy"""

unet_config_template = """input_size: 64
input_channels: 3
noise_model:
  block_list_base_channels: {channel_size}
  block_list_channels_mult: [1,2,4,8]
  num_res_blocks: {block_num}
  attention_heads: [1]
  attention_resolutions: [16]
  latent_space: 512
encoder_model:
  block_list_base_channels: {channel_size}
  block_list_channels_mult: [1,2,4,8,8]
  latent_space: 512
  num_res_blocks: {block_num}
  attention_heads: [1]
  attention_resolutions: [16]"""

latent_ddim_template = '''diffae:
  diffae_config: {diffae_config}
  checkpoint_file: {diffae_ckpt}
model:
  batch_size: {batch_size}
  total_samples: 50000000
data:
  base_dir: /common/cseos2g/papapalpi/
  train: data/train_float_256x256_64x64_latent_better.npy
  val: data/val_float_256x256_64x64_latent_better.npy
'''

submit_template = '''#!/bin/bash -l
#SBATCH --job-name={job_name}
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=20
#SBATCH --nodes=4
#SBATCH --mem=128gb
#SBATCH --time=168:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --constraint=gpu_32gb|gpu_48gb|gpu_80gb
#SBATCH --output=/common/cseos2g/papapalpi/logs/%x-%j.out

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH="/home/cseos2g/papapalpi/.local/lib/python3.10/site-packages/:$PYTHONPATH"
srun singularity exec --env "PYTHONPATH=/home/cseos2g/papapalpi/.local/lib/python3.10/site-packages/" --env "AUTOENCODER_CONFIG={config_name}" --nv $WORK/evelyn_container.sif python3 train.py fit --trainer.num_nodes 4  --trainer.devices 2'''

submit_latent_template = '''#!/bin/bash -l
#SBATCH --job-name={job_name}
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=20
#SBATCH --nodes=4
#SBATCH --mem=128gb
#SBATCH --time=168:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --output=/common/cseos2g/papapalpi/logs/%x-%j.out
#SBATCH --constraint={constraint}

export PYTHONPATH="/home/cseos2g/papapalpi/.local/lib/python3.10/site-packages/:$PYTHONPATH"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
srun singularity exec --env "PYTHONPATH=/home/cseos2g/papapalpi/.local/lib/python3.10/site-packages/" --env "LATENT_DDIM_CONFIG={config_name}" --nv $WORK/evelyn_container.sif python3 train_latent.py fit --trainer.num_nodes 4  --trainer.devices 2
'''

bs_grid = [24]
blr_grid = [0.000025]
ch_grid = [64]
block_grid = [2]

for ch in ch_grid:
	for block_num in block_grid:
		unet_config_path = f"config_files/unet_ch{ch}_block{block_num}.yaml"
		unet_config_content = unet_config_template.format(channel_size = ch, block_num=block_num)
		with open(unet_config_path, "w+") as f:
			f.write(unet_config_content)

for bs in bs_grid:
	for blr in blr_grid:
		for ch in ch_grid:
			for block_num in block_grid:
				blr = "{0:.6f}".format(blr)
				autoencoder_config_name = f"diffusion_model_bs{bs}_blr{blr}_ch{ch}_block{block_num}"
				autoencoder_config_path = f"config_files/diffusion_model_bs{bs}_blr{blr}_ch{ch}_block{block_num}.yaml"
				unet_config_path = f"config_files/unet_ch{ch}_block{block_num}.yaml"
				slurm_config_path = f"config_files/submit_bs{bs}_blr{blr}_ch{ch}_block{block_num}.slurm"
				job_name = f"grid_search_bs{bs}_blr{blr}_ch{ch}_block{block_num}"
				autoencoder_config_content = autoencoder_config_template.format(batch_size=bs, base_learning_rate=blr, channel_size=ch, unet_config = unet_config_path, block_num=block_num)
				slurm_config_content = submit_template.format(job_name = job_name, config_name = autoencoder_config_path)

				with open(autoencoder_config_path, "w+") as f:
					f.write(autoencoder_config_content)
				with open(slurm_config_path, "w+") as f:
					f.write(slurm_config_content)
				latent_config_path = f"config_files/latent_model_bs{bs}_blr{blr}_ch{ch}_block{block_num}.yaml"
				latent_slurm_config_path = f"config_files/latent_submit_bs{bs}_blr{blr}_ch{ch}_block{block_num}.slurm"
				latent_job_name = f"latent_grid_search_bs{bs}_blr{blr}_ch{ch}_block{block_num}"
				ckpt_file = f"grid_search_models/{autoencoder_config_name}/model.ckpt"
				constraint = None
				if ((ch == 128)):
					constraint = "gpu_32gb"
				else:
					constraint = "gpu_32gb"
				latent_slurm_config_content = submit_latent_template.format(job_name = latent_job_name, config_name = latent_config_path, constraint=constraint)
				latent_config_content = latent_ddim_template.format(batch_size = bs, diffae_config = autoencoder_config_path, diffae_ckpt = ckpt_file)
				with open(latent_config_path, "w+") as f:
					f.write(latent_config_content)
				with open(latent_slurm_config_path, "w+") as f:
					f.write(latent_slurm_config_content)
