import os
import sys
import yaml
from . import autoencoder

def generate_model(yaml_file):
    unet_config = yaml.safe_load(open(os.path.join(__path__[0], "..", yaml_file)))
    model = autoencoder.Autoencoder(unet_config)
    return model

def generate_ddim_model():
    model = autoencoder.DDIM()
    return model

__all__ = ['generate_model', 'generate_ddim_model']
