#!/bin/bash
export PYTHONPATH="/home/cseos2g/papapalpi/.local/lib/python3.9/site-packages/:$PYTHONPATH"
tensorboard --logdir lightning_logs/ --samples_per_plugin images=2000
