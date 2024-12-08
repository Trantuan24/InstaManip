#!/bin/bash
#SBATCH --job-name=instamanip
#SBATCH --partition-HCESC-H100
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-gpu=10
#SBATCH --gres=gpu:H100:8
#SBATCH --chdir=/home/bolinlai/InstaManip
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err

srun bash scripts/train.sh
