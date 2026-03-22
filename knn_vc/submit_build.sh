#!/bin/bash
#SBATCH --job-name=knnvc_build
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

module load python/3.10
module load cuda/11.8

source ../.venv/bin/activate
python scripts/build_matching_set.py
