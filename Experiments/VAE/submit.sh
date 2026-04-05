#!/bin/bash
#SBATCH --job-name=vae_voice_research
#SBATCH --account=def-zshakeri
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:08:00
#SBATCH --output=%x-%j.out

module load python/3.10 
module load cuda/11.8

source ../.env/bin/activate
python -m scripts.test.