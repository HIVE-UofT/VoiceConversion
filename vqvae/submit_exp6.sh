#!/bin/bash
#SBATCH --job-name=vqunet_exp6
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

module load python/3.10
module load cuda/11.8

source ../.env/bin/activate

# Step 1: Train VQ-UNet on WavLM features
python scripts/train_exp6.py

# Step 2: Convert all pre-surgery files
python scripts/inference_exp6.py

# Step 3: Evaluate
python scripts/evaluate_exp6.py --skip_f0
