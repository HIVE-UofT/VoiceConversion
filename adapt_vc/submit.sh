#!/bin/bash
#SBATCH --job-name=adapt_vc
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

module load python/3.10
module load cuda/11.8

source ../.env/bin/activate

# Step 1: Train AdaptVC on raw audio (WavLM features extracted on-the-fly)
python scripts/train.py

# Step 2: Convert all pre-surgery files
python scripts/inference.py

# Step 3: Evaluate
python scripts/evaluate.py --skip_f0
