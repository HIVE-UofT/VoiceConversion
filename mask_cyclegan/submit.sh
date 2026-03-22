#!/bin/bash
#SBATCH --job-name=mask_cyclegan_vc
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out

module load python/3.10
module load cuda/11.8

source ../.venv/bin/activate

# Train or evaluate based on MODE env variable
# Usage: sbatch submit.sh              (default: train)
# Usage: MODE=evaluate sbatch submit.sh (run test set evaluation)
MODE="${MODE:-train}"

if [ "$MODE" = "evaluate" ]; then
    python scripts/evaluate.py
else
    python scripts/train.py
fi
