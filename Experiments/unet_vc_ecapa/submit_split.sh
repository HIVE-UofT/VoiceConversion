#!/bin/bash
#SBATCH --job-name=unet_ecapa
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

module load python/3.10
source ../.env/bin/activate

echo "=========================================="
echo "  UNet-VC + ECAPA Loss — Train/Test Split"
echo "=========================================="

python scripts/train_split_v2.py --surgery Tonsill --n_test 5 --seed 42

echo "Done!"
