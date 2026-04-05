#!/bin/bash
#SBATCH --job-name=dla_vc_split
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

module load python/3.10
source ../.env/bin/activate

echo "=========================================="
echo "  DLA-VC — Train/Test Split"
echo "=========================================="

python scripts/train_split.py --surgery Tonsill --n_test 5 --seed 42

echo "Done!"
