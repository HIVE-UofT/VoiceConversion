#!/bin/bash
#SBATCH --job-name=unet_spk_split
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

module load python/3.10
source ../.env/bin/activate

echo "=========================================="
echo "  UNet-VC Speaker-Conditioned — Train/Test Split"
echo "=========================================="

python scripts/train_split.py --surgery Tonsill --n_test 5 --seed 42 \
    --output results_tonsill

python scripts/train_split.py --surgery Fess --n_test 5 --seed 42 \
    --output results_fess

python scripts/train_split.py --surgery Sept --n_test 5 --seed 42 \
    --output results_sept

echo "Done!"
