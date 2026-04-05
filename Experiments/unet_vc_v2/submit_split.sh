#!/bin/bash
#SBATCH --job-name=unet_v2_split
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

module load python/3.10
source ../.env/bin/activate

echo "=========================================="
echo "  UNet-VC v2 — Train/Test Split"
echo "=========================================="

# Tonsill — cross-patient
python scripts/train_split.py --surgery Tonsill --n_test 5 --seed 42 \
    --output results_tonsill_cross

# Tonsill — same-patient (ablation)
python scripts/train_split.py --surgery Tonsill --n_test 5 --seed 42 \
    --same_patient --output results_tonsill_same

# Fess
python scripts/train_split.py --surgery Fess --n_test 5 --seed 42 \
    --output results_fess_cross

# Sept
python scripts/train_split.py --surgery Sept --n_test 5 --seed 42 \
    --output results_sept_cross

echo "Done!"
