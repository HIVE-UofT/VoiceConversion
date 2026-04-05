#!/bin/bash
#SBATCH --job-name=all_kfold
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out

module load python/3.10

source .env/bin/activate

echo "=========================================="
echo "  Running all baselines (kNN, Mean-Shift, MKL, Linear)"
echo "  with held-out test set, all 3 surgery types"
echo "=========================================="

python run_all_kfold.py --surgery Tonsill Fess Sept --n_test 5 --seed 42

echo ""
echo "=========================================="
echo "  Now run UNet-VC k-fold separately:"
echo "  cd unet_vc && sbatch submit_kfold.sh"
echo "=========================================="
