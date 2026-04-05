#!/bin/bash
#SBATCH --job-name=baselines_train_test
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

module load python/3.10

source .env/bin/activate

echo "=========================================="
echo "  Computing baselines + train/test metrics"
echo "  for all surgery types"
echo "=========================================="

python compute_baselines_and_train_metrics.py \
    --surgery Tonsill Fess Sept \
    --seed 42 --n_test 5

echo ""
echo "Done!"
