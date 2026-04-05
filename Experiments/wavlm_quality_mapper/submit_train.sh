#!/bin/bash
#SBATCH --job-name=wlm_mapper
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

module load python/3.10
source ../.env/bin/activate

echo "=========================================="
echo "  Multi-Layer Quality Mapper Training"
echo "=========================================="

python scripts/train.py --surgery Tonsill --n_test 5 --seed 42

echo "Done!"
