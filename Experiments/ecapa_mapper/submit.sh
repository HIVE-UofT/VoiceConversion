#!/bin/bash
#SBATCH --job-name=ecapa_mapper
#SBATCH --account=def-zshakeri
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

module load python/3.10
source ../.env/bin/activate

echo "=========================================="
echo "  ECAPA Mapper: pre -> post surgery"
echo "=========================================="

python unet_mapper.py --surgery Tonsill --n_test 5 --seed 42

echo "Done!"
