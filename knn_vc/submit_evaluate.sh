#!/bin/bash
#SBATCH --job-name=knnvc_eval
#SBATCH --account=def-zshakeri
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

module load python/3.10

source ../.venv/bin/activate

python scripts/evaluate.py \
    --converted_dir knn_vc_converted
