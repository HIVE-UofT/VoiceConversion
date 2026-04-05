#!/bin/bash
#SBATCH --job-name=knnvc_infer
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

module load python/3.10
module load cuda/11.8

source ../.env/bin/activate

# Convert all pre-surgery test files
python scripts/inference.py \
    --input_dir /home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Tonsill/Speech/1 \
    --output_dir ../knn_vc_converted
