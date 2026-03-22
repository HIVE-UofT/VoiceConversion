#!/bin/bash
#SBATCH --job-name=unet_vc
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

module load python/3.10
module load cuda/11.8

source ../.env/bin/activate

# Step 1: Train U-Net feature transform
python scripts/train.py

# Step 2: Convert all pre-surgery files
python scripts/inference.py \
    --input_dir /home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Tonsill/Speech/1 \
    --output_dir converted

# Step 3: Evaluate
python scripts/evaluate.py \
    --converted_dir converted \
    --method_name "UNet-VC" \
    --skip_f0
