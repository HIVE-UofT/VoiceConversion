#!/bin/bash
#SBATCH --job-name=unet_vc_fess
#SBATCH --account=def-zshakeri
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out

module load python/3.10

source ../.env/bin/activate

# Step 1: Train U-Net on Fess data
python scripts/train.py \
    --pre_dir /home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Fess/Speech/1 \
    --post_dir /home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Fess/Speech/2 \
    --output checkpoints_fess

# Step 2: Convert all pre-surgery files
python scripts/inference.py \
    --input_dir /home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Fess/Speech/1 \
    --output_dir converted_fess \
    --checkpoint checkpoints_fess/best_model.pt

# Step 3: Evaluate
python scripts/evaluate.py \
    --converted_dir converted_fess \
    --pre_dir /home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Fess/Speech/1 \
    --post_dir /home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Fess/Speech/2 \
    --method_name "UNet-VC (Fess)" \
    --skip_f0
